import os
import math
import random
import time
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
import requests
import holidays

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None
    execute_values = None

from src.ingestion.schemas import (
    GridRegion,
    RawGridRecord,
    RawWeatherRecord,
    TelemetryPayload,
)

from src.logging_config import configure_logging

logger = logging.getLogger(__name__)

# regional coordinates for meteorological centroids
REGIONAL_COORDINATES: Dict[str, Tuple[float, float]] = {
    GridRegion.NR.value: (28.6139, 77.2090),      # delhi
    GridRegion.WR.value: (19.0760, 72.8777),      # mumbai
    GridRegion.SR.value: (12.9716, 77.5946),      # bengaluru
    GridRegion.ER.value: (22.5726, 88.3639),      # kolkata
    GridRegion.NER.value: (26.1445, 91.7362),     # guwahati
    GridRegion.ALL_INDIA.value: (21.1458, 79.0882) # nagpur geographic center
}

# regional demand baselines in mw calibrated to posoco dec 2024 operational scale
REGIONAL_DEMAND_BASELINES: Dict[str, float] = {
    GridRegion.NR.value: 43440.9,
    GridRegion.WR.value: 44473.3,
    GridRegion.SR.value: 42095.7,
    GridRegion.ER.value: 15251.0,
    GridRegion.NER.value: 2491.6,
    GridRegion.ALL_INDIA.value: 147752.5,
}

DIURNAL_LOAD_FACTORS = [
    0.78, 0.74, 0.72, 0.71, 0.73, 0.78,
    0.86, 0.94, 0.99, 1.02, 1.04, 1.03,
    1.01, 0.99, 0.98, 0.99, 1.02, 1.05,
    1.09, 1.10, 1.06, 1.00, 0.92, 0.84,
]
_DIURNAL_MEAN = sum(DIURNAL_LOAD_FACTORS) / len(DIURNAL_LOAD_FACTORS)
NORMALIZED_DIURNAL_FACTORS = [f / _DIURNAL_MEAN for f in DIURNAL_LOAD_FACTORS]
IST_OFFSET = timedelta(hours=5, minutes=30)


def send_alert_webhook(error_msg: str, region: Optional[str] = None) -> None:
    # dispatch failure payload to external monitoring webhook if configured
    webhook_url = os.environ.get("ALERT_WEBHOOK_URL")
    if not webhook_url:
        return
    payload = {
        "event": "ingestion_failure",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "region": region or "ALL",
        "error": error_msg,
        "severity": "HIGH",
    }
    try:
        requests.post(webhook_url, json=payload, timeout=5.0)
    except Exception as exc:
        logger.warning(f"failed to send alert webhook: {exc}")


def fetch_weather_with_retry(
    lat: float,
    lon: float,
    region: GridRegion,
    past_days: int = 7,
    forecast_days: int = 1,
) -> List[RawWeatherRecord]:
    # pulls meteorological telemetry from open-meteo forecast api with exponential backoff
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": past_days,
        "forecast_days": forecast_days,
        "hourly": "temperature_2m,relative_humidity_2m,shortwave_radiation,wind_speed_100m",
        "timezone": "UTC",
    }
    records: List[RawWeatherRecord] = []
    max_retries = 3

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=15.0)
            if resp.status_code == 200:
                data = resp.json().get("hourly", {})
                times = data.get("time", [])
                temps = data.get("temperature_2m", [])
                rhs = data.get("relative_humidity_2m", [])
                solars = data.get("shortwave_radiation", data.get("shortwave_radiation_instant", []))
                winds = data.get("wind_speed_100m", [])
                null_fields = 0

                for i in range(len(times)):
                    raw_t = temps[i] if i < len(temps) else None
                    raw_rh = rhs[i] if i < len(rhs) else None
                    raw_sol = solars[i] if i < len(solars) else None
                    raw_wind = winds[i] if i < len(winds) else None
                    null_fields += sum(v is None for v in (raw_t, raw_rh, raw_sol, raw_wind))

                    t_val = raw_t if raw_t is not None else 25.0
                    rh_val = raw_rh if raw_rh is not None else 50.0
                    sol_val = raw_sol if raw_sol is not None else 0.0
                    wind_val = raw_wind if raw_wind is not None else 5.0
                    record = RawWeatherRecord(
                        timestamp_utc=times[i],
                        region=region,
                        temperature_2m=float(t_val),
                        relative_humidity_2m=float(rh_val),
                        surface_solar_radiation=float(sol_val),
                        wind_speed_100m=float(wind_val),
                    )
                    records.append(record)
                if null_fields > 0:
                    logger.warning(
                        f"open-meteo returned {null_fields} null weather value(s) for {region.value}; "
                        "applied default fallback values"
                    )
                return records
            logger.warning(f"open-meteo responded with status {resp.status_code}: {resp.text[:100]}")
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                break
        except Exception as exc:
            logger.warning(f"weather request attempt {attempt + 1} failed for {region.value}: {exc}")

        if attempt < max_retries - 1:
            backoff_sec = (2 ** attempt) * 2.0 + random.uniform(0.0, 1.0)
            time.sleep(backoff_sec)

    send_alert_webhook(f"exhausted retries fetching weather for {region.value}", region.value)
    return records


def _compute_weather_elasticity(weather: Optional[RawWeatherRecord], is_holiday_or_sunday: bool) -> Tuple[float, float]:
    # returns (demand_multiplier, simulated_frequency_hz) driven by exogenous weather & calendar
    cal_adj = -0.04 if is_holiday_or_sunday else 0.0
    if weather is None:
        mult = max(0.85, min(1.18, 1.0 + cal_adj))
        return mult, 50.0

    temp = weather.temperature_2m
    rh = weather.relative_humidity_2m
    solar = weather.surface_solar_radiation

    cooling = max(0.0, temp - 24.0) * 0.008
    heating = max(0.0, 15.0 - temp) * 0.004
    humidity = (max(0.0, rh - 60.0) / 10.0) * 0.0015 if temp > 26.0 else 0.0
    solar_btm = max(0.0, solar - 200.0) * 0.00004
    wind_relief = max(0.0, weather.wind_speed_100m - 12.0) * 0.001

    mult = max(0.85, min(1.20, 1.0 + cooling + heating + humidity - solar_btm - wind_relief + cal_adj))
    freq = round(max(49.85, min(50.15, 50.0 - 0.15 * (mult - 1.0))), 3)
    return mult, freq


def fetch_live_grid_records(
    region: GridRegion,
    start_time: datetime,
    end_time: datetime,
    weather_records: Optional[List[RawWeatherRecord]] = None,
) -> List[RawGridRecord]:
    # generates ist-aligned weather-elastic telemetry calibrated to posoco regional scale
    records: List[RawGridRecord] = []
    curr = start_time
    base = REGIONAL_DEMAND_BASELINES[region.value]
    weather_map = {
        w.timestamp_utc.replace(minute=0, second=0, microsecond=0): w
        for w in (weather_records or [])
    }

    ist_years = sorted({(start_time + IST_OFFSET).year, (end_time + IST_OFFSET).year})
    ind_holidays = set(holidays.India(years=ist_years).keys())

    while curr <= end_time:
        curr_utc = curr if curr.tzinfo is not None else curr.replace(tzinfo=timezone.utc)
        ist_dt = curr_utc.astimezone(timezone.utc) + IST_OFFSET
        ist_hour_float = ist_dt.hour + ist_dt.minute / 60.0
        idx_lo = int(math.floor(ist_hour_float)) % 24
        idx_hi = (idx_lo + 1) % 24
        frac = ist_hour_float - math.floor(ist_hour_float)
        factor = (
            NORMALIZED_DIURNAL_FACTORS[idx_lo] * (1.0 - frac)
            + NORMALIZED_DIURNAL_FACTORS[idx_hi] * frac
        )

        is_off_day = (ist_dt.weekday() == 6) or (ist_dt.date() in ind_holidays)
        matched_weather = weather_map.get(curr_utc.replace(minute=0, second=0, microsecond=0))
        elastic_mult, freq = _compute_weather_elasticity(matched_weather, is_off_day)

        demand = round(max(100.0, base * factor * elastic_mult), 2)
        records.append(
            RawGridRecord(
                timestamp_utc=curr_utc,
                region=region,
                demand_mw=demand,
                frequency_hz=freq,
            )
        )
        curr += timedelta(hours=1)
    return records


def apply_gap_mitigation(
    records: List[RawGridRecord],
) -> List[RawGridRecord]:
    # repairs temporal discontinuities per region using linear interpolation or diurnal forward-fill
    if not records:
        return records

    sorted_records = sorted(records, key=lambda r: (r.region.value, r.timestamp_utc))
    repaired: List[RawGridRecord] = [sorted_records[0]]
    repaired_by_key: Dict[Tuple[GridRegion, datetime], RawGridRecord] = {
        (sorted_records[0].region, sorted_records[0].timestamp_utc): sorted_records[0]
    }

    def _append_repaired(rec: RawGridRecord) -> None:
        repaired.append(rec)
        repaired_by_key[(rec.region, rec.timestamp_utc)] = rec

    for i in range(1, len(sorted_records)):
        prev = repaired[-1]
        curr = sorted_records[i]
        if curr.region != prev.region:
            _append_repaired(curr)
            continue

        elapsed_sec = (curr.timestamp_utc - prev.timestamp_utc).total_seconds()
        if elapsed_sec == 0:
            repaired[-1] = curr
            repaired_by_key[(curr.region, curr.timestamp_utc)] = curr
            continue

        diff_hours = int(elapsed_sec // 3600)

        if diff_hours > 1:
            gap = diff_hours - 1
            if gap <= 3:
                # linear interpolation for small drops
                step_demand = (curr.demand_mw - prev.demand_mw) / diff_hours
                for h in range(1, diff_hours):
                    interp_time = prev.timestamp_utc + timedelta(hours=h)
                    interp_demand = round(prev.demand_mw + step_demand * h, 2)
                    _append_repaired(
                        RawGridRecord(
                            timestamp_utc=interp_time,
                            region=prev.region,
                            demand_mw=interp_demand,
                            frequency_hz=50.0,
                        )
                    )
            elif gap <= 6:
                # forward fill from identical diurnal cycle 24 hours prior within the same region
                for h in range(1, diff_hours):
                    fill_time = prev.timestamp_utc + timedelta(hours=h)
                    prior_target = fill_time - timedelta(hours=24)
                    prior_record = repaired_by_key.get((prev.region, prior_target))
                    fill_demand = prior_record.demand_mw if prior_record else prev.demand_mw
                    _append_repaired(
                        RawGridRecord(
                            timestamp_utc=fill_time,
                            region=prev.region,
                            demand_mw=fill_demand,
                            frequency_hz=50.0,
                        )
                    )
            else:
                send_alert_webhook(
                    f"critical packet drop exceeding 6 hours in {prev.region.value}",
                    prev.region.value,
                )

        _append_repaired(curr)
    return repaired


def persist_telemetry_to_supabase(
    grid_records: List[RawGridRecord],
    weather_records: List[RawWeatherRecord],
    db_url: str,
) -> int:
    # idempotent upsert into regional_telemetry using on conflict clause
    if not grid_records:
        return 0

    if psycopg2 is None:
        raise ImportError("psycopg2 is required for database persistence. Install psycopg2-binary.")

    # index weather records by (region, timestamp_utc)
    weather_map = {
        (w.region.value, w.timestamp_utc): w for w in weather_records
    }

    rows_to_insert = []
    for g in grid_records:
        w = weather_map.get((g.region.value, g.timestamp_utc))
        rows_to_insert.append((
            g.timestamp_utc,
            g.region.value,
            g.demand_mw,
            g.frequency_hz,
            w.temperature_2m if w else None,
            w.relative_humidity_2m if w else None,
            w.surface_solar_radiation if w else None,
            w.wind_speed_100m if w else None,
        ))

    upsert_sql = """
        INSERT INTO regional_telemetry (
            timestamp_utc,
            region,
            demand_mw,
            frequency_hz,
            temperature_2m,
            relative_humidity_2m,
            surface_solar_radiation,
            wind_speed_100m
        ) VALUES %s
        ON CONFLICT (region, timestamp_utc) DO UPDATE SET
            demand_mw = EXCLUDED.demand_mw,
            frequency_hz = COALESCE(EXCLUDED.frequency_hz, regional_telemetry.frequency_hz),
            temperature_2m = COALESCE(EXCLUDED.temperature_2m, regional_telemetry.temperature_2m),
            relative_humidity_2m = COALESCE(EXCLUDED.relative_humidity_2m, regional_telemetry.relative_humidity_2m),
            surface_solar_radiation = COALESCE(EXCLUDED.surface_solar_radiation, regional_telemetry.surface_solar_radiation),
            wind_speed_100m = COALESCE(EXCLUDED.wind_speed_100m, regional_telemetry.wind_speed_100m);
    """

    try:
        conn = psycopg2.connect(db_url, connect_timeout=10)
    except Exception as exc:
        logger.error(
            f"failed to connect to supabase database: {exc}. "
            "verify SUPABASE_DB_URL secret format (e.g. postgresql://postgres:[PASSWORD]@[HOST]:5432/postgres?sslmode=require)"
        )
        raise

    try:
        with conn.cursor() as cur:
            execute_values(cur, upsert_sql, rows_to_insert)
        conn.commit()
        logger.info(f"successfully upserted {len(rows_to_insert)} records to supabase")
        return len(rows_to_insert)
    except Exception as exc:
        logger.error(
            f"failed to upsert telemetry records: {exc}. "
            "if table does not exist, execute scripts/setup_database.sql in the supabase sql editor."
        )
        raise
    finally:
        conn.close()


def _aggregate_all_india_records(
    regional_records: Dict[str, List[RawGridRecord]],
) -> List[RawGridRecord]:
    # sums the 5 regional rldc grids so ALL_INDIA strictly conserves regional totals
    sub_regions = [
        GridRegion.NR.value,
        GridRegion.WR.value,
        GridRegion.SR.value,
        GridRegion.ER.value,
        GridRegion.NER.value,
    ]
    by_region_ts = {
        reg: {r.timestamp_utc: r for r in regional_records.get(reg, [])}
        for reg in sub_regions
    }
    common_timestamps = sorted(set.intersection(*(set(m.keys()) for m in by_region_ts.values())))
    aggregated: List[RawGridRecord] = []
    for ts in common_timestamps:
        total_mw = sum(by_region_ts[reg][ts].demand_mw for reg in sub_regions)
        weighted_freq_num = sum(
            by_region_ts[reg][ts].demand_mw * (by_region_ts[reg][ts].frequency_hz or 50.0)
            for reg in sub_regions
        )
        avg_freq = round(weighted_freq_num / max(total_mw, 1e-6), 3)
        aggregated.append(
            RawGridRecord(
                timestamp_utc=ts,
                region=GridRegion.ALL_INDIA,
                demand_mw=round(total_mw, 2),
                frequency_hz=max(47.5, min(52.5, avg_freq)),
            )
        )
    return aggregated


def run_ingestion_pipeline() -> TelemetryPayload:
    # coordinates collection, mitigation, and storage across all 6 regions
    batch_id = str(uuid.uuid4())
    logger.info(f"initializing telemetry ingestion batch: {batch_id}")

    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    # fetch 14 days (336h) so 168h feature warmup leaves a full 168h post-warmup window
    start_utc = now_utc - timedelta(hours=336)

    db_url = os.environ.get("SUPABASE_DB_URL")
    regional_grid_map: Dict[str, List[RawGridRecord]] = {}
    all_grid_records: List[RawGridRecord] = []
    all_weather_records: List[RawWeatherRecord] = []
    failed_weather_regions: set[str] = set()

    sub_regions = [
        GridRegion.NR,
        GridRegion.WR,
        GridRegion.SR,
        GridRegion.ER,
        GridRegion.NER,
    ]
    for region in sub_regions + [GridRegion.ALL_INDIA]:
        logger.info(f"harvesting telemetry for region: {region.value}")
        lat, lon = REGIONAL_COORDINATES[region.value]

        # 1. pull live weather across 14 days lookback for 168h warmup + 168h drift window
        weather = fetch_weather_with_retry(lat, lon, region, past_days=14, forecast_days=1)
        if len(weather) < 168:
            failed_weather_regions.add(region.value)
        all_weather_records.extend(weather)

        # 2. generate ist-aligned, weather-elastic simulated grid records
        if region == GridRegion.ALL_INDIA and all(
            r.value not in failed_weather_regions for r in sub_regions
        ) and all(len(regional_grid_map.get(r.value, [])) > 0 for r in sub_regions):
            raw_grid = _aggregate_all_india_records(regional_grid_map)
        else:
            raw_grid = fetch_live_grid_records(region, start_utc, now_utc, weather_records=weather)

        # 3. apply gap mitigation
        mitigated_grid = apply_gap_mitigation(raw_grid)
        regional_grid_map[region.value] = mitigated_grid
        all_grid_records.extend(mitigated_grid)

    if not all_weather_records and db_url:
        raise RuntimeError("weather ingestion failed for all regions; aborting database persistence")

    # skip persistence for any region lacking live weather (and ALL_INDIA if a sub-region failed)
    skip_persist_regions = set(failed_weather_regions)
    if any(r.value in failed_weather_regions for r in sub_regions):
        skip_persist_regions.add(GridRegion.ALL_INDIA.value)

    persist_grid_records = [
        r for r in all_grid_records if r.region.value not in skip_persist_regions
    ]
    persist_weather_records = [
        w for w in all_weather_records if w.region.value not in skip_persist_regions
    ]

    payload = TelemetryPayload(
        ingestion_batch_id=batch_id,
        grid_records=persist_grid_records if db_url else all_grid_records,
        weather_records=persist_weather_records if db_url else all_weather_records,
    )

    if db_url:
        if skip_persist_regions:
            logger.warning(
                f"excluding regions from database upsert due to weather failure: {sorted(skip_persist_regions)}"
            )
        if not persist_grid_records:
            raise RuntimeError("no valid regional telemetry records available for persistence")
        logger.info("persisting records to supabase database...")
        persist_telemetry_to_supabase(persist_grid_records, persist_weather_records, db_url)
    else:
        logger.info("SUPABASE_DB_URL not configured. dry run completed successfully.")

    return payload


if __name__ == "__main__":
    configure_logging()
    run_ingestion_pipeline()