import os
import math
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
        "hourly": "temperature_2m,relative_humidity_2m,shortwave_radiation_instant,wind_speed_100m",
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
                solars = data.get("shortwave_radiation_instant", data.get("surface_solar_radiation", []))
                winds = data.get("wind_speed_100m", [])

                for i in range(len(times)):
                    t_val = temps[i] if i < len(temps) and temps[i] is not None else 25.0
                    rh_val = rhs[i] if i < len(rhs) and rhs[i] is not None else 50.0
                    sol_val = solars[i] if i < len(solars) and solars[i] is not None else 0.0
                    wind_val = winds[i] if i < len(winds) and winds[i] is not None else 5.0
                    record = RawWeatherRecord(
                        timestamp_utc=times[i],
                        region=region,
                        temperature_2m=float(t_val),
                        relative_humidity_2m=float(rh_val),
                        surface_solar_radiation=float(sol_val),
                        wind_speed_100m=float(wind_val),
                    )
                    records.append(record)
                return records
            logger.warning(f"open-meteo responded with status {resp.status_code}: {resp.text[:100]}")
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                break
        except Exception as exc:
            logger.warning(f"weather request attempt {attempt + 1} failed for {region.value}: {exc}")

        if attempt < max_retries - 1:
            backoff_sec = (2 ** attempt) * 2.0
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
    # repairs temporal discontinuities using linear interpolation or diurnal forward-fill
    if not records:
        return records

    sorted_records = sorted(records, key=lambda r: r.timestamp_utc)
    repaired: List[RawGridRecord] = [sorted_records[0]]

    for i in range(1, len(sorted_records)):
        prev = repaired[-1]
        curr = sorted_records[i]
        diff_hours = int((curr.timestamp_utc - prev.timestamp_utc).total_seconds() // 3600)

        if diff_hours > 1:
            gap = diff_hours - 1
            if gap <= 3:
                # linear interpolation for small drops
                step_demand = (curr.demand_mw - prev.demand_mw) / diff_hours
                for h in range(1, diff_hours):
                    interp_time = prev.timestamp_utc + timedelta(hours=h)
                    interp_demand = round(prev.demand_mw + step_demand * h, 2)
                    repaired.append(
                        RawGridRecord(
                            timestamp_utc=interp_time,
                            region=prev.region,
                            demand_mw=interp_demand,
                            frequency_hz=50.0,
                        )
                    )
            elif gap <= 6:
                # forward fill from identical diurnal cycle 24 hours prior
                for h in range(1, diff_hours):
                    fill_time = prev.timestamp_utc + timedelta(hours=h)
                    # locate t - 24h record if available in history
                    prior_target = fill_time - timedelta(hours=24)
                    prior_record = next(
                        (r for r in repaired if r.timestamp_utc == prior_target),
                        None,
                    )
                    fill_demand = prior_record.demand_mw if prior_record else prev.demand_mw
                    repaired.append(
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

        repaired.append(curr)
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
        conn = psycopg2.connect(db_url)
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


def run_ingestion_pipeline() -> TelemetryPayload:
    # coordinates collection, mitigation, and storage across all 6 regions
    batch_id = str(uuid.uuid4())
    logger.info(f"initializing telemetry ingestion batch: {batch_id}")

    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_utc = now_utc - timedelta(hours=168)

    all_grid_records: List[RawGridRecord] = []
    all_weather_records: List[RawWeatherRecord] = []

    for region in GridRegion:
        logger.info(f"harvesting telemetry for region: {region.value}")
        lat, lon = REGIONAL_COORDINATES[region.value]

        # 1. pull live weather across 7+ days lookback for 168h rolling lags
        weather = fetch_weather_with_retry(lat, lon, region, past_days=7, forecast_days=1)
        all_weather_records.extend(weather)

        # 2. generate ist-aligned, weather-elastic simulated grid records
        raw_grid = fetch_live_grid_records(region, start_utc, now_utc, weather_records=weather)

        # 3. apply gap mitigation
        mitigated_grid = apply_gap_mitigation(raw_grid)
        all_grid_records.extend(mitigated_grid)

    db_url = os.environ.get("SUPABASE_DB_URL")
    if not all_weather_records and db_url:
        raise RuntimeError("weather ingestion failed for all regions; aborting database persistence")

    payload = TelemetryPayload(
        ingestion_batch_id=batch_id,
        grid_records=all_grid_records,
        weather_records=all_weather_records,
    )

    if db_url:
        logger.info("persisting records to supabase database...")
        persist_telemetry_to_supabase(all_grid_records, all_weather_records, db_url)
    else:
        logger.info("SUPABASE_DB_URL not configured. dry run completed successfully.")

    return payload


if __name__ == "__main__":
    run_ingestion_pipeline()