import pytest
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError

from src.ingestion.schemas import (
    GridRegion,
    RawGridRecord,
    RawWeatherRecord,
    TelemetryPayload,
)
from src.ingestion.fetch_telemetry import (
    apply_gap_mitigation,
    send_alert_webhook,
)


def test_raw_grid_record_valid():
    # asserts valid telemetry instantiation with utc coercion
    now_utc = datetime.now(timezone.utc)
    record = RawGridRecord(
        timestamp_utc=now_utc,
        region=GridRegion.NR,
        demand_mw=50000.0,
        frequency_hz=50.02,
    )
    assert record.region == GridRegion.NR
    assert record.demand_mw == 50000.0
    assert record.frequency_hz == 50.02
    assert record.timestamp_utc.tzinfo == timezone.utc


def test_raw_grid_record_ist_coercion():
    # asserts unlocalized timestamp is coerced from ist to utc (-5h30m)
    record = RawGridRecord(
        timestamp_utc="2026-05-15T12:00:00",
        region=GridRegion.WR,
        demand_mw=65000.0,
    )
    assert record.timestamp_utc.tzinfo == timezone.utc
    assert record.timestamp_utc.hour == 6
    assert record.timestamp_utc.minute == 30


def test_raw_grid_record_iso_z_coercion():
    # asserts iso 8601 string with zulu suffix parses correctly
    record = RawGridRecord(
        timestamp_utc="2026-05-15T06:30:00Z",
        region=GridRegion.ALL_INDIA,
        demand_mw=210000.0,
    )
    assert record.timestamp_utc.tzinfo == timezone.utc
    assert record.timestamp_utc.hour == 6
    assert record.timestamp_utc.minute == 30


def test_raw_grid_record_demand_boundaries():
    # asserts demand must strictly be positive and below physical grid capacity
    now_utc = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        RawGridRecord(timestamp_utc=now_utc, region=GridRegion.NR, demand_mw=0.0)

    with pytest.raises(ValidationError):
        RawGridRecord(timestamp_utc=now_utc, region=GridRegion.NR, demand_mw=360000.0)


def test_raw_grid_record_frequency_boundaries():
    # asserts grid frequency must remain within operational tripping bands
    now_utc = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        RawGridRecord(timestamp_utc=now_utc, region=GridRegion.SR, demand_mw=45000.0, frequency_hz=47.2)

    with pytest.raises(ValidationError):
        RawGridRecord(timestamp_utc=now_utc, region=GridRegion.SR, demand_mw=45000.0, frequency_hz=52.8)


def test_raw_weather_record_valid():
    # asserts valid meteorological observation parsing
    now_utc = datetime.now(timezone.utc)
    record = RawWeatherRecord(
        timestamp_utc=now_utc,
        region=GridRegion.ER,
        temperature_2m=28.5,
        relative_humidity_2m=65.0,
        surface_solar_radiation=450.0,
        wind_speed_100m=6.2,
    )
    assert record.temperature_2m == 28.5
    assert record.relative_humidity_2m == 65.0
    assert record.timestamp_utc.tzinfo == timezone.utc


def test_raw_weather_record_boundaries():
    # asserts physical boundary enforcement for meteorological telemetry
    now_utc = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        RawWeatherRecord(
            timestamp_utc=now_utc,
            region=GridRegion.ER,
            temperature_2m=65.0,
            relative_humidity_2m=50.0,
            surface_solar_radiation=100.0,
            wind_speed_100m=5.0,
        )

    with pytest.raises(ValidationError):
        RawWeatherRecord(
            timestamp_utc=now_utc,
            region=GridRegion.ER,
            temperature_2m=25.0,
            relative_humidity_2m=110.0,
            surface_solar_radiation=100.0,
            wind_speed_100m=5.0,
        )


def test_telemetry_payload_container():
    # asserts batch payload validation containing grid and weather records
    now_utc = datetime.now(timezone.utc)
    grid_r = RawGridRecord(timestamp_utc=now_utc, region=GridRegion.NER, demand_mw=3200.0)
    weather_r = RawWeatherRecord(
        timestamp_utc=now_utc,
        region=GridRegion.NER,
        temperature_2m=22.0,
        relative_humidity_2m=80.0,
        surface_solar_radiation=150.0,
        wind_speed_100m=3.5,
    )
    payload = TelemetryPayload(
        ingestion_batch_id="test-batch-001",
        grid_records=[grid_r],
        weather_records=[weather_r],
    )
    assert payload.ingestion_batch_id == "test-batch-001"
    assert len(payload.grid_records) == 1
    assert len(payload.weather_records) == 1


def test_gap_mitigation_linear_interpolation():
    # asserts linear interpolation for contiguous gaps of 3 hours or less
    t0 = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)
    r1 = RawGridRecord(timestamp_utc=t0, region=GridRegion.NR, demand_mw=50000.0)
    r2 = RawGridRecord(timestamp_utc=t0 + timedelta(hours=3), region=GridRegion.NR, demand_mw=53000.0)

    mitigated = apply_gap_mitigation([r1, r2])
    assert len(mitigated) == 4
    # hourly steps should be evenly spaced by 1000 mw
    assert mitigated[1].demand_mw == 51000.0
    assert mitigated[2].demand_mw == 52000.0
    assert mitigated[1].timestamp_utc == t0 + timedelta(hours=1)
    assert mitigated[2].timestamp_utc == t0 + timedelta(hours=2)


def test_gap_mitigation_diurnal_forward_fill():
    # asserts diurnal cycle forward fill for gaps between 4 and 6 hours
    t0 = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    records = []
    # populate full prior 24 hours
    for h in range(24):
        records.append(
            RawGridRecord(
                timestamp_utc=t0 + timedelta(hours=h),
                region=GridRegion.WR,
                demand_mw=60000.0 + (h * 100.0),
            )
        )
    # introduce a 5-hour gap on the next day
    gap_start = t0 + timedelta(hours=24)
    gap_end = t0 + timedelta(hours=30)
    records.append(RawGridRecord(timestamp_utc=gap_start, region=GridRegion.WR, demand_mw=65000.0))
    records.append(RawGridRecord(timestamp_utc=gap_end, region=GridRegion.WR, demand_mw=66000.0))

    mitigated = apply_gap_mitigation(records)
    # 24 initial + 1 gap_start + 5 missing + 1 gap_end = 31 records
    assert len(mitigated) == 31
    # verify t-24h retrieval for missing records
    fill_record = next(r for r in mitigated if r.timestamp_utc == t0 + timedelta(hours=26))
    prior_record = next(r for r in records if r.timestamp_utc == t0 + timedelta(hours=2))
    assert fill_record.demand_mw == prior_record.demand_mw


def test_send_alert_webhook_disabled(monkeypatch):
    # asserts webhook dispatch gracefully handles missing configuration
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    # should execute without raising exceptions
    send_alert_webhook("test failure message", "NR")


def test_weather_elasticity_and_ist_diurnal_peak():
    from datetime import datetime, timezone
    from src.ingestion.schemas import GridRegion, RawWeatherRecord
    from src.ingestion.fetch_telemetry import fetch_live_grid_records

    # Morning IST peak (~10:30 IST = 05:00 UTC) vs pre-dawn IST trough (~03:30 IST = 22:00 UTC prior day)
    t_trough = datetime(2024, 3, 12, 22, 0, tzinfo=timezone.utc)
    t_peak = datetime(2024, 3, 13, 5, 0, tzinfo=timezone.utc)

    cool_wx = [
        RawWeatherRecord(
            timestamp_utc=t_trough, region=GridRegion.NR,
            temperature_2m=22.0, relative_humidity_2m=45.0,
            surface_solar_radiation=0.0, wind_speed_100m=4.0,
        ),
        RawWeatherRecord(
            timestamp_utc=t_peak, region=GridRegion.NR,
            temperature_2m=22.0, relative_humidity_2m=45.0,
            surface_solar_radiation=0.0, wind_speed_100m=4.0,
        ),
    ]
    recs_cool = fetch_live_grid_records(GridRegion.NR, t_trough, t_peak, weather_records=cool_wx)
    by_ts_cool = {r.timestamp_utc: r.demand_mw for r in recs_cool}
    assert by_ts_cool[t_peak] > by_ts_cool[t_trough]

    hot_wx = [
        RawWeatherRecord(
            timestamp_utc=t_peak, region=GridRegion.NR,
            temperature_2m=40.0, relative_humidity_2m=75.0,
            surface_solar_radiation=100.0, wind_speed_100m=4.0,
        ),
    ]
    recs_hot = fetch_live_grid_records(GridRegion.NR, t_peak, t_peak, weather_records=hot_wx)
    assert recs_hot[0].demand_mw > by_ts_cool[t_peak]


def test_run_ingestion_pipeline_requests_14_day_backfill(monkeypatch):
    from src.ingestion import fetch_telemetry as ft

    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    captured_past_days = []

    def fake_fetch_weather(lat, lon, region, past_days=7, forecast_days=1):
        captured_past_days.append(past_days)
        return []

    monkeypatch.setattr(ft, "fetch_weather_with_retry", fake_fetch_weather)
    payload = ft.run_ingestion_pipeline()
    assert captured_past_days == [14] * len(GridRegion)
    # 337 hours per region * 6 regions = 2022 records (168h warmup + 169h post-warmup)
    assert len(payload.grid_records) == 337 * len(GridRegion)