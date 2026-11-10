from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.config import Settings
from src.db.client import DatabaseClient
from src.schemas.telemetry import (
    AnomalyRecord,
    DriftRecord,
    ForecastRecord,
    RegionEnum,
    TelemetryRecord,
)


def test_telemetry_record_valid():
    ts = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)
    record = TelemetryRecord(
        timestamp=ts,
        region="NR",
        metric_value=48210.5,
    )
    assert record.region == "NR"
    assert record.metric_value == 48210.5
    assert record.metric_type == "DEMAND_MW"
    assert record.timestamp.tzinfo == timezone.utc
    assert record.id is not None


def test_telemetry_record_rejects_negative_or_zero_value():
    ts = datetime.now(timezone.utc)
    with pytest.raises(ValidationError, match="metric_value must be strictly positive"):
        TelemetryRecord(timestamp=ts, region="NR", metric_value=0.0)

    with pytest.raises(ValidationError, match="metric_value must be strictly positive"):
        TelemetryRecord(timestamp=ts, region="NR", metric_value=-150.0)


def test_telemetry_record_rejects_invalid_region():
    ts = datetime.now(timezone.utc)
    with pytest.raises(ValidationError, match="invalid region"):
        TelemetryRecord(timestamp=ts, region="INVALID_ZONE", metric_value=12000.0)


def test_telemetry_record_naive_datetime_conversion():
    naive_ts = datetime(2026, 11, 10, 8, 30, 0)
    record = TelemetryRecord(timestamp=naive_ts, region="wr", metric_value=32100.0)
    assert record.timestamp.tzinfo == timezone.utc
    assert record.region == "WR"


def test_forecast_record_valid_and_bounds():
    origin = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)
    target = origin + timedelta(hours=1)
    record = ForecastRecord(
        forecast_origin=origin,
        target_timestamp=target,
        region="SR",
        p10=38000.0,
        p50=41000.0,
        p90=44500.0,
    )
    assert record.p10 <= record.p50 <= record.p90
    assert record.region == "SR"


def test_forecast_record_rejects_target_before_origin():
    origin = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)
    target = origin - timedelta(hours=1)
    with pytest.raises(ValidationError, match="target_timestamp must be strictly after"):
        ForecastRecord(
            forecast_origin=origin,
            target_timestamp=target,
            region="NR",
            p10=100.0,
            p50=200.0,
            p90=300.0,
        )


def test_forecast_record_rejects_quantile_crossing():
    origin = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)
    target = origin + timedelta(hours=1)
    with pytest.raises(ValidationError, match="quantile crossing detected"):
        ForecastRecord(
            forecast_origin=origin,
            target_timestamp=target,
            region="NR",
            p10=45000.0,
            p50=41000.0,
            p90=40000.0,
        )


def test_anomaly_record_valid():
    ts = datetime.now(timezone.utc)
    record = AnomalyRecord(
        timestamp=ts,
        region="ER",
        metric_value=25000.0,
        anomaly_score=0.88,
        is_anomaly=True,
        top_shap_features={"lag_24h_diff": 3.42, "rolling_std_7d": 1.15},
    )
    assert record.is_anomaly is True
    assert "lag_24h_diff" in record.top_shap_features


def test_drift_record_validation():
    ref_start = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    ref_end = datetime(2026, 10, 31, 23, 59, 59, tzinfo=timezone.utc)
    curr_start = datetime(2026, 11, 1, 0, 0, 0, tzinfo=timezone.utc)
    curr_end = datetime(2026, 11, 8, 0, 0, 0, tzinfo=timezone.utc)

    record = DriftRecord(
        reference_window_start=ref_start,
        reference_window_end=ref_end,
        current_window_start=curr_start,
        current_window_end=curr_end,
        drift_share=0.15,
        drifted_features={"lag_24h": {"p_value": 0.01}},
    )
    assert record.drift_share == 0.15

    with pytest.raises(ValidationError, match="reference_window_end cannot precede"):
        DriftRecord(
            reference_window_start=ref_end,
            reference_window_end=ref_start,
            current_window_start=curr_start,
            current_window_end=curr_end,
            drift_share=0.15,
        )


def test_db_client_sqlite_crud(tmp_path: Path):
    db_file = tmp_path / "test_telemetry.db"
    client = DatabaseClient(mode="sqlite", sqlite_path=db_file)

    now = datetime(2026, 11, 10, 10, 0, 0, tzinfo=timezone.utc)
    records = [
        TelemetryRecord(
            timestamp=now - timedelta(hours=i),
            region="NR",
            metric_value=40000.0 + i * 100,
        )
        for i in range(5)
    ]

    inserted = client.upsert_telemetry(records)
    assert inserted == 5

    # test idempotent upsert on conflict
    duplicate_record = TelemetryRecord(
        timestamp=now,
        region="NR",
        metric_value=49999.0,
    )
    client.upsert_telemetry([duplicate_record])

    recent = client.get_recent_telemetry("NR", limit=10)
    assert len(recent) == 5
    # most recent record timestamp should have updated metric_value
    assert recent[0]["metric_value"] == 49999.0

    # test forecasts
    origin = now
    forecasts = [
        ForecastRecord(
            forecast_origin=origin,
            target_timestamp=origin + timedelta(hours=h),
            region="NR",
            p10=39000.0,
            p50=42000.0,
            p90=45000.0,
        )
        for h in range(1, 4)
    ]
    assert client.upsert_forecasts(forecasts) == 3
    fetched_forecasts = client.get_latest_forecasts("NR")
    assert len(fetched_forecasts) == 3

    # test anomalies
    anomalies = [
        AnomalyRecord(
            timestamp=now,
            region="NR",
            metric_value=55000.0,
            anomaly_score=0.92,
            is_anomaly=True,
            top_shap_features={"lag_24h": 4.1},
        )
    ]
    assert client.upsert_anomalies(anomalies) == 1
    fetched_anomalies = client.get_recent_anomalies("NR")
    assert len(fetched_anomalies) == 1
    assert fetched_anomalies[0]["is_anomaly"] is True
    assert fetched_anomalies[0]["top_shap_features"]["lag_24h"] == 4.1

    # test drift
    drift = DriftRecord(
        reference_window_start=now - timedelta(days=30),
        reference_window_end=now - timedelta(days=7),
        current_window_start=now - timedelta(days=7),
        current_window_end=now,
        drift_share=0.20,
        drifted_features={"lag_168h": {"drift_detected": True}},
        report_url="/reports/drift_report.html",
    )
    result = client.insert_drift(drift)
    assert result["drift_share"] == 0.20
    latest_drift = client.get_latest_drift()
    assert latest_drift is not None
    assert latest_drift["drift_share"] == 0.20
    assert latest_drift["drifted_features"]["lag_168h"]["drift_detected"] is True


def test_settings_initialization():
    settings = Settings()
    assert settings.telemetry_region == "NR"
    assert settings.batch_interval_hours == 6
    assert settings.forecast_horizon_hours == 72