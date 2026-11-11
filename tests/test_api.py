import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_db
from src.api.main import create_app
from src.config import Settings
from src.db.client import DatabaseClient
from src.schemas.telemetry import (
    AnomalyRecord,
    DriftRecord,
    ForecastRecord,
    TelemetryRecord,
)


@pytest.fixture
def test_setup(tmp_path: Path) -> tuple[TestClient, DatabaseClient, Settings]:
    db_file = tmp_path / "test_api.db"
    settings = Settings(
        supabase_url=None,
        supabase_key=None,
        sqlite_db_path=db_file,
        model_store_path=tmp_path / "models",
    )
    db = DatabaseClient(settings)

    app = create_app(settings=settings, db_client=db)
    app.dependency_overrides[get_db] = lambda: db

    client = TestClient(app)
    return client, db, settings


def test_root_and_health(test_setup: tuple[TestClient, DatabaseClient, Settings]) -> None:
    client, _, _ = test_setup

    resp_root = client.get("/")
    assert resp_root.status_code == 200
    data_root = resp_root.json()
    assert data_root["app"] == "Grid Telemetry Engine API"
    assert data_root["status"] == "healthy"
    assert data_root["domain"] == "https://grid.maheshkakde.com"

    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    data_health = resp_health.json()
    assert data_health["status"] == "ok"
    assert data_health["database_mode"] == "sqlite"


def test_telemetry_recent_and_historical(
    test_setup: tuple[TestClient, DatabaseClient, Settings]
) -> None:
    client, db, _ = test_setup
    now = datetime.now(timezone.utc).replace(microsecond=0)

    records = [
        TelemetryRecord(
            timestamp=now - timedelta(hours=i),
            region="NR",
            metric_value=45000.0 + i * 100,
            metric_type="demand",
        )
        for i in range(5)
    ]
    db.upsert_telemetry(records)

    # test recent endpoint
    resp = client.get("/api/telemetry/recent?region=NR&limit=5")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 5
    assert items[0]["region"] == "NR"

    # test historical endpoint
    resp_hist = client.get("/api/telemetry/historical?region=NR&limit=5")
    assert resp_hist.status_code == 200
    items_hist = resp_hist.json()
    assert len(items_hist) == 5
    # historical should be ascending order
    assert items_hist[0]["timestamp"] <= items_hist[-1]["timestamp"]


def test_invalid_region_returns_400(
    test_setup: tuple[TestClient, DatabaseClient, Settings]
) -> None:
    client, _, _ = test_setup
    resp = client.get("/api/telemetry/recent?region=INVALID_ZONE")
    assert resp.status_code == 400
    assert "invalid region" in resp.json()["detail"].lower()


def test_forecasts_latest(test_setup: tuple[TestClient, DatabaseClient, Settings]) -> None:
    client, db, _ = test_setup
    origin = datetime.now(timezone.utc).replace(microsecond=0)

    forecast_records = [
        ForecastRecord(
            forecast_origin=origin,
            target_timestamp=origin + timedelta(hours=step),
            region="NR",
            p10=40000.0 + step * 10,
            p50=45000.0 + step * 10,
            p90=50000.0 + step * 10,
            model_version="lgbm-qreg-v1.0",
        )
        for step in range(1, 4)
    ]
    db.upsert_forecasts(forecast_records)

    resp = client.get("/api/forecasts/latest?region=NR&limit=3")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    for f in data:
        assert f["region"] == "NR"
        assert f["p10"] <= f["p50"] <= f["p90"]
        assert f["model_version"] == "lgbm-qreg-v1.0"


def test_anomalies_recent_and_filtering(
    test_setup: tuple[TestClient, DatabaseClient, Settings]
) -> None:
    client, db, _ = test_setup
    now = datetime.now(timezone.utc).replace(microsecond=0)

    records = [
        AnomalyRecord(
            timestamp=now - timedelta(hours=1),
            region="NR",
            metric_value=75000.0,
            anomaly_score=0.92,
            is_anomaly=True,
            top_shap_features={"metric_value_roll_mean_24h": 0.45, "hour": 0.22},
        ),
        AnomalyRecord(
            timestamp=now - timedelta(hours=2),
            region="NR",
            metric_value=45000.0,
            anomaly_score=0.15,
            is_anomaly=False,
            top_shap_features={},
        ),
    ]
    db.upsert_anomalies(records)

    # fetch all
    resp_all = client.get("/api/anomalies/recent?region=NR")
    assert resp_all.status_code == 200
    assert len(resp_all.json()) == 2

    # fetch flagged only
    resp_flagged = client.get("/api/anomalies/recent?region=NR&anomalies_only=true")
    assert resp_flagged.status_code == 200
    flagged_data = resp_flagged.json()
    assert len(flagged_data) == 1
    assert flagged_data[0]["is_anomaly"] is True
    assert "metric_value_roll_mean_24h" in flagged_data[0]["top_shap_features"]


def test_drift_status_and_report(
    test_setup: tuple[TestClient, DatabaseClient, Settings], tmp_path: Path
) -> None:
    client, db, _ = test_setup

    # test empty status
    resp_empty = client.get("/api/drift/status")
    assert resp_empty.status_code == 200
    assert resp_empty.json()["status"] == "no_drift_recorded"

    # test 404 report
    resp_404 = client.get("/api/drift/report?report_path=non_existent_file.html")
    assert resp_404.status_code == 404

    # create a dummy html report file
    reports_dir = tmp_path / "reports" / "drift"
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_file = reports_dir / "drift_test.html"
    html_file.write_text("<html><body>Drift Report Content</body></html>", encoding="utf-8")

    now = datetime.now(timezone.utc)
    record = DriftRecord(
        run_timestamp=now,
        reference_window_start=now - timedelta(days=30),
        reference_window_end=now - timedelta(days=7),
        current_window_start=now - timedelta(days=7),
        current_window_end=now,
        drift_share=0.05,
        drifted_features={"metric_value": {"drift_detected": False, "drift_score": 0.8}},
        report_url=str(html_file),
    )
    db.insert_drift(record)

    resp_status = client.get("/api/drift/status")
    assert resp_status.status_code == 200
    status_data = resp_status.json()
    assert status_data["status"] == "ok"
    assert status_data["data"]["drift_share"] == 0.05

    resp_report = client.get("/api/drift/report")
    assert resp_report.status_code == 200
    assert "text/html" in resp_report.headers["content-type"]
    assert "Drift Report Content" in resp_report.text


def test_endpoint_latency_under_200ms(
    test_setup: tuple[TestClient, DatabaseClient, Settings]
) -> None:
    client, db, _ = test_setup
    now = datetime.now(timezone.utc)

    # populate sample records across all entities
    db.upsert_telemetry([
        TelemetryRecord(
            timestamp=now - timedelta(hours=i),
            region="ALL",
            metric_value=50000.0,
            metric_type="demand",
        )
        for i in range(24)
    ])
    db.upsert_forecasts([
        ForecastRecord(
            forecast_origin=now,
            target_timestamp=now + timedelta(hours=step),
            region="ALL",
            p10=45000.0,
            p50=50000.0,
            p90=55000.0,
        )
        for step in range(1, 10)
    ])

    endpoints = [
        "/health",
        "/api/telemetry/recent?region=ALL&limit=24",
        "/api/telemetry/historical?region=ALL&limit=24",
        "/api/forecasts/latest?region=ALL&limit=10",
        "/api/anomalies/recent?region=ALL&limit=10",
        "/api/drift/status",
    ]

    for ep in endpoints:
        start_time = time.perf_counter()
        resp = client.get(ep)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        assert resp.status_code == 200, f"failed on {ep}"
        assert latency_ms < 200.0, f"endpoint {ep} latency {latency_ms:.2f}ms exceeded 200ms limit"