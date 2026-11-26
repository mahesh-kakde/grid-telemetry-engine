import os
from datetime import datetime, timedelta, timezone
import pytest


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _read_repo_file(*parts: str) -> str:
    with open(os.path.join(ROOT_DIR, *parts), "r", encoding="utf-8") as f:
        return f.read()


def test_sql_schemas_and_container_ci_contracts():
    # verifies rls, constraints, aligned healthchecks, and clean runtime dependencies offline
    setup_sql = _read_repo_file("scripts", "setup_database.sql").lower()
    migrate_sql = _read_repo_file("scripts", "migrate_v2.sql").lower()
    cleanup_sql = _read_repo_file("scripts", "cleanup_retention.sql").lower()

    for table in ("regional_telemetry", "model_forecasts", "telemetry_anomalies", "precomputed_snapshots"):
        assert table in setup_sql
        assert f"alter table {table} enable row level security" in setup_sql
        assert f"{table} enable row level security" in migrate_sql

    assert "constraint uq_region_timestamp unique (region, timestamp_utc)" in setup_sql
    assert "constraint uq_forecast_vintage unique" in setup_sql
    assert "constraint uq_anomaly_region_time unique (region, timestamp_utc)" in setup_sql
    assert "delete from regional_telemetry" in cleanup_sql
    assert "delete from model_forecasts" in cleanup_sql
    assert "delete from telemetry_anomalies" in cleanup_sql

    req_txt = _read_repo_file("requirements.txt").lower()
    req_dev_txt = _read_repo_file("requirements-dev.txt").lower()
    dockerfile = _read_repo_file("Dockerfile")
    render_yaml = _read_repo_file("render.yaml")
    ingest_yml = _read_repo_file(".github", "workflows", "ingest.yml")

    assert "pytest" not in req_txt
    assert "pytest" in req_dev_txt
    assert "/_stcore/health" not in dockerfile
    assert "/health" in dockerfile
    assert "healthCheckPath: /health" in render_yaml
    assert "python -m src.monitoring.drift_check" not in ingest_yml
    assert "python -m src.models.precompute" in ingest_yml


def test_precompute_dispatches_consolidated_drift_webhook(monkeypatch):
    from unittest.mock import MagicMock, patch
    from src.models.precompute import _dispatch_drift_alert_webhook

    mock_post = MagicMock()
    mock_post.return_value.status_code = 200
    snapshots = [
        {"region": "ALL_INDIA", "drift_summary": {"drift_alarm": False, "drift_ratio": 0.1, "drifted_features": 1, "recommended_quantile_alpha": [0.1, 0.9]}},
        {"region": "NR", "drift_summary": {"drift_alarm": True, "drift_ratio": 0.42, "drifted_features": 5, "recommended_quantile_alpha": [0.05, 0.95]}},
    ]

    with patch("requests.post", mock_post):
        _dispatch_drift_alert_webhook(snapshots, alert_webhook_url="https://hooks.example/alert")

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["event"] == "dataset_drift_alert"
    assert [r["region"] for r in call_kwargs["json"]["regions"]] == ["NR"]


@pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES_URL"),
    reason="requires live PostgreSQL container via TEST_POSTGRES_URL",
)
def test_live_postgres_migrations_upserts_snapshots_and_retention():
    import psycopg2
    from src.ingestion.schemas import GridRegion, RawGridRecord, RawWeatherRecord
    from src.ingestion.fetch_telemetry import persist_telemetry_to_supabase
    from src.ingestion.cleanup import prune_expired_records
    from src.models.explainers import persist_anomaly_events_to_supabase
    from src.models.infer_and_detect import (
        backfill_forecast_actuals,
        fetch_telemetry_window_from_db,
        persist_forecasts_to_supabase,
    )
    from src.models.precompute import (
        load_drift_report_html_from_supabase,
        load_snapshot_from_supabase,
        persist_precomputed_snapshots_to_supabase,
    )

    db_url = os.environ["TEST_POSTGRES_URL"]
    setup_sql = _read_repo_file("scripts", "setup_database.sql")
    migrate_sql = _read_repo_file("scripts", "migrate_v2.sql")

    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            # run schema + v2 migration twice to prove idempotency
            cur.execute(setup_sql)
            cur.execute(migrate_sql)
            cur.execute(migrate_sql)
            conn.commit()

            cur.execute(
                """
                SELECT relname, relrowsecurity
                FROM pg_class
                WHERE relname IN ('regional_telemetry', 'model_forecasts', 'telemetry_anomalies', 'precomputed_snapshots')
                  AND relkind = 'r';
                """
            )
            rls_map = dict(cur.fetchall())
            assert rls_map == {
                "regional_telemetry": True,
                "model_forecasts": True,
                "telemetry_anomalies": True,
                "precomputed_snapshots": True,
            }
    finally:
        conn.close()

    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    stale_utc = now_utc - timedelta(days=120)

    # 1. telemetry upserts (conflict overwrite + retention test row)
    grid_recs = [
        RawGridRecord(timestamp_utc=now_utc, region=GridRegion.NR, demand_mw=42000.0, frequency_hz=50.0),
        RawGridRecord(timestamp_utc=stale_utc, region=GridRegion.NR, demand_mw=41000.0, frequency_hz=49.98),
    ]
    weather_recs = [
        RawWeatherRecord(
            timestamp_utc=now_utc,
            region=GridRegion.NR,
            temperature_2m=31.0,
            relative_humidity_2m=52.0,
            surface_solar_radiation=320.0,
            wind_speed_100m=4.2,
        )
    ]
    assert persist_telemetry_to_supabase(grid_recs, weather_recs, db_url) == 2
    assert persist_telemetry_to_supabase([grid_recs[0].model_copy(update={"demand_mw": 42500.0})], weather_recs, db_url) == 1

    window_df = fetch_telemetry_window_from_db("NR", hours=24, db_url=db_url)
    assert window_df is not None and len(window_df) == 2
    assert float(window_df["demand_mw"][-1]) == pytest.approx(42500.0)

    # 2. forecasts + realized demand backfill + anomalies upserts
    assert persist_forecasts_to_supabase(
        [
            {
                "forecast_generated_at": now_utc - timedelta(hours=1),
                "target_timestamp_utc": now_utc,
                "region": "NR",
                "forecast_p10": 41000.0,
                "forecast_p50": 42400.0,
                "forecast_p90": 43800.0,
                "actual_demand_mw": None,
                "model_version": "2.0.0",
            }
        ],
        db_url,
    ) == 1
    assert backfill_forecast_actuals(db_url) >= 1

    assert persist_anomaly_events_to_supabase(
        [
            {
                "timestamp_utc": now_utc,
                "region": "NR",
                "actual_demand_mw": 42500.0,
                "predicted_p10_mw": 41000.0,
                "predicted_p50_mw": 42400.0,
                "predicted_p90_mw": 43800.0,
                "anomaly_score": -0.02,
                "is_structural": False,
                "residual_breach": False,
                "top_shap_features": [{"feature": "demand_mw", "shap_value": 100.0, "actual_value": 42500.0}],
            }
        ],
        db_url,
    ) == 1

    # 3. precomputed snapshot upsert & round-trip load
    assert persist_precomputed_snapshots_to_supabase(
        [
            {
                "region": "NR",
                "model_version": "2.0.0",
                "generated_at": now_utc.isoformat(),
                "widened": False,
                "forecasts": [],
                "drift_report_html": "<html><body>Live DB Drift HTML</body></html>",
            }
        ],
        db_url,
    ) == 1
    loaded_snap = load_snapshot_from_supabase("NR", db_url=db_url)
    assert loaded_snap is not None and loaded_snap["snapshot_source"] == "supabase_live"
    assert "Live DB Drift HTML" in (load_drift_report_html_from_supabase("NR", db_url=db_url) or "")

    # 4. retention pruning: dry_run previews stale row; real run deletes stale row and keeps fresh row
    dry_res = prune_expired_records(db_url=db_url, retention_days=90, dry_run=True)
    assert dry_res["deleted_telemetry"] == 1
    live_res = prune_expired_records(db_url=db_url, retention_days=90, dry_run=False)
    assert live_res["deleted_telemetry"] == 1

    window_after = fetch_telemetry_window_from_db("NR", hours=24, db_url=db_url)
    assert window_after is not None and len(window_after) == 1