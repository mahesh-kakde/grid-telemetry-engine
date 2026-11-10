import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import httpx
import pytest

from src.db.client import DatabaseClient
from src.ingestion.fetcher import TelemetryFetcher
from src.ingestion.runner import IngestionRunner


def test_fetcher_fallback_series_count_and_types():
    fetcher = TelemetryFetcher()
    now = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)
    records = fetcher._generate_fallback_series("NR", now, 24)

    assert len(records) == 24
    for r in records:
        assert r.region == "NR"
        assert r.metric_type == "DEMAND_MW"
        assert r.metric_value > 20000.0
        assert r.timestamp.tzinfo == timezone.utc


def test_fetcher_all_regions():
    fetcher = TelemetryFetcher()
    records = fetcher.fetch_latest(region="ALL", limit=6)
    assert len(records) == 5 * 6
    found_regions = {r.region for r in records}
    assert found_regions == {"NR", "WR", "SR", "ER", "NER"}


def test_fetcher_rejects_invalid_region():
    fetcher = TelemetryFetcher()
    with pytest.raises(ValueError, match="unsupported region"):
        fetcher.fetch_latest(region="INVALID_ZONE", limit=12)


def test_fetcher_mocked_api_success():
    fetcher = TelemetryFetcher()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "records": [
            {
                "timestamp": "2026-11-10T12:00:00Z",
                "region": "NR",
                "metric_value": 47520.0,
                "metric_type": "DEMAND_MW",
            }
        ]
    }

    with patch("httpx.Client.get", return_value=mock_response):
        records = fetcher.fetch_latest(
            region="NR",
            limit=1,
            endpoint_url="https://api.example.com/telemetry",
        )
        assert len(records) == 1
        assert records[0].metric_value == 47520.0
        assert records[0].region == "NR"


def test_fetcher_fallback_on_api_failure():
    fetcher = TelemetryFetcher(max_retries=2, backoff_factor=0.01)

    # mock http error raising exception on every call
    with patch(
        "httpx.Client.get",
        side_effect=httpx.ConnectError("network unreachable"),
    ):
        records = fetcher.fetch_latest(
            region="WR",
            limit=12,
            endpoint_url="https://api.example.com/unreachable",
        )
        # should fall back to synthetic series without failing
        assert len(records) == 12
        assert records[0].region == "WR"


def test_ingestion_runner_dry_run():
    runner = IngestionRunner()
    result = runner.run(region="NR", limit=12, dry_run=True)
    assert result["status"] == "success"
    assert result["count"] == 12
    assert result["upserted"] == 0
    assert result["dry_run"] is True


def test_ingestion_runner_persists_to_db(tmp_path: Path):
    db_file = tmp_path / "telemetry_test.db"
    db_client = DatabaseClient(mode="sqlite", sqlite_path=db_file)
    runner = IngestionRunner(db_client=db_client)

    result = runner.run(region="SR", limit=24, dry_run=False)
    assert result["status"] == "success"
    assert result["upserted"] == 24

    stored = db_client.get_recent_telemetry("SR", limit=50)
    assert len(stored) == 24

    # re-run to verify idempotent upsert
    re_result = runner.run(region="SR", limit=24, dry_run=False)
    assert re_result["status"] == "success"
    assert re_result["upserted"] == 24

    stored_again = db_client.get_recent_telemetry("SR", limit=50)
    assert len(stored_again) == 24


def test_ingestion_runner_export_json(tmp_path: Path):
    out_file = tmp_path / "export.json"
    runner = IngestionRunner()
    result = runner.run(region="NER", limit=6, dry_run=True, export_path=out_file)

    assert result["status"] == "success"
    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 6
    assert data[0]["region"] == "NER"