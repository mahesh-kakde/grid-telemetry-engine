import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.ingestion.cleanup import (
    calculate_cutoff_timestamp,
    prune_expired_records,
)


def test_cleanup_cutoff_calculation():
    # asserts utc cutoff timestamp corresponds to specified retention days
    now_utc = datetime.now(timezone.utc)
    cutoff_90 = calculate_cutoff_timestamp(retention_days=90)
    expected_diff = now_utc - cutoff_90

    # allow small execution delta within 2 seconds
    assert abs(expected_diff.total_seconds() - 90 * 86400) < 2
    assert cutoff_90.tzinfo == timezone.utc

    cutoff_30 = calculate_cutoff_timestamp(retention_days=30)
    diff_30 = now_utc - cutoff_30
    assert abs(diff_30.total_seconds() - 30 * 86400) < 2


def test_cleanup_skipped_when_no_db_url(monkeypatch):
    # asserts graceful skipped status when database url is unconfigured
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    result = prune_expired_records(db_url=None)

    assert result["status"] == "skipped"
    assert result["reason"] == "missing_db_url"
    assert result["total_deleted"] == 0
    assert result["deleted_telemetry"] == 0


def test_cleanup_dry_run_with_mock():
    # asserts dry run queries counts without issuing delete statements or committing
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = (15,)

    with patch("psycopg2.connect", return_value=mock_conn):
        result = prune_expired_records(
            db_url="postgresql://mock:mock@localhost:5432/mock_db",
            retention_days=90,
            dry_run=True,
        )

    assert result["status"] == "success"
    assert result["dry_run"] is True
    assert result["deleted_telemetry"] == 15
    assert result["deleted_forecasts"] == 15
    assert result["deleted_anomalies"] == 15
    assert result["total_deleted"] == 45

    # verify executed queries are SELECT COUNT
    calls = mock_cursor.execute.call_args_list
    assert len(calls) == 3
    for call in calls:
        query = call[0][0].upper()
        assert "SELECT COUNT(*)" in query
        assert "DELETE" not in query

    mock_conn.commit.assert_not_called()
    mock_conn.close.assert_called_once()


def test_cleanup_execution_with_mock():
    # asserts active execution performs parameterized delete and commits transaction
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 25
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with patch("psycopg2.connect", return_value=mock_conn):
        result = prune_expired_records(
            db_url="postgresql://mock:mock@localhost:5432/mock_db",
            retention_days=90,
            dry_run=False,
        )

    assert result["status"] == "success"
    assert result["dry_run"] is False
    assert result["deleted_telemetry"] == 25
    assert result["deleted_forecasts"] == 25
    assert result["deleted_anomalies"] == 25
    assert result["total_deleted"] == 75

    # verify executed queries are DELETE FROM
    calls = mock_cursor.execute.call_args_list
    assert len(calls) == 3
    tables_checked = []
    for call in calls:
        query = call[0][0].upper()
        assert "DELETE FROM" in query
        if "REGIONAL_TELEMETRY" in query:
            tables_checked.append("regional_telemetry")
        elif "MODEL_FORECASTS" in query:
            tables_checked.append("model_forecasts")
        elif "TELEMETRY_ANOMALIES" in query:
            tables_checked.append("telemetry_anomalies")

    assert set(tables_checked) == {"regional_telemetry", "model_forecasts", "telemetry_anomalies"}
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()


def test_cleanup_sql_script_exists():
    # verifies standalone cleanup sql procedure exists and covers all target tables
    sql_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "cleanup_retention.sql")
    assert os.path.exists(sql_path), "cleanup_retention.sql must exist in scripts/"
    with open(sql_path, "r", encoding="utf-8") as f:
        content = f.read().lower()

    assert "delete from regional_telemetry" in content
    assert "delete from model_forecasts" in content
    assert "delete from telemetry_anomalies" in content
    assert "90" in content
    assert "retention" in content