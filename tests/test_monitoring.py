import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.db.client import DatabaseClient
from src.monitoring.drift_suite import DriftSuite
from src.monitoring.drift_runner import DriftRunner
from src.monitoring.report_exporter import export_drift_report, extract_drift_summary
from src.schemas.telemetry import DriftRecord


@pytest.fixture
def temp_db(tmp_path: Path) -> DatabaseClient:
    db_file = tmp_path / "test_drift.db"
    settings = Settings(
        supabase_url=None,
        supabase_key=None,
        sqlite_db_path=db_file,
        model_store_path=tmp_path / "models",
    )
    client = DatabaseClient(settings)
    return client


def test_identical_distributions_no_drift() -> None:
    data = {
        "metric_value": [10.0, 20.0, 30.0, 40.0, 50.0] * 20,
        "lag_24h": [100.0, 200.0, 300.0, 400.0, 500.0] * 20,
    }
    ref_df = pd.DataFrame(data)
    curr_df = pd.DataFrame(data)

    suite = DriftSuite(drift_share_threshold=0.5)
    result = suite.run(ref_df, curr_df)

    assert result.drift_share == 0.0
    assert result.dataset_drift is False
    assert result.number_of_drifted_columns == 0
    assert result.number_of_columns == 2
    for feat_info in result.drifted_features.values():
        assert feat_info["drift_detected"] is False


def test_shifted_distribution_detected() -> None:
    ref_df = pd.DataFrame({
        "metric_value": [10.0, 20.0, 30.0, 40.0, 50.0] * 20,
        "lag_24h": [100.0, 200.0, 300.0, 400.0, 500.0] * 20,
    })
    curr_df = pd.DataFrame({
        "metric_value": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0] * 20,
        "lag_24h": [10000.0, 20000.0, 30000.0, 40000.0, 50000.0] * 20,
    })

    suite = DriftSuite(drift_share_threshold=0.5)
    result = suite.run(ref_df, curr_df)

    assert result.drift_share == 1.0
    assert result.dataset_drift is True
    assert result.number_of_drifted_columns == 2
    for feat_info in result.drifted_features.values():
        assert feat_info["drift_detected"] is True


def test_partial_feature_drift() -> None:
    ref_df = pd.DataFrame({
        "metric_value": [10.0, 20.0, 30.0, 40.0, 50.0] * 20,
        "stable_feature": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
    })
    curr_df = pd.DataFrame({
        "metric_value": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0] * 20,
        "stable_feature": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
    })

    suite = DriftSuite(drift_share_threshold=0.5)
    result = suite.run(ref_df, curr_df)

    assert result.drifted_features["metric_value"]["drift_detected"] is True
    assert result.drifted_features["stable_feature"]["drift_detected"] is False
    assert result.number_of_drifted_columns == 1
    assert result.drift_share == 0.5
    assert result.dataset_drift is True


def test_export_drift_report(tmp_path: Path) -> None:
    df = pd.DataFrame({"load": [10.0, 20.0, 30.0, 40.0, 50.0] * 10})
    suite = DriftSuite()
    result = suite.run(df, df)

    report_path = export_drift_report(result.report, reports_dir=tmp_path / "reports")
    assert report_path.exists()
    assert report_path.stat().st_size > 1000
    assert report_path.suffix == ".html"

    summary = extract_drift_summary(result)
    assert "drift_share" in summary
    assert "dataset_drift" in summary
    assert summary["dataset_drift"] is False


def test_drift_runner_sqlite_persistence(temp_db: DatabaseClient, tmp_path: Path) -> None:
    runner = DriftRunner(
        settings=temp_db.settings,
        db_client=temp_db,
        reports_dir=tmp_path / "drift_reports",
    )

    # run monitoring for 2 days ref and 1 day curr to keep test fast
    record, report_path, suite_result = runner.run(
        region="NR",
        reference_days=2,
        current_days=1,
        drift_share_threshold=0.5,
        save_db=True,
    )

    assert isinstance(record, DriftRecord)
    assert report_path.exists()
    assert report_path.stat().st_size > 1000
    assert 0.0 <= record.drift_share <= 1.0

    # verify persistence and retrieval from sqlite
    latest = temp_db.get_latest_drift()
    assert latest is not None
    assert latest["id"] == str(record.id)
    assert latest["drift_share"] == record.drift_share
    assert isinstance(latest["drifted_features"], dict)
    assert len(latest["drifted_features"]) > 0