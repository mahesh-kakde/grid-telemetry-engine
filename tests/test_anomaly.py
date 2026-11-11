from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import polars as pl
import pytest

from src.db.client import DatabaseClient
from src.features.pipeline import FeaturePipeline
from src.ingestion.fetcher import TelemetryFetcher
from src.models.anomaly_detector import AnomalyDetector
from src.models.anomaly_runner import AnomalyRunner
from src.models.explainability import ExplainabilityEngine


def test_anomaly_detector_flags_outlier():
    np.random.seed(42)
    # 300 normal samples with slight noise
    X = np.random.normal(loc=50.0, scale=2.0, size=(300, 5))
    # inject extreme anomalous stress event at index 150
    X[150] = [200.0, 250.0, 300.0, 180.0, 220.0]

    detector = AnomalyDetector(contamination=0.02)
    detector.fit(X, feature_names=["f0", "f1", "f2", "f3", "f4"])
    anomaly_mask, scores = detector.score(X)

    # the injected spike must be flagged as anomaly
    assert anomaly_mask[150] is np.True_ or anomaly_mask[150] is True
    assert scores[150] > 0.5
    # typical inlier must have lower score
    assert scores[0] < scores[150]


def test_anomaly_detector_save_load(tmp_path: Path):
    X = np.random.normal(loc=10.0, scale=1.0, size=(100, 4))
    detector = AnomalyDetector(contamination=0.03)
    detector.fit(X, feature_names=["a", "b", "c", "d"])

    mask_orig, scores_orig = detector.score(X)
    save_dir = tmp_path / "models" / "anomaly"
    detector.save(save_dir)

    loaded = AnomalyDetector.load(save_dir)
    mask_loaded, scores_loaded = loaded.score(X)

    assert np.array_equal(mask_orig, mask_loaded)
    assert np.allclose(scores_orig, scores_loaded)


def test_explainability_engine_top_k():
    X = np.random.normal(loc=0.0, scale=1.0, size=(200, 6))
    feature_names = [
        "diff_1h",
        "diff_24h",
        "roll_zscore_24h",
        "roll_std_24h",
        "sin_24h",
        "cos_24h",
    ]

    # inject extreme outlier driven primarily by diff_24h and roll_zscore_24h
    X[100] = [1.0, 50.0, 40.0, 1.2, 0.5, -0.5]

    detector = AnomalyDetector(contamination=0.02)
    detector.fit(X, feature_names=feature_names)

    explainer = ExplainabilityEngine(model=detector.model, feature_names=feature_names)
    top_features = explainer.explain_instance(X[100], top_k=3)

    assert len(top_features) == 3
    # top attributions must include dominant drivers
    top_keys = list(top_features.keys())
    assert "diff_24h" in top_keys or "roll_zscore_24h" in top_keys


def test_batch_explain_only_evaluates_flagged():
    X = np.random.normal(loc=0.0, scale=1.0, size=(50, 4))
    feature_names = ["f0", "f1", "f2", "f3"]
    detector = AnomalyDetector(contamination=0.05)
    detector.fit(X, feature_names=feature_names)

    mask = np.zeros(50, dtype=bool)
    mask[5] = True
    mask[25] = True

    explainer = ExplainabilityEngine(model=detector.model, feature_names=feature_names)
    results = explainer.batch_explain(X, anomaly_mask=mask, top_k=2)

    assert len(results) == 50
    # unflagged rows must be empty dicts to conserve storage and compute
    assert results[0] == {}
    assert results[10] == {}
    # flagged rows must contain exactly 2 top features
    assert len(results[5]) == 2
    assert len(results[25]) == 2


def test_anomaly_runner_e2e_persistence(tmp_path: Path):
    db_file = tmp_path / "anomaly_test.db"
    model_dir = tmp_path / "models" / "anomaly"
    db_client = DatabaseClient(mode="sqlite", sqlite_path=db_file)

    fetcher = TelemetryFetcher()
    records = fetcher.fetch_latest(region="NR", limit=300)

    # inject a synthetic heatwave peak load surge at row 250
    records[250].metric_value = 85000.0
    db_client.upsert_telemetry(records)

    runner = AnomalyRunner(
        db_client=db_client,
        model_dir=model_dir,
    )

    res = runner.run(region="NR", limit=300, fit_new=True)
    assert res["status"] == "success"
    assert res["total_evaluated"] > 0
    assert res["anomalies_flagged"] >= 1

    stored = db_client.get_recent_anomalies("NR", limit=150)
    assert len(stored) > 0
    # verify at least one flagged anomaly exists with shap explanations
    flagged = [r for r in stored if r["is_anomaly"]]
    assert len(flagged) >= 1
    assert isinstance(flagged[0]["top_shap_features"], dict)
    assert len(flagged[0]["top_shap_features"]) > 0