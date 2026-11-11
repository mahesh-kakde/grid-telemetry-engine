from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import polars as pl
import pytest

from src.db.client import DatabaseClient
from src.ingestion.fetcher import TelemetryFetcher
from src.models.forecast_dataset import (
    ForecastDataset,
    compute_coverage,
    compute_mase,
    compute_pinball_loss,
    compute_wape,
)
from src.models.forecast_inference import ForecastEngine
from src.models.quantile_regressor import QuantileRegressor
from src.models.train_forecast import evaluate_walk_forward


def test_metrics_calculations():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])
    wape = compute_wape(y_true, y_pred)
    assert abs(wape - (20.0 / 600.0 * 100.0)) < 1e-4

    p10 = np.array([90.0, 180.0, 280.0])
    p90 = np.array([120.0, 220.0, 320.0])
    cov = compute_coverage(y_true, p10, p90)
    assert cov == 100.0

    p_loss = compute_pinball_loss(y_true, y_pred, 0.50)
    assert p_loss > 0.0


def test_quantile_regressor_monotonicity():
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.maximum(100.0, 1000.0 + X[:, 0] * 50.0 + np.random.randn(200) * 20.0)

    reg = QuantileRegressor(n_estimators=30)
    reg.fit(X, y)

    X_test = np.random.randn(50, 10)
    preds = reg.predict(X_test)

    assert "p10" in preds and "p50" in preds and "p90" in preds
    p10, p50, p90 = preds["p10"], preds["p50"], preds["p90"]

    assert len(p10) == 50
    assert np.all(p10 >= 0.0)
    assert np.all(p10 <= p50)
    assert np.all(p50 <= p90)


def test_quantile_regressor_save_load(tmp_path: Path):
    X = np.random.randn(100, 5)
    y = 500.0 + X[:, 0] * 30.0

    reg = QuantileRegressor(n_estimators=20)
    reg.fit(X, y, feature_names=["f0", "f1", "f2", "f3", "f4"])
    preds_before = reg.predict(X[:10])

    model_dir = tmp_path / "models" / "forecast"
    reg.save(model_dir)

    loaded_reg = QuantileRegressor.load(model_dir)
    preds_after = loaded_reg.predict(X[:10])

    assert np.allclose(preds_before["p10"], preds_after["p10"])
    assert np.allclose(preds_before["p50"], preds_after["p50"])
    assert np.allclose(preds_before["p90"], preds_after["p90"])


def test_walk_forward_evaluation_acceptance_criteria():
    fetcher = TelemetryFetcher()
    records = fetcher.fetch_latest(region="NR", limit=1000)

    df = pl.DataFrame(
        {
            "timestamp": [r.timestamp.isoformat() for r in records],
            "region": [r.region for r in records],
            "metric_value": [r.metric_value for r in records],
        }
    )

    results = evaluate_walk_forward(df, n_splits=5, test_window=72, n_estimators=60)

    # acceptance criteria: wape < 8.0%, coverage in [70%, 90%]
    assert results["mean_wape"] < 8.0
    assert 70.0 <= results["mean_coverage"] <= 90.0
    assert results["mean_mase"] < 1.0


def test_recursive_72h_forecast_engine(tmp_path: Path):
    fetcher = TelemetryFetcher()
    records = fetcher.fetch_latest(region="WR", limit=400)
    df = pl.DataFrame(
        {
            "timestamp": [r.timestamp.isoformat() for r in records],
            "region": [r.region for r in records],
            "metric_value": [r.metric_value for r in records],
        }
    )

    ds = ForecastDataset()
    feat_df = ds.extract_features(df)
    X, y, fcols = ds.get_feature_target_arrays(feat_df)

    reg = QuantileRegressor(n_estimators=40)
    reg.fit(X, y, feature_names=fcols)

    engine = ForecastEngine(regressor=reg)
    forecasts = engine.generate_72h_forecast(df, region="WR", horizon_hours=72)

    assert len(forecasts) == 72
    origin = forecasts[0].forecast_origin
    for i, fc in enumerate(forecasts, 1):
        assert fc.region == "WR"
        assert fc.target_timestamp == origin + timedelta(hours=i)
        assert fc.p10 <= fc.p50 <= fc.p90
        assert fc.p10 > 0.0


def test_forecast_engine_db_persistence(tmp_path: Path):
    db_file = tmp_path / "forecast_test.db"
    db_client = DatabaseClient(mode="sqlite", sqlite_path=db_file)

    fetcher = TelemetryFetcher()
    records = fetcher.fetch_latest(region="SR", limit=300)
    df = pl.DataFrame(
        {
            "timestamp": [r.timestamp.isoformat() for r in records],
            "region": [r.region for r in records],
            "metric_value": [r.metric_value for r in records],
        }
    )

    ds = ForecastDataset()
    feat_df = ds.extract_features(df)
    X, y, fcols = ds.get_feature_target_arrays(feat_df)
    reg = QuantileRegressor(n_estimators=30)
    reg.fit(X, y, feature_names=fcols)

    engine = ForecastEngine(regressor=reg)
    forecasts = engine.generate_72h_forecast(df, region="SR", horizon_hours=72)
    assert len(forecasts) == 72

    db_client.upsert_forecasts(forecasts)
    stored = db_client.get_latest_forecasts("SR", limit=100)
    assert len(stored) == 72

    # verify idempotent upsert
    db_client.upsert_forecasts(forecasts)
    stored_again = db_client.get_latest_forecasts("SR", limit=100)
    assert len(stored_again) == 72