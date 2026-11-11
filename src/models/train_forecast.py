import argparse
import logging
import sys
from pathlib import Path
from typing import Any
import numpy as np
import polars as pl

from src.config import Settings, get_settings
from src.ingestion.fetcher import TelemetryFetcher
from src.models.forecast_dataset import (
    ForecastDataset,
    compute_coverage,
    compute_mase,
    compute_pinball_loss,
    compute_wape,
)
from src.models.quantile_regressor import QuantileRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_forecast")


def evaluate_walk_forward(
    df: pl.DataFrame,
    n_splits: int = 5,
    test_window: int = 72,
    n_estimators: int = 80,
) -> dict[str, Any]:
    dataset = ForecastDataset()
    features_df = dataset.extract_features(df)
    X, y, feature_names = dataset.get_feature_target_arrays(features_df)
    splits = dataset.get_walk_forward_splits(
        features_df, n_splits=n_splits, test_window=test_window
    )

    fold_results: list[dict[str, float]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
        regressor = QuantileRegressor(n_estimators=n_estimators)
        regressor.fit(X[train_idx], y[train_idx], feature_names=feature_names)
        preds = regressor.predict(X[test_idx])

        y_test = y[test_idx]
        wape = compute_wape(y_test, preds["p50"])
        cov = compute_coverage(y_test, preds["p10"], preds["p90"])
        mase = compute_mase(y_test, preds["p50"])
        loss_p50 = compute_pinball_loss(y_test, preds["p50"], 0.50)

        fold_results.append(
            {
                "fold": fold_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "wape": wape,
                "coverage": cov,
                "mase": mase,
                "pinball_p50": loss_p50,
            }
        )
        logger.info(
            "fold %d/%d: train=%d, test=%d | wape=%.2f%%, coverage=%.1f%%, mase=%.3f",
            fold_idx,
            n_splits,
            len(train_idx),
            len(test_idx),
            wape,
            cov,
            mase,
        )

    avg_wape = float(np.mean([r["wape"] for r in fold_results]))
    avg_cov = float(np.mean([r["coverage"] for r in fold_results]))
    avg_mase = float(np.mean([r["mase"] for r in fold_results]))

    logger.info(
        "walk-forward evaluation complete: mean wape=%.2f%%, mean coverage=%.1f%%, mean mase=%.3f",
        avg_wape,
        avg_cov,
        avg_mase,
    )
    return {
        "folds": fold_results,
        "mean_wape": avg_wape,
        "mean_coverage": avg_cov,
        "mean_mase": avg_mase,
    }


def train_production_models(
    df: pl.DataFrame,
    model_dir: Path | str | None = None,
    settings: Settings | None = None,
    n_estimators: int = 150,
) -> QuantileRegressor:
    cfg = settings or get_settings()
    out_dir = Path(model_dir or cfg.model_store_path / "forecast")

    dataset = ForecastDataset()
    features_df = dataset.extract_features(df)
    X, y, feature_names = dataset.get_feature_target_arrays(features_df)

    logger.info(
        "fitting production quantile models on %d observations (%d features)",
        len(y),
        len(feature_names),
    )
    regressor = QuantileRegressor(n_estimators=n_estimators)
    regressor.fit(X, y, feature_names=feature_names)
    regressor.save(out_dir)
    logger.info("saved trained quantile models to %s", out_dir)

    return regressor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train LightGBM quantile regression models with walk-forward CV."
    )
    parser.add_argument("--region", type=str, default="NR", help="target electrical region")
    parser.add_argument("--limit", type=int, default=1000, help="number of historical records")
    parser.add_argument(
        "--evaluate-cv", action="store_true", help="run 5-fold temporal walk-forward evaluation"
    )
    parser.add_argument(
        "--save-models", action="store_true", help="train and save production models"
    )

    args = parser.parse_args()
    fetcher = TelemetryFetcher()
    records = fetcher.fetch_latest(region=args.region, limit=args.limit)

    df = pl.DataFrame(
        {
            "timestamp": [r.timestamp.isoformat() for r in records],
            "region": [r.region for r in records],
            "metric_value": [r.metric_value for r in records],
        }
    )

    if args.evaluate_cv:
        res = evaluate_walk_forward(df, n_splits=5, test_window=72)
        if res["mean_wape"] > 8.0:
            logger.error("walk-forward wape exceeded 8.0%% threshold: %.2f%%", res["mean_wape"])
            return 1

    if args.save_models or not args.evaluate_cv:
        train_production_models(df)

    return 0


if __name__ == "__main__":
    sys.exit(main())