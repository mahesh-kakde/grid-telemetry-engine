import numpy as np
import polars as pl

from src.features.pipeline import FeaturePipeline


def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # weighted absolute percentage error in percentage
    total_actual = float(np.sum(y_true))
    if total_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual * 100.0)


def compute_coverage(y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> float:
    # empirical coverage of 80% prediction interval
    within_interval = (y_true >= p10) & (y_true <= p90)
    return float(np.mean(within_interval) * 100.0)


def compute_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    # pinball / quantile loss for given quantile alpha
    diff = y_true - y_pred
    loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)
    return float(np.mean(loss))


def compute_mase(y_true: np.ndarray, y_pred: np.ndarray, seasonality: int = 24) -> float:
    # mean absolute scaled error against seasonal lag baseline
    n = len(y_true)
    if n <= seasonality:
        return 1.0
    scale = float(np.mean(np.abs(y_true[seasonality:] - y_true[:-seasonality])))
    if scale == 0:
        return 0.0
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return float(mae / scale)


class ForecastDataset:
    def __init__(
        self,
        target_col: str = "metric_value",
        timestamp_col: str = "timestamp",
        region_col: str = "region",
    ) -> None:
        self.target_col = target_col
        self.timestamp_col = timestamp_col
        self.region_col = region_col
        self.pipeline = FeaturePipeline(
            target_col=target_col,
            timestamp_col=timestamp_col,
            region_col=region_col,
            drop_nulls=True,
        )

    def extract_features(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        return self.pipeline.transform(df)

    def get_feature_target_arrays(
        self,
        df_features: pl.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        feature_cols = self.pipeline.get_feature_names(include_target=False)
        X = df_features.select(feature_cols).to_numpy()
        y = df_features[self.target_col].to_numpy()
        return X, y, feature_cols

    def get_walk_forward_splits(
        self,
        df_features: pl.DataFrame,
        n_splits: int = 5,
        test_window: int = 72,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        # temporal walk-forward split indices ensuring test is strictly after train
        total_rows = df_features.height
        min_train_rows = total_rows - (n_splits * test_window)
        if min_train_rows < test_window:
            raise ValueError(
                f"insufficient rows ({total_rows}) for {n_splits} splits of window {test_window}"
            )

        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(n_splits):
            train_end = min_train_rows + (i * test_window)
            test_end = train_end + test_window
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)
            splits.append((train_idx, test_idx))

        return splits