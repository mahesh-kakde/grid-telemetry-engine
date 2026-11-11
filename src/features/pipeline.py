import logging
import sys
from typing import Any
import polars as pl

from src.features.lag_features import get_lag_expressions
from src.features.rolling_features import (
    get_base_rolling_expressions,
    get_derived_rolling_expressions,
)
from src.features.time_features import get_time_expressions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("feature_pipeline")


class FeaturePipeline:
    def __init__(
        self,
        target_col: str = "metric_value",
        timestamp_col: str = "timestamp",
        region_col: str = "region",
        lag_hours: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        drop_nulls: bool = True,
    ) -> None:
        self.target_col = target_col
        self.timestamp_col = timestamp_col
        self.region_col = region_col
        self.lag_hours = lag_hours or [1, 2, 3, 24, 48, 168]
        self.rolling_windows = rolling_windows or [6, 24, 168]
        self.drop_nulls = drop_nulls

    def _prepare_lazy_frame(self, df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
        lf = df.lazy() if isinstance(df, pl.DataFrame) else df

        # parse iso timestamp strings to timezone-aware utc datetime if necessary
        schema = lf.collect_schema()
        ts_type = schema.get(self.timestamp_col)
        if ts_type == pl.String:
            lf = lf.with_columns(
                pl.col(self.timestamp_col).str.to_datetime(time_zone="UTC")
            )

        # ensure deterministic temporal ordering per region
        return lf.sort([self.region_col, self.timestamp_col])

    def transform_lazy(self, df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
        lf = self._prepare_lazy_frame(df)

        time_exprs = get_time_expressions(self.timestamp_col)
        lag_exprs = get_lag_expressions(
            target_col=self.target_col,
            group_col=self.region_col,
            lags=self.lag_hours,
        )
        base_roll_exprs = get_base_rolling_expressions(
            target_col=self.target_col,
            group_col=self.region_col,
            windows=self.rolling_windows,
        )

        all_base_exprs = time_exprs + lag_exprs + base_roll_exprs
        transformed = lf.with_columns(all_base_exprs)

        # derived interaction ratios after base rolling metrics exist
        derived_exprs = get_derived_rolling_expressions(
            target_col=self.target_col,
            windows=self.rolling_windows,
        )
        if derived_exprs:
            transformed = transformed.with_columns(derived_exprs)

        if self.drop_nulls:
            # prune warm-up burn-in rows where 168h lags are undefined
            transformed = transformed.drop_nulls()

        return transformed

    def transform(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        lazy_pipeline = self.transform_lazy(df)
        return lazy_pipeline.collect()

    def get_feature_names(self, include_target: bool = False) -> list[str]:
        # generate empty mock frame to infer exact feature column headers
        mock_df = pl.DataFrame(
            {
                self.timestamp_col: ["2026-11-10T00:00:00+00:00"],
                self.region_col: ["NR"],
                self.target_col: [50000.0],
            }
        )
        transformed = self.transform_lazy(mock_df)
        cols = transformed.collect_schema().names()

        excluded = {self.timestamp_col, self.region_col, "id", "created_at", "metric_type"}
        if not include_target:
            excluded.add(self.target_col)

        return [c for c in cols if c not in excluded]


def run_demo_pipeline() -> dict[str, Any]:
    from src.ingestion.fetcher import TelemetryFetcher

    logger.info("fetching sample telemetry records for feature extraction")
    fetcher = TelemetryFetcher()
    records = fetcher.fetch_latest(region="ALL", limit=240)

    raw_dicts = [
        {
            "timestamp": r.timestamp.isoformat(),
            "region": r.region,
            "metric_value": r.metric_value,
        }
        for r in records
    ]
    raw_df = pl.DataFrame(raw_dicts)

    pipeline = FeaturePipeline(drop_nulls=True)
    features_df = pipeline.transform(raw_df)
    feature_cols = pipeline.get_feature_names()

    logger.info(
        "pipeline complete: raw rows=%d -> feature rows=%d, feature cols=%d",
        raw_df.height,
        features_df.height,
        len(feature_cols),
    )
    return {
        "raw_count": raw_df.height,
        "feature_count": features_df.height,
        "num_features": len(feature_cols),
    }


def main() -> int:
    try:
        run_demo_pipeline()
        return 0
    except Exception as exc:
        logger.error("pipeline execution failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())