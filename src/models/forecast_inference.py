import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import numpy as np
import polars as pl

from src.config import Settings, get_settings
from src.db.client import DatabaseClient, get_db_client
from src.features.pipeline import FeaturePipeline
from src.models.quantile_regressor import QuantileRegressor
from src.schemas.telemetry import ForecastRecord

logger = logging.getLogger(__name__)


class ForecastEngine:
    def __init__(
        self,
        regressor: QuantileRegressor | None = None,
        model_dir: Path | str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_dir = Path(model_dir or self.settings.model_store_path / "forecast")
        self.regressor = regressor
        if self.regressor is None and (self.model_dir / "meta.json").exists():
            self.regressor = QuantileRegressor.load(self.model_dir)

        self.pipeline = FeaturePipeline(drop_nulls=False)

    def generate_72h_forecast(
        self,
        historical_df: pl.DataFrame,
        region: str = "NR",
        horizon_hours: int = 72,
    ) -> list[ForecastRecord]:
        if self.regressor is None:
            raise RuntimeError("quantile regressor is not loaded or fitted")

        # filter and sort historical observations for target region
        cleaned_region = region.strip().upper()
        reg_df = historical_df.filter(pl.col("region") == cleaned_region)

        # parse string timestamps if necessary
        if reg_df.schema["timestamp"] == pl.String:
            reg_df = reg_df.with_columns(
                pl.col("timestamp").str.to_datetime(time_zone="UTC")
            )

        reg_df = reg_df.sort("timestamp").tail(240)

        if reg_df.height < 170:
            raise ValueError(
                f"insufficient history for region {region}: requires at least 170 rows, got {reg_df.height}"
            )

        raw_origin = reg_df["timestamp"][-1]
        if isinstance(raw_origin, str):
            origin_time = datetime.fromisoformat(raw_origin)
        else:
            origin_time = raw_origin

        if origin_time.tzinfo is None:
            origin_time = origin_time.replace(tzinfo=timezone.utc)

        buffer_df = reg_df.clone()
        feature_cols = self.regressor.feature_names
        forecast_records: list[ForecastRecord] = []

        # recursive multi-step forecasting loop over target horizon
        for step in range(1, horizon_hours + 1):
            target_time = origin_time + timedelta(hours=step)

            # append prospective row with placeholder target value
            dummy_row = pl.DataFrame(
                {
                    "timestamp": [target_time],
                    "region": [cleaned_region],
                    "metric_value": [float(buffer_df["metric_value"][-1])],
                }
            )

            window_df = pl.concat([buffer_df.tail(239), dummy_row])
            features_df = self.pipeline.transform(window_df)
            feature_vector = features_df.select(feature_cols).row(-1)
            X_step = np.array([feature_vector], dtype=np.float64)

            # predict quantiles with guaranteed non-crossing order
            preds = self.regressor.predict(X_step)
            p10_val = float(preds["p10"][0])
            p50_val = float(preds["p50"][0])
            p90_val = float(preds["p90"][0])

            record = ForecastRecord(
                forecast_origin=origin_time,
                target_timestamp=target_time,
                region=cleaned_region,
                p10=p10_val,
                p50=p50_val,
                p90=p90_val,
                model_version="v1.0",
            )
            forecast_records.append(record)

            # update buffer with median prediction as realized step
            simulated_row = pl.DataFrame(
                {
                    "timestamp": [target_time],
                    "region": [cleaned_region],
                    "metric_value": [p50_val],
                }
            )
            buffer_df = pl.concat([buffer_df.tail(239), simulated_row])

        return forecast_records

    def generate_and_save(
        self,
        region: str = "NR",
        horizon_hours: int = 72,
        db_client: DatabaseClient | None = None,
    ) -> list[ForecastRecord]:
        db = db_client or get_db_client(self.settings)

        # load recent telemetry from database or fetch fallback
        recent = db.get_recent_telemetry(region, limit=240)
        if len(recent) < 170:
            from src.ingestion.fetcher import TelemetryFetcher

            fetcher = TelemetryFetcher(self.settings)
            fresh_records = fetcher.fetch_latest(region=region, limit=240)
            db.upsert_telemetry(fresh_records)
            recent = db.get_recent_telemetry(region, limit=240)

        # reverse to chronological order
        chronological = list(reversed(recent))
        df = pl.DataFrame(
            {
                "timestamp": [r["timestamp"] for r in chronological],
                "region": [r["region"] for r in chronological],
                "metric_value": [r["metric_value"] for r in chronological],
            }
        )

        forecasts = self.generate_72h_forecast(
            historical_df=df,
            region=region,
            horizon_hours=horizon_hours,
        )
        db.upsert_forecasts(forecasts)
        logger.info(
            "persisted %d forecast records for region %s to %s",
            len(forecasts),
            region,
            db.mode,
        )
        return forecasts