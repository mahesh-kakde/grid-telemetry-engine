import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import polars as pl

from src.config import Settings, get_settings
from src.db.client import DatabaseClient, get_db_client
from src.features.pipeline import FeaturePipeline
from src.ingestion.fetcher import TelemetryFetcher
from src.models.anomaly_detector import AnomalyDetector
from src.models.explainability import ExplainabilityEngine
from src.schemas.telemetry import AnomalyRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("anomaly_runner")


class AnomalyRunner:
    def __init__(
        self,
        settings: Settings | None = None,
        db_client: DatabaseClient | None = None,
        model_dir: Path | str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db = db_client or get_db_client(self.settings)
        self.model_dir = Path(model_dir or self.settings.model_store_path / "anomaly")
        self.pipeline = FeaturePipeline(drop_nulls=True)
        self.detector: AnomalyDetector | None = None
        self.explainer: ExplainabilityEngine | None = None

    def _ensure_models(
        self, X: np.ndarray, feature_names: list[str], fit_new: bool = False
    ) -> None:
        meta_file = self.model_dir / "anomaly_meta.json"
        if not fit_new and meta_file.exists():
            self.detector = AnomalyDetector.load(self.model_dir)
        else:
            self.detector = AnomalyDetector(contamination=0.02)
            self.detector.fit(X, feature_names=feature_names)
            self.detector.save(self.model_dir)

        self.explainer = ExplainabilityEngine(
            model=self.detector.model,
            feature_names=self.detector.feature_names,
        )

    def run(
        self,
        region: str = "NR",
        limit: int = 240,
        fit_new: bool = False,
    ) -> dict[str, Any]:
        cleaned_region = region.strip().upper()
        logger.info(
            "running anomaly detection & shap explainability for region %s", cleaned_region
        )

        recent = self.db.get_recent_telemetry(cleaned_region, limit=limit)
        if len(recent) < 170:
            fetcher = TelemetryFetcher(self.settings)
            fresh = fetcher.fetch_latest(region=cleaned_region, limit=limit)
            self.db.upsert_telemetry(fresh)
            recent = self.db.get_recent_telemetry(cleaned_region, limit=limit)

        chronological = list(reversed(recent))
        raw_df = pl.DataFrame(
            {
                "timestamp": [r["timestamp"] for r in chronological],
                "region": [r["region"] for r in chronological],
                "metric_value": [r["metric_value"] for r in chronological],
            }
        )

        features_df = self.pipeline.transform(raw_df)
        feature_names = self.pipeline.get_feature_names(include_target=False)
        X = features_df.select(feature_names).to_numpy()

        self._ensure_models(X, feature_names, fit_new=fit_new)
        assert self.detector is not None and self.explainer is not None

        anomaly_mask, scores = self.detector.score(X)
        explanations = self.explainer.batch_explain(X, anomaly_mask=anomaly_mask, top_k=3)

        records: list[AnomalyRecord] = []
        timestamps = features_df["timestamp"].to_list()
        metric_values = features_df["metric_value"].to_list()

        for i in range(len(timestamps)):
            ts = timestamps[i]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            record = AnomalyRecord(
                timestamp=ts,
                region=cleaned_region,
                metric_value=float(metric_values[i]),
                anomaly_score=float(scores[i]),
                is_anomaly=bool(anomaly_mask[i]),
                top_shap_features=explanations[i],
            )
            records.append(record)

        self.db.upsert_anomalies(records)
        anomalies_count = int(np.sum(anomaly_mask))

        logger.info(
            "evaluated %d records: flagged %d structural anomalies (%.2f%%)",
            len(records),
            anomalies_count,
            anomalies_count / len(records) * 100.0 if records else 0.0,
        )

        return {
            "status": "success",
            "region": cleaned_region,
            "total_evaluated": len(records),
            "anomalies_flagged": anomalies_count,
            "anomaly_rate": round(anomalies_count / len(records) * 100.0, 2) if records else 0.0,
            "max_score": float(np.max(scores)) if len(scores) else 0.0,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run structural anomaly detection and SHAP attribution."
    )
    parser.add_argument("--region", type=str, default="NR", help="target electrical region")
    parser.add_argument("--limit", type=int, default=240, help="number of historical records")
    parser.add_argument("--fit-new", action="store_true", help="force fitting new detector")

    args = parser.parse_args()
    runner = AnomalyRunner()
    res = runner.run(region=args.region, limit=args.limit, fit_new=args.fit_new)
    return 0 if res["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())