import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import polars as pl

from src.config import Settings, get_settings
from src.db.client import DatabaseClient, get_db_client
from src.features.pipeline import FeaturePipeline
from src.ingestion.fetcher import TelemetryFetcher
from src.monitoring.drift_suite import DriftSuite, DriftSuiteResult
from src.monitoring.report_exporter import export_drift_report
from src.schemas.telemetry import DriftRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("drift_runner")


class DriftRunner:
    def __init__(
        self,
        settings: Settings | None = None,
        db_client: DatabaseClient | None = None,
        reports_dir: str | Path = "reports/drift",
    ) -> None:
        self.settings = settings or get_settings()
        self.db = db_client or get_db_client(self.settings)
        self.reports_dir = Path(reports_dir)
        self.pipeline = FeaturePipeline(drop_nulls=True)

    def _normalize_timestamp(self, ts: Any) -> datetime:
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts

    def run(
        self,
        region: str = "NATIONAL",
        reference_days: int = 30,
        current_days: int = 7,
        drift_share_threshold: float = 0.5,
        save_db: bool = True,
    ) -> tuple[DriftRecord, Path, DriftSuiteResult]:
        cleaned_region = region.strip().upper()
        logger.info(
            "running evidently drift monitoring for region %s (ref=%d days, curr=%d days)",
            cleaned_region,
            reference_days,
            current_days,
        )

        warmup_hours = max(self.pipeline.lag_hours) if self.pipeline.drop_nulls else 0
        eval_hours = (reference_days + current_days) * 24
        total_fetch_hours = eval_hours + warmup_hours
        recent = self.db.get_recent_telemetry(cleaned_region, limit=total_fetch_hours)

        # fallback to fetcher if insufficient historical rows in database
        if len(recent) < total_fetch_hours:
            logger.info("insufficient historical records in db (%d < %d); fetching from source", len(recent), total_fetch_hours)
            fetcher = TelemetryFetcher(self.settings)
            fresh = fetcher.fetch_latest(region=cleaned_region, limit=total_fetch_hours)
            self.db.upsert_telemetry(fresh)
            recent = self.db.get_recent_telemetry(cleaned_region, limit=total_fetch_hours)

        chronological = list(reversed(recent))
        raw_df = pl.DataFrame(
            {
                "timestamp": [r["timestamp"] for r in chronological],
                "region": [r["region"] for r in chronological],
                "metric_value": [r["metric_value"] for r in chronological],
            }
        )

        features_df = self.pipeline.transform(raw_df)
        total_rows = len(features_df)
        current_rows = current_days * 24

        if total_rows <= current_rows:
            split_idx = max(int(total_rows * 0.7), 1)
        else:
            split_idx = total_rows - current_rows

        ref_features = features_df[:split_idx]
        curr_features = features_df[split_idx:]

        if len(ref_features) == 0 or len(curr_features) == 0:
            raise ValueError("not enough samples after feature extraction to form reference and current sets")

        ref_start = self._normalize_timestamp(ref_features["timestamp"][0])
        ref_end = self._normalize_timestamp(ref_features["timestamp"][-1])
        curr_start = self._normalize_timestamp(curr_features["timestamp"][0])
        curr_end = self._normalize_timestamp(curr_features["timestamp"][-1])

        suite = DriftSuite(drift_share_threshold=drift_share_threshold)
        suite_result = suite.run(ref_features, curr_features)

        report_path = export_drift_report(suite_result.report, reports_dir=self.reports_dir)
        logger.info("drift report exported to %s", report_path)

        record = DriftRecord(
            run_timestamp=datetime.now(timezone.utc),
            reference_window_start=ref_start,
            reference_window_end=ref_end,
            current_window_start=curr_start,
            current_window_end=curr_end,
            drift_share=suite_result.drift_share,
            drifted_features=suite_result.drifted_features,
            report_url=str(report_path),
        )

        if save_db:
            self.db.insert_drift(record)
            logger.info("drift record persisted to database with id %s", record.id)

        return record, report_path, suite_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidently AI drift monitoring runner")
    parser.add_argument("--region", type=str, default="NATIONAL", help="grid region identifier")
    parser.add_argument("--reference-days", type=int, default=30, help="reference window in days")
    parser.add_argument("--current-days", type=int, default=7, help="current evaluation window in days")
    parser.add_argument("--reports-dir", type=str, default="reports/drift", help="output directory for reports")
    parser.add_argument("--threshold", type=float, default=0.5, help="dataset drift share threshold")
    parser.add_argument("--dry-run", action="store_true", help="run drift check without writing to database")
    args = parser.parse_args()

    runner = DriftRunner(reports_dir=args.reports_dir)
    record, report_path, suite_result = runner.run(
        region=args.region,
        reference_days=args.reference_days,
        current_days=args.current_days,
        drift_share_threshold=args.threshold,
        save_db=not args.dry_run,
    )

    print("\n--- Drift Monitoring Summary ---")
    print(f"Region: {args.region}")
    print(f"Dataset Drift Detected: {suite_result.dataset_drift}")
    print(f"Drift Share: {suite_result.drift_share:.2%}")
    print(f"Drifted Columns: {suite_result.number_of_drifted_columns} / {suite_result.number_of_columns}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()