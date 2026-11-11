import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.config import Settings, get_settings
from src.db.client import DatabaseClient, get_db_client
from src.ingestion.fetcher import TelemetryFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingestion_runner")


class IngestionRunner:
    def __init__(
        self,
        settings: Settings | None = None,
        db_client: DatabaseClient | None = None,
        fetcher: TelemetryFetcher | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db = db_client or get_db_client(self.settings)
        self.fetcher = fetcher or TelemetryFetcher(self.settings)

    def run(
        self,
        region: str = "NR",
        limit: int = 24,
        dry_run: bool = False,
        endpoint_url: str | None = None,
        export_path: Path | str | None = None,
    ) -> dict[str, Any]:
        logger.info(
            "starting telemetry ingestion: region=%s, limit=%d, dry_run=%s",
            region,
            limit,
            dry_run,
        )

        records = self.fetcher.fetch_latest(
            region=region,
            limit=limit,
            endpoint_url=endpoint_url,
        )

        if not records:
            logger.warning("no telemetry records retrieved for region %s", region)
            return {"status": "empty", "count": 0, "region": region}

        min_val = min(r.metric_value for r in records)
        max_val = max(r.metric_value for r in records)
        start_ts = min(r.timestamp for r in records)
        end_ts = max(r.timestamp for r in records)

        logger.info(
            "fetched %d records (%s to %s) | min: %.1f mw, max: %.1f mw",
            len(records),
            start_ts.isoformat(),
            end_ts.isoformat(),
            min_val,
            max_val,
        )

        upserted = 0
        if not dry_run:
            upserted = self.db.upsert_telemetry(records)
            logger.info("successfully persisted %d records to %s database", upserted, self.db.mode)
        else:
            logger.info("dry-run active: skipping database persistence")

        if export_path:
            out_file = Path(export_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            payload = [r.model_dump(mode="json") for r in records]
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info("exported %d records to %s", len(records), out_file)

        return {
            "status": "success",
            "count": len(records),
            "upserted": upserted,
            "region": region,
            "start_time": start_ts.isoformat(),
            "end_time": end_ts.isoformat(),
            "min_mw": min_val,
            "max_mw": max_val,
            "dry_run": dry_run,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest regional grid telemetry and persist to database."
    )
    parser.add_argument(
        "--region",
        type=str,
        default="NR",
        help="electrical region (NR, WR, SR, ER, NER, or ALL)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=24,
        help="number of hourly records to fetch",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="number of days to fetch (converts to limit = days * 24)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and validate records without writing to database",
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default=None,
        help="custom telemetry api endpoint url",
    )
    parser.add_argument(
        "--export-path",
        type=str,
        default=None,
        help="optional destination json file path",
    )

    args = parser.parse_args()
    limit = (args.days * 24) if args.days is not None else args.limit

    runner = IngestionRunner()
    result = runner.run(
        region=args.region,
        limit=limit,
        dry_run=args.dry_run,
        endpoint_url=args.endpoint_url,
        export_path=args.export_path,
    )
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())