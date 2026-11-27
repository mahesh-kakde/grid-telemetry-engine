import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

try:
    import psycopg2
except ImportError:
    psycopg2 = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 90


def calculate_cutoff_timestamp(retention_days: int = DEFAULT_RETENTION_DAYS) -> datetime:
    # computes utc cutoff timestamp relative to current wall clock
    if int(retention_days) < 1:
        raise ValueError(f"retention_days must be at least 1, got {retention_days}")
    return datetime.now(timezone.utc) - timedelta(days=int(retention_days))


def prune_expired_records(
    db_url: Optional[str] = None,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    dry_run: bool = False,
) -> Dict[str, Any]:
    # deletes records older than retention_days from telemetry, forecasts, and anomalies
    cutoff_ts = calculate_cutoff_timestamp(retention_days)
    resolved_db_url = db_url or os.environ.get("SUPABASE_DB_URL")
    if not resolved_db_url:
        logger.info("no supabase db url provided or configured; skipping retention pruning")
        return {
            "status": "skipped",
            "reason": "missing_db_url",
            "retention_days": retention_days,
            "deleted_telemetry": 0,
            "deleted_forecasts": 0,
            "deleted_anomalies": 0,
            "total_deleted": 0,
        }

    if psycopg2 is None:
        raise ImportError("psycopg2 is required for database operations. install psycopg2-binary.")

    logger.info(
        f"{'dry-run: evaluating' if dry_run else 'executing'} database retention cleanup "
        f"(retention: {retention_days} days, cutoff: {cutoff_ts.isoformat()})..."
    )

    conn = psycopg2.connect(resolved_db_url, connect_timeout=10)
    try:
        with conn.cursor() as cur:
            if dry_run:
                # preview counts without deleting
                cur.execute(
                    "SELECT COUNT(*) FROM regional_telemetry WHERE timestamp_utc < %s;",
                    (cutoff_ts,),
                )
                del_telemetry = int(cur.fetchone()[0])

                cur.execute(
                    "SELECT COUNT(*) FROM model_forecasts WHERE target_timestamp_utc < %s;",
                    (cutoff_ts,),
                )
                del_forecasts = int(cur.fetchone()[0])

                cur.execute(
                    "SELECT COUNT(*) FROM telemetry_anomalies WHERE timestamp_utc < %s;",
                    (cutoff_ts,),
                )
                del_anomalies = int(cur.fetchone()[0])
            else:
                # 1. regional physical telemetry
                cur.execute(
                    "DELETE FROM regional_telemetry WHERE timestamp_utc < %s;",
                    (cutoff_ts,),
                )
                del_telemetry = int(cur.rowcount)

                # 2. model forecast repository
                cur.execute(
                    "DELETE FROM model_forecasts WHERE target_timestamp_utc < %s;",
                    (cutoff_ts,),
                )
                del_forecasts = int(cur.rowcount)

                # 3. anomaly explainability events
                cur.execute(
                    "DELETE FROM telemetry_anomalies WHERE timestamp_utc < %s;",
                    (cutoff_ts,),
                )
                del_anomalies = int(cur.rowcount)

                conn.commit()

        total = del_telemetry + del_forecasts + del_anomalies
        logger.info(
            f"retention cleanup complete: {del_telemetry} telemetry, "
            f"{del_forecasts} forecasts, {del_anomalies} anomalies "
            f"({'identified' if dry_run else 'pruned'}, total: {total})"
        )
        return {
            "status": "success",
            "dry_run": dry_run,
            "cutoff_utc": cutoff_ts.isoformat(),
            "retention_days": retention_days,
            "deleted_telemetry": del_telemetry,
            "deleted_forecasts": del_forecasts,
            "deleted_anomalies": del_anomalies,
            "total_deleted": total,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="enforce rolling retention pruning on supabase postgresql")
    parser.add_argument(
        "--retention-days",
        type=int,
        default=int(os.environ.get("RETENTION_DAYS", DEFAULT_RETENTION_DAYS)),
        help="number of days to retain active historical telemetry",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count and report expired rows without deleting them",
    )
    args = parser.parse_args()

    prune_expired_records(retention_days=args.retention_days, dry_run=args.dry_run)