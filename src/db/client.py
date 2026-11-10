import json
import sqlite3
from pathlib import Path
from typing import Any, Literal
from src.config import Settings, get_settings
from src.schemas.telemetry import (
    AnomalyRecord,
    DriftRecord,
    ForecastRecord,
    TelemetryRecord,
)

try:
    from supabase import Client as SupabaseClient, create_client
except ImportError:
    SupabaseClient = None  # type: ignore[misc,assignment]
    create_client = None  # type: ignore[assignment]


class DatabaseClient:
    def __init__(
        self,
        settings: Settings | None = None,
        mode: Literal["auto", "supabase", "sqlite"] = "auto",
        sqlite_path: Path | str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.sqlite_path = Path(sqlite_path or self.settings.sqlite_db_path)
        self.mode = self._resolve_mode(mode)
        self.supabase_client: SupabaseClient | None = None

        if self.mode == "supabase":
            if create_client is None:
                # fallback to sqlite if supabase library is not installed
                self.mode = "sqlite"
            else:
                try:
                    self.supabase_client = create_client(
                        self.settings.supabase_url or "",
                        self.settings.supabase_key or "",
                    )
                except Exception:
                    # fallback to sqlite on connection or credentials initialization failure
                    self.mode = "sqlite"

        if self.mode == "sqlite":
            self._init_sqlite()

    def _resolve_mode(self, requested: str) -> str:
        if requested in ("supabase", "sqlite"):
            return requested
        if self.settings.has_supabase_credentials and create_client is not None:
            return "supabase"
        return "sqlite"

    def _get_sqlite_conn(self) -> sqlite3.Connection:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self) -> None:
        with self._get_sqlite_conn() as conn:
            conn.executescript(
                """
                create table if not exists telemetry_raw (
                    id text primary key,
                    timestamp text not null,
                    region text not null,
                    metric_value real not null,
                    metric_type text not null default 'DEMAND_MW',
                    created_at text not null,
                    unique (region, timestamp, metric_type)
                );

                create index if not exists idx_sqlite_telemetry_region_ts
                    on telemetry_raw (region, timestamp desc);

                create table if not exists forecasts (
                    id text primary key,
                    forecast_origin text not null,
                    target_timestamp text not null,
                    region text not null,
                    p10 real not null,
                    p50 real not null,
                    p90 real not null,
                    model_version text not null default 'v1.0',
                    created_at text not null,
                    unique (forecast_origin, target_timestamp, region, model_version)
                );

                create index if not exists idx_sqlite_forecasts_target
                    on forecasts (region, target_timestamp);

                create table if not exists anomalies (
                    id text primary key,
                    timestamp text not null,
                    region text not null,
                    metric_value real not null,
                    anomaly_score real not null,
                    is_anomaly integer not null default 0,
                    top_shap_features text not null default '{}',
                    created_at text not null,
                    unique (region, timestamp)
                );

                create index if not exists idx_sqlite_anomalies_region_ts
                    on anomalies (region, timestamp desc);

                create table if not exists drift_metrics (
                    id text primary key,
                    run_timestamp text not null,
                    reference_window_start text not null,
                    reference_window_end text not null,
                    current_window_start text not null,
                    current_window_end text not null,
                    drift_share real not null,
                    drifted_features text not null default '{}',
                    report_url text,
                    created_at text not null
                );
                """
            )

    # telemetry ingestion methods
    def upsert_telemetry(self, records: list[TelemetryRecord]) -> int:
        if not records:
            return 0

        if self.mode == "supabase" and self.supabase_client is not None:
            payload = [
                {
                    "id": str(r.id),
                    "timestamp": r.timestamp.isoformat(),
                    "region": r.region,
                    "metric_value": r.metric_value,
                    "metric_type": r.metric_type,
                    "created_at": r.created_at.isoformat(),
                }
                for r in records
            ]
            self.supabase_client.table("telemetry_raw").upsert(
                payload, on_conflict="region,timestamp,metric_type"
            ).execute()
            return len(records)

        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            rows = [
                (
                    str(r.id),
                    r.timestamp.isoformat(),
                    r.region,
                    r.metric_value,
                    r.metric_type,
                    r.created_at.isoformat(),
                )
                for r in records
            ]
            cursor.executemany(
                """
                insert into telemetry_raw (id, timestamp, region, metric_value, metric_type, created_at)
                values (?, ?, ?, ?, ?, ?)
                on conflict(region, timestamp, metric_type) do update set
                    metric_value = excluded.metric_value,
                    created_at = excluded.created_at
                """,
                rows,
            )
            return len(records)

    def get_recent_telemetry(self, region: str, limit: int = 168) -> list[dict[str, Any]]:
        cleaned_region = region.strip().upper()
        if self.mode == "supabase" and self.supabase_client is not None:
            response = (
                self.supabase_client.table("telemetry_raw")
                .select("*")
                .eq("region", cleaned_region)
                .order("timestamp", desc=True)
                .limit(limit)
                .execute()
            )
            return response.data or []

        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                select id, timestamp, region, metric_value, metric_type, created_at
                from telemetry_raw
                where region = ?
                order by timestamp desc
                limit ?
                """,
                (cleaned_region, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    # forecast output methods
    def upsert_forecasts(self, records: list[ForecastRecord]) -> int:
        if not records:
            return 0

        if self.mode == "supabase" and self.supabase_client is not None:
            payload = [
                {
                    "id": str(r.id),
                    "forecast_origin": r.forecast_origin.isoformat(),
                    "target_timestamp": r.target_timestamp.isoformat(),
                    "region": r.region,
                    "p10": r.p10,
                    "p50": r.p50,
                    "p90": r.p90,
                    "model_version": r.model_version,
                    "created_at": r.created_at.isoformat(),
                }
                for r in records
            ]
            self.supabase_client.table("forecasts").upsert(
                payload, on_conflict="forecast_origin,target_timestamp,region,model_version"
            ).execute()
            return len(records)

        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            rows = [
                (
                    str(r.id),
                    r.forecast_origin.isoformat(),
                    r.target_timestamp.isoformat(),
                    r.region,
                    r.p10,
                    r.p50,
                    r.p90,
                    r.model_version,
                    r.created_at.isoformat(),
                )
                for r in records
            ]
            cursor.executemany(
                """
                insert into forecasts (
                    id, forecast_origin, target_timestamp, region, p10, p50, p90, model_version, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(forecast_origin, target_timestamp, region, model_version) do update set
                    p10 = excluded.p10,
                    p50 = excluded.p50,
                    p90 = excluded.p90,
                    created_at = excluded.created_at
                """,
                rows,
            )
            return len(records)

    def get_latest_forecasts(self, region: str, limit: int = 72) -> list[dict[str, Any]]:
        cleaned_region = region.strip().upper()
        if self.mode == "supabase" and self.supabase_client is not None:
            response = (
                self.supabase_client.table("forecasts")
                .select("*")
                .eq("region", cleaned_region)
                .order("forecast_origin", desc=True)
                .order("target_timestamp", desc=False)
                .limit(limit)
                .execute()
            )
            return response.data or []

        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                select id, forecast_origin, target_timestamp, region, p10, p50, p90, model_version, created_at
                from forecasts
                where region = ?
                order by forecast_origin desc, target_timestamp asc
                limit ?
                """,
                (cleaned_region, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    # anomaly methods
    def upsert_anomalies(self, records: list[AnomalyRecord]) -> int:
        if not records:
            return 0

        if self.mode == "supabase" and self.supabase_client is not None:
            payload = [
                {
                    "id": str(r.id),
                    "timestamp": r.timestamp.isoformat(),
                    "region": r.region,
                    "metric_value": r.metric_value,
                    "anomaly_score": r.anomaly_score,
                    "is_anomaly": r.is_anomaly,
                    "top_shap_features": r.top_shap_features,
                    "created_at": r.created_at.isoformat(),
                }
                for r in records
            ]
            self.supabase_client.table("anomalies").upsert(
                payload, on_conflict="region,timestamp"
            ).execute()
            return len(records)

        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            rows = [
                (
                    str(r.id),
                    r.timestamp.isoformat(),
                    r.region,
                    r.metric_value,
                    r.anomaly_score,
                    1 if r.is_anomaly else 0,
                    json.dumps(r.top_shap_features),
                    r.created_at.isoformat(),
                )
                for r in records
            ]
            cursor.executemany(
                """
                insert into anomalies (
                    id, timestamp, region, metric_value, anomaly_score, is_anomaly, top_shap_features, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(region, timestamp) do update set
                    metric_value = excluded.metric_value,
                    anomaly_score = excluded.anomaly_score,
                    is_anomaly = excluded.is_anomaly,
                    top_shap_features = excluded.top_shap_features,
                    created_at = excluded.created_at
                """,
                rows,
            )
            return len(records)

    def get_recent_anomalies(self, region: str, limit: int = 50) -> list[dict[str, Any]]:
        cleaned_region = region.strip().upper()
        if self.mode == "supabase" and self.supabase_client is not None:
            response = (
                self.supabase_client.table("anomalies")
                .select("*")
                .eq("region", cleaned_region)
                .order("timestamp", desc=True)
                .limit(limit)
                .execute()
            )
            return response.data or []

        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                select id, timestamp, region, metric_value, anomaly_score, is_anomaly, top_shap_features, created_at
                from anomalies
                where region = ?
                order by timestamp desc
                limit ?
                """,
                (cleaned_region, limit),
            )
            results = []
            for row in cursor.fetchall():
                data = dict(row)
                data["is_anomaly"] = bool(data["is_anomaly"])
                if isinstance(data.get("top_shap_features"), str):
                    data["top_shap_features"] = json.loads(data["top_shap_features"])
                results.append(data)
            return results

    # drift tracking methods
    def insert_drift(self, record: DriftRecord) -> dict[str, Any]:
        if self.mode == "supabase" and self.supabase_client is not None:
            payload = {
                "id": str(record.id),
                "run_timestamp": record.run_timestamp.isoformat(),
                "reference_window_start": record.reference_window_start.isoformat(),
                "reference_window_end": record.reference_window_end.isoformat(),
                "current_window_start": record.current_window_start.isoformat(),
                "current_window_end": record.current_window_end.isoformat(),
                "drift_share": record.drift_share,
                "drifted_features": record.drifted_features,
                "report_url": record.report_url,
                "created_at": record.created_at.isoformat(),
            }
            response = self.supabase_client.table("drift_metrics").insert(payload).execute()
            return (response.data or [payload])[0]

        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                insert into drift_metrics (
                    id, run_timestamp, reference_window_start, reference_window_end,
                    current_window_start, current_window_end, drift_share, drifted_features,
                    report_url, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(record.id),
                    record.run_timestamp.isoformat(),
                    record.reference_window_start.isoformat(),
                    record.reference_window_end.isoformat(),
                    record.current_window_start.isoformat(),
                    record.current_window_end.isoformat(),
                    record.drift_share,
                    json.dumps(record.drifted_features),
                    record.report_url,
                    record.created_at.isoformat(),
                ),
            )
            return {
                "id": str(record.id),
                "run_timestamp": record.run_timestamp.isoformat(),
                "drift_share": record.drift_share,
            }

    def get_latest_drift(self) -> dict[str, Any] | None:
        if self.mode == "supabase" and self.supabase_client is not None:
            response = (
                self.supabase_client.table("drift_metrics")
                .select("*")
                .order("run_timestamp", desc=True)
                .limit(1)
                .execute()
            )
            return response.data[0] if response.data else None

        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                select id, run_timestamp, reference_window_start, reference_window_end,
                       current_window_start, current_window_end, drift_share, drifted_features,
                       report_url, created_at
                from drift_metrics
                order by run_timestamp desc
                limit 1
                """
            )
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            if isinstance(data.get("drifted_features"), str):
                data["drifted_features"] = json.loads(data["drifted_features"])
            return data


def get_db_client(
    settings: Settings | None = None,
    mode: Literal["auto", "supabase", "sqlite"] = "auto",
    sqlite_path: Path | str | None = None,
) -> DatabaseClient:
    return DatabaseClient(settings=settings, mode=mode, sqlite_path=sqlite_path)