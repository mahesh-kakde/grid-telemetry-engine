import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import pandas as pd

from src.logging_config import configure_logging

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

REGIONS = ["ALL_INDIA", "NR", "WR", "SR", "ER", "NER"]
MODEL_VERSION = "2.0.0"
DEFAULT_PRECOMPUTED_DIR = os.path.join(root_dir, "artifacts", "precomputed")
DEFAULT_HTML_REPORT_PATH = os.path.join(root_dir, "static", "reports", "drift_report.html")

try:
    import psycopg2
    from psycopg2.extras import execute_values, Json
except ImportError:
    psycopg2 = None
    execute_values = None
    Json = None

logger = logging.getLogger(__name__)


def get_precomputed_path(region: str, output_dir: str = DEFAULT_PRECOMPUTED_DIR) -> str:
    # returns absolute path to region precomputed json
    return os.path.join(output_dir, f"{region}.json")


def load_snapshot_from_supabase(
    region: str,
    db_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    # fetches live precomputed dashboard snapshot for region from supabase
    target_db = db_url or os.environ.get("SUPABASE_DB_URL")
    if not target_db or psycopg2 is None:
        return None
    conn = None
    try:
        conn = psycopg2.connect(target_db, connect_timeout=5)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT payload, generated_at, model_version
                FROM precomputed_snapshots
                WHERE region = %s;
                """,
                (region,),
            )
            row = cur.fetchone()
            if not row:
                return None
            payload = row[0]
            if isinstance(payload, str):
                payload = json.loads(payload)
            if isinstance(payload, dict):
                payload.setdefault("generated_at", row[1].isoformat() if hasattr(row[1], "isoformat") else str(row[1]))
                payload.setdefault("model_version", row[2])
                payload["snapshot_source"] = "supabase_live"
            return payload
    except Exception as exc:
        logger.warning(f"failed to load snapshot for {region} from supabase: {exc}")
        return None
    finally:
        if conn is not None:
            conn.close()


def load_all_snapshots_from_supabase(
    regions: Optional[List[str]] = None,
    db_url: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    # fetches all regional snapshots in one database round-trip to avoid repeated handshakes on region switch
    target_db = db_url or os.environ.get("SUPABASE_DB_URL")
    if not target_db or psycopg2 is None:
        return {}
    target_regions = list(regions or REGIONS)
    if not target_regions:
        return {}
    conn = None
    try:
        conn = psycopg2.connect(target_db, connect_timeout=4)
        snapshots: Dict[str, Dict[str, Any]] = {}
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = 5000;")
            cur.execute(
                """
                SELECT region, payload, generated_at, model_version
                FROM precomputed_snapshots
                WHERE region = ANY(%s);
                """,
                (target_regions,),
            )
            for reg_code, payload, generated_at, model_version in cur.fetchall():
                if isinstance(payload, str):
                    payload = json.loads(payload)
                if isinstance(payload, dict):
                    payload.setdefault(
                        "generated_at",
                        generated_at.isoformat() if hasattr(generated_at, "isoformat") else str(generated_at),
                    )
                    payload.setdefault("model_version", model_version)
                    payload["snapshot_source"] = "supabase_live"
                    snapshots[str(reg_code)] = payload
        return snapshots
    except Exception as exc:
        logger.warning(f"failed to batch load regional snapshots from supabase: {exc}")
        return {}
    finally:
        if conn is not None:
            conn.close()


def load_drift_report_html_from_supabase(
    region: str = "ALL_INDIA",
    db_url: Optional[str] = None,
) -> Optional[str]:
    # fetches latest persisted html drift report from supabase precomputed_snapshots
    target_db = db_url or os.environ.get("SUPABASE_DB_URL")
    if not target_db or psycopg2 is None:
        return None
    conn = None
    try:
        conn = psycopg2.connect(target_db, connect_timeout=5)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT drift_report_html FROM precomputed_snapshots WHERE region = %s;",
                (region,),
            )
            row = cur.fetchone()
            return row[0] if row and row[0] else None
    except Exception as exc:
        logger.warning(f"failed to load drift report html for {region} from supabase: {exc}")
        return None
    finally:
        if conn is not None:
            conn.close()


def load_precomputed_data(
    region: str,
    output_dir: str = DEFAULT_PRECOMPUTED_DIR,
    prefer_supabase: bool = True,
) -> Optional[Dict[str, Any]]:
    # reads live supabase snapshot when requested and configured, falling back cleanly to bundled local json
    if prefer_supabase:
        live_data = load_snapshot_from_supabase(region)
        if live_data is not None:
            return live_data

    path = get_precomputed_path(region, output_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            if isinstance(payload, dict):
                payload.setdefault("snapshot_source", "local_snapshot")
            return payload
    except Exception as exc:
        logger.warning(f"failed to load precomputed snapshot for {region}: {exc}")
        return None


def load_precomputed_forecasts(region: str, output_dir: str = DEFAULT_PRECOMPUTED_DIR) -> Optional[List[Dict[str, Any]]]:
    # extracts parsed forecast list with pandas datetimes
    data = load_precomputed_data(region, output_dir)
    if not data or "forecasts" not in data:
        return None
    parsed = []
    for fc in data["forecasts"]:
        parsed.append({
            "target_timestamp_utc": pd.to_datetime(fc["target_timestamp_utc"]),
            "region": fc["region"],
            "forecast_p10": float(fc["forecast_p10"]),
            "forecast_p50": float(fc["forecast_p50"]),
            "forecast_p90": float(fc["forecast_p90"]),
            "model_version": fc.get("model_version", MODEL_VERSION),
        })
    return parsed


def load_precomputed_anomaly(region: str, output_dir: str = DEFAULT_PRECOMPUTED_DIR) -> Optional[Dict[str, Any]]:
    # extracts nominal anomaly state with top-3 shap drivers
    data = load_precomputed_data(region, output_dir)
    if not data or "nominal_anomaly" not in data:
        return None
    return data["nominal_anomaly"]


def load_precomputed_drift(region: str, output_dir: str = DEFAULT_PRECOMPUTED_DIR) -> Optional[Dict[str, Any]]:
    # extracts feature drift metrics summary
    data = load_precomputed_data(region, output_dir)
    if not data or "drift_summary" not in data:
        return None
    return data["drift_summary"]


def precompute_region(
    region: str,
    bundle: Any,
    explainer: Any,
    feature_cols: Optional[List[str]] = None,
    db_url: Optional[str] = None,
    use_live_weather: Optional[bool] = None,
) -> Dict[str, Any]:
    # precalculates 72h forecast, nominal anomaly, and drift metrics for single region
    import numpy as np
    from src.features.build_features import compute_telemetry_features
    from src.models.load_artifacts import load_reference_baseline, resolve_calendar_timezone
    from src.models.infer_and_detect import (
        fetch_live_or_baseline_history,
        fetch_weather_forecast_map,
        generate_72h_multi_step_forecast,
        predict_quantiles,
        detect_structural_anomalies,
    )
    from src.monitoring.drift_check import (
        evaluate_feature_drift,
        build_html_drift_report,
        select_drift_observation_window,
    )

    logger.info(f"precomputing telemetry artifacts for region: {region}")
    target_db = db_url or os.environ.get("SUPABASE_DB_URL")
    live_weather_enabled = bool(target_db) if use_live_weather is None else bool(use_live_weather)

    ref_df = load_reference_baseline(region).copy()
    df_history, source_tag = fetch_live_or_baseline_history(
        region=region,
        lookback_hours=360,
        bundle=bundle,
        db_url=target_db,
    )
    cal_tz = resolve_calendar_timezone(getattr(bundle, "quantile_calibration", None), region)
    feat_pl = compute_telemetry_features(df_history, drop_nulls_flag=True, calendar_tz=cal_tz)
    curr_feat_df = ref_df.copy() if feat_pl.is_empty() else feat_pl.to_pandas()

    # 1. statistical drift evaluation (requires full 168h post-warmup live window across 12 covariates)
    live_feat_df = curr_feat_df if (source_tag == "supabase_live" and not feat_pl.is_empty()) else None
    curr_window, drift_window_source = select_drift_observation_window(
        live_feat_df, ref_df, window_hours=168
    )
    drift = evaluate_feature_drift(ref_df, curr_window, feature_cols=None)
    widened = bool(drift["drift_alarm"])
    generated_at = datetime.now(timezone.utc).isoformat()

    drift_summary = {
        "region": region,
        "observation_hours": int(drift["observation_hours"]),
        "window_source": drift_window_source,
        "drift_alarm": widened,
        "drift_ratio": float(drift["drift_ratio"]),
        "drifted_features": int(drift["drifted_features"]),
        "total_features": int(drift["total_features"]),
        "recommended_quantile_alpha": list(drift["recommended_quantile_alpha"]),
        "feature_metrics": drift["feature_metrics"],
        "report_available": True,
    }
    drift_report_html = build_html_drift_report(drift_summary, region=region, generated_at=generated_at)

    # 2. 72-hour recursive multi-step quantile forecast (widened automatically if drift alarm active)
    wx_map = fetch_weather_forecast_map(region, forecast_days=4) if live_weather_enabled else None
    raw_forecasts = generate_72h_multi_step_forecast(
        df_history,
        region,
        bundle,
        widened=widened,
        weather_forecast_map=wx_map,
    )
    serialized_forecasts = [
        {
            "target_timestamp_utc": fc["target_timestamp_utc"].isoformat()
            if isinstance(fc["target_timestamp_utc"], (datetime, pd.Timestamp))
            else str(fc["target_timestamp_utc"]),
            "region": fc["region"],
            "forecast_p10": round(float(fc["forecast_p10"]), 2),
            "forecast_p50": round(float(fc["forecast_p50"]), 2),
            "forecast_p90": round(float(fc["forecast_p90"]), 2),
            "model_version": fc.get("model_version", MODEL_VERSION),
        }
        for fc in raw_forecasts
    ]

    # 3. latest observation anomaly scoring and top-3 shap drivers
    feature_row = curr_feat_df.iloc[-1]
    features_df = pd.DataFrame([feature_row[bundle.feature_columns]])
    p10, p50, p90 = predict_quantiles(features_df, region, bundle, widened=widened)
    actual_demand = float(feature_row["demand_mw"])
    scores, is_structural, trigger_b = detect_structural_anomalies(
        X=features_df,
        y_actual=np.array([actual_demand]),
        p10=p10,
        p50=p50,
        p90=p90,
        region=region,
        bundle=bundle,
    )
    top_shap = explainer.explain_anomaly_point(feature_row, region, top_k=3)
    anomaly_shap = explainer.explain_anomaly_score(
        feature_row, region, actual_demand_mw=actual_demand, top_k=3
    )
    obs_ts = feature_row.get("timestamp_utc")
    obs_iso = obs_ts.isoformat() if hasattr(obs_ts, "isoformat") else generated_at
    nominal_anomaly = {
        "region": region,
        "timestamp_utc": obs_iso,
        "actual_demand_mw": round(actual_demand, 2),
        "predicted_p10_mw": round(float(p10[0]), 2),
        "predicted_p50_mw": round(float(p50[0]), 2),
        "predicted_p90_mw": round(float(p90[0]), 2),
        "anomaly_score": round(float(scores[0]), 4),
        "is_structural": bool(is_structural[0]),
        "residual_breach": bool(trigger_b[0]),
        "top_shap_features": top_shap,
        "anomaly_shap_features": anomaly_shap,
    }

    return {
        "region": region,
        "model_version": MODEL_VERSION,
        "generated_at": generated_at,
        "data_source": source_tag,
        "widened": widened,
        "forecasts": serialized_forecasts,
        "nominal_anomaly": nominal_anomaly,
        "drift_summary": drift_summary,
        "drift_report_html": drift_report_html,
    }


def persist_precomputed_snapshots_to_supabase(
    snapshots: List[Dict[str, Any]],
    db_url: str,
) -> int:
    # idempotent upsert of precomputed dashboard snapshots into supabase
    if not snapshots:
        return 0
    if psycopg2 is None:
        raise ImportError("psycopg2 is required for database persistence. Install psycopg2-binary.")

    rows = []
    for snap in snapshots:
        clean_payload = {k: v for k, v in snap.items() if k != "drift_report_html"}
        rows.append((
            snap["region"],
            snap.get("model_version", MODEL_VERSION),
            snap["generated_at"],
            Json(clean_payload),
            snap.get("drift_report_html"),
        ))

    upsert_sql = """
        INSERT INTO precomputed_snapshots (
            region,
            model_version,
            generated_at,
            payload,
            drift_report_html
        ) VALUES %s
        ON CONFLICT (region) DO UPDATE SET
            model_version = EXCLUDED.model_version,
            generated_at = EXCLUDED.generated_at,
            payload = EXCLUDED.payload,
            drift_report_html = EXCLUDED.drift_report_html,
            updated_at = now();
    """

    conn = psycopg2.connect(db_url, connect_timeout=10)
    try:
        with conn.cursor() as cur:
            execute_values(cur, upsert_sql, rows)
        conn.commit()
        logger.info(f"successfully upserted {len(rows)} snapshots to precomputed_snapshots")
        return len(rows)
    finally:
        conn.close()


def _dispatch_drift_alert_webhook(
    snapshots: List[Dict[str, Any]],
    alert_webhook_url: Optional[str] = None,
) -> None:
    # notifies optional ops webhook when any region triggers statistical covariate drift
    webhook_url = alert_webhook_url or os.environ.get("ALERT_WEBHOOK_URL")
    if not webhook_url:
        return
    drifted_regions = [
        {
            "region": snap["region"],
            "drift_ratio": snap["drift_summary"]["drift_ratio"],
            "drifted_features": snap["drift_summary"]["drifted_features"],
            "recommended_quantile_alpha": snap["drift_summary"]["recommended_quantile_alpha"],
        }
        for snap in snapshots
        if snap.get("drift_summary", {}).get("drift_alarm")
    ]
    if not drifted_regions:
        return
    try:
        import requests

        res = requests.post(
            webhook_url,
            json={"event": "dataset_drift_alert", "regions": drifted_regions},
            timeout=10,
        )
        logger.info(f"dispatched consolidated drift alert webhook: status code {res.status_code}")
    except Exception as exc:
        logger.warning(f"failed to dispatch consolidated drift webhook: {exc}")


def precompute_all(
    output_dir: str = DEFAULT_PRECOMPUTED_DIR,
    regions: Optional[List[str]] = None,
    db_url: Optional[str] = None,
    use_live_weather: Optional[bool] = None,
    alert_webhook_url: Optional[str] = None,
) -> Dict[str, str]:
    # executes batch precomputations across all designated regions
    from src.models.load_artifacts import ModelArtifactsBundle, load_feature_columns
    from src.models.explainers import AnomalySHAPExplainer
    from src.models.infer_and_detect import SUB_REGIONS, reconcile_all_india_forecasts

    os.makedirs(output_dir, exist_ok=True)
    target_regions = regions or REGIONS
    target_db = db_url or os.environ.get("SUPABASE_DB_URL")

    logger.info("warming up model artifacts bundle and shap explainer for batch precomputation...")
    bundle = ModelArtifactsBundle()
    explainer = AnomalySHAPExplainer(bundle)
    feature_cols = load_feature_columns()

    saved_paths: Dict[str, str] = {}
    snapshots_by_region: Dict[str, Dict[str, Any]] = {}

    for region in target_regions:
        snapshots_by_region[region] = precompute_region(
            region,
            bundle,
            explainer,
            feature_cols=feature_cols,
            db_url=target_db,
            use_live_weather=use_live_weather,
        )

    if "ALL_INDIA" in snapshots_by_region and all(r in snapshots_by_region for r in SUB_REGIONS):
        sub_forecasts = {r: snapshots_by_region[r]["forecasts"] for r in SUB_REGIONS}
        snapshots_by_region["ALL_INDIA"]["forecasts"] = reconcile_all_india_forecasts(
            snapshots_by_region["ALL_INDIA"]["forecasts"],
            sub_forecasts,
            round_digits=2,
        )

    snapshots_for_db: List[Dict[str, Any]] = []
    for region in target_regions:
        full_snap = snapshots_by_region[region]
        snapshots_for_db.append(full_snap)

        json_payload = {k: v for k, v in full_snap.items() if k != "drift_report_html"}
        out_path = os.path.join(output_dir, f"{region}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2)
        saved_paths[region] = out_path
        logger.info(f"saved precomputed artifact: {out_path}")

        if region == "ALL_INDIA" and full_snap.get("drift_report_html"):
            os.makedirs(os.path.dirname(DEFAULT_HTML_REPORT_PATH), exist_ok=True)
            with open(DEFAULT_HTML_REPORT_PATH, "w", encoding="utf-8") as f:
                f.write(full_snap["drift_report_html"])
            logger.info(f"updated standalone drift report: {DEFAULT_HTML_REPORT_PATH}")

    # export manifest index
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "regions": target_regions,
        "file_count": len(saved_paths),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    if target_db:
        persist_precomputed_snapshots_to_supabase(snapshots_for_db, target_db)

    _dispatch_drift_alert_webhook(snapshots_for_db, alert_webhook_url=alert_webhook_url)
    logger.info(f"batch precomputation complete for {len(saved_paths)} regions")
    return saved_paths


if __name__ == "__main__":
    import argparse
    configure_logging()
    parser = argparse.ArgumentParser(description="batch precompute forecasts, anomaly states, and drift metrics")
    parser.add_argument("--output-dir", default=DEFAULT_PRECOMPUTED_DIR, help="directory for serialized artifacts")
    parser.add_argument("--regions", nargs="*", default=None, help="target regional grid codes")
    args = parser.parse_args()

    precompute_all(output_dir=args.output_dir, regions=args.regions)