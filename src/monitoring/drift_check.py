import os
import sys
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
import requests

try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except (ImportError, ModuleNotFoundError):
        try:
            from evidently.legacy.report import Report
            from evidently.legacy.metric_preset import DataDriftPreset
        except (ImportError, ModuleNotFoundError):
            from evidently import Report
            from evidently.presets.drift import DataDriftPreset
except Exception:
    # fallback when evidently encounters dependency conflicts in host environment
    Report = None
    DataDriftPreset = None

from src.models.load_artifacts import (
    load_reference_baseline,
    load_feature_columns,
    REGIONS,
    get_artifacts_dir,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CYCLICAL_CATEGORICAL_FEATURES = [
    "sin_hour_k1",
    "cos_hour_k1",
    "sin_hour_k2",
    "cos_hour_k2",
    "sin_doy",
    "cos_doy",
    "day_of_week",
    "is_holiday",
]
DRIFT_ALARM_THRESHOLD = 0.30
KS_PVALUE_THRESHOLD = 0.05
WASSERSTEIN_THRESHOLD = 0.10
DEFAULT_REPORT_PATH = os.path.join("static", "reports", "drift_report.html")


def evaluate_feature_drift(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "demand_mw",
) -> Dict[str, Any]:
    # evaluates ks test for continuous features and wasserstein distance for cyclical/categorical features
    if feature_cols is None:
        feature_cols = load_feature_columns()

    eval_cols = [c for c in feature_cols if c in ref_df.columns and c in curr_df.columns]
    if target_col in ref_df.columns and target_col in curr_df.columns:
        eval_cols.append(target_col)

    metrics = {}
    for col in eval_cols:
        u = ref_df[col].dropna().to_numpy()
        v = curr_df[col].dropna().to_numpy()

        if len(u) == 0 or len(v) == 0:
            continue

        if col in CYCLICAL_CATEGORICAL_FEATURES:
            # 1d wasserstein distance for cyclical and discrete calendar indicators
            w_score = float(wasserstein_distance(u, v))
            is_drifted = bool(w_score >= WASSERSTEIN_THRESHOLD)
            metrics[col] = {
                "test": "wasserstein",
                "score": round(w_score, 4),
                "threshold": WASSERSTEIN_THRESHOLD,
                "drift_detected": is_drifted,
            }
        else:
            # two-sample kolmogorov-smirnov test for continuous features and demand target
            ks_res = ks_2samp(u, v)
            is_drifted = bool(ks_res.pvalue < KS_PVALUE_THRESHOLD)
            metrics[col] = {
                "test": "kolmogorov_smirnov",
                "statistic": round(float(ks_res.statistic), 4),
                "p_value": round(float(ks_res.pvalue), 6),
                "threshold": KS_PVALUE_THRESHOLD,
                "drift_detected": is_drifted,
            }

    total_features = len(metrics)
    drifted_features = sum(1 for m in metrics.values() if m["drift_detected"])
    drift_ratio = round(drifted_features / total_features, 4) if total_features > 0 else 0.0
    drift_alarm = drift_ratio > DRIFT_ALARM_THRESHOLD

    return {
        "total_features": total_features,
        "drifted_features": drifted_features,
        "drift_ratio": drift_ratio,
        "drift_alarm": drift_alarm,
        "retraining_advisory": drift_alarm,
        "recommended_quantile_alpha": [0.05, 0.50, 0.95] if drift_alarm else [0.10, 0.50, 0.90],
        "feature_metrics": metrics,
    }


def generate_evidently_drift_report(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    output_path: str = DEFAULT_REPORT_PATH,
) -> str:
    # generates self-contained interactive evidently data drift html dashboard
    abs_output = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    # align columns for report computation
    shared_cols = [c for c in ref_df.columns if c in curr_df.columns]
    ref_slice = ref_df[shared_cols].dropna()
    curr_slice = curr_df[shared_cols].dropna()

    if Report is not None and DataDriftPreset is not None:
        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_slice, current_data=curr_slice)
            report.save_html(abs_output)
            logger.info(f"evidently drift report saved to {abs_output}")
            return abs_output
        except Exception as exc:
            logger.warning(f"evidently report execution failed: {exc}. generating fallback html.")

    # fallback html report if evidently runtime is unavailable
    html_fallback = f"""<!DOCTYPE html>
<html>
<head><title>Drift Report</title>
<style>body {{ font-family: -apple-system, sans-serif; padding: 24px; color: #111; }}</style>
</head>
<body>
<h2>Statistical Drift Diagnostic Report</h2>
<p>Reference baseline records: {len(ref_slice)} | Live window records: {len(curr_slice)}</p>
<p>Kolmogorov-Smirnov and 1D Wasserstein distance evaluations completed programmatically.</p>
</body>
</html>"""
    with open(abs_output, "w", encoding="utf-8") as f:
        f.write(html_fallback)
    logger.info(f"fallback drift report written to {abs_output}")
    return abs_output


def fetch_trailing_telemetry_window(
    db_url: Optional[str] = None,
    region: str = "ALL_INDIA",
    window_hours: int = 168,
) -> pd.DataFrame:
    # retrieves trailing 7-day feature window from postgres or falls back to baseline slice
    if db_url and psycopg2 is not None:
        try:
            from src.features.build_features import compute_telemetry_features
            import polars as pl

            conn = psycopg2.connect(db_url)
            query = f"""
                SELECT timestamp_utc, region, demand_mw, frequency_hz,
                       temperature_2m, relative_humidity_2m,
                       surface_solar_radiation, wind_speed_100m
                FROM regional_telemetry
                WHERE region = %s
                ORDER BY timestamp_utc DESC
                LIMIT %s;
            """
            pdf = pd.read_sql(query, conn, params=(region, window_hours + 168))
            conn.close()

            if len(pdf) >= 168:
                pdf = pdf.sort_values("timestamp_utc").reset_index(drop=True)
                pl_df = pl.from_pandas(pdf)
                feat_df = compute_telemetry_features(pl_df, drop_nulls_flag=False)
                return feat_df.tail(window_hours).to_pandas()
        except Exception as exc:
            logger.warning(f"failed to fetch telemetry from database: {exc}. using baseline fallback")

    # baseline fallback slice
    ref_df = load_reference_baseline(region)
    return ref_df.tail(window_hours).copy()


def run_drift_monitoring_pipeline(
    db_url: Optional[str] = None,
    region: str = "ALL_INDIA",
    report_output_path: str = DEFAULT_REPORT_PATH,
    render_trigger_url: Optional[str] = None,
) -> Dict[str, Any]:
    # executes 7-day vs 30-day baseline statistical comparison and exports html report
    logger.info(f"initiating statistical drift audit for region '{region}'...")
    ref_df = load_reference_baseline(region)
    curr_df = fetch_trailing_telemetry_window(db_url=db_url, region=region, window_hours=168)

    feature_cols = load_feature_columns()
    drift_summary = evaluate_feature_drift(ref_df, curr_df, feature_cols=feature_cols)

    # generate standalone html dashboard
    report_path = generate_evidently_drift_report(ref_df, curr_df, output_path=report_output_path)
    drift_summary["report_path"] = report_path
    drift_summary["region"] = region

    logger.info(
        f"drift audit complete for {region}: "
        f"{drift_summary['drifted_features']}/{drift_summary['total_features']} features drifted "
        f"(ratio: {drift_summary['drift_ratio']:.2f}, alarm: {drift_summary['drift_alarm']})"
    )

    # dispatch automated webhook if drift alarm triggered
    if drift_summary["drift_alarm"] and render_trigger_url:
        try:
            payload = {
                "event": "dataset_drift_alert",
                "region": region,
                "drift_ratio": drift_summary["drift_ratio"],
                "drifted_features": drift_summary["drifted_features"],
                "recommended_quantile_alpha": drift_summary["recommended_quantile_alpha"],
            }
            res = requests.post(render_trigger_url, json=payload, timeout=10)
            logger.info(f"dispatched drift alert webhook: status code {res.status_code}")
        except Exception as exc:
            logger.warning(f"failed to dispatch drift webhook: {exc}")

    return drift_summary


if __name__ == "__main__":
    db_url = os.environ.get("SUPABASE_DB_URL")
    trigger_url = os.environ.get("RENDER_TRIGGER_URL")
    result = run_drift_monitoring_pipeline(db_url=db_url, render_trigger_url=trigger_url)
    print(json.dumps(result, indent=2))