import os
import sys
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

# ensure project root is present in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

REGIONS = ["ALL_INDIA", "NR", "WR", "SR", "ER", "NER"]
MODEL_VERSION = "1.0.0"
DEFAULT_PRECOMPUTED_DIR = os.path.join(root_dir, "artifacts", "precomputed")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_precomputed_path(region: str, output_dir: str = DEFAULT_PRECOMPUTED_DIR) -> str:
    # returns absolute path to region precomputed json
    return os.path.join(output_dir, f"{region}.json")


def load_precomputed_data(region: str, output_dir: str = DEFAULT_PRECOMPUTED_DIR) -> Optional[Dict[str, Any]]:
    # reads serialized precomputed json payload for region
    path = get_precomputed_path(region, output_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
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
) -> Dict[str, Any]:
    # precalculates 72h forecast, nominal anomaly, and drift metrics for single region
    import numpy as np
    import polars as pl
    from src.models.load_artifacts import load_reference_baseline, load_feature_columns
    from src.models.infer_and_detect import (
        generate_72h_multi_step_forecast,
        predict_quantiles,
        detect_structural_anomalies,
    )
    from src.monitoring.drift_check import (
        evaluate_feature_drift,
        fetch_trailing_telemetry_window,
    )

    logger.info(f"precomputing telemetry artifacts for region: {region}")
    ref_df = load_reference_baseline(region).copy()
    if feature_cols is None:
        feature_cols = load_feature_columns()

    # 1. 72-hour recursive multi-step quantile forecast
    history_tail = ref_df.tail(168).copy()
    base_ts = datetime.now(timezone.utc) - timedelta(hours=len(history_tail))
    records = []
    for idx, (_, row) in enumerate(history_tail.iterrows()):
        records.append({
            "timestamp_utc": base_ts + timedelta(hours=idx),
            "region": region,
            "demand_mw": float(row["demand_mw"]),
            "frequency_hz": 50.0,
            "temperature_2m": 30.0,
            "relative_humidity_2m": 50.0,
            "surface_solar_radiation": 350.0,
            "wind_speed_100m": 4.0,
        })
    df_history = pl.DataFrame(records)
    raw_forecasts = generate_72h_multi_step_forecast(df_history, region, bundle)
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

    # 2. nominal anomaly scoring and top-3 shap drivers
    feature_row = ref_df.iloc[-1]
    features_df = pd.DataFrame([feature_row[bundle.feature_columns]])
    p10, p50, p90 = predict_quantiles(features_df, region, bundle)
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
    nominal_anomaly = {
        "region": region,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "actual_demand_mw": round(actual_demand, 2),
        "predicted_p10_mw": round(float(p10[0]), 2),
        "predicted_p50_mw": round(float(p50[0]), 2),
        "predicted_p90_mw": round(float(p90[0]), 2),
        "anomaly_score": round(float(scores[0]), 4),
        "is_structural": bool(is_structural[0]),
        "residual_breach": bool(trigger_b[0]),
        "top_shap_features": top_shap,
    }

    # 3. statistical drift evaluation
    curr_df = fetch_trailing_telemetry_window(region=region, window_hours=168)
    drift = evaluate_feature_drift(ref_df, curr_df, feature_cols=feature_cols)
    drift_summary = {
        "region": region,
        "drift_alarm": bool(drift["drift_alarm"]),
        "drift_ratio": float(drift["drift_ratio"]),
        "drifted_features": int(drift["drifted_features"]),
        "total_features": int(drift["total_features"]),
        "recommended_quantile_alpha": str(drift["recommended_quantile_alpha"]),
        "feature_metrics": drift["feature_metrics"],
        "report_available": True,
    }

    return {
        "region": region,
        "model_version": MODEL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "forecasts": serialized_forecasts,
        "nominal_anomaly": nominal_anomaly,
        "drift_summary": drift_summary,
    }


def precompute_all(
    output_dir: str = DEFAULT_PRECOMPUTED_DIR,
    regions: Optional[List[str]] = None,
) -> Dict[str, str]:
    # executes batch precomputations across all designated regions
    from src.models.load_artifacts import ModelArtifactsBundle, load_feature_columns
    from src.models.explainers import AnomalySHAPExplainer

    os.makedirs(output_dir, exist_ok=True)
    target_regions = regions or REGIONS

    logger.info("warming up model artifacts bundle and shap explainer for batch precomputation...")
    bundle = ModelArtifactsBundle()
    explainer = AnomalySHAPExplainer(bundle)
    feature_cols = load_feature_columns()

    saved_paths: Dict[str, str] = {}
    for region in target_regions:
        payload = precompute_region(region, bundle, explainer, feature_cols=feature_cols)
        out_path = os.path.join(output_dir, f"{region}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        saved_paths[region] = out_path
        logger.info(f"saved precomputed artifact: {out_path}")

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

    logger.info(f"batch precomputation complete for {len(saved_paths)} regions")
    return saved_paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="batch precompute forecasts, anomaly states, and drift metrics")
    parser.add_argument("--output-dir", default=DEFAULT_PRECOMPUTED_DIR, help="directory for serialized artifacts")
    parser.add_argument("--regions", nargs="*", default=None, help="target regional grid codes")
    args = parser.parse_args()

    precompute_all(output_dir=args.output_dir, regions=args.regions)