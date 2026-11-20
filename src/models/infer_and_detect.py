import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import polars as pl

try:
    import psycopg2
    from psycopg2.extras import execute_values, Json
except ImportError:
    psycopg2 = None
    execute_values = None
    Json = None

from src.models.load_artifacts import (
    ModelArtifactsBundle,
    REGIONS,
    QUANTILES,
    get_artifacts_dir,
)
from src.features.build_features import compute_telemetry_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_VERSION = "lgb-quantile-v2.0"


def enforce_quantile_monotonicity(
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # applies isotonic sorting across quantiles to guarantee p10 <= p50 <= p90
    stacked = np.stack([p10, p50, p90], axis=1)
    sorted_stacked = np.sort(stacked, axis=1)
    return sorted_stacked[:, 0], sorted_stacked[:, 1], sorted_stacked[:, 2]


def predict_quantiles(
    X: pd.DataFrame,
    region: str,
    bundle: Optional[ModelArtifactsBundle] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # produces multi-horizon quantile demand forecasts with monotonicity enforcement
    if bundle is None:
        bundle = ModelArtifactsBundle()

    feature_cols = bundle.feature_columns
    features_subset = X[feature_cols]

    p10_raw = bundle.lgb_models[region][0.10].predict(features_subset)
    p50_raw = bundle.lgb_models[region][0.50].predict(features_subset)
    p90_raw = bundle.lgb_models[region][0.90].predict(features_subset)

    p10, p50, p90 = enforce_quantile_monotonicity(p10_raw, p50_raw, p90_raw)
    return p10, p50, p90


def detect_structural_anomalies(
    X: pd.DataFrame,
    y_actual: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    region: str,
    bundle: Optional[ModelArtifactsBundle] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # dual-trigger anomaly detection: isolation forest score and predictive iqr envelope breach
    if bundle is None:
        bundle = ModelArtifactsBundle()

    feature_cols = bundle.feature_columns
    features_subset = X[feature_cols]

    # trigger a: isolation forest score below contamination threshold
    iso_model = bundle.isolation_forests[region]
    iso_scores = iso_model.score_samples(features_subset)
    # predict returns -1 for outliers based on calibrated contamination
    trigger_a = (iso_model.predict(features_subset) == -1)

    # trigger b: residual iqr violation
    iqr = p90 - p10
    upper_envelope = p90 + (1.5 * iqr)
    lower_envelope = p10 - (1.5 * iqr)
    trigger_b = (y_actual > upper_envelope) | (y_actual < lower_envelope)

    # verified structural anomaly requires both isolation and predictive breach
    is_structural = trigger_a & trigger_b
    return iso_scores, is_structural, trigger_b


def generate_72h_multi_step_forecast(
    df_history: pl.DataFrame,
    region: str,
    bundle: Optional[ModelArtifactsBundle] = None,
) -> List[Dict[str, Any]]:
    # generates 72-hour direct multi-step forecast vectors from latest historical state
    if bundle is None:
        bundle = ModelArtifactsBundle()

    # filter and sort historical telemetry for region
    df_reg = df_history.filter(pl.col("region") == region).sort("timestamp_utc")
    if len(df_reg) < 168:
        raise ValueError(f"insufficient historical lookback for {region}: requires at least 168 hours")

    forecast_generated_at = datetime.now(timezone.utc)
    latest_ts = df_reg["timestamp_utc"].max()

    # iteratively project forward 72 hours
    extended_df = df_reg.clone()
    forecasts = []

    for step in range(1, 73):
        target_ts = latest_ts + timedelta(hours=step)
        # build updated features on recent trailing window to bound memory allocations
        window_slice = extended_df.tail(180)
        features_df = compute_telemetry_features(window_slice, drop_nulls_flag=False)
        last_row = features_df.tail(1).to_pandas()

        p10, p50, p90 = predict_quantiles(last_row, region, bundle)

        forecasts.append({
            "forecast_generated_at": forecast_generated_at,
            "target_timestamp_utc": target_ts,
            "region": region,
            "forecast_p10": float(p10[0]),
            "forecast_p50": float(p50[0]),
            "forecast_p90": float(p90[0]),
            "actual_demand_mw": None,
            "model_version": MODEL_VERSION,
        })

        # append median forecast as surrogate for subsequent autoregressive step
        new_row = pl.DataFrame([{
            "timestamp_utc": target_ts,
            "region": region,
            "demand_mw": float(p50[0]),
            "frequency_hz": 50.0,
            "temperature_2m": float(df_reg["temperature_2m"].tail(1)[0]),
            "relative_humidity_2m": float(df_reg["relative_humidity_2m"].tail(1)[0]),
            "surface_solar_radiation": float(df_reg["surface_solar_radiation"].tail(1)[0]),
            "wind_speed_100m": float(df_reg["wind_speed_100m"].tail(1)[0]),
        }])
        extended_df = pl.concat([extended_df, new_row])

    del extended_df
    import gc
    gc.collect()
    return forecasts


def persist_forecasts_to_supabase(
    forecasts: List[Dict[str, Any]],
    db_url: str,
) -> int:
    # idempotent upsert into model_forecasts table
    if not forecasts:
        return 0
    if psycopg2 is None:
        raise ImportError("psycopg2 is required for database persistence. Install psycopg2-binary.")

    rows = [
        (
            f["forecast_generated_at"],
            f["target_timestamp_utc"],
            f["region"],
            f["forecast_p10"],
            f["forecast_p50"],
            f["forecast_p90"],
            f["actual_demand_mw"],
            f["model_version"],
        )
        for f in forecasts
    ]

    upsert_sql = """
        INSERT INTO model_forecasts (
            forecast_generated_at,
            target_timestamp_utc,
            region,
            forecast_p10,
            forecast_p50,
            forecast_p90,
            actual_demand_mw,
            model_version
        ) VALUES %s
        ON CONFLICT (model_version, region, target_timestamp_utc) DO UPDATE SET
            forecast_generated_at = EXCLUDED.forecast_generated_at,
            forecast_p10 = EXCLUDED.forecast_p10,
            forecast_p50 = EXCLUDED.forecast_p50,
            forecast_p90 = EXCLUDED.forecast_p90,
            actual_demand_mw = COALESCE(EXCLUDED.actual_demand_mw, model_forecasts.actual_demand_mw);
    """

    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            execute_values(cur, upsert_sql, rows)
        conn.commit()
        logger.info(f"successfully upserted {len(rows)} forecast rows to supabase")
        return len(rows)
    finally:
        conn.close()


def run_inference_and_detection() -> Dict[str, Any]:
    # batch execution runner invoked by scheduled etl pipeline
    logger.info("initializing batch forecasting and anomaly detection engine...")
    bundle = ModelArtifactsBundle()

    # load latest baseline data for synthesis or querying
    results = {}
    for region in REGIONS:
        ref_df = bundle.reference_baselines[region]
        X = ref_df[bundle.feature_columns]
        y = ref_df["demand_mw"].to_numpy()

        p10, p50, p90 = predict_quantiles(X, region, bundle)
        iso_scores, is_structural, trigger_b = detect_structural_anomalies(
            X, y, p10, p50, p90, region, bundle
        )

        n_anomalies = int(np.sum(is_structural))
        logger.info(f"region {region}: evaluated {len(y)} records, flagged {n_anomalies} structural anomalies")
        results[region] = {
            "records": len(y),
            "anomalies_detected": n_anomalies,
            "mean_demand_p50": float(np.mean(p50)),
        }

    db_url = os.environ.get("SUPABASE_DB_URL")
    if db_url:
        logger.info("generating 72h multi-step forecasts and persisting to supabase...")
        # build historical dataframe from baselines
        combined_rows = []
        for reg in REGIONS:
            ref = bundle.reference_baselines[reg].copy()
            ref["region"] = reg
            combined_rows.append(ref)
        history_df = pl.from_pandas(pd.concat(combined_rows, ignore_index=True))

        all_forecasts = []
        for reg in REGIONS:
            forecasts = generate_72h_multi_step_forecast(history_df, reg, bundle)
            all_forecasts.extend(forecasts)
        persist_forecasts_to_supabase(all_forecasts, db_url)
    else:
        logger.info("SUPABASE_DB_URL not configured. completed dry-run inference successfully.")

    return results


if __name__ == "__main__":
    run_inference_and_detection()