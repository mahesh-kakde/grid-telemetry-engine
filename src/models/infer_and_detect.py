import os
import math
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import polars as pl

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None
    execute_values = None

from src.models.load_artifacts import (
    ModelArtifactsBundle,
    REGIONS,
)
from src.features.build_features import (
    compute_telemetry_features,
    get_indian_holidays,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_VERSION = "2.0.0"
IST_OFFSET = timedelta(hours=5, minutes=30)


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
    widened: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # produces conformal-calibrated quantile demand forecasts with monotonicity enforcement
    if bundle is None:
        bundle = ModelArtifactsBundle()

    feature_cols = bundle.feature_columns
    features_subset = X[feature_cols].astype(float)

    p10_raw = bundle.lgb_models[region][0.10].predict(features_subset)
    p50_raw = bundle.lgb_models[region][0.50].predict(features_subset)
    p90_raw = bundle.lgb_models[region][0.90].predict(features_subset)

    cal = getattr(bundle, "quantile_calibration", {}).get(region, {})
    margin_key = "cqr_margin_90" if widened else "cqr_margin_80"
    margin = float(cal.get(margin_key, 0.0))

    p10_cal = np.maximum(0.0, p10_raw - margin)
    p90_cal = p90_raw + margin

    p10, p50, p90 = enforce_quantile_monotonicity(p10_cal, p50_raw, p90_cal)
    return p10, p50, p90


def _build_anomaly_feature_matrix(
    X: pd.DataFrame,
    y_actual: np.ndarray,
    anom_cols: List[str],
) -> pd.DataFrame:
    # aligns isolation forest inputs with live observed demand_mw and short-term shock deltas
    y_arr = np.asarray(y_actual, dtype=float)
    n_rows = len(X)
    anom_data: Dict[str, np.ndarray] = {}

    if "demand_mw" in anom_cols:
        anom_data["demand_mw"] = y_arr

    if "demand_delta_1h" in anom_cols:
        if "demand_lag_1h" in X.columns:
            anom_data["demand_delta_1h"] = y_arr - np.asarray(X["demand_lag_1h"], dtype=float)
        elif "demand_delta_1h" in X.columns:
            anom_data["demand_delta_1h"] = np.asarray(X["demand_delta_1h"], dtype=float)
        else:
            anom_data["demand_delta_1h"] = np.zeros(n_rows, dtype=float)

    if "demand_delta_24h" in anom_cols:
        if "demand_lag_24h" in X.columns:
            anom_data["demand_delta_24h"] = y_arr - np.asarray(X["demand_lag_24h"], dtype=float)
        elif "demand_delta_24h" in X.columns:
            anom_data["demand_delta_24h"] = np.asarray(X["demand_delta_24h"], dtype=float)
        else:
            anom_data["demand_delta_24h"] = np.zeros(n_rows, dtype=float)

    if "demand_zscore_24h" in anom_cols:
        if "demand_rolling_mean_24h" in X.columns and "demand_rolling_std_24h" in X.columns:
            mean_24 = np.asarray(X["demand_rolling_mean_24h"], dtype=float)
            std_24 = np.maximum(np.asarray(X["demand_rolling_std_24h"], dtype=float), 1.0)
            anom_data["demand_zscore_24h"] = (y_arr - mean_24) / std_24
        elif "demand_zscore_24h" in X.columns:
            anom_data["demand_zscore_24h"] = np.asarray(X["demand_zscore_24h"], dtype=float)
        else:
            anom_data["demand_zscore_24h"] = np.zeros(n_rows, dtype=float)

    for col in anom_cols:
        if col not in anom_data:
            if col in X.columns:
                anom_data[col] = np.asarray(X[col], dtype=float)
            else:
                anom_data[col] = np.zeros(n_rows, dtype=float)

    return pd.DataFrame(anom_data, index=X.index)[anom_cols].astype(float)


def detect_structural_anomalies(
    X: pd.DataFrame,
    y_actual: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    region: str,
    bundle: Optional[ModelArtifactsBundle] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # dual-trigger anomaly detection: observation-aware isolation forest + predictive iqr envelope breach
    if bundle is None:
        bundle = ModelArtifactsBundle()

    anom_cols = getattr(bundle, "anomaly_feature_columns", bundle.feature_columns)
    X_anom = _build_anomaly_feature_matrix(X, y_actual, anom_cols)

    # trigger a: observation-aware isolation forest joint surprise
    iso_model = bundle.isolation_forests[region]
    iso_scores = iso_model.score_samples(X_anom)
    trigger_a = (iso_model.predict(X_anom) == -1)

    # trigger b: residual iqr envelope violation
    iqr = np.maximum(p90 - p10, 1e-6)
    upper_envelope = p90 + (1.5 * iqr)
    lower_envelope = np.maximum(0.0, p10 - (1.5 * iqr))
    y_arr = np.asarray(y_actual, dtype=float)
    trigger_b = (y_arr > upper_envelope) | (y_arr < lower_envelope)

    # verified structural anomaly requires both isolation outlier and predictive envelope breach
    is_structural = trigger_a & trigger_b
    return iso_scores, is_structural, trigger_b


def fetch_telemetry_window_from_db(
    region: str,
    hours: int = 360,
    db_url: Optional[str] = None,
) -> Optional[pl.DataFrame]:
    # queries trailing hourly telemetry from supabase regional_telemetry table via cursor
    target_db = db_url or os.environ.get("SUPABASE_DB_URL")
    if not target_db or psycopg2 is None:
        return None

    query = """
        SELECT
            timestamp_utc,
            region,
            demand_mw,
            frequency_hz,
            temperature_2m,
            relative_humidity_2m,
            surface_solar_radiation,
            wind_speed_100m
        FROM regional_telemetry
        WHERE region = %s
        ORDER BY timestamp_utc DESC
        LIMIT %s;
    """
    conn = None
    try:
        conn = psycopg2.connect(target_db)
        with conn.cursor() as cur:
            cur.execute(query, (region, hours))
            rows = cur.fetchall()
            if not rows:
                return None
            col_names = [desc[0] for desc in cur.description]
    except Exception as exc:
        logger.warning(f"failed to query regional_telemetry for {region}: {exc}")
        return None
    finally:
        if conn is not None:
            conn.close()

    rows_asc = list(reversed(rows))
    df_pd = pd.DataFrame(rows_asc, columns=col_names)
    df_pd["timestamp_utc"] = pd.to_datetime(df_pd["timestamp_utc"], utc=True).dt.as_unit("us")
    for num_col in ["demand_mw", "frequency_hz", "temperature_2m", "relative_humidity_2m", "surface_solar_radiation", "wind_speed_100m"]:
        df_pd[num_col] = pd.to_numeric(df_pd[num_col], errors="coerce").astype(float)
    return pl.from_pandas(df_pd)


def build_baseline_history_pl(
    region: str,
    lookback_hours: int = 168,
    bundle: Optional[ModelArtifactsBundle] = None,
) -> pl.DataFrame:
    # builds hourly polars history using real reference weather & demand from training baseline
    if bundle is None:
        bundle = ModelArtifactsBundle()
    ref_df = bundle.reference_baselines[region].tail(lookback_hours).copy()
    end_ts = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_ts = end_ts - timedelta(hours=len(ref_df) - 1)

    records = []
    for idx, (_, row) in enumerate(ref_df.iterrows()):
        records.append({
            "timestamp_utc": start_ts + timedelta(hours=idx),
            "region": region,
            "demand_mw": float(row["demand_mw"]),
            "frequency_hz": 50.0,
            "temperature_2m": float(row["temperature_2m"]),
            "relative_humidity_2m": float(row["relative_humidity_2m"]),
            "surface_solar_radiation": float(row["surface_solar_radiation"]),
            "wind_speed_100m": float(row["wind_speed_100m"]),
        })
    return pl.DataFrame(records).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"))
    )


def fetch_live_or_baseline_history(
    region: str,
    lookback_hours: int = 240,
    bundle: Optional[ModelArtifactsBundle] = None,
    db_url: Optional[str] = None,
    max_staleness_hours: int = 72,
) -> Tuple[pl.DataFrame, str]:
    # loads live telemetry from supabase when fresh and sufficient, with baseline fallback
    live_df = fetch_telemetry_window_from_db(region=region, hours=lookback_hours, db_url=db_url)
    if live_df is not None and len(live_df) >= 168:
        latest_ts = live_df["timestamp_utc"].max()
        latest_dt = latest_ts.to_pydatetime() if isinstance(latest_ts, pd.Timestamp) else latest_ts
        if latest_dt.tzinfo is None:
            latest_dt = latest_dt.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - latest_dt.astimezone(timezone.utc)).total_seconds() / 3600.0
        if age_hours <= max_staleness_hours:
            if age_hours > 12.0:
                logger.warning(f"supabase telemetry for {region} is {age_hours:.1f}h old")
            return live_df, "supabase_live"
        logger.warning(
            f"supabase telemetry for {region} is stale ({age_hours:.1f}h > {max_staleness_hours}h); "
            "falling back to baseline"
        )
    return build_baseline_history_pl(region, lookback_hours=max(168, lookback_hours), bundle=bundle), "baseline_fallback"


def fetch_weather_forecast_map(
    region: str,
    forecast_days: int = 4,
) -> Dict[datetime, Dict[str, float]]:
    # pulls open-meteo 72h+ exogenous weather forecast mapped by whole-hour utc timestamp
    from src.ingestion.schemas import GridRegion
    from src.ingestion.fetch_telemetry import REGIONAL_COORDINATES, fetch_weather_with_retry

    lat, lon = REGIONAL_COORDINATES[region]
    try:
        weather_records = fetch_weather_with_retry(
            lat=lat,
            lon=lon,
            region=GridRegion(region),
            past_days=0,
            forecast_days=forecast_days,
        )
    except Exception as exc:
        logger.warning(f"weather forecast fetch failed for {region}; falling back to diurnal persistence: {exc}")
        return {}

    return {
        w.timestamp_utc.replace(minute=0, second=0, microsecond=0): {
            "temperature_2m": float(w.temperature_2m),
            "relative_humidity_2m": float(w.relative_humidity_2m),
            "surface_solar_radiation": float(w.surface_solar_radiation),
            "wind_speed_100m": float(w.wind_speed_100m),
        }
        for w in weather_records
    }


def _align_hourly_rollout_history(df_reg: pl.DataFrame, region: str) -> pl.DataFrame:
    # reindexes rollout history onto a uniform 1h grid so gaps never alias ring-buffer lags
    ts_dtype = df_reg.schema["timestamp_utc"]
    if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
        ts_expr = pl.col("timestamp_utc").dt.convert_time_zone("UTC").dt.cast_time_unit("us")
    elif isinstance(ts_dtype, pl.Datetime):
        ts_expr = pl.col("timestamp_utc").dt.replace_time_zone("UTC").dt.cast_time_unit("us")
    else:
        ts_expr = pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"))

    df_norm = (
        df_reg.with_columns(ts_expr.alias("timestamp_utc"))
        .unique(subset=["region", "timestamp_utc"], keep="last")
        .sort("timestamp_utc")
    )
    df_grid = (
        df_norm.upsample(time_column="timestamp_utc", every="1h", group_by="region")
        .with_columns(pl.col("region").forward_fill())
    )

    missing_slots = int(df_grid["demand_mw"].null_count())
    if missing_slots > 0:
        logger.warning(
            f"detected {missing_slots} missing hourly slot(s) in {region} rollout history; "
            "applying diurnal t-24h and linear backfill"
        )
        fill_cols = [
            col
            for col in [
                "demand_mw",
                "temperature_2m",
                "relative_humidity_2m",
                "surface_solar_radiation",
                "wind_speed_100m",
            ]
            if col in df_grid.columns
        ]
        # preserve diurnal cycles first via t-24h/t-48h/t+24h, then interpolate any remainder
        df_grid = df_grid.with_columns([
            pl.coalesce([
                pl.col(c),
                pl.col(c).shift(24),
                pl.col(c).shift(48),
                pl.col(c).shift(-24),
            ]).alias(c)
            for c in fill_cols
        ]).with_columns([
            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
            for c in fill_cols
        ])

    return df_grid


def generate_72h_multi_step_forecast(
    df_history: pl.DataFrame,
    region: str,
    bundle: Optional[ModelArtifactsBundle] = None,
    widened: bool = False,
    weather_forecast_map: Optional[Dict[datetime, Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    # generates 72-hour iterative multi-step quantile forecasts using ring-buffer updates
    if bundle is None:
        bundle = ModelArtifactsBundle()

    df_reg = df_history.filter(pl.col("region") == region).sort("timestamp_utc")
    if len(df_reg) < 168:
        raise ValueError(f"insufficient historical lookback for {region}: requires at least 168 hours")

    df_aligned = _align_hourly_rollout_history(df_reg, region)
    tail_df = df_aligned.tail(240)
    latest_ts = tail_df["timestamp_utc"].max()
    if isinstance(latest_ts, pd.Timestamp):
        latest_utc = latest_ts.to_pydatetime()
    else:
        latest_utc = latest_ts
    if latest_utc.tzinfo is None:
        latest_utc = latest_utc.replace(tzinfo=timezone.utc)
    else:
        latest_utc = latest_utc.astimezone(timezone.utc)

    forecast_generated_at = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    years = sorted({
        (latest_utc + IST_OFFSET).year,
        (latest_utc + timedelta(hours=72) + IST_OFFSET).year,
    })
    holidays_set = get_indian_holidays(years)

    demand_hist: List[float] = [float(x) for x in tail_df["demand_mw"].to_list()]
    temp_hist: List[float] = [float(x if x is not None else 25.0) for x in tail_df["temperature_2m"].to_list()]
    rh_hist: List[float] = [float(x if x is not None else 55.0) for x in tail_df["relative_humidity_2m"].to_list()]
    solar_hist: List[float] = [float(x if x is not None else 0.0) for x in tail_df["surface_solar_radiation"].to_list()]
    wind_hist: List[float] = [float(x if x is not None else 4.0) for x in tail_df["wind_speed_100m"].to_list()]

    weather_map = weather_forecast_map or {}
    feature_cols = bundle.feature_columns
    forecasts: List[Dict[str, Any]] = []

    for step in range(1, 73):
        target_ts = latest_utc + timedelta(hours=step)
        hour = target_ts.hour
        day_of_week = target_ts.isoweekday()
        day_of_year = target_ts.timetuple().tm_yday
        ist_date = (target_ts + IST_OFFSET).date()
        is_holiday = 1 if ist_date in holidays_set else 0

        wx = weather_map.get(target_ts.replace(minute=0, second=0, microsecond=0))
        if wx is not None:
            temp = wx["temperature_2m"]
            rh = wx["relative_humidity_2m"]
            solar = wx["surface_solar_radiation"]
            wind = wx["wind_speed_100m"]
        else:
            # repeat t-24h diurnal weather profile rather than freezing a single hour
            temp = temp_hist[-24]
            rh = rh_hist[-24]
            solar = solar_hist[-24]
            wind = wind_hist[-24]

        row_feats = {
            "sin_hour_k1": math.sin(2.0 * math.pi * hour / 24.0),
            "cos_hour_k1": math.cos(2.0 * math.pi * hour / 24.0),
            "sin_hour_k2": math.sin(4.0 * math.pi * hour / 24.0),
            "cos_hour_k2": math.cos(4.0 * math.pi * hour / 24.0),
            "sin_doy": math.sin(2.0 * math.pi * day_of_year / 365.25),
            "cos_doy": math.cos(2.0 * math.pi * day_of_year / 365.25),
            "day_of_week": float(day_of_week),
            "is_holiday": float(is_holiday),
            "demand_lag_1h": demand_hist[-1],
            "demand_lag_2h": demand_hist[-2],
            "demand_lag_3h": demand_hist[-3],
            "demand_lag_24h": demand_hist[-24],
            "demand_lag_48h": demand_hist[-48],
            "demand_lag_168h": demand_hist[-168],
            "demand_rolling_mean_6h": float(np.mean(demand_hist[-6:])),
            "demand_rolling_mean_24h": float(np.mean(demand_hist[-24:])),
            "demand_rolling_mean_168h": float(np.mean(demand_hist[-168:])),
            "demand_rolling_std_6h": float(np.std(demand_hist[-6:], ddof=1)),
            "demand_rolling_std_24h": float(np.std(demand_hist[-24:], ddof=1)),
            "demand_rolling_std_168h": float(np.std(demand_hist[-168:], ddof=1)),
            "temperature_2m": temp,
            "relative_humidity_2m": rh,
            "surface_solar_radiation": solar,
            "wind_speed_100m": wind,
            "temp_delta_24h": temp - temp_hist[-24],
            "rh_delta_24h": rh - rh_hist[-24],
        }
        X_step = pd.DataFrame([[row_feats[c] for c in feature_cols]], columns=feature_cols)
        p10, p50, p90 = predict_quantiles(X_step, region, bundle, widened=widened)

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

        demand_hist.append(float(p50[0]))
        temp_hist.append(temp)
        rh_hist.append(rh)
        solar_hist.append(solar)
        wind_hist.append(wind)

    return forecasts


def persist_forecasts_to_supabase(
    forecasts: List[Dict[str, Any]],
    db_url: str,
) -> int:
    # idempotent vintage-preserving upsert into model_forecasts table
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
        ON CONFLICT (model_version, region, forecast_generated_at, target_timestamp_utc) DO UPDATE SET
            forecast_p10 = EXCLUDED.forecast_p10,
            forecast_p50 = EXCLUDED.forecast_p50,
            forecast_p90 = EXCLUDED.forecast_p90,
            actual_demand_mw = COALESCE(EXCLUDED.actual_demand_mw, model_forecasts.actual_demand_mw);
    """

    try:
        conn = psycopg2.connect(db_url)
    except Exception as exc:
        logger.error(
            f"failed to connect to supabase database: {exc}. "
            "verify SUPABASE_DB_URL secret format (e.g. postgresql://postgres:[PASSWORD]@[HOST]:5432/postgres?sslmode=require)"
        )
        raise

    try:
        with conn.cursor() as cur:
            execute_values(cur, upsert_sql, rows)
        conn.commit()
        logger.info(f"successfully upserted {len(rows)} forecast rows to supabase")
        return len(rows)
    except Exception as exc:
        logger.error(
            f"failed to upsert forecast records: {exc}. "
            "if schema needs migration, execute scripts/migrate_v2.sql in the supabase sql editor."
        )
        raise
    finally:
        conn.close()


def backfill_forecast_actuals(db_url: str) -> int:
    # populates realized demand_mw into past forecast rows for live error evaluation
    if not db_url or psycopg2 is None:
        return 0
    sql = """
        UPDATE model_forecasts mf
        SET actual_demand_mw = rt.demand_mw
        FROM regional_telemetry rt
        WHERE mf.region = rt.region
          AND mf.target_timestamp_utc = rt.timestamp_utc
          AND mf.actual_demand_mw IS NULL;
    """
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            updated = cur.rowcount
        conn.commit()
        logger.info(f"backfilled {updated} realized demand values into model_forecasts")
        return max(0, int(updated))
    finally:
        conn.close()


def run_inference_and_detection() -> Dict[str, Any]:
    # batch execution runner invoked by scheduled etl pipeline
    from src.models.explainers import AnomalySHAPExplainer, persist_anomaly_events_to_supabase
    from src.monitoring.drift_check import evaluate_feature_drift, select_drift_observation_window

    logger.info("initializing batch forecasting and anomaly detection engine...")
    bundle = ModelArtifactsBundle()
    explainer = AnomalySHAPExplainer(bundle)
    db_url = os.environ.get("SUPABASE_DB_URL")

    results: Dict[str, Any] = {}
    all_forecasts: List[Dict[str, Any]] = []
    all_anomaly_events: List[Dict[str, Any]] = []

    for region in REGIONS:
        df_history, source_tag = fetch_live_or_baseline_history(
            region=region,
            lookback_hours=360,
            bundle=bundle,
            db_url=db_url,
        )
        feat_pl = compute_telemetry_features(df_history, drop_nulls_flag=True)
        if feat_pl.is_empty():
            ref_pd = bundle.reference_baselines[region].copy()
        else:
            ref_pd = feat_pl.to_pandas()

        eval_window = ref_pd.tail(48).copy()
        X = eval_window[bundle.feature_columns]
        y = eval_window["demand_mw"].to_numpy()

        p10, p50, p90 = predict_quantiles(X, region, bundle)
        iso_scores, is_structural, trigger_b = detect_structural_anomalies(
            X, y, p10, p50, p90, region, bundle
        )

        n_anomalies = int(np.sum(is_structural))
        logger.info(
            f"region {region} ({source_tag}): evaluated {len(y)} records, flagged {n_anomalies} structural anomalies"
        )
        results[region] = {
            "source": source_tag,
            "records": len(y),
            "anomalies_detected": n_anomalies,
            "mean_demand_p50": float(np.mean(p50)),
        }

        if db_url:
            flagged_indices = np.where(is_structural | trigger_b)[0]
            # always store latest nominal/alert state plus any flagged events
            indices_to_persist = sorted(set(flagged_indices.tolist() + [len(eval_window) - 1]))
            for idx_pos in indices_to_persist:
                row_s = eval_window.iloc[idx_pos]
                ts_val = row_s.get("timestamp_utc") or datetime.now(timezone.utc)
                top_shap = explainer.explain_anomaly_point(row_s, region, top_k=3)
                all_anomaly_events.append({
                    "timestamp_utc": ts_val,
                    "region": region,
                    "actual_demand_mw": float(y[idx_pos]),
                    "predicted_p10_mw": float(p10[idx_pos]),
                    "predicted_p50_mw": float(p50[idx_pos]),
                    "predicted_p90_mw": float(p90[idx_pos]),
                    "anomaly_score": float(iso_scores[idx_pos]),
                    "is_structural": bool(is_structural[idx_pos]),
                    "residual_breach": bool(trigger_b[idx_pos]),
                    "top_shap_features": top_shap,
                })

            baseline_ref = bundle.reference_baselines[region]
            live_feat_df = None if feat_pl.is_empty() else ref_pd
            drift_obs_df, _ = select_drift_observation_window(
                live_feat_df, baseline_ref, window_hours=168
            )
            drift_res = evaluate_feature_drift(baseline_ref, drift_obs_df)
            widened = bool(drift_res.get("drift_alarm", False))
            wx_map = fetch_weather_forecast_map(region, forecast_days=4)
            forecasts = generate_72h_multi_step_forecast(
                df_history,
                region,
                bundle,
                widened=widened,
                weather_forecast_map=wx_map,
            )
            all_forecasts.extend(forecasts)

    if db_url:
        logger.info("persisting anomaly events and 72h forecasts to supabase...")
        persist_anomaly_events_to_supabase(all_anomaly_events, db_url)
        persist_forecasts_to_supabase(all_forecasts, db_url)
        backfill_forecast_actuals(db_url)
    else:
        logger.info("SUPABASE_DB_URL not configured. dry-run inference completed successfully.")

    return results


if __name__ == "__main__":
    run_inference_and_detection()