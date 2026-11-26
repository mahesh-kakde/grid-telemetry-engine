import os
import sys

# ensure project root is present in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import polars as pl

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.models.load_artifacts import (
    ModelArtifactsBundle,
    REGIONS,
    load_reference_baseline,
    load_feature_columns,
)
from src.models.infer_and_detect import (
    generate_72h_multi_step_forecast,
    predict_quantiles,
    detect_structural_anomalies,
    fetch_live_or_baseline_history,
    MODEL_VERSION,
)
from src.models.explainers import AnomalySHAPExplainer
from src.monitoring.drift_check import (
    evaluate_feature_drift,
    DEFAULT_REPORT_PATH,
    fetch_trailing_telemetry_window,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# shared in-memory state container
app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # warm up model artifacts and shap explainers lazily
    logger.info("initializing in-memory model artifacts bundle...")
    bundle = ModelArtifactsBundle()
    app_state["bundle"] = bundle

    logger.info("initializing shap anomaly explainer...")
    explainer = AnomalySHAPExplainer(bundle)
    app_state["explainer"] = explainer
    logger.info("application runtime warmup complete")
    yield
    app_state.clear()


app = FastAPI(
    title="Grid Telemetry Engine API",
    description="Optional developer REST service for probabilistic grid forecasting and anomaly detection",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# restricted cors origins (no wildcard credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8501",
        "http://localhost:8501",
        "https://grid.maheshkakde.com",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.exception_handler(404)
async def custom_404_handler(request, exc):
    from fastapi.responses import HTMLResponse
    return HTMLResponse(
        content="""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>404 Not Found | grid-telemetry-engine</title>
    <style>
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #f8fafc;
            color: #0f172a;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }
        .error-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 56px 36px;
            text-align: center;
            max-width: 520px;
            box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.05);
        }
        .error-code { font-size: 88px; font-weight: 800; line-height: 1; margin-bottom: 8px; letter-spacing: -0.04em; }
        .error-title { font-size: 22px; font-weight: 700; color: #1e293b; margin-bottom: 12px; }
        .error-desc { font-size: 14.5px; color: #64748b; line-height: 1.6; margin-bottom: 32px; }
        .actions { display: flex; gap: 16px; justify-content: center; }
        .btn-home {
            background-color: #2563eb; color: #ffffff; text-decoration: none; font-weight: 600;
            font-size: 14px; padding: 10px 28px; border-radius: 6px; box-shadow: 0 1px 3px rgba(37, 99, 235, 0.25);
        }
        .btn-home:hover { background-color: #1d4ed8; }
        .btn-portfolio {
            background-color: #ffffff; color: #0f172a; text-decoration: none; font-weight: 600;
            font-size: 14px; padding: 10px 28px; border-radius: 6px; border: 1px solid #cbd5e1;
        }
        .btn-portfolio:hover { background-color: #f8fafc; }
    </style>
</head>
<body>
    <div class="error-card">
        <div class="error-code">404</div>
        <div class="error-title">Page Not Found</div>
        <div class="error-desc">The power grid telemetry dispatch route or resource you are looking for does not exist or has been moved.</div>
        <div class="actions">
            <a href="https://grid.maheshkakde.com" class="btn-home">Home</a>
            <a href="https://maheshkakde.com" class="btn-portfolio" target="_blank" rel="noopener noreferrer">Portfolio</a>
        </div>
    </div>
</body>
</html>""",
        status_code=404,
    )


class ForecastRequest(BaseModel):
    region: str = Field("ALL_INDIA", description="Regional load despatch grid (NR, WR, SR, ER, NER, ALL_INDIA)")
    lookback_hours: int = Field(168, ge=168, le=720, description="Hours of lookback telemetry")


class AnomalyCheckRequest(BaseModel):
    region: str = Field("ALL_INDIA", description="Regional grid identifier")
    actual_demand_mw: Optional[float] = Field(None, description="Observed active grid demand in MW")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, Any]:
    # supervisor health probe endpoint
    bundle_ready = "bundle" in app_state and app_state["bundle"] is not None
    return {
        "status": "healthy" if bundle_ready else "initializing",
        "service": "grid-telemetry-engine-api",
        "model_version": MODEL_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_loaded": bundle_ready,
    }


@app.get("/regions")
def get_supported_regions() -> Dict[str, List[str]]:
    # lists active regional load despatch areas
    return {"regions": REGIONS}


@app.post("/forecast")
def generate_forecast(req: ForecastRequest) -> Dict[str, Any]:
    # produces 72-hour recursive multi-step quantile predictions via snapshot cache or model bundle
    if req.region not in REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown region '{req.region}'. supported regions: {REGIONS}",
        )

    # instant return from precomputed snapshot when standard 168h lookback requested
    if req.lookback_hours == 168:
        from src.models.precompute import load_precomputed_data
        precomputed = load_precomputed_data(req.region)
        if precomputed and "forecasts" in precomputed:
            return {
                "region": req.region,
                "model_version": MODEL_VERSION,
                "forecast_generated_at": precomputed.get("generated_at", datetime.now(timezone.utc).isoformat()),
                "forecast_horizon_hours": len(precomputed["forecasts"]),
                "forecasts": precomputed["forecasts"],
            }

    bundle: ModelArtifactsBundle = app_state.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="model bundle not initialized")

    df_history, _ = fetch_live_or_baseline_history(
        req.region,
        lookback_hours=req.lookback_hours,
        bundle=bundle,
    )

    forecast_records = generate_72h_multi_step_forecast(df_history, req.region, bundle)
    serialized_forecasts = [
        {
            "target_timestamp_utc": fc["target_timestamp_utc"].isoformat(),
            "region": fc["region"],
            "forecast_p10": round(fc["forecast_p10"], 2),
            "forecast_p50": round(fc["forecast_p50"], 2),
            "forecast_p90": round(fc["forecast_p90"], 2),
            "model_version": fc["model_version"],
        }
        for fc in forecast_records
    ]

    return {
        "region": req.region,
        "model_version": MODEL_VERSION,
        "forecast_generated_at": datetime.now(timezone.utc).isoformat(),
        "forecast_horizon_hours": len(serialized_forecasts),
        "forecasts": serialized_forecasts,
    }


@app.post("/anomalies")
def evaluate_anomaly(req: AnomalyCheckRequest) -> Dict[str, Any]:
    # evaluates anomaly status and computes shap physical feature attributions
    if req.region not in REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown region '{req.region}'. supported regions: {REGIONS}",
        )

    # instant return from precomputed nominal snapshot when no custom demand is provided
    if req.actual_demand_mw is None:
        from src.models.precompute import load_precomputed_anomaly
        precomputed = load_precomputed_anomaly(req.region)
        if precomputed is not None:
            return precomputed

    bundle: ModelArtifactsBundle = app_state.get("bundle")
    explainer: AnomalySHAPExplainer = app_state.get("explainer")
    if bundle is None or explainer is None:
        raise HTTPException(status_code=503, detail="inference runtime not initialized")

    from src.features.build_features import compute_telemetry_features
    df_history, _ = fetch_live_or_baseline_history(req.region, lookback_hours=240, bundle=bundle)
    feat_pl = compute_telemetry_features(df_history, drop_nulls_flag=True)
    feat_df = bundle.reference_baselines[req.region] if feat_pl.is_empty() else feat_pl.to_pandas()
    feature_row = feat_df.iloc[-1].copy()
    features_df = pd.DataFrame([feature_row[bundle.feature_columns]])

    p10, p50, p90 = predict_quantiles(features_df, req.region, bundle)
    actual_demand = float(req.actual_demand_mw) if req.actual_demand_mw is not None else float(feature_row["demand_mw"])

    scores, is_structural, trigger_b = detect_structural_anomalies(
        X=features_df,
        y_actual=np.array([actual_demand]),
        p10=p10,
        p50=p50,
        p90=p90,
        region=req.region,
        bundle=bundle,
    )

    # compute top-3 baseline forecast drivers (mw) and observation-aware anomaly drivers
    top_shap = explainer.explain_anomaly_point(feature_row, req.region, top_k=3)
    anomaly_shap = explainer.explain_anomaly_score(
        feature_row, req.region, actual_demand_mw=actual_demand, top_k=3
    )

    return {
        "region": req.region,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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


@app.get("/drift")
def get_drift_metrics(region: str = Query("ALL_INDIA", description="Regional grid identifier")) -> Dict[str, Any]:
    # returns statistical drift indicators and advisory recommendations
    if region not in REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown region '{region}'. supported regions: {REGIONS}",
        )

    from src.models.precompute import load_precomputed_drift
    precomputed = load_precomputed_drift(region)
    if precomputed is not None:
        return precomputed

    ref_df = load_reference_baseline(region)
    curr_df = fetch_trailing_telemetry_window(region=region, window_hours=168)

    summary = evaluate_feature_drift(ref_df, curr_df, feature_cols=None)
    summary["region"] = region
    summary["report_available"] = os.path.exists(DEFAULT_REPORT_PATH)
    return summary


@app.get("/drift/report")
def get_drift_report_html():
    # serves standalone generated evidently html report
    if not os.path.exists(DEFAULT_REPORT_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="drift report not yet generated. run src.monitoring.drift_check first.",
        )
    return FileResponse(DEFAULT_REPORT_PATH, media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=False)