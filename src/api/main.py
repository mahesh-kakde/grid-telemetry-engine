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
from fastapi.responses import JSONResponse, FileResponse
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
    # warm up model artifacts and shap explainers during boot
    logger.info("initializing in-memory model artifacts bundle...")
    bundle = ModelArtifactsBundle()
    bundle.preload_all()
    app_state["bundle"] = bundle

    logger.info("initializing shap anomaly explainer...")
    explainer = AnomalySHAPExplainer(bundle)
    app_state["explainer"] = explainer
    logger.info("application runtime warmup complete")
    yield
    app_state.clear()


app = FastAPI(
    title="Grid Telemetry Engine API",
    description="Internal loopback REST service for probabilistic grid forecasting and anomaly detection",
    version="2.0.0",
    lifespan=lifespan,
)

# permissive loopback cors policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    # produces 72-hour recursive multi-step quantile predictions
    if req.region not in REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown region '{req.region}'. supported regions: {REGIONS}",
        )

    bundle: ModelArtifactsBundle = app_state.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="model bundle not initialized")

    # synthesize historical telemetry frame from reference baseline
    ref_df = load_reference_baseline(req.region).tail(req.lookback_hours).copy()
    base_ts = datetime.now(timezone.utc) - timedelta(hours=len(ref_df))
    records = []
    for idx, (_, row) in enumerate(ref_df.iterrows()):
        records.append({
            "timestamp_utc": base_ts + timedelta(hours=idx),
            "region": req.region,
            "demand_mw": float(row["demand_mw"]),
            "frequency_hz": 50.0,
            "temperature_2m": 30.0,
            "relative_humidity_2m": 50.0,
            "surface_solar_radiation": 350.0,
            "wind_speed_100m": 4.0,
        })
    df_history = pl.DataFrame(records)

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

    bundle: ModelArtifactsBundle = app_state.get("bundle")
    explainer: AnomalySHAPExplainer = app_state.get("explainer")
    if bundle is None or explainer is None:
        raise HTTPException(status_code=503, detail="inference runtime not initialized")

    ref_df = load_reference_baseline(req.region)
    feature_row = ref_df.iloc[-1]
    features_df = pd.DataFrame([feature_row[bundle.feature_columns]])

    p10, p50, p90 = predict_quantiles(features_df, req.region, bundle)
    actual_demand = req.actual_demand_mw if req.actual_demand_mw is not None else float(feature_row["demand_mw"])

    scores, is_structural, trigger_b = detect_structural_anomalies(
        X=features_df,
        y_actual=np.array([actual_demand]),
        p10=p10,
        p50=p50,
        p90=p90,
        region=req.region,
        bundle=bundle,
    )

    # compute top-3 shap drivers
    top_shap = explainer.explain_anomaly_point(feature_row, req.region, top_k=3)

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
    }


@app.get("/drift")
def get_drift_metrics(region: str = Query("ALL_INDIA", description="Regional grid identifier")) -> Dict[str, Any]:
    # returns statistical drift indicators and advisory recommendations
    if region not in REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown region '{region}'. supported regions: {REGIONS}",
        )

    ref_df = load_reference_baseline(region)
    curr_df = fetch_trailing_telemetry_window(region=region, window_hours=168)
    feature_cols = load_feature_columns()

    summary = evaluate_feature_drift(ref_df, curr_df, feature_cols=feature_cols)
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