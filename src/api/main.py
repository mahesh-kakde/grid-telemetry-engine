import os
import sys

# ensure project root is present in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import logging
import hmac
import math
import time
import threading
from collections import OrderedDict, deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from contextlib import asynccontextmanager, contextmanager
import numpy as np
import pandas as pd

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.models import precompute as precompute_mod
from src.models.load_artifacts import (
    ModelArtifactsBundle,
    REGIONS,
    load_reference_baseline,
    resolve_calendar_timezone,
    validate_artifacts_presence,
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

STATIC_REPORTS_DIR = os.path.dirname(DEFAULT_REPORT_PATH)
COMPUTE_RATE_PATHS = frozenset({"/forecast", "/anomalies", "/drift"})
RATE_EXEMPT_PATHS = frozenset({"/health", "/healthz"})
MAX_INFERENCE_LOOKBACK_HOURS = 240
LIVE_FORECAST_CACHE_TTL_SEC = 300.0
LIVE_FORECAST_CACHE_MAX_ENTRIES = 24

API_SECURITY_HEADERS: Tuple[Tuple[str, str], ...] = (
    ("strict-transport-security", "max-age=31536000; includeSubDomains"),
    ("x-content-type-options", "nosniff"),
    ("x-frame-options", "SAMEORIGIN"),
    ("referrer-policy", "strict-origin-when-cross-origin"),
    ("permissions-policy", "camera=(), microphone=(), geolocation=()"),
    (
        "content-security-policy",
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "frame-ancestors 'self'; "
        "base-uri 'self'; "
        "object-src 'none'",
    ),
)

# shared in-memory state container
app_state: Dict[str, Any] = {}
_INFERENCE_SEMAPHORE = threading.BoundedSemaphore(value=2)
_LIVE_CACHE_LOCK = threading.Lock()
_LIVE_FORECAST_CACHE: "OrderedDict[Tuple[str, int, bool], Tuple[float, Dict[str, Any]]]" = OrderedDict()


class SlidingWindowRateLimiter:
    """Bounded per-client sliding-window rate limiter safe for single-worker ASGI."""

    def __init__(self, max_clients: int = 2048, window_sec: float = 60.0) -> None:
        self._max_clients = max_clients
        self._window_sec = window_sec
        self._lock = threading.Lock()
        self._buckets: "OrderedDict[Tuple[str, str], deque[float]]" = OrderedDict()

    def allow(self, client_key: str, bucket: str, limit: int) -> Tuple[bool, int]:
        if limit <= 0:
            return True, 0
        now = time.monotonic()
        cutoff = now - self._window_sec
        compound_key = (client_key, bucket)
        with self._lock:
            timestamps = self._buckets.get(compound_key)
            if timestamps is None:
                if len(self._buckets) >= self._max_clients:
                    self._buckets.popitem(last=False)
                timestamps = deque()
                self._buckets[compound_key] = timestamps
            else:
                self._buckets.move_to_end(compound_key)

            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= limit:
                retry_after = max(1, math.ceil(self._window_sec - (now - timestamps[0])))
                return False, retry_after

            timestamps.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


rate_limiter = SlidingWindowRateLimiter()


def _parse_positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        parsed = int(raw)
        return parsed if parsed > 0 else default
    except ValueError:
        return default


def _get_configured_api_token() -> Optional[str]:
    token = os.environ.get("API_AUTH_TOKEN") or os.environ.get("GRID_API_KEY")
    if token is None:
        return None
    stripped = token.strip()
    return stripped if stripped else None


def verify_api_key(request: Request) -> None:
    """Validates Bearer or X-API-Key token when API auth is configured."""
    expected = _get_configured_api_token()
    if expected is None:
        return

    candidate: Optional[str] = request.headers.get("x-api-key")
    if candidate is None:
        auth_header = request.headers.get("authorization", "")
        parts = auth_header.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            candidate = parts[1].strip()

    if not candidate or not hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing api credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _apply_security_headers(response: Response) -> Response:
    for header_name, header_val in API_SECURITY_HEADERS:
        response.headers.setdefault(header_name, header_val)
    response.headers.setdefault("cache-control", "no-store")
    return response


def _validate_region(region: str) -> str:
    if region not in REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown region '{region}'. supported regions: {REGIONS}",
        )
    return region


def _resolve_widened_from_snapshot(snapshot: Optional[Dict[str, Any]]) -> bool:
    if not snapshot:
        return False
    if "widened" in snapshot:
        return bool(snapshot["widened"])
    drift = snapshot.get("drift_summary") or {}
    return bool(drift.get("drift_alarm", False))


@contextmanager
def _inference_slot():
    # cap concurrent live lightgbm rollouts to protect render 512mb cpu/ram budget
    acquired = _INFERENCE_SEMAPHORE.acquire(timeout=2.0)
    if not acquired:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="inference capacity saturated; retry shortly",
            headers={"Retry-After": "5"},
        )
    try:
        yield
    finally:
        _INFERENCE_SEMAPHORE.release()


def _get_cached_live_forecast(cache_key: Tuple[str, int, bool]) -> Optional[Dict[str, Any]]:
    now = time.monotonic()
    with _LIVE_CACHE_LOCK:
        entry = _LIVE_FORECAST_CACHE.get(cache_key)
        if entry is None:
            return None
        expires_at, payload = entry
        if now >= expires_at:
            _LIVE_FORECAST_CACHE.pop(cache_key, None)
            return None
        _LIVE_FORECAST_CACHE.move_to_end(cache_key)
        return dict(payload)


def _set_cached_live_forecast(cache_key: Tuple[str, int, bool], payload: Dict[str, Any]) -> None:
    expires_at = time.monotonic() + LIVE_FORECAST_CACHE_TTL_SEC
    with _LIVE_CACHE_LOCK:
        if cache_key in _LIVE_FORECAST_CACHE:
            _LIVE_FORECAST_CACHE.move_to_end(cache_key)
        elif len(_LIVE_FORECAST_CACHE) >= LIVE_FORECAST_CACHE_MAX_ENTRIES:
            _LIVE_FORECAST_CACHE.popitem(last=False)
        _LIVE_FORECAST_CACHE[cache_key] = (expires_at, dict(payload))


@asynccontextmanager
async def lifespan(app: FastAPI):
    rate_limiter.reset()
    with _LIVE_CACHE_LOCK:
        _LIVE_FORECAST_CACHE.clear()

    if _get_configured_api_token() is None:
        logger.warning(
            "API_AUTH_TOKEN / GRID_API_KEY not configured; rest endpoints allow unauthenticated requests"
        )

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
    with _LIVE_CACHE_LOCK:
        _LIVE_FORECAST_CACHE.clear()


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
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


@app.middleware("http")
async def security_and_rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    if request.method != "OPTIONS" and path not in RATE_EXEMPT_PATHS:
        client_host = request.client.host if request.client else "unknown"
        is_compute = path in COMPUTE_RATE_PATHS
        bucket = "compute" if is_compute else "general"
        limit = (
            _parse_positive_int_env("API_COMPUTE_RATE_LIMIT_RPM", 15)
            if is_compute
            else _parse_positive_int_env("API_RATE_LIMIT_RPM", 60)
        )
        allowed, retry_after = rate_limiter.allow(client_host, bucket, limit)
        if not allowed:
            denied = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "rate limit exceeded; retry shortly"},
                headers={"Retry-After": str(retry_after)},
            )
            return _apply_security_headers(denied)

    response = await call_next(request)
    return _apply_security_headers(response)


@app.exception_handler(404)
async def custom_404_handler(request, exc):
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
    model_config = ConfigDict(allow_inf_nan=False)

    region: str = Field("ALL_INDIA", description="Regional load despatch grid (NR, WR, SR, ER, NER, ALL_INDIA)")
    lookback_hours: int = Field(168, ge=168, le=720, description="Hours of lookback telemetry")


class AnomalyCheckRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    region: str = Field("ALL_INDIA", description="Regional grid identifier")
    actual_demand_mw: Optional[float] = Field(
        None,
        ge=0.0,
        le=1_000_000.0,
        description="Observed active grid demand in MW",
    )


@app.get("/health", status_code=status.HTTP_200_OK)
@app.get("/healthz", status_code=status.HTTP_200_OK)
def health_check(response: Response) -> Dict[str, Any]:
    # supervisor health probe endpoint
    bundle_ready = "bundle" in app_state and app_state["bundle"] is not None
    try:
        artifacts_ready = validate_artifacts_presence()
    except Exception as exc:
        logger.warning(f"api health artifact validation failed: {exc}")
        artifacts_ready = False

    is_ready = bundle_ready and artifacts_ready
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "healthy" if is_ready else "initializing",
        "service": "grid-telemetry-engine-api",
        "model_version": MODEL_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_loaded": bundle_ready,
        "artifacts_ready": artifacts_ready,
    }


@app.get("/regions", dependencies=[Depends(verify_api_key)])
def get_supported_regions() -> Dict[str, List[str]]:
    # lists active regional load despatch areas
    return {"regions": REGIONS}


@app.post("/forecast", dependencies=[Depends(verify_api_key)])
def generate_forecast(req: ForecastRequest) -> Dict[str, Any]:
    # produces 72-hour recursive multi-step quantile predictions via snapshot cache or model bundle
    _validate_region(req.region)

    precomputed = precompute_mod.load_precomputed_data(req.region)
    widened = _resolve_widened_from_snapshot(precomputed)

    # instant return from precomputed snapshot when standard 168h lookback requested
    if req.lookback_hours == 168 and precomputed and "forecasts" in precomputed:
        return {
            "region": req.region,
            "model_version": MODEL_VERSION,
            "forecast_generated_at": precomputed.get("generated_at", datetime.now(timezone.utc).isoformat()),
            "forecast_horizon_hours": len(precomputed["forecasts"]),
            "widened": widened,
            "forecasts": precomputed["forecasts"],
        }

    # cap effective lookback to the 240h rollout ring-buffer and deduplicate repeated custom lookbacks
    effective_lookback = min(req.lookback_hours, MAX_INFERENCE_LOOKBACK_HOURS)
    cache_key = (req.region, effective_lookback, widened)
    cached_payload = _get_cached_live_forecast(cache_key)
    if cached_payload is not None:
        return cached_payload

    bundle: ModelArtifactsBundle = app_state.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="model bundle not initialized")

    with _inference_slot():
        # re-check cache inside slot in case a concurrent caller populated it while waiting
        cached_payload = _get_cached_live_forecast(cache_key)
        if cached_payload is not None:
            return cached_payload

        df_history, _ = fetch_live_or_baseline_history(
            req.region,
            lookback_hours=effective_lookback,
            bundle=bundle,
        )
        forecast_records = generate_72h_multi_step_forecast(
            df_history,
            req.region,
            bundle,
            widened=widened,
        )

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

    result = {
        "region": req.region,
        "model_version": MODEL_VERSION,
        "forecast_generated_at": datetime.now(timezone.utc).isoformat(),
        "forecast_horizon_hours": len(serialized_forecasts),
        "widened": widened,
        "forecasts": serialized_forecasts,
    }
    _set_cached_live_forecast(cache_key, result)
    return result


@app.post("/anomalies", dependencies=[Depends(verify_api_key)])
def evaluate_anomaly(req: AnomalyCheckRequest) -> Dict[str, Any]:
    # evaluates anomaly status and computes shap physical feature attributions with cqr widening parity
    _validate_region(req.region)

    precomputed = precompute_mod.load_precomputed_data(req.region)
    widened = _resolve_widened_from_snapshot(precomputed)

    # instant return from precomputed nominal snapshot when no custom demand is provided
    if req.actual_demand_mw is None and precomputed and precomputed.get("nominal_anomaly") is not None:
        nominal = dict(precomputed["nominal_anomaly"])
        nominal.setdefault("widened", widened)
        return nominal

    bundle: ModelArtifactsBundle = app_state.get("bundle")
    explainer: AnomalySHAPExplainer = app_state.get("explainer")
    if bundle is None or explainer is None:
        raise HTTPException(status_code=503, detail="inference runtime not initialized")

    from src.features.build_features import compute_telemetry_features

    with _inference_slot():
        df_history, _ = fetch_live_or_baseline_history(req.region, lookback_hours=240, bundle=bundle)
        cal_tz = resolve_calendar_timezone(bundle.quantile_calibration, req.region)
        feat_pl = compute_telemetry_features(df_history, drop_nulls_flag=True, calendar_tz=cal_tz)
        feat_df = bundle.reference_baselines[req.region] if feat_pl.is_empty() else feat_pl.to_pandas()
        feature_row = feat_df.iloc[-1].copy()
        features_df = pd.DataFrame([feature_row[bundle.feature_columns]])

        p10, p50, p90 = predict_quantiles(features_df, req.region, bundle, widened=widened)
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
        "widened": widened,
        "anomaly_score": round(float(scores[0]), 4),
        "is_structural": bool(is_structural[0]),
        "residual_breach": bool(trigger_b[0]),
        "top_shap_features": top_shap,
        "anomaly_shap_features": anomaly_shap,
    }


@app.get("/drift", dependencies=[Depends(verify_api_key)])
def get_drift_metrics(region: str = Query("ALL_INDIA", description="Regional grid identifier")) -> Dict[str, Any]:
    # returns statistical drift indicators and advisory recommendations
    _validate_region(region)

    precomputed = precompute_mod.load_precomputed_drift(region)
    if precomputed is not None:
        return precomputed

    ref_df = load_reference_baseline(region)
    curr_df = fetch_trailing_telemetry_window(region=region, window_hours=168)

    summary = evaluate_feature_drift(ref_df, curr_df, feature_cols=None)
    summary["region"] = region
    summary["report_available"] = os.path.exists(DEFAULT_REPORT_PATH)
    return summary


@app.get("/drift/report", dependencies=[Depends(verify_api_key)])
def get_drift_report_html(region: str = Query("ALL_INDIA", description="Regional grid identifier")):
    # serves regional drift diagnostic html report from supabase or static fallback
    _validate_region(region)

    html_content = precompute_mod.load_drift_report_html_from_supabase(region)
    if html_content:
        return HTMLResponse(content=html_content, media_type="text/html")

    regional_report_path = (
        DEFAULT_REPORT_PATH
        if region == "ALL_INDIA"
        else os.path.join(STATIC_REPORTS_DIR, f"drift_report_{region}.html")
    )
    if os.path.exists(regional_report_path):
        return FileResponse(regional_report_path, media_type="text/html")
    if os.path.exists(DEFAULT_REPORT_PATH):
        return FileResponse(DEFAULT_REPORT_PATH, media_type="text/html")

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="drift report not yet generated. run src.monitoring.drift_check first.",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=False)