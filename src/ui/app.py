from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
import streamlit as st

from src.config import get_settings
from src.db.client import get_db_client
from src.ingestion.fetcher import TelemetryFetcher
from src.ui.components.anomaly_cards import render_anomaly_section
from src.ui.components.drift_badge import render_drift_section
from src.ui.components.forecast_chart import render_forecast_figure

st.set_page_config(
    page_title="Grid Telemetry Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# apple-inspired light design system
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1d1d1f;
    }
    .stApp {
        background-color: #fbfbfd;
    }
    div[data-testid="stSidebar"] {
        background-color: #f5f5f7;
        border-right: 1px solid #e5e5ea;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e5ea;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #86868b;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1d1d1f;
        margin-top: 0.35rem;
        letter-spacing: -0.02em;
    }
    .metric-delta {
        font-size: 0.82rem;
        color: #6e6e73;
        margin-top: 0.25rem;
    }
    .section-header {
        margin-top: 2rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #e5e5ea;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1d1d1f;
        letter-spacing: -0.01em;
    }
    .section-subtitle {
        font-size: 0.85rem;
        color: #6e6e73;
        margin-top: 0.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_db():
    settings = get_settings()
    return get_db_client(settings)


def load_telemetry(region: str, limit: int = 168) -> list[dict[str, Any]]:
    db = get_db()
    records = db.get_recent_telemetry(region=region, limit=limit)
    if not records:
        fetcher = TelemetryFetcher()
        fresh = fetcher.fetch_latest(region=region, limit=limit)
        db.upsert_telemetry(fresh)
        records = db.get_recent_telemetry(region=region, limit=limit)
    return records


def load_forecasts(region: str, limit: int = 72) -> list[dict[str, Any]]:
    db = get_db()
    return db.get_latest_forecasts(region=region, limit=limit)


def load_anomalies(region: str, limit: int = 50) -> list[dict[str, Any]]:
    db = get_db()
    return db.get_recent_anomalies(region=region, limit=limit)


def load_drift() -> dict[str, Any] | None:
    db = get_db()
    return db.get_latest_drift()


# sidebar controls
with st.sidebar:
    st.markdown("### Grid Telemetry Engine")
    st.markdown(
        "<div style='font-size:0.85rem; color:#6e6e73; margin-bottom:1.2rem;'>Live Indian Power Grid Forecasting, Anomaly Scoring, and Drift Suite.</div>",
        unsafe_allow_html=True,
    )

    region_options = {
        "ALL": "National Grid (ALL)",
        "NR": "Northern Region (NR)",
        "WR": "Western Region (WR)",
        "SR": "Southern Region (SR)",
        "ER": "Eastern Region (ER)",
        "NER": "North-Eastern Region (NER)",
    }
    selected_region = st.selectbox(
        "Grid Control Region",
        options=list(region_options.keys()),
        format_func=lambda x: region_options[x],
        index=0,
    )

    history_window = st.slider("Historical Window (Hours)", min_value=24, max_value=336, value=72, step=24)

    st.markdown("---")
    st.markdown("#### Engine Specifications")
    st.markdown(
        """
        - **Forecast Model**: LightGBM Multi-Step Pinball ($p_{10}, p_{50}, p_{90}$)
        - **Horizon**: 72-Hour Recursive Autoregressive
        - **Anomaly Model**: Isolation Forest (2% Contamination)
        - **Explainability**: SHAP TreeExplainer Local Attribution
        - **Drift Suite**: Evidently AI Kolmogorov-Smirnov / Wasserstein
        - **Feature SIMD**: Polars LazyFrame (< 50ms)
        """
    )

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size:0.8rem; color:#86868b;'>
            Production Endpoint: <a href="https://grid.maheshkakde.com" target="_blank" style="color:#0071e3; text-decoration:none;">grid.maheshkakde.com</a><br>
            REST API Swagger: <a href="http://localhost:8000/docs" target="_blank" style="color:#0071e3; text-decoration:none;">/docs</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# main dashboard layout
st.markdown("## National & Regional Grid Telemetry Dashboard")
st.markdown(
    f"<div style='font-size:0.95rem; color:#6e6e73; margin-top:-0.5rem; margin-bottom:1.5rem;'>Monitoring telemetry, predictive quantile uncertainty, and distribution drift for <b>{region_options[selected_region]}</b>.</div>",
    unsafe_allow_html=True,
)

# load data
recent_telemetry = load_telemetry(selected_region, limit=history_window)
forecasts = load_forecasts(selected_region, limit=72)
anomalies = load_anomalies(selected_region, limit=50)
drift_status = load_drift()

# top metrics strip
curr_demand = recent_telemetry[0]["metric_value"] if recent_telemetry else 0.0
peak_24h = max([r["metric_value"] for r in recent_telemetry[:24]], default=0.0)
min_24h = min([r["metric_value"] for r in recent_telemetry[:24]], default=0.0)
proj_peak = max([f["p50"] for f in forecasts], default=0.0) if forecasts else 0.0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Current Grid Demand</div>
            <div class="metric-value">{curr_demand:,.0f} <span style="font-size:1rem; font-weight:500;">MW</span></div>
            <div class="metric-delta">Region: {selected_region}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">24h Peak Demand</div>
            <div class="metric-value">{peak_24h:,.0f} <span style="font-size:1rem; font-weight:500;">MW</span></div>
            <div class="metric-delta">Trough: {min_24h:,.0f} MW</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">72h Projected Median Peak</div>
            <div class="metric-value">{proj_peak:,.0f} <span style="font-size:1rem; font-weight:500;">MW</span></div>
            <div class="metric-delta">Model: LightGBM Pinball</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    flagged_count = sum(1 for a in anomalies if a.get("is_anomaly"))
    drift_share = float(drift_status.get("drift_share", 0.0)) if drift_status else 0.0
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">System Health & Drift</div>
            <div class="metric-value" style="font-size:1.5rem; color:{'#166534' if drift_share < 0.5 else '#991b1b'};">
                {'Nominal' if drift_share < 0.5 else 'Drift Detected'}
            </div>
            <div class="metric-delta">Drift Share: {drift_share:.1%} | Anomalies: {flagged_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# section 1: 72h forecast chart
st.markdown(
    """
    <div class="section-header">
        <div class="section-title">72-Hour Quantile Forecast & Confidence Ribbon</div>
        <div class="section-subtitle">LightGBM recursive roll-forward predicting p10 (lower), p50 (median), and p90 (upper) uncertainty bounds.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if recent_telemetry or forecasts:
    fig = render_forecast_figure(recent_telemetry, forecasts, region=selected_region)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("No forecast data available for this region.")

# section 2: anomalies & shap attribution
st.markdown(
    """
    <div class="section-header">
        <div class="section-title">Structural Anomalies & SHAP Root-Cause Attribution</div>
        <div class="section-subtitle">Isolation Forest flags structural deviations; SHAP TreeExplainer computes local driving feature importance.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
render_anomaly_section(anomalies)

# section 3: continuous drift monitoring
st.markdown(
    """
    <div class="section-header">
        <div class="section-title">Evidently AI Continuous Drift Monitoring</div>
        <div class="section-subtitle">Evaluating rolling 7-day live telemetry against the 30-day training reference baseline.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
render_drift_section(drift_status)