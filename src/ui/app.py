import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ensure repository root is in sys.path when executed directly by streamlit runner
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

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

# design system styling
st.markdown(
    """
    <style>
    /* global typography */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif !important;
        color: #1d1d1f;
        -webkit-font-smoothing: antialiased;
    }
    .stApp {
        background-color: #fbfbfd;
    }

    /* enforce sidebar persistent visibility */
    section[data-testid="stSidebar"] {
        background-color: #f5f5f7 !important;
        border-right: 1px solid rgba(0, 0, 0, 0.08) !important;
        display: block !important;
        min-width: 280px !important;
    }
    button[data-testid="baseButton-headerNoPadding"],
    div[data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    /* bento grid cards */
    .card {
        background: #ffffff;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 18px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.03), 0 2px 6px -1px rgba(0, 0, 0, 0.02);
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }
    .card:hover {
        box-shadow: 0 8px 30px -4px rgba(0, 0, 0, 0.05);
    }
    .metric-eyebrow {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #86868b;
    }
    .metric-hero {
        font-size: 2.25rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #1d1d1f;
        line-height: 1.1;
        margin-top: 0.35rem;
    }
    .metric-unit {
        font-size: 1.05rem;
        font-weight: 500;
        color: #86868b;
        margin-left: 0.25rem;
    }
    .metric-footer {
        font-size: 0.82rem;
        color: #6e6e73;
        margin-top: 0.5rem;
    }

    /* status pills */
    .pill-success {
        background: rgba(52, 199, 89, 0.12);
        color: #1b8738;
        border: 1px solid rgba(52, 199, 89, 0.25);
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 10px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .pill-danger {
        background: rgba(255, 59, 48, 0.12);
        color: #d70015;
        border: 1px solid rgba(255, 59, 48, 0.25);
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 10px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .pill-neutral {
        background: #f5f5f7;
        color: #1d1d1f;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 3px 10px;
        display: inline-flex;
        align-items: center;
    }
    .button-primary {
        background: #0071e3;
        color: #ffffff !important;
        text-decoration: none !important;
        padding: 0.45rem 1.1rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        box-shadow: 0 1px 2px rgba(0, 113, 227, 0.2);
        transition: background 0.15s ease;
    }
    .button-primary:hover {
        background: #0077ed;
    }

    /* segmented control styling */
    div[data-testid="stSegmentedControl"] {
        background-color: #efeff1 !important;
        border-radius: 9999px !important;
        padding: 4px !important;
        border: 1px solid rgba(0, 0, 0, 0.04) !important;
    }
    div[data-testid="stSegmentedControl"] button {
        border-radius: 9999px !important;
        border: none !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        color: #6e6e73 !important;
        transition: all 0.15s ease !important;
        padding: 4px 14px !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background-color: #ffffff !important;
        color: #1d1d1f !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        font-weight: 600 !important;
    }

    /* section headings */
    .section-header {
        margin-top: 2.75rem;
        margin-bottom: 1.25rem;
    }
    .section-eyebrow {
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #86868b;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .section-title {
        font-size: 1.45rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #1d1d1f;
    }
    .section-subtitle {
        font-size: 0.9rem;
        color: #6e6e73;
        margin-top: 0.25rem;
        line-height: 1.4;
    }

    /* header cleanup */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    footer {
        visibility: hidden;
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


# persistent sidebar inspector
with st.sidebar:
    st.markdown(
        """
        <div style="padding-top: 0.5rem; margin-bottom: 1.25rem;">
            <div class="metric-eyebrow">SYSTEM INSPECTOR</div>
            <div style="font-size: 1.15rem; font-weight: 600; color: #1d1d1f; margin-top: 0.2rem;">Telemetry Core</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history_window = st.slider("Historical Lookback (Hours)", min_value=24, max_value=336, value=72, step=24)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="metric-eyebrow">ENGINE ARCHITECTURE</div>
        <div style="font-size: 0.85rem; color: #6e6e73; line-height: 1.6; margin-top: 0.5rem;">
            <b>Forecasting</b>: LightGBM Multi-Step Pinball (p10, p50, p90)<br>
            <b>Roll-Forward</b>: 72h Recursive Autoregressive<br>
            <b>Anomalies</b>: Isolation Forest (2% Contamination)<br>
            <b>Attribution</b>: SHAP TreeExplainer<br>
            <b>Monitoring</b>: Evidently AI KS & Wasserstein<br>
            <b>Pipeline</b>: Polars SIMD LazyFrame
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="metric-eyebrow">API ENDPOINTS</div>
        <div style="font-size: 0.82rem; color: #6e6e73; line-height: 1.6; margin-top: 0.5rem;">
            FastAPI OpenAPI: <a href="/docs" target="_blank" style="color: #0071e3; text-decoration: none; font-weight: 500;">/docs ↗</a><br>
            Health Probe: <a href="/health" target="_blank" style="color: #0071e3; text-decoration: none; font-weight: 500;">/health ↗</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# top minimal navbar
st.markdown(
    """
    <div style="display:flex; justify-content:space-between; align-items:center; padding-bottom: 1.25rem; margin-bottom: 1.5rem; border-bottom: 1px solid rgba(0,0,0,0.06);">
        <div style="font-size:0.85rem; font-weight:600; letter-spacing:0.04em; color:#1d1d1f;">
            GRID TELEMETRY ENGINE
        </div>
        <div style="font-size:0.82rem; color:#86868b;">
            NTP Synchronized
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# hero headline
st.markdown(
    """
    <div>
        <div class="metric-eyebrow">INDIAN POWER GRID TELEMETRY</div>
        <div style="font-size: 2.75rem; font-weight: 700; letter-spacing: -0.035em; color: #1d1d1f; line-height: 1.08; margin-top: 0.4rem; margin-bottom: 0.65rem;">
            Demand Forecasting & Predictive Uncertainty.
        </div>
        <div style="font-size: 1.12rem; color: #6e6e73; line-height: 1.45; max-width: 740px; margin-bottom: 1.5rem;">
            Continuous 72-hour recursive quantile forecasts, Isolation Forest structural anomaly detection with SHAP attributions, and Evidently statistical drift monitoring.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# segmented control for direct manipulation
region_options = {
    "ALL": "National Grid",
    "NR": "Northern (NR)",
    "WR": "Western (WR)",
    "SR": "Southern (SR)",
    "ER": "Eastern (ER)",
    "NER": "North-Eastern (NER)",
}

selected_region = st.segmented_control(
    "Control Region",
    options=list(region_options.keys()),
    default="ALL",
    format_func=lambda x: region_options[x],
    label_visibility="collapsed",
) or "ALL"

st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

# load data
recent_telemetry = load_telemetry(selected_region, limit=history_window)
forecasts = load_forecasts(selected_region, limit=72)
anomalies = load_anomalies(selected_region, limit=50)
drift_status = load_drift()

# top bento metrics strip
curr_demand = recent_telemetry[0]["metric_value"] if recent_telemetry else 0.0
peak_24h = max([r["metric_value"] for r in recent_telemetry[:24]], default=0.0)
min_24h = min([r["metric_value"] for r in recent_telemetry[:24]], default=0.0)
proj_peak = max([f["p50"] for f in forecasts], default=0.0) if forecasts else 0.0
flagged_count = sum(1 for a in anomalies if a.get("is_anomaly"))
drift_share = float(drift_status.get("drift_share", 0.0)) if drift_status else 0.0
is_drift = drift_share >= 0.5

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="card" style="padding: 1.5rem 1.75rem;">
            <div class="metric-eyebrow">CURRENT DEMAND</div>
            <div class="metric-hero">
                {curr_demand:,.0f} <span class="metric-unit">MW</span>
            </div>
            <div class="metric-footer">
                <span class="pill-neutral">Region: {selected_region}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="card" style="padding: 1.5rem 1.75rem;">
            <div class="metric-eyebrow">24H PEAK DEMAND</div>
            <div class="metric-hero">
                {peak_24h:,.0f} <span class="metric-unit">MW</span>
            </div>
            <div class="metric-footer">
                Trough: {min_24h:,.0f} MW
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="card" style="padding: 1.5rem 1.75rem;">
            <div class="metric-eyebrow">72H PROJECTED PEAK</div>
            <div class="metric-hero">
                {proj_peak:,.0f} <span class="metric-unit">MW</span>
            </div>
            <div class="metric-footer">
                Median (p50) Pinball Horizon
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="card" style="padding: 1.5rem 1.75rem;">
            <div class="metric-eyebrow">SYSTEM STABILITY</div>
            <div class="metric-hero" style="font-size: 1.65rem; color: {'#d70015' if is_drift else '#1b8738'}; margin-top: 0.6rem;">
                {'Drift Detected' if is_drift else 'Nominal State'}
            </div>
            <div class="metric-footer">
                Drift: {drift_share:.1%} • Anomalies: {flagged_count}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# section 1: 72h forecast chart
st.markdown(
    """
    <div class="section-header">
        <div class="section-eyebrow">PREDICTIVE HORIZON</div>
        <div class="section-title">72-Hour Quantile Forecast & Confidence Ribbon</div>
        <div class="section-subtitle">
            LightGBM recursive roll-forward predicting p10 (lower boundary), p50 (median), and p90 (upper boundary) uncertainty bounds.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if recent_telemetry or forecasts:
    fig = render_forecast_figure(recent_telemetry, forecasts, region=selected_region)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("No forecast records available for this control region.")

# section 2: anomalies & shap attribution
st.markdown(
    """
    <div class="section-header">
        <div class="section-eyebrow">ANOMALY ENGINE</div>
        <div class="section-title">Structural Anomalies & SHAP Root-Cause Attribution</div>
        <div class="section-subtitle">
            Isolation Forest flags structural deviations; SHAP TreeExplainer computes localized driving feature importance.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
render_anomaly_section(anomalies)

# section 3: continuous drift monitoring
st.markdown(
    """
    <div class="section-header">
        <div class="section-eyebrow">CONTINUOUS MONITORING</div>
        <div class="section-title">Evidently AI Statistical Drift Suite</div>
        <div class="section-subtitle">
            Evaluating rolling 7-day live telemetry against the 30-day training reference baseline via Kolmogorov-Smirnov and Wasserstein tests.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
render_drift_section(drift_status)

# footer
st.markdown(
    """
    <div style="margin-top: 4rem; padding-top: 2rem; border-top: 1px solid rgba(0, 0, 0, 0.06); text-align: center; font-size: 0.85rem; color: #86868b;">
        Created by <a href="https://maheshkakde.com" target="_blank" style="color: #1d1d1f; font-weight: 500; text-decoration: none;">Mahesh Kakde</a> • <a href="https://github.com/mahesh-kakde/grid-telemetry-engine" target="_blank" style="color: #0071e3; text-decoration: none;">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)