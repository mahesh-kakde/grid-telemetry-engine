import os
import sys

# ensure project root is present in sys.path when running standalone streamlit commands
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import requests
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
import plotly.graph_objects as go

REGIONS = ["ALL_INDIA", "NR", "WR", "SR", "ER", "NER"]
MODEL_VERSION = "2.0.0"
DEFAULT_REPORT_PATH = os.path.join(root_dir, "static", "reports", "drift_report.html")

# page configuration
st.set_page_config(
    page_title="grid-telemetry-engine | Mahesh Kakde",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# custom css aligned with plan.html styling
st.markdown(
    """
    <style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #111111;
    }
    .metric-card {
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 12px 16px;
        background-color: #ffffff;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 12px;
        color: #555555;
        text-transform: uppercase;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 600;
        color: #111111;
    }
    .badge-normal {
        background-color: #e6f4ea;
        color: #137333;
        border: 1px solid #ceead6;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-alert {
        background-color: #fce8e6;
        color: #c5221f;
        border: 1px solid #fad2cf;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-info {
        background-color: #e8f0fe;
        color: #1a73e8;
        border: 1px solid #d2e3fc;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .mono-tag {
        font-family: Consolas, Monaco, monospace;
        background-color: #f4f4f4;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


@st.cache_resource
def get_cached_artifacts():
    # caches in-memory bundle with on-demand lazy loading
    from src.models.load_artifacts import ModelArtifactsBundle
    from src.models.explainers import AnomalySHAPExplainer
    bundle = ModelArtifactsBundle()
    explainer = AnomalySHAPExplainer(bundle)
    return bundle, explainer


def check_api_health() -> Tuple[bool, Dict[str, Any]]:
    # queries loopback api health probe
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        if res.status_code == 200:
            return True, res.json()
    except Exception:
        pass
    return False, {}


def fetch_forecasts(region: str) -> List[Dict[str, Any]]:
    # retrieves 72-hour forecast via loopback api or local model bundle fallback
    is_live, _ = check_api_health()
    if is_live:
        try:
            res = requests.post(
                f"{API_BASE_URL}/forecast",
                json={"region": region, "lookback_hours": 168},
                timeout=15,
            )
            if res.status_code == 200:
                data = res.json()
                parsed = []
                for fc in data["forecasts"]:
                    parsed.append({
                        "target_timestamp_utc": pd.to_datetime(fc["target_timestamp_utc"]),
                        "forecast_p10": fc["forecast_p10"],
                        "forecast_p50": fc["forecast_p50"],
                        "forecast_p90": fc["forecast_p90"],
                        "region": fc["region"],
                    })
                return parsed
        except Exception:
            pass

    # direct local execution fallback with deferred imports
    from src.models.load_artifacts import load_reference_baseline
    from src.models.infer_and_detect import generate_72h_multi_step_forecast
    bundle, _ = get_cached_artifacts()
    ref_df = load_reference_baseline(region).tail(168).copy()
    base_ts = datetime.now(timezone.utc) - timedelta(hours=len(ref_df))
    records = []
    for idx, (_, row) in enumerate(ref_df.iterrows()):
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
    return [
        {
            "target_timestamp_utc": pd.to_datetime(fc["target_timestamp_utc"]),
            "forecast_p10": fc["forecast_p10"],
            "forecast_p50": fc["forecast_p50"],
            "forecast_p90": fc["forecast_p90"],
            "region": fc["region"],
        }
        for fc in raw_forecasts
    ]


def evaluate_live_anomaly(region: str, custom_demand: Optional[float] = None) -> Dict[str, Any]:
    # evaluates anomaly detection status via api or local fallback
    is_live, _ = check_api_health()
    if is_live:
        try:
            payload = {"region": region}
            if custom_demand is not None:
                payload["actual_demand_mw"] = custom_demand
            res = requests.post(f"{API_BASE_URL}/anomalies", json=payload, timeout=10)
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass

    from src.models.load_artifacts import load_reference_baseline
    from src.models.infer_and_detect import predict_quantiles, detect_structural_anomalies
    bundle, explainer = get_cached_artifacts()
    ref_df = load_reference_baseline(region)
    feature_row = ref_df.iloc[-1]
    features_df = pd.DataFrame([feature_row[bundle.feature_columns]])

    p10, p50, p90 = predict_quantiles(features_df, region, bundle)
    actual = custom_demand if custom_demand is not None else float(feature_row["demand_mw"])

    scores, is_structural, trigger_b = detect_structural_anomalies(
        X=features_df,
        y_actual=np.array([actual]),
        p10=p10,
        p50=p50,
        p90=p90,
        region=region,
        bundle=bundle,
    )
    top_shap = explainer.explain_anomaly_point(feature_row, region, top_k=3)

    return {
        "region": region,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "actual_demand_mw": round(actual, 2),
        "predicted_p10_mw": round(float(p10[0]), 2),
        "predicted_p50_mw": round(float(p50[0]), 2),
        "predicted_p90_mw": round(float(p90[0]), 2),
        "anomaly_score": round(float(scores[0]), 4),
        "is_structural": bool(is_structural[0]),
        "residual_breach": bool(trigger_b[0]),
        "top_shap_features": top_shap,
    }


def fetch_drift_summary(region: str) -> Dict[str, Any]:
    # fetches statistical drift summary via api or local evaluation
    is_live, _ = check_api_health()
    if is_live:
        try:
            res = requests.get(f"{API_BASE_URL}/drift?region={region}", timeout=10)
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass

    from src.models.load_artifacts import load_reference_baseline
    from src.monitoring.drift_check import evaluate_feature_drift, fetch_trailing_telemetry_window
    ref_df = load_reference_baseline(region)
    curr_df = fetch_trailing_telemetry_window(region=region, window_hours=168)
    summary = evaluate_feature_drift(ref_df, curr_df)
    summary["region"] = region
    summary["report_available"] = os.path.exists(DEFAULT_REPORT_PATH)
    return summary


# header and navigation
st.title("National Power Grid Telemetry & Forecasting Engine")
st.markdown(
    "Live-Data Forecasting & Anomaly Engine with Drift Monitoring for India's 5 Regional Despatch Grids and Aggregate National Demand."
)

# sidebar controls
with st.sidebar:
    st.subheader("Grid Configuration")
    selected_region = st.selectbox(
        "Select Regional Grid",
        options=REGIONS,
        index=0,
        help="Choose one of India's 5 Regional Load Despatch Centres (RLDCs) or aggregate ALL_INDIA",
    )

    api_active, api_info = check_api_health()
    st.markdown("---")
    st.markdown("**Engine Runtime State**")
    if api_active:
        st.markdown('<span class="badge-normal">FastAPI Online</span>', unsafe_allow_html=True)
        st.caption(f"Loopback Service: `127.0.0.1:8000`")
    else:
        st.markdown('<span class="badge-info">Direct In-Memory</span>', unsafe_allow_html=True)
        st.caption("Native Model Preload Active")

    st.markdown(f"Model Version: <span class=\"mono-tag\">{MODEL_VERSION}</span>", unsafe_allow_html=True)
    st.markdown("Domain Ingress: <span class=\"mono-tag\">grid.maheshkakde.com</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("National Grid RLDCs: NR (North), WR (West), SR (South), ER (East), NER (North East)")

# tabs for main interface
tab_forecast, tab_anomalies, tab_drift, tab_system = st.tabs([
    "72-Hour Quantile Forecast",
    "Dual-Trigger Anomaly Detection",
    "Statistical Drift Monitoring",
    "System Architecture",
])

# tab 1: 72-hour forecast
with tab_forecast:
    st.subheader(f"72-Hour Probabilistic Demand Forecast: {selected_region}")
    forecasts = fetch_forecasts(selected_region)
    df_fc = pd.DataFrame(forecasts)

    # forecast metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Peak Forecast Load (p50)</div>
                <div class="metric-value">{df_fc['forecast_p50'].max():,.0f} MW</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Trough Forecast Load (p50)</div>
                <div class="metric-value">{df_fc['forecast_p50'].min():,.0f} MW</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        avg_band = (df_fc['forecast_p90'] - df_fc['forecast_p10']).mean()
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Mean Uncertainty Band (p90 - p10)</div>
                <div class="metric-value">{avg_band:,.0f} MW</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Quantile Monotonicity</div>
                <div class="metric-value"><span class="badge-normal">p10 ≤ p50 ≤ p90</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # plotly interactive load forecast curve
    fig = go.Figure()

    # confidence interval ribbon (p10 to p90)
    fig.add_trace(go.Scatter(
        x=df_fc["target_timestamp_utc"],
        y=df_fc["forecast_p90"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="p90 Upper Bound",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df_fc["target_timestamp_utc"],
        y=df_fc["forecast_p10"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0, 82, 204, 0.12)",
        name="p10-p90 Confidence Interval",
        hoverinfo="skip",
    ))

    # median forecast line
    fig.add_trace(go.Scatter(
        x=df_fc["target_timestamp_utc"],
        y=df_fc["forecast_p50"],
        mode="lines+markers",
        line=dict(color="#0052cc", width=2.5),
        marker=dict(size=4),
        name="Median Forecast (p50)",
        hovertemplate="<b>%{x|%d %b %H:%M UTC}</b><br>p50: %{y:,.0f} MW<extra></extra>",
    ))

    # bound lines for hover clarity
    fig.add_trace(go.Scatter(
        x=df_fc["target_timestamp_utc"],
        y=df_fc["forecast_p90"],
        mode="lines",
        line=dict(color="#172b4d", width=1, dash="dot"),
        name="p90 (90th Percentile)",
        hovertemplate="p90: %{y:,.0f} MW<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_fc["target_timestamp_utc"],
        y=df_fc["forecast_p10"],
        mode="lines",
        line=dict(color="#6b778c", width=1, dash="dot"),
        name="p10 (10th Percentile)",
        hovertemplate="p10: %{y:,.0f} MW<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=20, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            title="Forecast Target Timestamp (UTC)",
            showgrid=True,
            gridcolor="#f0f0f0",
        ),
        yaxis=dict(
            title="Active Power Demand (MW)",
            showgrid=True,
            gridcolor="#f0f0f0",
            tickformat=",.0f",
        ),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Tabular Forecast Data (72 Hours)"):
        st.dataframe(df_fc, use_container_width=True)

# tab 2: anomaly detection & shap
with tab_anomalies:
    st.subheader(f"Dual-Trigger Structural Anomaly Detection: {selected_region}")
    st.markdown(
        "A verified structural anomaly trips only when **Trigger A** (unsupervised Isolation Forest feature isolation) "
        "and **Trigger B** (predictive residual IQR envelope breach) both occur simultaneously."
    )

    # interactive test parameter injection
    col_input, col_status = st.columns([1, 2])
    with col_input:
        st.markdown("**Simulate Live Demand Injection**")
        anomaly_scenario = st.radio(
            "Evaluation Scenario:",
            ["Nominal Baseline Demand", "Extreme Demand Surge (+200,000 MW)", "Custom Value"],
            index=0,
        )
        custom_val = None
        if anomaly_scenario == "Extreme Demand Surge (+200,000 MW)":
            custom_val = 260000.0 if selected_region == "ALL_INDIA" else 150000.0
        elif anomaly_scenario == "Custom Value":
            custom_val = st.number_input("Demand (MW):", min_value=100.0, max_value=350000.0, value=75000.0)

    anomaly_res = evaluate_live_anomaly(selected_region, custom_demand=custom_val)

    with col_status:
        st.markdown("**Live Verification Status**")
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            st.markdown('<div class="metric-card"><div class="metric-label">Actual vs Predicted (p50)</div>'
                        f'<div class="metric-value">{anomaly_res["actual_demand_mw"]:,.0f} / {anomaly_res["predicted_p50_mw"]:,.0f} MW</div></div>',
                        unsafe_allow_html=True)
        with status_col2:
            trig_b_badge = '<span class="badge-alert">Breached</span>' if anomaly_res["residual_breach"] else '<span class="badge-normal">Within Envelope</span>'
            st.markdown(f'<div class="metric-card"><div class="metric-label">Trigger B (IQR Breach)</div>'
                        f'<div class="metric-value">{trig_b_badge}</div></div>',
                        unsafe_allow_html=True)
        with status_col3:
            struct_badge = '<span class="badge-alert">Structural Anomaly</span>' if anomaly_res["is_structural"] else '<span class="badge-normal">Nominal Operation</span>'
            st.markdown(f'<div class="metric-card"><div class="metric-label">Verified Anomaly (A ∧ B)</div>'
                        f'<div class="metric-value">{struct_badge}</div></div>',
                        unsafe_allow_html=True)

    # shap explainability section
    st.markdown("---")
    st.subheader("Top-3 Physical Feature Drivers (SHAP TreeExplainer)")
    st.markdown(
        "Feature attributions computed using pre-calculated $k=150$ background centroids and regional LightGBM median boosters:"
    )

    top_features = anomaly_res["top_shap_features"]
    shap_cols = st.columns(len(top_features))
    for idx, (col_ui, feat) in enumerate(zip(shap_cols, top_features)):
        with col_ui:
            direction_color = "#c5221f" if feat["shap_value"] > 0 else "#0052cc"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Rank {idx + 1}: <code class="mono-tag">{feat['feature']}</code></div>
                    <div class="metric-value" style="color: {direction_color};">
                        {feat['shap_value']:+,.1f} MW
                    </div>
                    <div style="font-size: 13px; color: #555555; margin-top: 4px;">
                        Observed Value: <b>{feat['actual_value']:,.2f}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # shap horizontal bar visualization
    df_shap = pd.DataFrame(top_features)
    fig_shap = go.Figure(go.Bar(
        x=df_shap["shap_value"],
        y=df_shap["feature"],
        orientation="h",
        marker=dict(
            color=["#c5221f" if v > 0 else "#0052cc" for v in df_shap["shap_value"]],
        ),
        text=[f"{v:+,.1f} MW" for v in df_shap["shap_value"]],
        textposition="outside",
    ))
    fig_shap.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis=dict(title="Shapley Value (MW Impact on Prediction)", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(autorange="reversed"),
        height=240,
    )
    st.plotly_chart(fig_shap, use_container_width=True)

# tab 3: statistical drift monitoring
with tab_drift:
    st.subheader(f"Statistical Covariate & Target Drift Monitoring: {selected_region}")
    drift_data = fetch_drift_summary(selected_region)

    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        alarm_badge = '<span class="badge-alert">Retraining Advisory</span>' if drift_data["drift_alarm"] else '<span class="badge-normal">Distributions Stable</span>'
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">System Drift Alarm</div>
                <div class="metric-value">{alarm_badge}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Drifted Feature Ratio</div>
                <div class="metric-value">{drift_data['drift_ratio']:.1%} ({drift_data['drifted_features']}/{drift_data['total_features']})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Alarm Threshold</div>
                <div class="metric-value">> 30.0%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d_col4:
        alphas = drift_data["recommended_quantile_alpha"]
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Recommended Quantile Alphas</div>
                <div class="metric-value"><span class="mono-tag">{alphas}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Per-Feature Statistical Test Results (7-Day Trailing vs 30-Day Baseline)**")
    metrics_records = []
    for f_name, m in drift_data["feature_metrics"].items():
        test_type = "Kolmogorov-Smirnov" if m["test"] == "kolmogorov_smirnov" else "1D Wasserstein"
        stat_val = f"{m.get('statistic', m.get('score')):.4f}"
        p_or_thresh = f"{m.get('p_value'):.6f}" if "p_value" in m else "N/A"
        thresh_val = f"{m['threshold']:.2f}"
        status_txt = "DRIFTED" if m["drift_detected"] else "STABLE"
        metrics_records.append({
            "Feature Name": f_name,
            "Statistical Test": test_type,
            "Statistic / Score": stat_val,
            "P-Value": p_or_thresh,
            "Threshold": thresh_val,
            "Drift Status": status_txt,
        })
    df_metrics = pd.DataFrame(metrics_records)
    st.dataframe(df_metrics, use_container_width=True)

    if os.path.exists(DEFAULT_REPORT_PATH):
        st.markdown("---")
        st.markdown("**Standalone Evidently AI Interactive HTML Report**")
        with open(DEFAULT_REPORT_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.download_button(
            label="Download Full Evidently HTML Report",
            data=html_content,
            file_name="drift_report.html",
            mime="text/html",
        )

# tab 4: architecture
with tab_system:
    st.subheader("System Architecture & Specifications")
    st.markdown(
        """
        <div class="metric-card">
            <b>1. Ingestion Cron Job:</b> GitHub Actions scheduled cron runs every 6 hours (<code>15 */6 * * *</code>),
            ingesting POSOCO / NLDC demand telemetry and Open-Meteo meteorological observations with Pydantic v2 validation.
        </div>
        <div class="metric-card">
            <b>2. Zero-Leakage Feature Pipeline:</b> Vectorized Polars LazyFrames computing autoregressive lags, rolling statistics,
            and continuous Fourier harmonics with a mandatory <code>shift(1)</code> causality fence.
        </div>
        <div class="metric-card">
            <b>3. Dual Model Engine:</b> LightGBM multi-step quantile regression (<code>p10, p50, p90</code>) with isotonic monotonicity sorting
            coupled with Scikit-Learn Isolation Forest for structural anomaly detection.
        </div>
        <div class="metric-card">
            <b>4. Explainability & Observability:</b> SHAP TreeExplainer with 150 k-means background centroids extracting top-3 physical drivers,
            and Evidently AI performing KS and Wasserstein drift evaluation.
        </div>
        <div class="metric-card">
            <b>5. Deployment & Edge Ingress:</b> Single multi-stage Docker container co-locating internal FastAPI (127.0.0.1:8000)
            and Streamlit dashboard exposed at <code>grid.maheshkakde.com</code> via Cloudflare Edge SSL.
        </div>
        """,
        unsafe_allow_html=True,
    )