import importlib.util
import os
import sys

# only inject repo root when invoked directly via `streamlit run src/ui/app.py` without package install
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if importlib.util.find_spec("src") is None and root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from datetime import datetime, timezone
import html
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from src.monitoring.drift_check import (
    FEATURE_METADATA,
    build_html_drift_report,
    format_drift_test_label,
)
from src.ui.charts import build_forecast_figure, build_shap_figure
from src.ui.components import (
    build_architecture_grid_html,
    build_dashboard_skeleton_html,
    build_drift_criteria_card_html,
    build_footer_html,
    build_health_summary_cards_html,
    build_implementation_details_html,
    build_kpi_row_html,
    build_nerds_math_cards_html,
    build_nerds_overview_kpis_html,
    build_nerds_readme_cta_html,
    build_notices_html,
    build_operation_loading_html,
    build_sentinel_gates_html,
    build_shap_driver_cards_html,
)
from src.ui.helpers import (
    REGION_FULL_NAMES,
    evaluate_snapshot_freshness,
    format_region_option,
)
from src.ui.styles import get_app_css

REGIONS = ["ALL_INDIA", "NR", "WR", "SR", "ER", "NER"]
FAVICON_PATH = os.path.join(root_dir, "static", "favicon.png")
PLOTLY_STATIC_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}

query_page = str(st.query_params.get("page", "")).lower()
query_error = str(st.query_params.get("error", "")).lower()
is_404 = query_page in ["404", "not_found", "notfound"] or query_error == "404"

if hasattr(st, "context") and getattr(st.context, "url", None):
    try:
        from urllib.parse import urlparse

        req_path = urlparse(st.context.url).path.strip("/").lower()
        if req_path and req_path not in ["", "app"]:
            is_404 = True
    except Exception:
        pass

app_page_title = (
    "404 - Page Not Found | grid-telemetry-engine"
    if is_404
    else "grid-telemetry-engine | Mahesh Kakde"
)

st.set_page_config(
    page_title=app_page_title,
    page_icon=FAVICON_PATH if os.path.exists(FAVICON_PATH) else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(get_app_css(), unsafe_allow_html=True)

if is_404:
    st.markdown(
        """
        <main style="max-width: 500px; margin: 4rem auto; background: #ffffff; border: 1px solid #d2d2d7;
                     border-radius: 18px; padding: 48px 36px; text-align: center;
                     box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04);">
            <div style="font-size: 72px; font-weight: 700; line-height: 1; margin-bottom: 10px;" aria-hidden="true">404</div>
            <h1 style="font-size: 22px !important; margin-bottom: 10px !important;">Page Not Found</h1>
            <p style="font-size: 14px; color: #515154; line-height: 1.55; margin-bottom: 26px;">
                The requested page or route does not exist or has been moved.
            </p>
            <div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;">
                <a href="/" target="_self" style="background: #0066cc; color: #ffffff; text-decoration: none;
                   font-weight: 600; font-size: 13.5px; padding: 10px 24px; border-radius: 999px;">Return to Dashboard</a>
                <a href="https://maheshkakde.com" target="_blank" rel="noopener noreferrer"
                   style="background: #ffffff; color: #1d1d1f; text-decoration: none; font-weight: 600;
                   font-size: 13.5px; padding: 10px 24px; border-radius: 999px; border: 1px solid #d2d2d7;">Portfolio</a>
            </div>
        </main>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


@st.cache_resource(show_spinner=False)
def get_cached_artifacts():
    # lazy-loaded ml bundle bounded to 2 regions for 512mb container headroom
    from src.models.explainers import AnomalySHAPExplainer
    from src.models.load_artifacts import ModelArtifactsBundle

    bundle = ModelArtifactsBundle(max_cached_regions=2)
    explainer = AnomalySHAPExplainer(bundle)
    return bundle, explainer


@st.cache_data(ttl=300, show_spinner=False)
def _load_all_cached_snapshots() -> Dict[str, Dict[str, Any]]:
    from src.models.precompute import load_all_snapshots_from_supabase, load_precomputed_data

    batch_snaps = load_all_snapshots_from_supabase(REGIONS)
    snapshots: Dict[str, Dict[str, Any]] = {}
    for reg in REGIONS:
        snap = batch_snaps.get(reg)
        if snap is None:
            snap = load_precomputed_data(reg, prefer_supabase=False)
        if snap is not None and snap.get("forecasts") and snap.get("nominal_anomaly") and snap.get("drift_summary"):
            snapshots[reg] = snap
    return snapshots


@st.cache_data(ttl=300, show_spinner=False)
def load_snapshot(region: str) -> Dict[str, Any]:
    from src.models.precompute import load_precomputed_data, precompute_region

    batch_snap = _load_all_cached_snapshots().get(region)
    if batch_snap is not None:
        return batch_snap

    snap = load_precomputed_data(region)
    if snap is not None and snap.get("forecasts") and snap.get("nominal_anomaly") and snap.get("drift_summary"):
        return snap

    bundle, explainer = get_cached_artifacts()
    snap = precompute_region(region, bundle, explainer, use_live_weather=False)
    snap["snapshot_source"] = "local_runtime_fallback"
    return snap


@st.cache_data(ttl=300, show_spinner=False)
def get_forecast_dataframe(region: str) -> pd.DataFrame:
    snap = load_snapshot(region)
    cols = ("target_timestamp_utc", "forecast_p10", "forecast_p50", "forecast_p90", "region")
    df = pd.DataFrame.from_records(snap["forecasts"], columns=cols)
    if not df.empty:
        df["target_timestamp_utc"] = pd.to_datetime(df["target_timestamp_utc"])
        num_cols = ["forecast_p10", "forecast_p50", "forecast_p90"]
        df[num_cols] = df[num_cols].astype(float)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def get_forecast_figure(region: str, horizon_hours: int, widened: bool):
    df_fc = get_forecast_dataframe(region)
    return build_forecast_figure(df_fc, horizon_hours=horizon_hours, widened=widened)


@st.cache_data(ttl=300, show_spinner=False)
def get_forecast_table_payload(region: str, horizon_hours: int) -> tuple[pd.DataFrame, str]:
    df_slice = get_forecast_dataframe(region).head(horizon_hours)
    df_export = pd.DataFrame({
        "Target Time (UTC)": df_slice["target_timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M"),
        "Lower p10 (MW)": df_slice["forecast_p10"].round(1),
        "Expected p50 (MW)": df_slice["forecast_p50"].round(1),
        "Upper p90 (MW)": df_slice["forecast_p90"].round(1),
    })
    df_display = pd.DataFrame({
        "Target Time (UTC)": df_export["Target Time (UTC)"],
        "Lower p10 (MW)": [f"{v:,.1f}" for v in df_export["Lower p10 (MW)"]],
        "Expected p50 (MW)": [f"{v:,.1f}" for v in df_export["Expected p50 (MW)"]],
        "Upper p90 (MW)": [f"{v:,.1f}" for v in df_export["Upper p90 (MW)"]],
    })
    return df_display, df_export.to_csv(index=False)


@st.cache_data(ttl=300, show_spinner=False)
def get_latest_feature_row(region: str) -> pd.Series:
    # cache history fetch + polars feature extraction independently of custom demand input
    from src.features.build_features import compute_telemetry_features
    from src.models.infer_and_detect import fetch_live_or_baseline_history
    from src.models.load_artifacts import resolve_calendar_timezone

    bundle, _ = get_cached_artifacts()
    cal_tz = resolve_calendar_timezone(bundle.quantile_calibration, region)
    df_history, _ = fetch_live_or_baseline_history(region, lookback_hours=240, bundle=bundle)
    feat_pl = compute_telemetry_features(df_history, drop_nulls_flag=True, calendar_tz=cal_tz)
    feat_df = bundle.reference_baselines[region] if feat_pl.is_empty() else feat_pl.to_pandas()
    return feat_df.iloc[-1].copy()


@st.cache_data(ttl=300, show_spinner=False)
def evaluate_live_anomaly(region: str, custom_demand: Optional[float] = None) -> Dict[str, Any]:
    snap = load_snapshot(region)
    if custom_demand is None:
        return snap["nominal_anomaly"]

    from src.models.infer_and_detect import detect_structural_anomalies, predict_quantiles

    bundle, explainer = get_cached_artifacts()
    feature_row = get_latest_feature_row(region)
    features_df = pd.DataFrame([feature_row[bundle.feature_columns]])

    widened = bool(snap.get("widened", False))
    p10, p50, p90 = predict_quantiles(features_df, region, bundle, widened=widened)
    actual = float(custom_demand)

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
    anomaly_shap = explainer.explain_anomaly_score(
        feature_row, region, actual_demand_mw=actual, top_k=3
    )

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
        "anomaly_shap_features": anomaly_shap,
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_cached_shap_figure(drivers_key: str, drivers: List[Dict[str, Any]], mode: str):
    return build_shap_figure(drivers, mode=mode)


@st.cache_data(ttl=300, show_spinner=False)
def get_drift_metrics_dataframe(region: str) -> pd.DataFrame:
    drift_data = load_snapshot(region)["drift_summary"]
    records = []
    for f_name, m in drift_data["feature_metrics"].items():
        meta = FEATURE_METADATA.get(f_name, {"name": f_name, "desc": "Telemetry parameter"})
        shifted = bool(m.get("drift_detected", False))
        records.append({
            "_shifted": shifted,
            "Parameter": str(meta["name"]),
            "Status": "Shifted" if shifted else "Nominal",
            "Observed": f"{m.get('statistic', m.get('score', 0.0)):.4f}",
            "Limit": f"{m['threshold']:.2f}",
            "P-Value": f"{m.get('p_value'):.6f}" if "p_value" in m else "—",
            "Test": format_drift_test_label(str(m.get("test", ""))),
            "Code": f_name,
        })

    df_out = pd.DataFrame(records)
    if not df_out.empty:
        df_out = df_out.sort_values(by=["_shifted", "Parameter"], ascending=[False, True]).drop(columns=["_shifted"])
    return df_out


@st.cache_data(ttl=300, show_spinner=False)
def get_cached_drift_report(region: str) -> str:
    snap = load_snapshot(region)
    return build_html_drift_report(
        snap["drift_summary"],
        region=region,
        generated_at=snap.get("generated_at"),
    )


def _render_plotly_chart(fig) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOTLY_STATIC_CONFIG)


def _safe_segmented_choice(selected: Any, state_key: str, default: Any) -> Any:
    if selected is None:
        return st.session_state.get(state_key, default)
    st.session_state[state_key] = selected
    return selected


# global sticky top navigation bar (brand heading left, sections right)
NAV_SECTIONS = ["Forecast", "Anomaly Check", "Model Health", "Info for Nerds"]
if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "Forecast"
if "active_region" not in st.session_state:
    st.session_state["active_region"] = "ALL_INDIA"

with st.container(key="global_header_shell"):
    with st.container(
        key="global_header_inner",
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="center",
    ):
        st.markdown(
            "<h1 class='global-brand-title'>National Power Grid Telemetry</h1>",
            unsafe_allow_html=True,
        )
        selected_nav_raw = st.segmented_control(
            "Section Navigation",
            options=NAV_SECTIONS,
            selection_mode="single",
            default=st.session_state["active_nav"],
            key="nav_control",
            label_visibility="collapsed",
        )

active_nav = _safe_segmented_choice(selected_nav_raw, "active_nav", "Forecast")
selected_region = st.session_state["active_region"]
region_full_name = REGION_FULL_NAMES.get(selected_region, selected_region)

if active_nav != "Info for Nerds":
    st.markdown(
        "<p class='scope-lead'>72-hour electricity demand forecasts, anomaly checks, and feature drift tracking.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='region-caption'>Region</div>", unsafe_allow_html=True)
    selected_region_raw = st.segmented_control(
        "Select Regional Grid",
        options=REGIONS,
        selection_mode="single",
        default=st.session_state["active_region"],
        format_func=format_region_option,
        key="region_control",
        label_visibility="collapsed",
        wrap=True,
    )
    selected_region = _safe_segmented_choice(selected_region_raw, "active_region", "ALL_INDIA")
    region_full_name = REGION_FULL_NAMES.get(selected_region, selected_region)

    # show structured skeleton window during cold session snapshot fetch so first paint is immediate
    dashboard_loader = st.empty()
    hydrated_regions = st.session_state.setdefault("_hydrated_regions", set())
    if selected_region not in hydrated_regions:
        dashboard_loader.markdown(build_dashboard_skeleton_html(region_full_name), unsafe_allow_html=True)

    active_snapshot = load_snapshot(selected_region)
    df_fc = get_forecast_dataframe(selected_region)
    nom_state = active_snapshot["nominal_anomaly"]
    drift_state = active_snapshot["drift_summary"]
    widened_flag = bool(active_snapshot.get("widened", False))
    freshness = evaluate_snapshot_freshness(active_snapshot, stale_after_hours=12.0)

    peak_mw = float(df_fc["forecast_p50"].max()) if not df_fc.empty else 0.0
    baseload_mw = float(df_fc["forecast_p50"].min()) if not df_fc.empty else 0.0
    avg_band_mw = float((df_fc["forecast_p90"] - df_fc["forecast_p10"]).mean()) if not df_fc.empty else 0.0

    hydrated_regions.update(REGIONS)
    dashboard_loader.empty()

if active_nav == "Forecast":
    st.markdown(
        build_kpi_row_html(
            region_code=selected_region,
            peak_mw=peak_mw,
            baseload_mw=baseload_mw,
            avg_band_mw=avg_band_mw,
            widened=widened_flag,
            nom_state=nom_state,
            drift_state=drift_state,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(build_notices_html(freshness, drift_state, widened_flag), unsafe_allow_html=True)

    hdr_col, ctrl_col = st.columns([3, 2], vertical_alignment="center")
    with hdr_col:
        st.markdown(f"## Demand Forecast ({html.escape(region_full_name)})")
        st.markdown(
            "<div class='section-lead'>Expected median load (p50) with calibrated 80% or 90% quantile bounds.</div>",
            unsafe_allow_html=True,
        )
    with ctrl_col:
        h_state_key = f"horizon_state_{selected_region}"
        if h_state_key not in st.session_state:
            st.session_state[h_state_key] = 24
        horizon_raw = st.segmented_control(
            "Forecast Window",
            options=[24, 48, 72],
            selection_mode="single",
            default=st.session_state[h_state_key],
            format_func=lambda h: f"Next {h}h",
            key=f"horizon_ctrl_{selected_region}",
            label_visibility="collapsed",
        )
        selected_horizon = int(_safe_segmented_choice(horizon_raw, h_state_key, 24))

    st.markdown(
        "<div class='chart-mobile-hint'><strong>Tip:</strong> Next 24h offers sharper hourly spacing on narrow screens.</div>",
        unsafe_allow_html=True,
    )

    _render_plotly_chart(get_forecast_figure(selected_region, selected_horizon, widened_flag))

    st.markdown(f"### Hourly Forecast Table (Next {selected_horizon}h)")
    df_table_display, forecast_csv = get_forecast_table_payload(selected_region, selected_horizon)
    st.dataframe(
        df_table_display,
        width="stretch",
        height=458,
        hide_index=True,
        column_config={
            "Target Time (UTC)": st.column_config.TextColumn("Target Time (UTC)", width=180),
            "Lower p10 (MW)": st.column_config.TextColumn("Lower p10 (MW)", width=150, alignment="right"),
            "Expected p50 (MW)": st.column_config.TextColumn("Expected p50 (MW)", width=160, alignment="right"),
            "Upper p90 (MW)": st.column_config.TextColumn("Upper p90 (MW)", width=150, alignment="right"),
        },
    )
    st.download_button(
        label=f"Export {selected_horizon}h Forecast (CSV)",
        data=forecast_csv,
        file_name=f"forecast_{selected_region.lower()}_{selected_horizon}h.csv",
        mime="text/csv",
        key=f"dl_fc_{selected_region}_{selected_horizon}",
    )

elif active_nav == "Anomaly Check":
    st.markdown(f"## Anomaly Check ({html.escape(region_full_name)})")
    st.markdown(
        "<div class='section-lead'>Triggers an alert only when both the Isolation Forest model and the 1.5×IQR quantile bounds flag the observation.</div>",
        unsafe_allow_html=True,
    )

    scen_state_key = f"scenario_state_{selected_region}"
    if scen_state_key not in st.session_state:
        st.session_state[scen_state_key] = "baseline"

    scen_labels = {
        "baseline": "Live Baseline",
        "surge": "Surge Test",
        "custom": "Custom Load",
    }
    scen_raw = st.segmented_control(
        "Evaluation Scenario",
        options=["baseline", "surge", "custom"],
        selection_mode="single",
        default=st.session_state[scen_state_key],
        format_func=lambda k: scen_labels[k],
        key=f"scenario_ctrl_{selected_region}",
        label_visibility="collapsed",
        wrap=True,
    )
    scenario = _safe_segmented_choice(scen_raw, scen_state_key, "baseline")

    custom_val: Optional[float] = None
    nom_p50 = float(nom_state.get("predicted_p50_mw", 50000.0))
    if scenario == "surge":
        custom_val = min(350_000.0, round(nom_p50 * 2.0 + 15000.0, 0))
    elif scenario == "custom":
        default_custom = min(350_000.0, max(100.0, float(round(nom_p50 * 1.35, 0))))
        custom_val = st.number_input(
            "Test Load Input (MW)",
            min_value=100.0,
            max_value=350_000.0,
            value=default_custom,
            step=1000.0,
            help="Defaults to +35% above baseline p50. Minor deviations (~15%) remain normal under dual-trigger checks.",
            key=f"custom_mw_{selected_region}",
        )

    evaluated_scenarios = st.session_state.setdefault("_evaluated_scenarios", set())
    eval_cache_key = (selected_region, None if custom_val is None else round(float(custom_val), 2))
    sentinel_loader = st.empty()
    if custom_val is not None and eval_cache_key not in evaluated_scenarios:
        sentinel_loader.markdown(
            build_operation_loading_html(
                "Evaluating Load Scenario",
                f"Running Isolation Forest, quantile bounds, and TreeSHAP for {custom_val:,.0f} MW...",
            ),
            unsafe_allow_html=True,
        )

    anomaly_res = evaluate_live_anomaly(selected_region, custom_demand=custom_val)
    evaluated_scenarios.add(eval_cache_key)
    sentinel_loader.empty()

    st.markdown(build_sentinel_gates_html(anomaly_res), unsafe_allow_html=True)

    anomaly_drivers = anomaly_res.get("anomaly_shap_features")
    baseline_drivers = anomaly_res["top_shap_features"]
    primary_drivers = anomaly_drivers if anomaly_drivers else baseline_drivers
    primary_mode = "anomaly" if anomaly_drivers else "forecast"

    st.markdown("### Feature Contributions (TreeSHAP)")
    st.markdown(
        "<div class='section-lead'>Top three load or weather features contributing to this result.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(build_shap_driver_cards_html(primary_drivers, mode=primary_mode), unsafe_allow_html=True)

    _render_plotly_chart(
        get_cached_shap_figure(
            f"{eval_cache_key}:{primary_mode}",
            primary_drivers,
            mode=primary_mode,
        )
    )

    st.markdown("### Contribution Breakdown")
    df_attr = pd.DataFrame([
        {
            "Rank": f"#{i + 1}",
            "Parameter": FEATURE_METADATA.get(f["feature"], {}).get("name", f["feature"]),
            "Code": f["feature"],
            "Contribution": f"{f['shap_value']:+.3f}" if primary_mode == "anomaly" else f"{f['shap_value']:+,.1f} MW",
            "Observed Value": f"{f['actual_value']:,.2f}",
        }
        for i, f in enumerate(primary_drivers)
    ])
    st.dataframe(
        df_attr,
        width="stretch",
        height="content",
        hide_index=True,
        column_config={
            "Rank": st.column_config.TextColumn("Rank", width=64),
            "Parameter": st.column_config.TextColumn("Parameter", width=220),
            "Code": st.column_config.TextColumn("Code", width=160),
            "Contribution": st.column_config.TextColumn("Contribution", width=110, alignment="right"),
            "Observed Value": st.column_config.TextColumn("Observed Value", width=120, alignment="right"),
        },
    )

    if primary_mode == "anomaly":
        st.markdown("### Baseline Forecast Drivers (LightGBM SHAP)")
        st.markdown(
            "<div class='section-lead'>Expected baseline load drivers prior to evaluating any demand override.</div>",
            unsafe_allow_html=True,
        )
        _render_plotly_chart(
            get_cached_shap_figure(
                f"{selected_region}:baseline:forecast",
                baseline_drivers,
                mode="forecast",
            )
        )

elif active_nav == "Model Health":
    st.markdown(f"## Model Health ({html.escape(region_full_name)})")
    st.markdown(
        "<div class='section-lead'>Compares the trailing 7-day window against the 30-day December 2024 baseline across 12 features.</div>",
        unsafe_allow_html=True,
    )

    drift_alarm = bool(drift_state["drift_alarm"])
    drifted_n = int(drift_state["drifted_features"])
    total_n = int(drift_state["total_features"])
    obs_hours = int(drift_state.get("observation_hours", 168))
    alphas_str = str(drift_state["recommended_quantile_alpha"])

    notice_cls = "notice-advisory" if drift_alarm else "notice-soft"
    notice_title = (
        "Seasonal Drift Alert (90% Interval Active)"
        if drift_alarm
        else "Winter Reference Baseline (December 2024)"
    )
    notice_copy = (
        f"{drifted_n} of {total_n} monitored features shifted (&gt;30% limit) relative to December 2024. "
        f"Prediction intervals automatically widen to {html.escape(alphas_str)}."
        if drift_alarm
        else f"{drifted_n} of {total_n} features shifted (30% limit). Reference values reflect December 2024 winter conditions (720h); "
        "seasonal weather shifts above 30% automatically widen prediction intervals to 90% CQR."
    )
    st.markdown(
        f"<div class='notice-bar {notice_cls}'>"
        f"<div><strong>{notice_title}:</strong> {notice_copy}</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        build_health_summary_cards_html(
            obs_hours=obs_hours,
            drifted_n=drifted_n,
            total_n=total_n,
            drift_ratio=float(drift_state["drift_ratio"]),
            alphas_str=alphas_str,
            drift_alarm=drift_alarm,
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### Feature Drift Breakdown")
    st.markdown(
        "<div class='chart-mobile-hint'><strong>Tip:</strong> Swipe horizontally across the table to inspect Observed, Limit, P-Value, Test, and Feature code.</div>",
        unsafe_allow_html=True,
    )
    df_metrics = get_drift_metrics_dataframe(selected_region)
    st.dataframe(
        df_metrics,
        width="stretch",
        height="content",
        hide_index=True,
        column_config={
            "Parameter": st.column_config.TextColumn("Parameter", width=176),
            "Status": st.column_config.TextColumn("Status", width=88),
            "Observed": st.column_config.TextColumn("Observed", width=76, alignment="right"),
            "Limit": st.column_config.TextColumn("Limit", width=62, alignment="right"),
            "P-Value": st.column_config.TextColumn("P-Value", width=82, alignment="right"),
            "Test": st.column_config.TextColumn("Statistical Test", width=140),
            "Code": st.column_config.TextColumn("Feature", width=150),
        },
    )

    st.markdown("### Statistical Drift Criteria")
    st.markdown(build_drift_criteria_card_html(), unsafe_allow_html=True)

    report_html = get_cached_drift_report(selected_region)
    report_region_label = "All India" if selected_region == "ALL_INDIA" else selected_region
    st.download_button(
        label=f"Download Drift Report ({report_region_label} HTML)",
        data=report_html,
        file_name=f"drift_report_{selected_region.lower()}.html",
        mime="text/html",
        key=f"dl_drift_{selected_region}",
    )

else:
    st.markdown("## Info for Nerds")
    st.markdown(
        "<div class='section-lead'>"
        "High-level system architecture, core formulas, and key engineering choices behind forecasting, anomaly detection, and drift monitoring."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(build_nerds_overview_kpis_html(), unsafe_allow_html=True)

    st.markdown("### How the Pipeline Works")
    st.markdown(build_architecture_grid_html(), unsafe_allow_html=True)

    st.markdown("### Core Decision & Calibration Formulas")
    st.markdown(build_nerds_math_cards_html(), unsafe_allow_html=True)

    st.markdown("### Engineering Highlights")
    st.markdown(build_implementation_details_html(), unsafe_allow_html=True)

    st.markdown("### System Stack Summary")
    df_stack_spec = pd.DataFrame([
        {
            "Layer": "Ingestion & Storage",
            "Technologies": "Open-Meteo, Pydantic v2, Supabase PostgreSQL",
            "Role": "Weather-elastic telemetry, schema checks, RLS storage & 90-day cleanup",
        },
        {
            "Layer": "Features & Models",
            "Technologies": "Polars, LightGBM, Scikit-Learn Isolation Forest",
            "Role": "26 causal features (shift(1)), 72h quantile bounds, dual-trigger sentinel",
        },
        {
            "Layer": "Explainability & Drift",
            "Technologies": "TreeSHAP, SciPy (Wasserstein, KS, TVD, Chi-Square)",
            "Role": "MW & outlier risk drivers, auto-widening from 80% to 90% CQR intervals",
        },
        {
            "Layer": "Serving & Automation",
            "Technologies": "GitHub Actions, FastAPI, Streamlit",
            "Role": "6h batch snapshots, REST API, single-process UI with security headers",
        },
    ])
    st.dataframe(
        df_stack_spec,
        width="stretch",
        height="content",
        hide_index=True,
        column_config={
            "Layer": st.column_config.TextColumn("Layer", width=180),
            "Technologies": st.column_config.TextColumn("Technologies", width=360),
            "Role": st.column_config.TextColumn("Role", width=620),
        },
    )

    st.markdown(build_nerds_readme_cta_html(), unsafe_allow_html=True)

st.markdown(build_footer_html(), unsafe_allow_html=True)