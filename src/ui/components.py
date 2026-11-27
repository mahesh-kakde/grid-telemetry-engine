import html
from typing import Any, Dict, List, Sequence, Tuple

from src.monitoring.drift_check import FEATURE_METADATA
from src.ui.helpers import format_info_tip


def _render_details_card(items: Sequence[Tuple[str, str]]) -> str:
    rows = "".join(
        f"<div class='detail-row'><span class='detail-key'>{label}</span>"
        f"<span class='detail-val'>{desc}</span></div>"
        for label, desc in items
    )
    return f"<div class='details-card'>{rows}</div>"


def build_dashboard_skeleton_html(region_label: str) -> str:
    safe_region = html.escape(region_label)
    kpi_cards = "".join(
        "<div class='skeleton-card'>"
        "<div class='skeleton-line' style='width:52%;height:11px;margin-bottom:12px;'></div>"
        "<div class='skeleton-line' style='width:68%;height:24px;margin-bottom:10px;'></div>"
        "<div class='skeleton-line' style='width:78%;height:11px;'></div>"
        "</div>"
        for _ in range(4)
    )
    return (
        f"<div class='skeleton-region-banner' role='status' aria-live='polite'>"
        f"<div class='loading-window'>"
        f"<div class='loading-head'>"
        f"<span class='loading-title'>Loading {safe_region} telemetry...</span>"
        f"<span class='loading-meta'>Syncing snapshot &amp; forecast bounds</span>"
        f"</div>"
        f"<div class='loading-bar-track'><div class='loading-bar-fill'></div></div>"
        f"</div>"
        f"<div class='kpi-grid'>{kpi_cards}</div>"
        f"<div class='notice-grid'>"
        f"<div class='skeleton-notice'><div class='skeleton-line' style='width:84%;height:12px;'></div></div>"
        f"<div class='skeleton-notice'><div class='skeleton-line' style='width:80%;height:12px;'></div></div>"
        f"</div>"
        f"<div class='skeleton-chart'>"
        f"<div class='skeleton-line' style='width:34%;height:15px;margin-bottom:16px;'></div>"
        f"<div class='skeleton-line' style='width:100%;height:265px;border-radius:10px;'></div>"
        f"</div>"
        f"</div>"
    )


def build_operation_loading_html(title: str, detail: str) -> str:
    safe_title = html.escape(title)
    safe_detail = html.escape(detail)
    return (
        "<div class='loading-window' role='status' aria-live='polite'>"
        "<div class='loading-head'>"
        f"<span class='loading-title'>{safe_title}</span>"
        f"<span class='loading-meta'>{safe_detail}</span>"
        "</div>"
        "<div class='loading-bar-track'><div class='loading-bar-fill'></div></div>"
        "</div>"
    )


def build_notices_html(
    freshness: Dict[str, Any],
    drift_state: Dict[str, Any],
    widened: bool,
) -> str:
    src_label = html.escape(str(freshness.get("source_label", "Snapshot")))
    gen_str = html.escape(str(freshness.get("generated_at_display", "Unknown")))
    drifted_count = int(drift_state.get("drifted_features", 0))
    total_count = int(drift_state.get("total_features", 12))

    sim_card = (
        "<div class='notice-bar'>"
        "<div><strong>Simulation Note:</strong> POSOCO-calibrated IST demand driven by Open-Meteo weather data (non-SCADA). "
        f"<span class='notice-meta'>Source: {src_label}, {gen_str}.</span></div>"
        "</div>"
    )

    if widened or bool(drift_state.get("drift_alarm", False)):
        base_card = (
            "<div class='notice-bar notice-advisory'>"
            "<div><strong>Baseline Advisory:</strong> "
            f"{drifted_count} of {total_count} features shifted vs. the December 2024 winter baseline (&gt;30% limit). "
            "Prediction interval widened to 90% CQR.</div>"
            "</div>"
        )
    else:
        status_word = "nominal" if drifted_count == 0 else "within 30% limit"
        base_card = (
            "<div class='notice-bar'>"
            f"<div><strong>Baseline Note:</strong> {drifted_count} of {total_count} features shifted vs. the "
            f"December 2024 winter baseline ({status_word}; 80% CQR active). "
            "Warmer seasons widen intervals to 90% CQR above 30% drift.</div>"
            "</div>"
        )

    blocks = [f"<div class='notice-grid'>{sim_card}{base_card}</div>"]

    freshness_warning = freshness.get("warning_message")
    if freshness_warning:
        blocks.append(
            "<div class='notice-stack'>"
            "<div class='notice-bar notice-advisory'>"
            f"<div><strong>Freshness Warning:</strong> {html.escape(str(freshness_warning))}</div>"
            "</div>"
            "</div>"
        )

    return "".join(blocks)


def build_kpi_row_html(
    region_code: str,
    peak_mw: float,
    baseload_mw: float,
    avg_band_mw: float,
    widened: bool,
    nom_state: Dict[str, Any],
    drift_state: Dict[str, Any],
) -> str:
    display_code = "All India" if region_code == "ALL_INDIA" else region_code
    safe_code = html.escape(display_code)
    cqr_title = "90% Interval" if widened else "80% Interval"

    nom_alert = bool(nom_state.get("is_structural", False))
    sentinel_cls = "status-err" if nom_alert else "status-ok"
    sentinel_txt = "Alert" if nom_alert else "Normal"
    actual_mw = float(nom_state.get("actual_demand_mw", 0.0))
    pred_p50 = float(nom_state.get("predicted_p50_mw", 0.0))

    drift_alarm = bool(drift_state.get("drift_alarm", False))
    drifted_n = int(drift_state.get("drifted_features", 0))
    total_n = int(drift_state.get("total_features", 12))
    stability_pct = (1.0 - float(drift_state.get("drift_ratio", 0.0))) * 100.0
    health_cls = "status-warn" if drift_alarm else "status-ok"
    health_txt = f"{stability_pct:.1f}% Stable" if not drift_alarm else "Drift Alert"

    tip_reserve = format_info_tip(
        "Calibrated p10 to p90 uncertainty interval. Quantiles are sorted so p10 <= p50 <= p90 across all 72 hours."
    )
    tip_sentinel = format_info_tip(
        "Dual-trigger check requiring both Isolation Forest outlier status and 1.5xIQR residual bounds.",
        tip_right=True,
    )
    tip_health = format_info_tip(
        "Compares the trailing 7-day window against the December 2024 30-day baseline across 12 features.",
        tip_right=True,
    )

    return (
        "<div class='kpi-grid'>"
        "<div class='kpi-card'>"
        f"<div class='kpi-label'><span>{safe_code} 72h Peak (p50)</span></div>"
        f"<div class='kpi-value'>{peak_mw:,.0f} MW</div>"
        f"<div class='kpi-meta'>Baseload {baseload_mw:,.0f} MW (median forecast)</div>"
        "</div>"
        "<div class='kpi-card'>"
        f"<div class='kpi-label'><span>Prediction Interval</span>{tip_reserve}</div>"
        f"<div class='kpi-value'>{cqr_title}</div>"
        f"<div class='kpi-meta'>Mean width {avg_band_mw:,.0f} MW (p10-p90)</div>"
        "</div>"
        "<div class='kpi-card'>"
        f"<div class='kpi-label'><span>Anomaly Check</span>{tip_sentinel}</div>"
        f"<div class='kpi-value'><span class='status-chip {sentinel_cls}'>{sentinel_txt}</span></div>"
        f"<div class='kpi-meta'>Observed {actual_mw:,.0f} vs {pred_p50:,.0f} MW expected</div>"
        "</div>"
        "<div class='kpi-card'>"
        f"<div class='kpi-label'><span>Model Drift</span>{tip_health}</div>"
        f"<div class='kpi-value'><span class='status-chip {health_cls}'>{health_txt}</span></div>"
        f"<div class='kpi-meta'>{drifted_n} of {total_n} features shifted vs Dec 2024</div>"
        "</div>"
        "</div>"
    )


def build_sentinel_gates_html(anomaly_res: Dict[str, Any]) -> str:
    actual_mw = float(anomaly_res["actual_demand_mw"])
    p10_mw = float(anomaly_res.get("predicted_p10_mw", 0.0))
    p50_mw = float(anomaly_res["predicted_p50_mw"])
    p90_mw = float(anomaly_res.get("predicted_p90_mw", 0.0))

    breach = bool(anomaly_res["residual_breach"])
    structural = bool(anomaly_res["is_structural"])
    iso_score = float(anomaly_res.get("anomaly_score", 0.0))
    # gate 1 (isolation forest) is inferred when gate 2 holds or from raw isolation score
    gate1_flagged = structural or (iso_score <= -0.57 and breach)

    env_cls = "status-err" if breach else "status-ok"
    env_txt = "Breached" if breach else "Within Bounds"
    gate1_cls = "status-err" if structural else ("status-warn" if gate1_flagged else "status-ok")
    gate1_txt = "Outlier" if structural else ("Elevated" if iso_score < -0.54 else "Normal")
    verdict_cls = "status-err" if structural else "status-ok"
    verdict_txt = "Anomaly Alert" if structural else "Normal"

    tip_gate1 = format_info_tip(
        "9-feature Isolation Forest scoring whether load diverges from weather and recent hours."
    )
    tip_gate2 = format_info_tip(
        "Expected tolerance range [p10 - 1.5*IQR, p90 + 1.5*IQR] derived from quantile bounds."
    )
    tip_verdict = format_info_tip(
        "Raises an alert only when both Isolation Forest and the 1.5xIQR bounds trigger simultaneously.",
        tip_right=True,
    )

    return (
        "<div class='grid-3'>"
        "<div class='metric-card'>"
        f"<div class='metric-label'><span>Gate 1: Isolation Forest</span>{tip_gate1}</div>"
        f"<div class='metric-value'><span class='status-chip {gate1_cls}'>{gate1_txt}</span></div>"
        f"<div class='metric-caption'>Score {iso_score:.3f}, observed {actual_mw:,.0f} MW</div>"
        "</div>"
        "<div class='metric-card'>"
        f"<div class='metric-label'><span>Gate 2: 1.5×IQR Bounds</span>{tip_gate2}</div>"
        f"<div class='metric-value'><span class='status-chip {env_cls}'>{env_txt}</span></div>"
        f"<div class='metric-caption'>Expected {p50_mw:,.0f} MW ({p10_mw:,.0f} to {p90_mw:,.0f} MW)</div>"
        "</div>"
        "<div class='metric-card'>"
        f"<div class='metric-label'><span>Combined Result</span>{tip_verdict}</div>"
        f"<div class='metric-value'><span class='status-chip {verdict_cls}'>{verdict_txt}</span></div>"
        "<div class='metric-caption'>Triggered only when Gate 1 and Gate 2 fire together</div>"
        "</div>"
        "</div>"
    )


def build_shap_driver_cards_html(drivers: List[Dict[str, Any]], mode: str) -> str:
    cards: List[str] = []
    is_anom = mode == "anomaly"
    for idx, feat in enumerate(drivers):
        raw_code = str(feat["feature"])
        meta = FEATURE_METADATA.get(raw_code, {"name": raw_code, "desc": "Telemetry parameter"})
        title = html.escape(str(meta["name"]))
        desc = html.escape(str(meta["desc"]))
        val = float(feat["shap_value"])
        obs = float(feat["actual_value"])
        color = "#b91c1c" if val > 0 else "#0066cc"

        if is_anom:
            val_str = f"{val:+.3f} risk"
            impact = "Increases anomaly risk" if val > 0 else "Matches baseline pattern"
        else:
            val_str = f"{val:+,.1f} MW"
            impact = "Pushes demand higher" if val > 0 else "Pulls demand lower"

        cards.append(
            "<div class='metric-card'>"
            f"<div class='metric-label'><span>Driver #{idx + 1}: {title}</span></div>"
            f"<div class='metric-value' style='color: {color};'>{val_str}</div>"
            f"<div class='metric-caption'><strong>{impact}</strong> (Value: {obs:,.2f})<br>{desc}</div>"
            "</div>"
        )
    return f"<div class='grid-3'>{''.join(cards)}</div>"


def build_nerds_overview_kpis_html() -> str:
    kpis = (
        ("Telemetry Scope", "6 Grids", "All India aggregate plus 5 regional grids"),
        ("Forecast Models", "18 Boosters", "LightGBM p10 / p50 / p90 with conformal margins"),
        ("Fault Detection", "Dual-Trigger", "Isolation Forest + 1.5×IQR bound agreement"),
        ("Refresh Cadence", "Every 6h", "Automated ETL, scoring, and snapshot caching"),
    )
    cards = "".join(
        "<div class='kpi-card'>"
        f"<div class='kpi-label'><span>{label}</span></div>"
        f"<div class='kpi-value'>{val}</div>"
        f"<div class='kpi-meta'>{meta}</div>"
        "</div>"
        for label, val, meta in kpis
    )
    return f"<div class='kpi-grid'>{cards}</div>"


def build_architecture_grid_html() -> str:
    steps = [
        (
            "1. Ingestion & Validation",
            "Live Weather + Scaled Load",
            "Pulls hourly Open-Meteo temperature, humidity, solar radiation, and wind across 6 regional hubs to drive a POSOCO-calibrated demand series with automated gap repair.",
        ),
        (
            "2. Feature Pipeline",
            "Leakage-Free Signals",
            "Resamples each series onto clean 1-hour steps in Asia/Kolkata, computes Indian holiday calendars and cyclical harmonics, and shifts rolling windows strictly into the past.",
        ),
        (
            "3. Forecast Engine",
            "72h Conformal Intervals",
            "Projects 1h to 72h demand recursively across p10, p50, and p90 LightGBM models, widens raw bands using calibrated error margins, and reconciles All India as the regional sum.",
        ),
        (
            "4. Anomaly Check",
            "Consensus Outlier Gate",
            "Evaluates incoming demand and real-time ramp features together, raising an alert only when both the Isolation Forest model and the 1.5×IQR quantile bounds trigger.",
        ),
        (
            "5. Explainability & Drift",
            "TreeSHAP & Adaptive Bands",
            "Explains both demand drivers (MW) and outlier risk via TreeSHAP, and compares trailing 7-day data vs December 2024 to auto-widen intervals to 90% above 30% drift.",
        ),
        (
            "6. Serving Layer",
            "Precomputed DB Snapshots",
            "Scheduled runs publish precomputed snapshots and HTML drift reports to Supabase PostgreSQL so page loads return instantly without re-running full 72-hour inference.",
        ),
    ]
    cards = [
        "<div class='spec-card'>"
        f"<div class='spec-step'>{html.escape(step)}</div>"
        f"<div class='spec-title'>{html.escape(title)}</div>"
        f"<div class='spec-body'>{html.escape(body)}</div>"
        "</div>"
        for step, title, body in steps
    ]
    return f"<div class='grid-2x3'>{''.join(cards)}</div>"


def build_nerds_math_cards_html() -> str:
    cards = [
        (
            "Quantile Calibration",
            "Conformal Intervals &amp; Reconciliation",
            "p10 = max(0, q10 − margin)\np90 = q90 + margin\np10 ≤ p50 ≤ p90  (isotonic sort)\nAll India p50 = Σ Regional p50",
            "Adds held-out residual margins to raw quantiles for calibrated 80% or 90% coverage while keeping the national forecast equal to the sum of the five grids.",
        ),
        (
            "Outlier Detection",
            "Dual-Trigger Consensus Rule",
            "Gate 1: Isolation Forest = −1\nGate 2: load &gt; p90 + 1.5 × IQR\n     or load &lt; max(0, p10 − 1.5 × IQR)\nAlert = Gate 1 AND Gate 2",
            "Prevents false alarms by requiring both a multivariate outlier score and a prediction-band breach before raising an anomaly alert.",
        ),
        (
            "Attribution &amp; Drift",
            "TreeSHAP &amp; Adaptive Coverage",
            "Drivers = TreeSHAP(model, X)\nDrift Ratio = Shifted / 12 Features\nBand = 90% CQR if Ratio &gt; 30% else 80%",
            "Decomposes every forecast and spike into per-feature impact and automatically widens uncertainty bands when seasonal weather diverges beyond 30%.",
        ),
    ]
    rendered = [
        "<div class='spec-card'>"
        f"<div class='spec-step'>{step}</div>"
        f"<div class='spec-title'>{title}</div>"
        f"<div class='formula-box'>{formula}</div>"
        f"<div class='spec-body'>{html.escape(body)}</div>"
        "</div>"
        for step, title, formula, body in cards
    ]
    return f"<div class='grid-3'>{''.join(rendered)}</div>"


def build_health_summary_cards_html(
    obs_hours: int,
    drifted_n: int,
    total_n: int,
    drift_ratio: float,
    alphas_str: str,
    drift_alarm: bool,
) -> str:
    env_desc = "Widened 90% prediction interval" if drift_alarm else "Standard 80% prediction interval"
    return (
        "<div class='grid-3'>"
        "<div class='metric-card'>"
        "<div class='metric-label'><span>Observation Window</span></div>"
        f"<div class='metric-value'>Trailing {int(obs_hours)}h</div>"
        "<div class='metric-caption'>Compared vs 720h Dec 2024 baseline</div>"
        "</div>"
        "<div class='metric-card'>"
        "<div class='metric-label'><span>Shifted Features</span></div>"
        f"<div class='metric-value'>{int(drifted_n)} / {int(total_n)}</div>"
        f"<div class='metric-caption'>Alert threshold &gt; 30.0% ({drift_ratio * 100:.1f}% observed)</div>"
        "</div>"
        "<div class='metric-card'>"
        "<div class='metric-label'><span>Active Quantile Alphas</span></div>"
        f"<div class='metric-value'>{html.escape(alphas_str)}</div>"
        f"<div class='metric-caption'>{env_desc}</div>"
        "</div>"
        "</div>"
    )


def build_drift_criteria_card_html() -> str:
    return _render_details_card((
        (
            "Continuous Physical Metrics (<code>demand_mw</code>, weather, rolling std)",
            "Flagged only when normalized 1D Wasserstein distance <code>W₁ / σ ≥ 0.50</code> and two-sample Kolmogorov-Smirnov <code>p &lt; 0.01</code>.",
        ),
        (
            "Cyclical Diurnal Harmonics (<code>sin_hour_k1</code>, <code>cos_hour_k1</code>)",
            "Flagged when 1D Wasserstein distance <code>W₁ ≥ 0.15</code>.",
        ),
        (
            "Discrete Calendar Indicators (<code>day_of_week</code>, <code>is_holiday</code>)",
            "Flagged when Total Variation Distance <code>TVD ≥ 0.15</code> and Chi-square <code>p &lt; 0.01</code>.",
        ),
    ))


def build_implementation_details_html() -> str:
    return _render_details_card((
        (
            "Data Source &amp; Weather Elasticity",
            "Open-Meteo weather feeds run live every 6 hours, driving an IST demand simulation calibrated to POSOCO Dec 2024 regional scales (disclosed as non-SCADA synthetic load).",
        ),
        (
            "Zero-Leakage Causal Design",
            "Hourly <code>1h</code> Polars upsampling prevents dropped packets from aliasing lags, while rolling averages and standard deviations shift strictly past target time via <code>.shift(1)</code>.",
        ),
        (
            "Storage, Security &amp; Retention",
            "Supabase PostgreSQL stores telemetry, forecast vintages, anomaly logs, and snapshots behind Row-Level Security (<code>RLS</code>), SHA-256 verified model artifacts, and 90-day pruning.",
        ),
        (
            "Low-Memory Serving Architecture",
            "One single-query batch load hydrates all 6 regions on startup; lazy-loaded models and TreeSHAP explainers use a 2-region LRU cap to fit cleanly inside a 512 MB container.",
        ),
    ))


def build_nerds_readme_cta_html() -> str:
    return (
        "<div class='readme-cta'>"
        "<div><strong>Want the full deep dive?</strong> For detailed technical implementation, exact mathematical formulas, "
        "feature schemas, database tables, and API specs, check "
        "<a href='https://github.com/mahesh-kakde/grid-telemetry-engine/blob/main/README.md' "
        "target='_blank' rel='noopener noreferrer'>README.md</a> on GitHub.</div>"
        "</div>"
    )


def build_footer_html(version: str = "") -> str:
    return (
        "<footer class='app-footer'>"
        "<div class='app-footer-inner'>"
        "<div class='footer-author'>Created by "
        "<a href='https://maheshkakde.com' target='_blank' rel='noopener noreferrer'>Mahesh Kakde</a>"
        "</div>"
        "<div class='footer-links'>"
        "<a href='https://github.com/mahesh-kakde/grid-telemetry-engine' target='_blank' rel='noopener noreferrer'>GitHub</a>"
        "</div>"
        "</div>"
        "</footer>"
    )