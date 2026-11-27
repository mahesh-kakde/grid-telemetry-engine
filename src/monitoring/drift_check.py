import os
import json
import html
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance
import requests

from src.logging_config import configure_logging
from src.models.load_artifacts import (
    load_reference_baseline,
    REGIONS,
)

logger = logging.getLogger(__name__)

CYCLICAL_FEATURES = [
    "sin_hour_k1",
    "cos_hour_k1",
    "sin_hour_k2",
    "cos_hour_k2",
    "sin_doy",
    "cos_doy",
]
CATEGORICAL_FEATURES = [
    "day_of_week",
    "is_holiday",
]
MONITORED_DRIFT_FEATURES = [
    "demand_mw",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_solar_radiation",
    "wind_speed_100m",
    "temp_delta_24h",
    "rh_delta_24h",
    "demand_rolling_std_24h",
    "sin_hour_k1",
    "cos_hour_k1",
    "day_of_week",
    "is_holiday",
]
DRIFT_ALARM_THRESHOLD = 0.30
KS_PVALUE_THRESHOLD = 0.01
WASSERSTEIN_NORM_THRESHOLD = 0.50
WASSERSTEIN_THRESHOLD = 0.15
TVD_THRESHOLD = 0.15
DEFAULT_REPORT_PATH = os.path.join("static", "reports", "drift_report.html")
REGION_NAMES = {
    "ALL_INDIA": "ALL_INDIA — National Grid",
    "NR": "NR — Northern Region",
    "WR": "WR — Western Region",
    "SR": "SR — Southern Region",
    "ER": "ER — Eastern Region",
    "NER": "NER — North Eastern Region",
}

FEATURE_METADATA: Dict[str, Dict[str, str]] = {
    "demand_mw": {
        "name": "Active Electrical Demand",
        "desc": "Total electrical load consumed across the regional transmission network (MW)",
    },
    "temperature_2m": {
        "name": "Surface Ambient Temperature",
        "desc": "Air temperature 2m above ground; primary driver for cooling/heating demand",
    },
    "relative_humidity_2m": {
        "name": "Atmospheric Relative Humidity",
        "desc": "Relative humidity percentage; amplifies heat index and AC air conditioning usage",
    },
    "surface_solar_radiation": {
        "name": "Solar Irradiance / Sunlight",
        "desc": "Shortwave solar radiation; offsets transmission load via rooftop and utility solar",
    },
    "wind_speed_100m": {
        "name": "Wind Speed at Turbine Height",
        "desc": "Wind velocity at 100m turbine hub height; correlates with regional wind generation",
    },
    "hour": {
        "name": "Hour of the Day (0-23)",
        "desc": "Diurnal cycle index capturing evening consumption peaks and overnight baseload",
    },
    "day_of_week": {
        "name": "Day of Week (Mon-Sun)",
        "desc": "Calendar day indicator capturing weekday industrial production vs weekend drops",
    },
    "day_of_year": {
        "name": "Day of Year (Seasonal Progress)",
        "desc": "Annual seasonal trajectory capturing monsoon, summer heatwaves, and winter cycles",
    },
    "is_holiday": {
        "name": "National Statutory Holiday",
        "desc": "Binary indicator for gazetted statutory holidays with commercial load drops",
    },
    "sin_hour_k1": {
        "name": "24h Diurnal Rhythm (Harmonic Sine)",
        "desc": "Continuous Fourier transform representing the primary 24-hour diurnal cycle",
    },
    "cos_hour_k1": {
        "name": "24h Diurnal Rhythm (Harmonic Cosine)",
        "desc": "Continuous Fourier transform representing the primary 24-hour diurnal cycle",
    },
    "sin_hour_k2": {
        "name": "12h Semi-Diurnal Peak (Harmonic Sine)",
        "desc": "Secondary 12-hour harmonic capturing morning and evening dual peak rhythms",
    },
    "cos_hour_k2": {
        "name": "12h Semi-Diurnal Peak (Harmonic Cosine)",
        "desc": "Secondary 12-hour harmonic capturing morning and evening dual peak rhythms",
    },
    "sin_doy": {
        "name": "Annual Seasonal Cycle (Sine)",
        "desc": "Continuous annual sinusoidal wave representing yearly seasonal shifts",
    },
    "cos_doy": {
        "name": "Annual Seasonal Cycle (Cosine)",
        "desc": "Continuous annual sinusoidal wave representing yearly seasonal shifts",
    },
    "demand_lag_1h": {
        "name": "Demand 1 Hour Ago (t - 1h)",
        "desc": "Immediate electrical demand baseline",
    },
    "demand_lag_2h": {
        "name": "Demand 2 Hours Ago (t - 2h)",
        "desc": "Two-hour autoregressive momentum indicator",
    },
    "demand_lag_3h": {
        "name": "Demand 3 Hours Ago (t - 3h)",
        "desc": "Three-hour autoregressive trajectory indicator",
    },
    "demand_lag_24h": {
        "name": "Same-Hour Demand Yesterday (24h Lag)",
        "desc": "Direct 24-hour lookback capturing yesterday's consumption at this exact time",
    },
    "demand_lag_48h": {
        "name": "Same-Hour Demand 2 Days Ago (48h Lag)",
        "desc": "48-hour lookback capturing multi-day diurnal persistence",
    },
    "demand_lag_168h": {
        "name": "Same-Hour Demand 1 Week Ago (168h Lag)",
        "desc": "Full 7-day lookback capturing weekly cyclic behavior at this exact hour and day",
    },
    "demand_rolling_mean_6h": {
        "name": "6-Hour Mean Demand Trend",
        "desc": "Short-term moving average capturing rapid intraday demand trajectory",
    },
    "demand_rolling_mean_24h": {
        "name": "24-Hour Mean Demand Trend",
        "desc": "Full diurnal moving average isolating underlying daily baseload",
    },
    "demand_rolling_mean_168h": {
        "name": "7-Day Mean Demand Baseline",
        "desc": "Weekly moving average indicating sustained regional demand expansion or contraction",
    },
    "demand_rolling_std_6h": {
        "name": "6-Hour Demand Volatility (Std Dev)",
        "desc": "Short-term demand variance capturing rapid intraday load turbulence",
    },
    "demand_rolling_std_24h": {
        "name": "24-Hour Demand Volatility (Std Dev)",
        "desc": "24-hour load standard deviation measuring intraday peak-to-trough amplitude",
    },
    "demand_rolling_std_168h": {
        "name": "7-Day Demand Volatility (Std Dev)",
        "desc": "Weekly demand standard deviation measuring medium-term grid variability",
    },
    "temp_delta_24h": {
        "name": "24-Hour Temperature Shift",
        "desc": "Day-over-day ambient temperature differential driving cooling demand surges",
    },
    "rh_delta_24h": {
        "name": "24-Hour Relative Humidity Shift",
        "desc": "Day-over-day humidity differential affecting perceived discomfort index and cooling",
    },
    "demand_delta_1h": {
        "name": "1-Hour Load Ramp (Δ1h)",
        "desc": "Instantaneous demand change compared to 1 hour earlier (MW)",
    },
    "demand_delta_24h": {
        "name": "24-Hour Load Shift (Δ24h)",
        "desc": "Diurnal demand deviation compared to the same hour yesterday (MW)",
    },
    "demand_zscore_24h": {
        "name": "24-Hour Rolling Load Z-Score",
        "desc": "Standardized load surprise relative to trailing 24-hour mean and volatility",
    },
}


def _evaluate_categorical_drift(u: np.ndarray, v: np.ndarray) -> Dict[str, Any]:
    """Evaluates discrete calendar indicator drift via Total Variation Distance gated by Chi-square significance."""
    cats = sorted(set(u.tolist()) | set(v.tolist()))
    counts = [(int(np.sum(u == c)), int(np.sum(v == c))) for c in cats]
    active = [(cu, cv) for cu, cv in counts if (cu + cv) > 0]
    if len(active) < 2:
        return {
            "test": "tvd_chi2",
            "statistic": 0.0,
            "p_value": 1.0,
            "threshold": TVD_THRESHOLD,
            "drift_detected": False,
        }
    c_u = np.array([pair[0] for pair in active], dtype=float)
    c_v = np.array([pair[1] for pair in active], dtype=float)
    tvd = 0.5 * float(np.sum(np.abs(c_u / len(u) - c_v / len(v))))
    _, p_val, _, _ = chi2_contingency([c_u, c_v])
    is_drifted = bool(tvd >= TVD_THRESHOLD and p_val < KS_PVALUE_THRESHOLD)
    return {
        "test": "tvd_chi2",
        "statistic": round(tvd, 4),
        "p_value": round(float(p_val), 6),
        "threshold": TVD_THRESHOLD,
        "drift_detected": is_drifted,
    }


def evaluate_feature_drift(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "demand_mw",
) -> Dict[str, Any]:
    # evaluates normalized wasserstein + ks on physical features, wasserstein on cyclical harmonics, and tvd + chi2 on discrete calendar indicators
    if feature_cols is None:
        eval_cols = [c for c in MONITORED_DRIFT_FEATURES if c in ref_df.columns and c in curr_df.columns]
    else:
        eval_cols = list(dict.fromkeys(
            [c for c in feature_cols if c in ref_df.columns and c in curr_df.columns]
            + ([target_col] if target_col in ref_df.columns and target_col in curr_df.columns else [])
        ))

    metrics = {}
    for col in eval_cols:
        u = pd.to_numeric(ref_df[col], errors="coerce").dropna().to_numpy(dtype=float)
        v = pd.to_numeric(curr_df[col], errors="coerce").dropna().to_numpy(dtype=float)

        if len(u) == 0 or len(v) == 0:
            continue

        if col in CYCLICAL_FEATURES:
            w_score = float(wasserstein_distance(u, v))
            is_drifted = bool(w_score >= WASSERSTEIN_THRESHOLD)
            metrics[col] = {
                "test": "wasserstein",
                "score": round(w_score, 4),
                "threshold": WASSERSTEIN_THRESHOLD,
                "drift_detected": is_drifted,
            }
        elif col in CATEGORICAL_FEATURES:
            metrics[col] = _evaluate_categorical_drift(u, v)
        else:
            ref_std = float(np.std(u))
            w_raw = float(wasserstein_distance(u, v))
            w_norm = w_raw / max(ref_std, 1e-6)
            ks_res = ks_2samp(u, v)
            is_drifted = bool(w_norm >= WASSERSTEIN_NORM_THRESHOLD and ks_res.pvalue < KS_PVALUE_THRESHOLD)
            metrics[col] = {
                "test": "wasserstein_ks",
                "statistic": round(w_norm, 4),
                "ks_statistic": round(float(ks_res.statistic), 4),
                "p_value": round(float(ks_res.pvalue), 6),
                "threshold": WASSERSTEIN_NORM_THRESHOLD,
                "drift_detected": is_drifted,
            }

    total_features = len(metrics)
    drifted_features = sum(1 for m in metrics.values() if m["drift_detected"])
    drift_ratio = round(drifted_features / total_features, 4) if total_features > 0 else 0.0
    drift_alarm = drift_ratio > DRIFT_ALARM_THRESHOLD

    return {
        "observation_hours": int(len(curr_df)),
        "total_features": total_features,
        "drifted_features": drifted_features,
        "drift_ratio": drift_ratio,
        "drift_alarm": drift_alarm,
        "retraining_advisory": drift_alarm,
        "recommended_quantile_alpha": [0.05, 0.50, 0.95] if drift_alarm else [0.10, 0.50, 0.90],
        "feature_metrics": metrics,
    }


def select_drift_observation_window(
    feat_df: Optional[pd.DataFrame],
    ref_df: pd.DataFrame,
    window_hours: int = 168,
) -> Tuple[pd.DataFrame, str]:
    # requires a full post-warmup observation window before running distribution tests
    if feat_df is not None and len(feat_df) >= window_hours:
        return feat_df.tail(window_hours).copy(), "live_window"
    avail = 0 if feat_df is None else len(feat_df)
    logger.warning(
        f"insufficient post-warmup rows ({avail} < {window_hours}) for drift audit; "
        "falling back to reference baseline tail"
    )
    return ref_df.tail(window_hours).copy(), "baseline_fallback"


DRIFT_TEST_LABELS = {
    "wasserstein_ks": "Wasserstein + KS",
    "tvd_chi2": "TVD + Chi-Square",
    "kolmogorov_smirnov": "Kolmogorov-Smirnov",
}


def format_drift_test_label(test_code: str) -> str:
    """Maps internal statistical test identifiers to operator-facing display labels."""
    return DRIFT_TEST_LABELS.get(test_code, "1D Wasserstein")


def build_html_drift_report(
    drift_summary: Dict[str, Any],
    region: str = "ALL_INDIA",
    generated_at: Optional[str] = None,
) -> str:
    # generates standalone html report mirroring the Apple-inspired dashboard design tokens
    full_region = html.escape(str(REGION_NAMES.get(region, region)))
    timestamp = html.escape(str(generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")))

    total_features = int(drift_summary.get("total_features", 0))
    drifted_features = int(drift_summary.get("drifted_features", 0))
    drift_ratio = float(drift_summary.get("drift_ratio", 0.0))
    drift_alarm = bool(drift_summary.get("drift_alarm", False))
    alphas = html.escape(str(drift_summary.get("recommended_quantile_alpha", [0.10, 0.50, 0.90])))
    obs_hours = int(drift_summary.get("observation_hours", 168))
    obs_label = (
        "Trailing 7-Day (168h)"
        if obs_hours == 168
        else html.escape(f"Trailing {obs_hours}h Window")
    )

    if drift_alarm:
        notice_class = "notice-alarm"
        notice_title = "Seasonal Drift Advisory"
        notice_desc = (
            f"Covariate feature drift exceeds the 30.0% alarm threshold against the December 2024 winter baseline "
            f"({drifted_features} of {total_features} features shifted, {drift_ratio * 100:.1f}%). "
            f"Quantile bounds widened automatically to {alphas}; warmer spring/summer weather "
            f"can trigger this guard intentionally."
        )
    else:
        notice_class = "notice-normal"
        notice_title = "Winter Reference Baseline (December 2024)"
        notice_desc = (
            f"{drifted_features} of {total_features} features shifted ({drift_ratio * 100:.1f}%, within the 30.0% limit). "
            f"Reference values reflect December 2024 winter conditions (720h); seasonal weather shifts above 30% "
            f"automatically widen prediction intervals to 90% CQR."
        )

    # assemble sorted table rows (shifted features first, then alphabetically by label)
    feature_metrics = drift_summary.get("feature_metrics", {})
    sorted_features = sorted(
        feature_metrics.items(),
        key=lambda item: (
            0 if bool(item[1].get("drift_detected", False)) else 1,
            str(FEATURE_METADATA.get(item[0], {}).get("name", item[0])),
        ),
    )

    table_rows = []
    for feat_name, metric_info in sorted_features:
        safe_feat = html.escape(str(feat_name))
        meta = FEATURE_METADATA.get(feat_name, {"name": feat_name, "desc": "Telemetry parameter"})
        friendly_title = html.escape(str(meta["name"]))
        test_type = str(metric_info.get("test", "unknown"))
        test_label = format_drift_test_label(test_type)
        stat_val = float(metric_info.get("statistic", metric_info.get("score", 0.0)))
        thresh_val = float(metric_info.get("threshold", 0.0))
        p_val_str = f"{metric_info['p_value']:.6f}" if "p_value" in metric_info else "&mdash;"
        is_drifted = bool(metric_info.get("drift_detected", False))
        status_html = (
            '<span class="status-shifted">Shifted</span>'
            if is_drifted
            else '<span class="status-nominal">Nominal</span>'
        )

        table_rows.append(
            f"<tr>"
            f"<td>{friendly_title}</td>"
            f"<td>{status_html}</td>"
            f"<td class='num'>{stat_val:.4f}</td>"
            f"<td class='num'>{thresh_val:.4f}</td>"
            f"<td class='num'>{p_val_str}</td>"
            f"<td>{test_label}</td>"
            f"<td><code>{safe_feat}</code></td>"
            f"</tr>"
        )
    table_rows_html = "\n".join(table_rows)

    drift_status_color = "#b91c1c" if drift_alarm else "#15803d"
    drift_status_text = "Warning" if drift_alarm else "Nominal"
    interval_desc = "Widened 90% CQR bounds" if drift_alarm else "Standard 80% prediction interval"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Statistical Drift Diagnostic Report - {full_region}</title>
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%230066cc'/%3E%3Cpolyline points='4,16 9,16 12,9 15,23 18,11 21,19 23,16 28,16' fill='none' stroke='%23ffffff' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E">
    <style>
        :root {{
            --bg-canvas: #f5f5f7;
            --bg-surface: #ffffff;
            --bg-note: #fbfbfd;
            --text-primary: #1d1d1f;
            --text-secondary: #515154;
            --text-muted: #6e6e73;
            --accent-blue: #0066cc;
            --border-hairline: #d2d2d7;
            --border-subtle: #e5e5ea;
            --status-ok-fg: #15803d;
            --status-warn-fg: #92400e;
            --status-warn-bg: #fffbeb;
            --status-warn-border: #fcd34d;
            --status-err-fg: #b91c1c;
            --radius-card: 16px;
            --radius-notice: 12px;
            --shadow-card: 0 1px 2px rgba(0, 0, 0, 0.04), 0 4px 14px rgba(0, 0, 0, 0.03);
        }}
        * {{
            box-sizing: border-box;
        }}
        body {{
            margin: 0;
            padding: 0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Inter", "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-canvas);
            color: var(--text-primary);
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}
        .site-header {{
            position: sticky;
            top: 0;
            z-index: 50;
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border-hairline);
        }}
        .site-header-inner {{
            max-width: 1240px;
            margin: 0 auto;
            padding: 14px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
        }}
        .brand-title {{
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.025em;
            color: var(--text-primary);
            line-height: 1.2;
        }}
        .report-badge {{
            font-size: 13.5px;
            font-weight: 500;
            color: var(--text-secondary);
        }}
        .report-container {{
            flex: 1 0 auto;
            width: 100%;
            max-width: 1240px;
            margin: 0 auto;
            padding: 28px 32px 48px;
        }}
        h1 {{
            font-size: 26px;
            font-weight: 700;
            margin: 0 0 6px 0;
            color: var(--text-primary);
            letter-spacing: -0.02em;
            line-height: 1.25;
        }}
        .subtitle {{
            font-size: 15px;
            color: var(--text-secondary);
            margin: 0 0 20px 0;
            line-height: 1.5;
        }}
        .meta-bar {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 16px;
            margin-bottom: 16px;
            padding: 16px 20px;
            background-color: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-card);
            box-shadow: var(--shadow-card);
        }}
        .meta-item {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}
        .meta-label {{
            font-size: 11.5px;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.06em;
        }}
        .meta-val {{
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
            font-variant-numeric: tabular-nums;
        }}
        .notice-bar {{
            padding: 14px 18px;
            border-radius: var(--radius-notice);
            font-size: 13.5px;
            line-height: 1.55;
            margin-bottom: 16px;
        }}
        .notice-normal {{
            background-color: var(--bg-note);
            border: 1px solid var(--border-subtle);
            color: var(--text-secondary);
        }}
        .notice-normal strong {{
            color: var(--text-primary);
        }}
        .notice-alarm {{
            background-color: var(--status-warn-bg);
            border: 1px solid var(--status-warn-border);
            color: var(--status-warn-fg);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 16px;
            margin-bottom: 8px;
        }}
        .metric-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-card);
            padding: 18px 20px;
            box-shadow: var(--shadow-card);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 118px;
        }}
        .metric-label {{
            font-size: 13.5px;
            color: var(--text-muted);
            font-weight: 500;
            margin-bottom: 6px;
        }}
        .metric-value {{
            font-size: 26px;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.025em;
            font-variant-numeric: tabular-nums;
            line-height: 1.15;
            margin-bottom: 6px;
        }}
        .metric-desc {{
            font-size: 13px;
            color: var(--text-secondary);
        }}
        h2 {{
            font-size: 19px;
            font-weight: 700;
            letter-spacing: -0.015em;
            margin: 32px 0 8px 0;
            color: var(--text-primary);
            line-height: 1.3;
        }}
        .table-wrap {{
            width: 100%;
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-card);
            box-shadow: var(--shadow-card);
            overflow: hidden;
        }}
        .table-scroll {{
            width: 100%;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13.5px;
        }}
        th {{
            background-color: var(--bg-note);
            border-bottom: 1px solid var(--border-subtle);
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
            font-size: 12px;
            color: var(--text-muted);
            white-space: nowrap;
        }}
        th.num, td.num {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        td {{
            border-bottom: 1px solid var(--border-subtle);
            padding: 11px 16px;
            color: var(--text-primary);
            vertical-align: middle;
        }}
        tbody tr:last-child td {{
            border-bottom: none;
        }}
        tbody tr:hover td {{
            background-color: #fafafc;
        }}
        .status-nominal {{
            color: var(--status-ok-fg);
            font-weight: 600;
        }}
        .status-shifted {{
            color: var(--status-err-fg);
            font-weight: 600;
        }}
        code {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            background-color: var(--bg-canvas);
            padding: 2px 6px;
            border-radius: 5px;
            font-size: 12.5px;
            color: var(--text-primary);
        }}
        .details-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-card);
            box-shadow: var(--shadow-card);
            overflow: hidden;
        }}
        .detail-row {{
            display: grid;
            grid-template-columns: minmax(220px, 1.35fr) minmax(260px, 2fr);
            gap: 16px;
            padding: 13px 20px;
            border-bottom: 1px solid var(--border-subtle);
            font-size: 13.5px;
            line-height: 1.5;
            align-items: center;
        }}
        .detail-row:last-child {{
            border-bottom: none;
        }}
        .detail-key {{
            font-weight: 600;
            color: var(--text-primary);
        }}
        .detail-val {{
            color: var(--text-secondary);
            font-variant-numeric: tabular-nums;
        }}
        .site-footer {{
            flex-shrink: 0;
            background: var(--bg-surface);
            border-top: 1px solid var(--border-hairline);
            margin-top: auto;
        }}
        .site-footer-inner {{
            max-width: 1240px;
            margin: 0 auto;
            padding: 14px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
            font-size: 13px;
            color: var(--text-muted);
        }}
        .site-footer a {{
            color: var(--accent-blue);
            text-decoration: none;
            font-weight: 500;
        }}
        .site-footer a:hover {{
            text-decoration: underline;
        }}
        @media (max-width: 900px) {{
            .metric-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
        }}
        @media (max-width: 640px) {{
            .site-header-inner,
            .site-footer-inner {{
                padding: 12px 16px;
            }}
            .report-container {{
                padding: 20px 16px 36px;
            }}
            h1 {{
                font-size: 22px;
            }}
            .metric-grid, .meta-bar {{
                grid-template-columns: 1fr;
                gap: 12px;
            }}
            .detail-row {{
                grid-template-columns: 1fr;
                gap: 4px;
                padding: 12px 16px;
            }}
            th, td {{
                padding: 10px 12px;
                font-size: 13px;
            }}
        }}
        @media print {{
            body {{
                background-color: #ffffff;
            }}
            .site-header {{
                position: static;
            }}
            .report-container {{
                padding: 16px 0;
                max-width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <header class="site-header">
        <div class="site-header-inner">
            <div class="brand-title">National Power Grid Telemetry</div>
            <div class="report-badge">Statistical Drift Diagnostic Report</div>
        </div>
    </header>

    <main class="report-container">
        <h1>Model Health Report ({full_region})</h1>
        <p class="subtitle">Compares the trailing {obs_hours}-hour telemetry window against the 30-day December 2024 winter baseline across {total_features} monitored features.</p>

        <div class="meta-bar">
            <div class="meta-item">
                <span class="meta-label">Regional Grid</span>
                <span class="meta-val">{full_region}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Baseline Window</span>
                <span class="meta-val">30-Day Reference (720h)</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Observation Window</span>
                <span class="meta-val">{obs_label}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Audit Generated</span>
                <span class="meta-val">{timestamp}</span>
            </div>
        </div>

        <div class="notice-bar {notice_class}">
            <strong>{notice_title}:</strong> {notice_desc}
        </div>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Monitored Features</div>
                <div class="metric-value">{total_features}</div>
                <div class="metric-desc">Physical telemetry contract</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Shifted Features</div>
                <div class="metric-value">{drifted_features} / {total_features}</div>
                <div class="metric-desc">Alert threshold &gt; 30.0% ({drift_ratio * 100:.1f}% observed)</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Drift Status</div>
                <div class="metric-value" style="color: {drift_status_color};">{drift_status_text}</div>
                <div class="metric-desc">Maximum tolerance 30.0%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Active Quantile Alphas</div>
                <div class="metric-value" style="font-size: 20px;">{alphas}</div>
                <div class="metric-desc">{interval_desc}</div>
            </div>
        </div>

        <h2>Feature Drift Breakdown</h2>
        <div class="table-wrap">
            <div class="table-scroll">
                <table>
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Status</th>
                            <th class="num">Observed</th>
                            <th class="num">Limit</th>
                            <th class="num">P-Value</th>
                            <th>Statistical Test</th>
                            <th>Feature</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows_html}
                    </tbody>
                </table>
            </div>
        </div>

        <h2>Statistical Drift Criteria</h2>
        <div class="details-card">
            <div class="detail-row">
                <div class="detail-key">Continuous Physical Metrics (demand_mw, weather, rolling std)</div>
                <div class="detail-val">Flagged when normalized 1D Wasserstein distance W&#x2081; / &#x03C3; &ge; 0.50 and two-sample Kolmogorov-Smirnov p &lt; 0.01.</div>
            </div>
            <div class="detail-row">
                <div class="detail-key">Cyclical Harmonics (sin_hour_k1, cos_hour_k1)</div>
                <div class="detail-val">Flagged when unnormalized 1D Wasserstein distance W&#x2081; &ge; 0.15 across the trailing window.</div>
            </div>
            <div class="detail-row">
                <div class="detail-key">Discrete Calendar Indicators (day_of_week, is_holiday)</div>
                <div class="detail-val">Flagged when Total Variation Distance TVD &ge; 0.15 and Chi-square p &lt; 0.01.</div>
            </div>
        </div>
    </main>

    <footer class="site-footer">
        <div class="site-footer-inner">
            <span>Created by <a href="https://maheshkakde.com" target="_blank" rel="noopener noreferrer">Mahesh Kakde</a></span>
            <a href="https://github.com/mahesh-kakde/grid-telemetry-engine" target="_blank" rel="noopener noreferrer">GitHub</a>
        </div>
    </footer>
</body>
</html>"""


def generate_evidently_drift_report(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    output_path: str = DEFAULT_REPORT_PATH,
    region: str = "ALL_INDIA",
    drift_summary: Optional[Dict[str, Any]] = None,
) -> str:
    # generates self-contained drift html report adhering to unified ui specifications
    abs_output = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    if drift_summary is None:
        drift_summary = evaluate_feature_drift(ref_df, curr_df, feature_cols=None)

    html_report = build_html_drift_report(drift_summary, region=region)
    with open(abs_output, "w", encoding="utf-8") as f:
        f.write(html_report)
    logger.info(f"drift diagnostic report written to {abs_output}")
    return abs_output


generate_drift_report = generate_evidently_drift_report


def fetch_trailing_telemetry_window(
    db_url: Optional[str] = None,
    region: str = "ALL_INDIA",
    window_hours: int = 168,
) -> pd.DataFrame:
    # retrieves trailing 7-day feature window from supabase cursor helper or baseline fallback
    from src.models.infer_and_detect import fetch_telemetry_window_from_db
    from src.models.load_artifacts import resolve_calendar_timezone
    from src.features.build_features import compute_telemetry_features

    pl_df = fetch_telemetry_window_from_db(
        region=region,
        hours=window_hours + 168,
        db_url=db_url,
    )
    ref_df = load_reference_baseline(region)
    cal_tz = resolve_calendar_timezone(region=region)
    live_feat_df = None
    if pl_df is not None and len(pl_df) >= window_hours + 168:
        feat_df = compute_telemetry_features(pl_df, drop_nulls_flag=True, calendar_tz=cal_tz)
        if not feat_df.is_empty():
            live_feat_df = feat_df.to_pandas()

    selected_df, _ = select_drift_observation_window(
        live_feat_df, ref_df, window_hours=window_hours
    )
    return selected_df


def run_drift_monitoring_pipeline(
    db_url: Optional[str] = None,
    region: str = "ALL_INDIA",
    report_output_path: str = DEFAULT_REPORT_PATH,
    render_trigger_url: Optional[str] = None,
) -> Dict[str, Any]:
    # executes 7-day vs 30-day baseline statistical comparison and exports html report
    logger.info(f"initiating statistical drift audit for region '{region}'...")
    ref_df = load_reference_baseline(region)
    curr_df = fetch_trailing_telemetry_window(db_url=db_url, region=region, window_hours=168)

    drift_summary = evaluate_feature_drift(ref_df, curr_df, feature_cols=None)

    # generate standalone html dashboard
    report_path = generate_evidently_drift_report(
        ref_df, curr_df, output_path=report_output_path, region=region, drift_summary=drift_summary
    )
    drift_summary["report_path"] = report_path
    drift_summary["region"] = region

    logger.info(
        f"drift audit complete for {region}: "
        f"{drift_summary['drifted_features']}/{drift_summary['total_features']} features drifted "
        f"(ratio: {drift_summary['drift_ratio']:.2f}, alarm: {drift_summary['drift_alarm']})"
    )

    # dispatch automated webhook if drift alarm triggered
    if drift_summary["drift_alarm"] and render_trigger_url:
        try:
            payload = {
                "event": "dataset_drift_alert",
                "region": region,
                "drift_ratio": drift_summary["drift_ratio"],
                "drifted_features": drift_summary["drifted_features"],
                "recommended_quantile_alpha": drift_summary["recommended_quantile_alpha"],
            }
            res = requests.post(render_trigger_url, json=payload, timeout=10)
            logger.info(f"dispatched drift alert webhook: status code {res.status_code}")
        except Exception as exc:
            logger.warning(f"failed to dispatch drift webhook: {exc}")

    return drift_summary


if __name__ == "__main__":
    configure_logging()
    db_url = os.environ.get("SUPABASE_DB_URL")
    trigger_url = os.environ.get("RENDER_TRIGGER_URL") or os.environ.get("ALERT_WEBHOOK_URL")
    all_results = {}
    for reg in REGIONS:
        out_path = DEFAULT_REPORT_PATH if reg == "ALL_INDIA" else os.path.join("static", "reports", f"drift_report_{reg}.html")
        all_results[reg] = run_drift_monitoring_pipeline(
            db_url=db_url,
            region=reg,
            report_output_path=out_path,
            render_trigger_url=trigger_url,
        )
    print(json.dumps(all_results, indent=2))