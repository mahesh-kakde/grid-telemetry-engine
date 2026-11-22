import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
import requests

try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except (ImportError, ModuleNotFoundError):
        try:
            from evidently.legacy.report import Report
            from evidently.legacy.metric_preset import DataDriftPreset
        except (ImportError, ModuleNotFoundError):
            from evidently import Report
            from evidently.presets.drift import DataDriftPreset
except Exception:
    # fallback when evidently encounters dependency conflicts in host environment
    Report = None
    DataDriftPreset = None

from src.models.load_artifacts import (
    load_reference_baseline,
    load_feature_columns,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CYCLICAL_CATEGORICAL_FEATURES = [
    "sin_hour_k1",
    "cos_hour_k1",
    "sin_hour_k2",
    "cos_hour_k2",
    "sin_doy",
    "cos_doy",
    "day_of_week",
    "is_holiday",
]
DRIFT_ALARM_THRESHOLD = 0.30
KS_PVALUE_THRESHOLD = 0.05
WASSERSTEIN_THRESHOLD = 0.10
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
}


def evaluate_feature_drift(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "demand_mw",
) -> Dict[str, Any]:
    # evaluates ks test for continuous features and wasserstein distance for cyclical/categorical features
    if feature_cols is None:
        feature_cols = load_feature_columns()

    eval_cols = [c for c in feature_cols if c in ref_df.columns and c in curr_df.columns]
    if target_col in ref_df.columns and target_col in curr_df.columns:
        eval_cols.append(target_col)

    metrics = {}
    for col in eval_cols:
        u = ref_df[col].dropna().to_numpy()
        v = curr_df[col].dropna().to_numpy()

        if len(u) == 0 or len(v) == 0:
            continue

        if col in CYCLICAL_CATEGORICAL_FEATURES:
            # 1d wasserstein distance for cyclical and discrete calendar indicators
            w_score = float(wasserstein_distance(u, v))
            is_drifted = bool(w_score >= WASSERSTEIN_THRESHOLD)
            metrics[col] = {
                "test": "wasserstein",
                "score": round(w_score, 4),
                "threshold": WASSERSTEIN_THRESHOLD,
                "drift_detected": is_drifted,
            }
        else:
            # two-sample kolmogorov-smirnov test for continuous features and demand target
            ks_res = ks_2samp(u, v)
            is_drifted = bool(ks_res.pvalue < KS_PVALUE_THRESHOLD)
            metrics[col] = {
                "test": "kolmogorov_smirnov",
                "statistic": round(float(ks_res.statistic), 4),
                "p_value": round(float(ks_res.pvalue), 6),
                "threshold": KS_PVALUE_THRESHOLD,
                "drift_detected": is_drifted,
            }

    total_features = len(metrics)
    drifted_features = sum(1 for m in metrics.values() if m["drift_detected"])
    drift_ratio = round(drifted_features / total_features, 4) if total_features > 0 else 0.0
    drift_alarm = drift_ratio > DRIFT_ALARM_THRESHOLD

    return {
        "total_features": total_features,
        "drifted_features": drifted_features,
        "drift_ratio": drift_ratio,
        "drift_alarm": drift_alarm,
        "retraining_advisory": drift_alarm,
        "recommended_quantile_alpha": [0.05, 0.50, 0.95] if drift_alarm else [0.10, 0.50, 0.90],
        "feature_metrics": metrics,
    }


def build_html_drift_report(
    drift_summary: Dict[str, Any],
    region: str = "ALL_INDIA",
    generated_at: Optional[str] = None,
) -> str:
    # generates standalone html report mirroring dashboard design standards and specifications
    full_region = REGION_NAMES.get(region, region)
    timestamp = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    total_features = drift_summary.get("total_features", 0)
    drifted_features = drift_summary.get("drifted_features", 0)
    drift_ratio = drift_summary.get("drift_ratio", 0.0)
    drift_alarm = drift_summary.get("drift_alarm", False)
    alphas = str(drift_summary.get("recommended_quantile_alpha", [0.10, 0.50, 0.90]))

    if drift_alarm:
        notice_class = "notice-alarm"
        notice_title = "Retraining Advisory Active"
        notice_desc = (
            f"Covariate feature drift exceeds the 30.0% alarm threshold "
            f"({drifted_features}/{total_features} features drifted, {drift_ratio * 100:.1f}%). "
            f"Model recalibration is advised; quantile prediction bounds recommended widening to {alphas}."
        )
    else:
        notice_class = "notice-normal"
        notice_title = "Distribution Stability Nominal"
        notice_desc = (
            f"Continuous and cyclical feature distributions match the 30-day reference baseline. "
            f"Covariate feature drift is within normal tolerance "
            f"({drifted_features}/{total_features} features drifted, {drift_ratio * 100:.1f}%). "
            f"Quantile intervals operating at standard [0.10, 0.50, 0.90]."
        )

    # assemble table rows for feature-by-feature evaluation
    table_rows = []
    feature_metrics = drift_summary.get("feature_metrics", {})
    for feat_name, metric_info in feature_metrics.items():
        meta = FEATURE_METADATA.get(feat_name, {"name": feat_name, "desc": "Telemetry parameter"})
        friendly_title = meta["name"]
        test_type = metric_info.get("test", "unknown")
        test_label = "Kolmogorov-Smirnov" if test_type == "kolmogorov_smirnov" else "Wasserstein Distance"
        stat_val = metric_info.get("statistic", metric_info.get("score", 0.0))
        thresh_val = metric_info.get("threshold", 0.0)
        p_val_str = f"{metric_info['p_value']:.6f}" if "p_value" in metric_info else "N/A (Wasserstein)"
        is_drifted = metric_info.get("drift_detected", False)
        status_html = (
            '<span style="color: #dc2626; font-weight: 600;">Drift Detected</span>'
            if is_drifted
            else '<span style="color: #059669; font-weight: 600;">Nominal</span>'
        )

        table_rows.append(
            f"<tr>"
            f"<td><code>{feat_name}</code><br><span style='font-size: 11.5px; color: #64748b; font-weight: 500;'>{friendly_title}</span></td>"
            f"<td>{test_label}</td>"
            f"<td>{stat_val:.4f}</td>"
            f"<td>{thresh_val:.4f}</td>"
            f"<td>{p_val_str}</td>"
            f"<td>{status_html}</td>"
            f"</tr>"
        )
    table_rows_html = "\n".join(table_rows)

    drift_status_color = "#dc2626" if drift_alarm else "#059669"
    drift_status_text = "Action Required" if drift_alarm else "Nominal"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Statistical Drift Diagnostic Report - {full_region}</title>
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%232563eb'/%3E%3Cpolyline points='4,16 9,16 12,9 15,23 18,11 21,19 23,16 28,16' fill='none' stroke='%23ffffff' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E">
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
            background-color: #f8fafc;
            color: #0f172a;
            line-height: 1.5;
            padding: 32px 16px;
            margin: 0;
            -webkit-font-smoothing: antialiased;
        }}
        .report-container {{
            max-width: 1040px;
            margin: 0 auto;
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 36px 40px;
            box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.05);
        }}
        .header-section {{
            margin-bottom: 24px;
            padding-bottom: 20px;
            border-bottom: 1px solid #e2e8f0;
        }}
        h1 {{
            font-size: 22px;
            font-weight: 700;
            margin: 0 0 6px 0;
            color: #0f172a;
            letter-spacing: -0.02em;
        }}
        .subtitle {{
            font-size: 13.5px;
            color: #64748b;
            margin: 0;
        }}
        .meta-bar {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-top: 18px;
            padding: 14px 18px;
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
        }}
        .meta-item {{
            display: flex;
            flex-direction: column;
        }}
        .meta-label {{
            font-size: 11px;
            color: #64748b;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 2px;
        }}
        .meta-val {{
            font-size: 13px;
            font-weight: 600;
            color: #0f172a;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin-bottom: 24px;
        }}
        @media (max-width: 768px) {{
            .report-container {{
                padding: 24px 20px;
            }}
            .metric-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        .metric-card {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px 18px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        }}
        .metric-label {{
            font-size: 11px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
            font-weight: 600;
        }}
        .metric-value {{
            font-size: 22px;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 4px;
        }}
        .metric-desc {{
            font-size: 11.5px;
            color: #64748b;
        }}
        .notice-card {{
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 28px;
            border: 1px solid #e2e8f0;
        }}
        .notice-normal {{
            border-left: 4px solid #059669;
            background-color: #f0fdf4;
            border-color: #bbf7d0;
            color: #166534;
        }}
        .notice-alarm {{
            border-left: 4px solid #dc2626;
            background-color: #fef2f2;
            border-color: #fecaca;
            color: #991b1b;
        }}
        .notice-title {{
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .notice-desc {{
            font-size: 13px;
            line-height: 1.5;
            opacity: 0.95;
        }}
        .section-header {{
            margin-bottom: 12px;
        }}
        h2 {{
            font-size: 16px;
            font-weight: 700;
            margin: 0 0 4px 0;
            color: #0f172a;
        }}
        .section-desc {{
            font-size: 13px;
            color: #64748b;
            margin: 0 0 14px 0;
        }}
        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 13px;
            margin-bottom: 32px;
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
        }}
        th {{
            background-color: #f8fafc;
            border-bottom: 2px solid #e2e8f0;
            padding: 11px 16px;
            text-align: left;
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #475569;
        }}
        td {{
            border-bottom: 1px solid #f1f5f9;
            padding: 11px 16px;
            color: #1e293b;
            vertical-align: middle;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        tr:hover td {{
            background-color: #f8fafc;
        }}
        code {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            background-color: #f1f5f9;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
            color: #0f172a;
        }}
        .footer {{
            margin-top: 32px;
            padding-top: 16px;
            border-top: 1px solid #e2e8f0;
            font-size: 12px;
            color: #64748b;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}
        @media print {{
            body {{
                background-color: #ffffff;
                padding: 0;
            }}
            .report-container {{
                border: none;
                box-shadow: none;
                padding: 0;
                max-width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <div class="header-section">
            <h1>National Power Grid Telemetry & Forecasting Engine</h1>
            <div class="subtitle">Statistical Covariate & Target Drift Diagnostic Report · Continuous MLOps Reliability Audit</div>
            
            <div class="meta-bar">
                <div class="meta-item">
                    <span class="meta-label">Regional Grid</span>
                    <span class="meta-val">{full_region}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Engine Version</span>
                    <span class="meta-val">v1.0.0</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Baseline Window</span>
                    <span class="meta-val">30-Day Reference (720h)</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Observation Window</span>
                    <span class="meta-val">Trailing 7-Day (168h)</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Audit Generated</span>
                    <span class="meta-val">{timestamp}</span>
                </div>
            </div>
        </div>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total Monitored Features</div>
                <div class="metric-value">{total_features}</div>
                <div class="metric-desc">Pydantic physical contract</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Drifted Features</div>
                <div class="metric-value" style="color: {drift_status_color};">{drifted_features} <span style="font-size: 14px; font-weight: 500; color: #64748b;">({drift_ratio * 100:.1f}%)</span></div>
                <div class="metric-desc">Status: <span style="font-weight: 600; color: {drift_status_color};">{drift_status_text}</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Watchdog Threshold</div>
                <div class="metric-value">30.0%</div>
                <div class="metric-desc">Maximum allowable shift</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Recommended Alpha Bounds</div>
                <div class="metric-value" style="font-size: 18px; font-family: monospace;">{alphas}</div>
                <div class="metric-desc">Operating safety envelope</div>
            </div>
        </div>

        <div class="notice-card {notice_class}">
            <div class="notice-title">{notice_title}</div>
            <div class="notice-desc">{notice_desc}</div>
        </div>

        <div class="section-header">
            <h2>Feature-by-Feature Statistical Drift Audit</h2>
            <div class="section-desc">Each physical parameter is tested for statistical distribution drift. Continuous variables use the two-sample Kolmogorov-Smirnov test (significance threshold p &lt; 0.05); cyclical harmonics use 1D Wasserstein distance (threshold W1 &ge; 0.10).</div>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Audit Test</th>
                    <th>Observed Metric</th>
                    <th>Safe Threshold</th>
                    <th>P-Value / Score</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {table_rows_html}
            </tbody>
        </table>

        <div class="footer">
            <div>Created by <a href="https://maheshkakde.com" target="_blank" style="color: #2563eb; text-decoration: none; font-weight: 600;">Mahesh Kakde</a></div>
        </div>
    </div>
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
        feature_cols = load_feature_columns()
        drift_summary = evaluate_feature_drift(ref_df, curr_df, feature_cols=feature_cols)

    html_report = build_html_drift_report(drift_summary, region=region)
    with open(abs_output, "w", encoding="utf-8") as f:
        f.write(html_report)
    logger.info(f"drift diagnostic report written to {abs_output}")
    return abs_output


def fetch_trailing_telemetry_window(
    db_url: Optional[str] = None,
    region: str = "ALL_INDIA",
    window_hours: int = 168,
) -> pd.DataFrame:
    # retrieves trailing 7-day feature window from postgres or falls back to baseline slice
    if db_url and psycopg2 is not None:
        try:
            from src.features.build_features import compute_telemetry_features
            import polars as pl

            conn = psycopg2.connect(db_url)
            try:
                query = f"""
                    SELECT timestamp_utc, region, demand_mw, frequency_hz,
                           temperature_2m, relative_humidity_2m,
                           surface_solar_radiation, wind_speed_100m
                    FROM regional_telemetry
                    WHERE region = %s
                    ORDER BY timestamp_utc DESC
                    LIMIT %s;
                """
                pdf = pd.read_sql(query, conn, params=(region, window_hours + 168))
            finally:
                conn.close()

            if len(pdf) >= 168:
                pdf = pdf.sort_values("timestamp_utc").reset_index(drop=True)
                pl_df = pl.from_pandas(pdf)
                feat_df = compute_telemetry_features(pl_df, drop_nulls_flag=False)
                return feat_df.tail(window_hours).to_pandas()
        except Exception as exc:
            logger.warning(f"failed to fetch telemetry from database: {exc}. using baseline fallback")

    # baseline fallback slice
    ref_df = load_reference_baseline(region)
    return ref_df.tail(window_hours).copy()


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

    feature_cols = load_feature_columns()
    drift_summary = evaluate_feature_drift(ref_df, curr_df, feature_cols=feature_cols)

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
    db_url = os.environ.get("SUPABASE_DB_URL")
    trigger_url = os.environ.get("RENDER_TRIGGER_URL")
    result = run_drift_monitoring_pipeline(db_url=db_url, render_trigger_url=trigger_url)
    print(json.dumps(result, indent=2))