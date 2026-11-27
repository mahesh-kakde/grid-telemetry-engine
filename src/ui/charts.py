from typing import Any, Dict, List
import pandas as pd
import plotly.graph_objects as go

from src.monitoring.drift_check import FEATURE_METADATA

_UI_FONT_FAMILY = "-apple-system, BlinkMacSystemFont, sans-serif"
_HOVER_LABEL_STYLE = {
    "bgcolor": "rgba(29, 29, 31, 0.94)",
    "font_size": 11.5,
    "font_color": "#f5f5f7",
    "font_family": "-apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif",
    "bordercolor": "#3a3a3c",
}
_AXIS_COMMON = {
    "showgrid": True,
    "gridcolor": "#f2f2f7",
    "linecolor": "#e5e5ea",
    "automargin": True,
}

SHAP_SHORT_LABELS = {
    "demand_mw": "Active Demand",
    "demand_delta_1h": "Load Ramp (1h)",
    "demand_delta_24h": "Load Shift (24h)",
    "demand_zscore_24h": "Load Z-Score (24h)",
    "demand_lag_1h": "Demand (1h Lag)",
    "demand_lag_2h": "Demand (2h Lag)",
    "demand_lag_3h": "Demand (3h Lag)",
    "demand_lag_24h": "Demand (24h Lag)",
    "demand_lag_48h": "Demand (48h Lag)",
    "demand_lag_168h": "Demand (168h Lag)",
    "demand_rolling_mean_6h": "Mean Demand (6h)",
    "demand_rolling_mean_24h": "Mean Demand (24h)",
    "demand_rolling_mean_168h": "Mean Baseline (7d)",
    "demand_rolling_std_6h": "Volatility (6h)",
    "demand_rolling_std_24h": "Volatility (24h)",
    "demand_rolling_std_168h": "Volatility (7d)",
    "temperature_2m": "Temperature (2m)",
    "relative_humidity_2m": "Relative Humidity",
    "surface_solar_radiation": "Solar Radiation",
    "wind_speed_100m": "Wind Speed (100m)",
    "temp_delta_24h": "Temp Shift (24h)",
    "rh_delta_24h": "Humidity Shift (24h)",
    "hour": "Hour of Day",
    "day_of_week": "Day of Week",
    "day_of_year": "Day of Year",
    "is_holiday": "Holiday (IST)",
    "sin_hour_k1": "Diurnal Sin (24h)",
    "cos_hour_k1": "Diurnal Cos (24h)",
    "sin_hour_k2": "Semi-Diurnal Sin",
    "cos_hour_k2": "Semi-Diurnal Cos",
    "sin_doy": "Annual Sin",
    "cos_doy": "Annual Cos",
}


def _anchor_for_pos(pos: int, n_pts: int) -> str:
    if pos <= 2:
        return "left"
    return "right" if pos >= n_pts - 3 else "center"


def build_forecast_figure(
    df_fc: pd.DataFrame,
    horizon_hours: int = 72,
    widened: bool = False,
) -> go.Figure:
    # minimalist apple-style chart focusing on expected load trajectory and uncertainty band
    ci_label = "90% Widened CQR Band" if widened else "80% CQR Band"
    df_plot = df_fc.head(horizon_hours) if horizon_hours and horizon_hours < len(df_fc) else df_fc

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_plot["target_timestamp_utc"],
        y=df_plot["forecast_p90"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="p90 Upper",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df_plot["target_timestamp_utc"],
        y=df_plot["forecast_p10"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0, 102, 204, 0.10)",
        name=ci_label,
        hoverinfo="skip",
    ))

    dot_size = 5 if horizon_hours <= 24 else (4 if horizon_hours <= 48 else 0)
    mode = "lines+markers" if dot_size > 0 else "lines"
    fig.add_trace(go.Scatter(
        x=df_plot["target_timestamp_utc"],
        y=df_plot["forecast_p50"],
        mode=mode,
        line=dict(color="#0066cc", width=2.8, shape="spline", smoothing=0.45),
        marker=dict(size=dot_size, color="#0066cc"),
        name="Expected Demand (p50)",
        hovertemplate="<b>%{x|%d %b, %H:%M UTC}</b><br>Expected: <b>%{y:,.0f} MW</b><extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_plot["target_timestamp_utc"],
        y=df_plot["forecast_p90"],
        mode="lines",
        line=dict(color="#8e8e93", width=1.1, dash="dot"),
        name="Upper Limit (p90)",
        hovertemplate="Upper Limit (p90): %{y:,.0f} MW<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_plot["target_timestamp_utc"],
        y=df_plot["forecast_p10"],
        mode="lines",
        line=dict(color="#aeaeb2", width=1.1, dash="dot"),
        name="Lower Limit (p10)",
        hovertemplate="Lower Limit (p10): %{y:,.0f} MW<extra></extra>",
    ))

    y_range = None
    if not df_plot.empty:
        p50_vals = df_plot["forecast_p50"].to_numpy()
        n_pts = len(p50_vals)
        peak_pos = int(p50_vals.argmax())
        min_pos = int(p50_vals.argmin())
        peak_val = float(p50_vals[peak_pos])
        min_val = float(p50_vals[min_pos])

        extrema_specs = (
            (peak_pos, peak_val, f"Peak {peak_val:,.0f} MW", "#b91c1c", -26, "#991b1b", "rgba(254, 242, 242, 0.94)", "#fecaca"),
            (min_pos, min_val, f"Baseload {min_val:,.0f} MW", "#0066cc", 26, "#0055b3", "rgba(239, 246, 255, 0.94)", "#bfdbfe"),
        )
        for pos, val, label, arrow_color, ay, font_color, bgcolor, bordercolor in extrema_specs:
            fig.add_annotation(
                x=df_plot["target_timestamp_utc"].iloc[pos],
                y=val,
                text=label,
                showarrow=True,
                arrowhead=0,
                arrowwidth=1.2,
                arrowcolor=arrow_color,
                ax=0,
                ay=ay,
                xanchor=_anchor_for_pos(pos, n_pts),
                font=dict(size=10.5, color=font_color, family=_UI_FONT_FAMILY),
                bgcolor=bgcolor,
                bordercolor=bordercolor,
                borderwidth=1,
                borderpad=4,
            )

        y_low = float(df_plot["forecast_p10"].min())
        y_high = float(df_plot["forecast_p90"].max())
        y_span = max(y_high - y_low, 100.0)
        y_range = [max(0.0, y_low - y_span * 0.14), y_high + y_span * 0.16]

    if horizon_hours <= 24:
        dtick_val, tick_format = 4 * 3600 * 1000, "%H:%M<br>%d %b"
    elif horizon_hours <= 48:
        dtick_val, tick_format = 8 * 3600 * 1000, "%H:%M<br>%d %b"
    else:
        dtick_val, tick_format = 12 * 3600 * 1000, "%d %b<br>%H:%M"

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        margin=dict(l=34, r=24, t=80, b=74),
        hovermode="x unified",
        hoverlabel=_HOVER_LABEL_STYLE,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.18,
            xanchor="center",
            x=0.5,
            font=dict(size=10.5, color="#515154"),
        ),
        xaxis=dict(
            **_AXIS_COMMON,
            title="Target Time (UTC)",
            title_font=dict(size=11, color="#6e6e73"),
            title_standoff=6,
            tickfont=dict(size=10, color="#6e6e73"),
            dtick=dtick_val,
            tickformat=tick_format,
            zeroline=False,
        ),
        yaxis=dict(
            **_AXIS_COMMON,
            title="Active Demand (MW)",
            title_font=dict(size=11, color="#6e6e73"),
            tickfont=dict(size=10, color="#6e6e73"),
            tickformat=",.0f",
            range=y_range,
            zeroline=False,
        ),
        height=396,
    )
    return fig


def build_shap_figure(top_features: List[Dict[str, Any]], mode: str = "forecast") -> go.Figure:
    feature_codes = [f["feature"] for f in top_features]
    feature_names = [
        SHAP_SHORT_LABELS.get(code, FEATURE_METADATA.get(code, {}).get("name", code))
        for code in feature_codes
    ]
    shap_vals = [float(f["shap_value"]) for f in top_features]
    is_anomaly_mode = mode == "anomaly"
    bar_text = [
        f"{v:+.3f} risk" if is_anomaly_mode else f"{v:+,.1f} MW"
        for v in shap_vals
    ]
    hover_tpl = (
        "<b>%{y}</b><br>Anomaly Risk Shift: <b>%{x:+.3f}</b><extra></extra>"
        if is_anomaly_mode
        else "<b>%{y}</b><br>Projected Load Impact: <b>%{x:+,.1f} MW</b><extra></extra>"
    )
    x_title = (
        "Anomaly Contribution (+ pushes toward alert)"
        if is_anomaly_mode
        else "Shapley Load Contribution (MW)"
    )

    fig_shap = go.Figure(go.Bar(
        x=shap_vals,
        y=feature_names,
        orientation="h",
        marker=dict(
            color=["#dc2626" if v > 0 else "#0066cc" for v in shap_vals],
            line=dict(width=0),
        ),
        text=bar_text,
        textposition="auto",
        insidetextanchor="middle",
        textfont=dict(size=10.5, color="#ffffff"),
        hovertemplate=hover_tpl,
        cliponaxis=False,
    ))
    fig_shap.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        margin=dict(l=126, r=32, t=22, b=54),
        hovermode="closest",
        hoverlabel=_HOVER_LABEL_STYLE,
        xaxis=dict(
            **_AXIS_COMMON,
            title=x_title,
            title_font=dict(size=11, color="#6e6e73"),
            tickfont=dict(size=10, color="#6e6e73"),
            zeroline=True,
            zerolinecolor="#d2d2d7",
            zerolinewidth=1.5,
            rangemode="tozero",
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=11, color="#1d1d1f"),
            linecolor="#e5e5ea",
            automargin=True,
        ),
        height=288,
    )
    return fig_shap