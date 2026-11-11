from typing import Any
import pandas as pd
import plotly.graph_objects as go


def render_forecast_figure(
    history: list[dict[str, Any]],
    forecasts: list[dict[str, Any]],
    region: str = "ALL",
) -> go.Figure:
    fig = go.Figure()

    # historical actual telemetry trace
    if history:
        hist_df = pd.DataFrame(history)
        if "timestamp" in hist_df.columns and "metric_value" in hist_df.columns:
            hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
            hist_df = hist_df.sort_values("timestamp")

            fig.add_trace(
                go.Scatter(
                    x=hist_df["timestamp"],
                    y=hist_df["metric_value"],
                    name="Actual Demand",
                    mode="lines",
                    line=dict(color="#1e293b", width=2),
                    hovertemplate="%{x|%b %d, %H:%M UTC}<br>Actual: <b>%{y:,.0f} MW</b><extra></extra>",
                )
            )

    # 72-hour forecast traces with quantile ribbon
    if forecasts:
        fc_df = pd.DataFrame(forecasts)
        if "target_timestamp" in fc_df.columns and "p50" in fc_df.columns:
            fc_df["target_timestamp"] = pd.to_datetime(fc_df["target_timestamp"])
            fc_df = fc_df.sort_values("target_timestamp")

            # lower bound p10
            fig.add_trace(
                go.Scatter(
                    x=fc_df["target_timestamp"],
                    y=fc_df["p10"],
                    name="p10 (Lower Bound)",
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # upper bound p90 with shaded ribbon
            fig.add_trace(
                go.Scatter(
                    x=fc_df["target_timestamp"],
                    y=fc_df["p90"],
                    name="80% Confidence Interval [p10-p90]",
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(0, 113, 227, 0.12)",
                    hovertemplate="%{x|%b %d, %H:%M UTC}<br>p90: <b>%{y:,.0f} MW</b><extra></extra>",
                )
            )

            # median forecast p50
            fig.add_trace(
                go.Scatter(
                    x=fc_df["target_timestamp"],
                    y=fc_df["p50"],
                    name="Forecast Median (p50)",
                    mode="lines",
                    line=dict(color="#0071e3", width=2.5, dash="solid"),
                    hovertemplate="%{x|%b %d, %H:%M UTC}<br>p50: <b>%{y:,.0f} MW</b><extra></extra>",
                )
            )

    # apple-inspired clean light layout
    fig.update_layout(
        title=dict(
            text=f"72-Hour Demand Forecast Ribbon & Actuals ({region})",
            font=dict(size=16, color="#1d1d1f", family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
            x=0.01,
            y=0.96,
        ),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#e2e8f0",
            tickformat="%b %d\n%H:%M",
            tickfont=dict(size=11, color="#64748b"),
        ),
        yaxis=dict(
            title=dict(text="Demand (MW)", font=dict(size=12, color="#475569")),
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#e2e8f0",
            tickformat=",.0f",
            tickfont=dict(size=11, color="#64748b"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color="#475569"),
            bgcolor="rgba(255, 255, 255, 0.9)",
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=60, r=20, t=60, b=40),
        hovermode="x unified",
        height=450,
    )

    return fig