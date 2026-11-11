from typing import Any
import pandas as pd
import plotly.graph_objects as go


def render_forecast_figure(
    history: list[dict[str, Any]],
    forecasts: list[dict[str, Any]],
    region: str = "ALL",
) -> go.Figure:
    """Renders a clean minimalist 72h forecast chart with p10-p90 quantile ribbon."""
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
                    line=dict(color="#1d1d1f", width=2.2, shape="spline", smoothing=0.3),
                    hovertemplate="%{x|%b %d, %H:%M UTC}<br>Actual: <b>%{y:,.0f} MW</b><extra></extra>",
                )
            )

    # 72-hour forecast traces with quantile ribbon
    if forecasts:
        fc_df = pd.DataFrame(forecasts)
        if "target_timestamp" in fc_df.columns and "p50" in fc_df.columns:
            fc_df["target_timestamp"] = pd.to_datetime(fc_df["target_timestamp"])
            fc_df = fc_df.sort_values("target_timestamp")

            # lower bound p10 (foundation line for interval shading)
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

            # upper bound p90 with translucent uncertainty ribbon
            fig.add_trace(
                go.Scatter(
                    x=fc_df["target_timestamp"],
                    y=fc_df["p90"],
                    name="80% Confidence Interval [p10-p90]",
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(0, 113, 227, 0.09)",
                    hovertemplate="%{x|%b %d, %H:%M UTC}<br>p90 (Upper): <b>%{y:,.0f} MW</b><extra></extra>",
                )
            )

            # median forecast p50
            fig.add_trace(
                go.Scatter(
                    x=fc_df["target_timestamp"],
                    y=fc_df["p50"],
                    name="Forecast Median (p50)",
                    mode="lines",
                    line=dict(color="#0071e3", width=2.4, shape="spline", smoothing=0.3),
                    hovertemplate="%{x|%b %d, %H:%M UTC}<br>p50 (Median): <b>%{y:,.0f} MW</b><extra></extra>",
                )
            )

    # clean light layout: zero border, subtle horizontal rules
    fig.update_layout(
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif',
            color="#1d1d1f",
        ),
        xaxis=dict(
            title="",
            showgrid=False,
            showline=False,
            zeroline=False,
            tickformat="%b %d\n%H:%M",
            tickfont=dict(size=11, color="#86868b"),
        ),
        yaxis=dict(
            title=dict(text="Demand (MW)", font=dict(size=11, color="#86868b")),
            showgrid=True,
            gridcolor="rgba(0, 0, 0, 0.05)",
            gridwidth=1,
            showline=False,
            zeroline=False,
            tickformat=",.0f",
            tickfont=dict(size=11, color="#86868b"),
            side="right",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color="#6e6e73"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=20, r=60, t=40, b=40),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#ffffff",
            font_size=12,
            font_family='-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif',
            font_color="#1d1d1f",
            bordercolor="rgba(0, 0, 0, 0.08)",
        ),
        height=420,
    )

    return fig