from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
from src.ui.components.anomaly_cards import render_shap_bar_chart
from src.ui.components.forecast_chart import render_forecast_figure


def test_render_forecast_figure_structure() -> None:
    now = datetime.now(timezone.utc)
    history = [
        {"timestamp": (now - timedelta(hours=i)).isoformat(), "metric_value": 45000.0 + i * 50}
        for i in range(24)
    ]
    forecasts = [
        {
            "target_timestamp": (now + timedelta(hours=i)).isoformat(),
            "p10": 42000.0,
            "p50": 46000.0,
            "p90": 50000.0,
        }
        for i in range(1, 10)
    ]

    fig = render_forecast_figure(history=history, forecasts=forecasts, region="NR")
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data if t.name]
    assert "Actual Demand" in trace_names
    assert "Forecast Median (p50)" in trace_names
    assert "80% Confidence Interval [p10-p90]" in trace_names
    assert fig.layout.paper_bgcolor == "#ffffff"


def test_render_forecast_figure_empty() -> None:
    fig = render_forecast_figure(history=[], forecasts=[], region="ALL")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_render_shap_bar_chart() -> None:
    shap_dict = {
        "metric_value_roll_mean_24h": 0.52,
        "hour": -0.21,
        "metric_value_lag_1h": 0.15,
    }
    fig = render_shap_bar_chart(shap_dict)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"
    assert fig.data[0].orientation == "h"
    assert len(fig.data[0].x) == 3


def test_render_shap_bar_chart_empty() -> None:
    fig = render_shap_bar_chart({})
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0