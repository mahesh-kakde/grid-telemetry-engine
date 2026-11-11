from typing import Any
import plotly.graph_objects as go
import streamlit as st


def render_shap_bar_chart(shap_features: dict[str, float]) -> go.Figure:
    """Renders a horizontal micro-bar chart of SHAP feature contributions."""
    if not shap_features:
        return go.Figure()

    sorted_features = sorted(shap_features.items(), key=lambda x: abs(x[1]))
    names = [k for k, _ in sorted_features]
    values = [v for _, v in sorted_features]
    colors = ["#ff3b30" if v > 0 else "#0071e3" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=0),
                cornerradius=4,
            ),
            hovertemplate="Feature: <b>%{y}</b><br>SHAP Impact: <b>%{x:+.4f}</b><extra></extra>",
        )
    )

    fig.update_layout(
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif',
            color="#1d1d1f",
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=130,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0, 0, 0, 0.04)",
            linecolor="rgba(0, 0, 0, 0.06)",
            tickfont=dict(size=10, color="#86868b"),
            zeroline=True,
            zerolinecolor="rgba(0, 0, 0, 0.15)",
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#1d1d1f"),
            showgrid=False,
            showline=False,
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            font_size=11,
            bordercolor="rgba(0, 0, 0, 0.08)",
        ),
    )
    return fig


def render_anomaly_section(anomalies: list[dict[str, Any]]) -> None:
    """Renders the structural anomaly detection section with bento grid cards."""
    flagged = [a for a in anomalies if a.get("is_anomaly")]
    total_eval = len(anomalies)
    rate = (len(flagged) / total_eval * 100) if total_eval > 0 else 0.0

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="card" style="padding: 1.5rem 1.75rem;">
                <div class="metric-eyebrow">ISOLATION FOREST CONTAMINATION</div>
                <div style="display:flex; align-items:baseline; gap:0.5rem; margin-top:0.4rem;">
                    <div style="font-size:2.4rem; font-weight:700; letter-spacing:-0.03em; color:{'#ff3b30' if flagged else '#1d1d1f'}; line-height:1;">
                        {len(flagged)}
                    </div>
                    <div style="font-size:0.95rem; color:#86868b; font-weight:500;">flagged events</div>
                </div>
                <div style="font-size:0.85rem; color:#6e6e73; margin-top:0.5rem;">
                    Calibrated contamination target: 2.0% threshold
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="card" style="padding: 1.5rem 1.75rem;">
                <div class="metric-eyebrow">OBSERVED ANOMALY RATE</div>
                <div style="display:flex; align-items:baseline; gap:0.5rem; margin-top:0.4rem;">
                    <div style="font-size:2.4rem; font-weight:700; letter-spacing:-0.03em; color:#1d1d1f; line-height:1;">
                        {rate:.2f}%
                    </div>
                    <div style="font-size:0.95rem; color:#86868b; font-weight:500;">over {total_eval} points</div>
                </div>
                <div style="font-size:0.85rem; color:#6e6e73; margin-top:0.5rem;">
                    Rolling window temporal evaluation
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    if not flagged:
        st.markdown(
            """
            <div class="card" style="padding: 2.5rem 1.5rem; text-align: center;">
                <div style="font-size:1rem; font-weight:600; color:#1d1d1f;">Nominal Grid State</div>
                <div style="font-size:0.88rem; color:#86868b; margin-top:0.35rem; max-width:480px; margin-left:auto; margin-right:auto;">
                    No structural anomalies detected in recent telemetry. Load fluctuations remain strictly within nominal 2% variance boundaries.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <div style="font-size:1rem; font-weight:600; color:#1d1d1f; margin-bottom:0.75rem; letter-spacing:-0.01em;">
            Recent Outliers & Local SHAP Attribution
        </div>
        """,
        unsafe_allow_html=True,
    )

    for anom in flagged[:6]:
        ts = anom.get("timestamp", "N/A")
        val = float(anom.get("metric_value", 0.0))
        score = float(anom.get("anomaly_score", 0.0))
        region = anom.get("region", "N/A")
        shap_dict = anom.get("top_shap_features") or {}

        with st.container():
            st.markdown(
                f"""
                <div class="card" style="padding: 1.25rem 1.5rem; margin-bottom: 0.85rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.75rem;">
                        <div style="display:flex; align-items:center; gap:0.6rem;">
                            <span class="pill-danger">Outlier Event</span>
                            <span style="font-weight:600; color:#1d1d1f; font-size:0.95rem;">{ts}</span>
                            <span class="pill-neutral">Region {region}</span>
                        </div>
                        <div style="display:flex; align-items:center; gap:0.5rem;">
                            <span style="font-size:0.85rem; color:#6e6e73;">Load:</span>
                            <span style="font-weight:600; color:#1d1d1f; font-size:0.95rem;">{val:,.0f} MW</span>
                            <span style="font-size:0.82rem; color:#86868b; margin-left:0.4rem;">Score: {score:.3f}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if shap_dict:
                fig = render_shap_bar_chart(shap_dict)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})