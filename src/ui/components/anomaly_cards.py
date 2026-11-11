from typing import Any
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_shap_bar_chart(shap_features: dict[str, float]) -> go.Figure:
    if not shap_features:
        return go.Figure()

    sorted_features = sorted(shap_features.items(), key=lambda x: abs(x[1]))
    names = [k for k, _ in sorted_features]
    values = [v for _, v in sorted_features]
    colors = ["#dc2626" if v > 0 else "#2563eb" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="Feature: %{y}<br>SHAP Contribution: <b>%{x:+.4f}</b><extra></extra>",
        )
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=140,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#e2e8f0",
            tickfont=dict(size=10, color="#64748b"),
            zerolinecolor="#cbd5e1",
        ),
        yaxis=dict(
            tickfont=dict(size=10, color="#334155"),
            showgrid=False,
        ),
    )
    return fig


def render_anomaly_section(anomalies: list[dict[str, Any]]) -> None:
    flagged = [a for a in anomalies if a.get("is_anomaly")]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:1.2rem; box-shadow:0 1px 3px rgba(0,0,0,0.02);">
                <div style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em; color:#64748b; font-weight:600;">Flagged Stress Events</div>
                <div style="font-size:1.8rem; font-weight:700; color:#dc2626; margin-top:0.3rem;">{len(flagged)}</div>
                <div style="font-size:0.85rem; color:#64748b; margin-top:0.2rem;">Evaluated via Isolation Forest (2% contamination)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        total_eval = len(anomalies)
        rate = (len(flagged) / total_eval * 100) if total_eval > 0 else 0.0
        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:1.2rem; box-shadow:0 1px 3px rgba(0,0,0,0.02);">
                <div style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em; color:#64748b; font-weight:600;">Total Evaluated Timestamps</div>
                <div style="font-size:1.8rem; font-weight:700; color:#1e293b; margin-top:0.3rem;">{total_eval}</div>
                <div style="font-size:0.85rem; color:#64748b; margin-top:0.2rem;">Observed Contamination Rate: {rate:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    if not flagged:
        st.markdown(
            """
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:1rem; text-align:center; color:#64748b; font-size:0.95rem;">
                No structural anomalies detected in recent telemetry. Grid operation within nominal variance.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown("##### Recent Structural Outliers & SHAP Attribution")
    for anom in flagged[:6]:
        ts = anom.get("timestamp", "N/A")
        val = float(anom.get("metric_value", 0.0))
        score = float(anom.get("anomaly_score", 0.0))
        region = anom.get("region", "N/A")
        shap_dict = anom.get("top_shap_features") or {}

        with st.container():
            st.markdown(
                f"""
                <div style="background:#ffffff; border:1px solid #e2e8f0; border-left:4px solid #dc2626; border-radius:8px; padding:1rem; margin-bottom:0.75rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-weight:600; color:#1e293b; font-size:0.95rem;">{ts}</span>
                            <span style="color:#64748b; font-size:0.85rem; margin-left:0.5rem;">Region: {region}</span>
                        </div>
                        <div>
                            <span style="background:#fee2e2; color:#991b1b; padding:0.2rem 0.6rem; border-radius:9999px; font-size:0.8rem; font-weight:600;">
                                Load: {val:,.0f} MW
                            </span>
                            <span style="background:#f1f5f9; color:#475569; padding:0.2rem 0.6rem; border-radius:9999px; font-size:0.8rem; font-weight:600; margin-left:0.3rem;">
                                Score: {score:.3f}
                            </span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if shap_dict:
                fig = render_shap_bar_chart(shap_dict)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})