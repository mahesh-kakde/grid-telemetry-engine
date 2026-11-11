from pathlib import Path
from typing import Any
import streamlit as st


def render_drift_section(drift_data: dict[str, Any] | None) -> None:
    if not drift_data:
        st.markdown(
            """
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:1.2rem; text-align:center; color:#64748b;">
                No evidently drift monitoring runs recorded yet. Execute drift evaluation via CLI or GitHub action.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    share = float(drift_data.get("drift_share", 0.0))
    drifted_features = drift_data.get("drifted_features") or {}
    report_url = drift_data.get("report_url")
    run_ts = drift_data.get("run_timestamp", "N/A")

    drift_flagged = share >= 0.5
    badge_bg = "#fef2f2" if drift_flagged else "#f0fdf4"
    badge_color = "#991b1b" if drift_flagged else "#166534"
    badge_border = "#fca5a5" if drift_flagged else "#86efac"
    badge_text = "Dataset Drift Detected" if drift_flagged else "Distribution Stable (No Drift)"

    drifted_cols = [k for k, v in drifted_features.items() if v.get("drift_detected")]
    total_cols = len(drifted_features)

    col1, col2, col3 = st.columns([1.5, 1, 1])

    with col1:
        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:1.2rem; box-shadow:0 1px 3px rgba(0,0,0,0.02);">
                <div style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em; color:#64748b; font-weight:600;">Status</div>
                <div style="margin-top:0.4rem;">
                    <span style="background:{badge_bg}; color:{badge_color}; border:1px solid {badge_border}; padding:0.35rem 0.75rem; border-radius:9999px; font-size:0.88rem; font-weight:600;">
                        {badge_text}
                    </span>
                </div>
                <div style="font-size:0.8rem; color:#64748b; margin-top:0.6rem;">Last evaluated: {run_ts}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:1.2rem; box-shadow:0 1px 3px rgba(0,0,0,0.02);">
                <div style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em; color:#64748b; font-weight:600;">Drift Share</div>
                <div style="font-size:1.8rem; font-weight:700; color:{badge_color}; margin-top:0.3rem;">{share:.1%}</div>
                <div style="font-size:0.8rem; color:#64748b; margin-top:0.2rem;">Threshold: 50.0%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:1.2rem; box-shadow:0 1px 3px rgba(0,0,0,0.02);">
                <div style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em; color:#64748b; font-weight:600;">Feature Drift</div>
                <div style="font-size:1.8rem; font-weight:700; color:#1e293b; margin-top:0.3rem;">{len(drifted_cols)} / {total_cols}</div>
                <div style="font-size:0.8rem; color:#64748b; margin-top:0.2rem;">Drifted vs. Evaluated</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    # feature drift details table
    if drifted_features:
        with st.expander("Detailed Feature Drift Test Results (KS Test & Wasserstein)"):
            rows = []
            for feat_name, feat_meta in drifted_features.items():
                rows.append({
                    "Feature": feat_name,
                    "Drift Detected": "Yes" if feat_meta.get("drift_detected") else "No",
                    "Score / p-value": f"{float(feat_meta.get('drift_score', 0.0)):.4e}",
                    "Test Name": feat_meta.get("stattest_name", "N/A"),
                    "Threshold": feat_meta.get("stattest_threshold", 0.05),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

    # link or embed html report if available
    if report_url:
        report_path = Path(report_url)
        if report_path.exists():
            st.markdown(
                f"""
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:0.85rem 1rem; margin-top:0.5rem; display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.88rem; color:#334155;">Evidently interactive HTML report generated at <code>{report_path.name}</code></span>
                    <a href="/api/drift/report" target="_blank" style="text-decoration:none; background:#0071e3; color:#ffffff; padding:0.4rem 0.85rem; border-radius:6px; font-size:0.85rem; font-weight:500;">
                        Open Interactive Report
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )