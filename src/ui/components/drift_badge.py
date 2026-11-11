from pathlib import Path
from typing import Any
import streamlit as st


def render_drift_section(drift_data: dict[str, Any] | None) -> None:
    """Renders the continuous distribution drift monitoring section."""
    if not drift_data:
        st.markdown(
            """
            <div class="card" style="padding: 2.5rem 1.5rem; text-align: center;">
                <div style="font-size:1rem; font-weight:600; color:#1d1d1f;">No Drift Evaluations Recorded</div>
                <div style="font-size:0.88rem; color:#86868b; margin-top:0.35rem; max-width:480px; margin-left:auto; margin-right:auto;">
                    Execute the drift monitoring pipeline via CLI (<code>python -m src.monitoring.drift_runner</code>) or scheduled workflow to populate statistical tests.
                </div>
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
    drifted_cols = [k for k, v in drifted_features.items() if v.get("drift_detected")]
    total_cols = len(drifted_features)

    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.markdown(
            f"""
            <div class="card" style="padding: 1.5rem 1.75rem;">
                <div class="metric-eyebrow">DISTRIBUTION STABILITY</div>
                <div style="margin-top:0.6rem;">
                    <span class="{'pill-danger' if drift_flagged else 'pill-success'}" style="font-size:0.9rem; padding: 5px 14px;">
                        {'● Dataset Drift Detected' if drift_flagged else '● Distribution Stable'}
                    </span>
                </div>
                <div style="font-size:0.82rem; color:#86868b; margin-top:0.75rem;">
                    Evaluated: {run_ts}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="card" style="padding: 1.5rem 1.75rem;">
                <div class="metric-eyebrow">DRIFT SHARE RATIO</div>
                <div style="font-size:2.4rem; font-weight:700; letter-spacing:-0.03em; color:{'#ff3b30' if drift_flagged else '#1d1d1f'}; line-height:1; margin-top:0.4rem;">
                    {share:.1%}
                </div>
                <div style="font-size:0.82rem; color:#86868b; margin-top:0.6rem;">
                    Acceptance threshold: 50.0%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="card" style="padding: 1.5rem 1.75rem;">
                <div class="metric-eyebrow">DRIFTED FEATURES</div>
                <div style="display:flex; align-items:baseline; gap:0.4rem; margin-top:0.4rem;">
                    <div style="font-size:2.4rem; font-weight:700; letter-spacing:-0.03em; color:#1d1d1f; line-height:1;">
                        {len(drifted_cols)}
                    </div>
                    <div style="font-size:1.1rem; color:#86868b; font-weight:500;">/ {total_cols}</div>
                </div>
                <div style="font-size:0.82rem; color:#86868b; margin-top:0.6rem;">
                    KS & Wasserstein hypotheses
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.85rem;'></div>", unsafe_allow_html=True)

    # feature drift details table inside an expander
    if drifted_features:
        with st.expander("Inspect Statistical Drift Breakdown (KS-Test & Wasserstein Distance)"):
            rows = []
            for feat_name, feat_meta in drifted_features.items():
                is_drifted = bool(feat_meta.get("drift_detected"))
                rows.append({
                    "Feature Identifier": feat_name,
                    "Status": "Drifted" if is_drifted else "Nominal",
                    "Score / p-value": f"{float(feat_meta.get('drift_score', 0.0)):.4e}",
                    "Test Mechanism": feat_meta.get("stattest_name", "N/A"),
                    "Significance Threshold": feat_meta.get("stattest_threshold", 0.05),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

    # interactive report button
    if report_url:
        report_path = Path(report_url)
        st.markdown(
            f"""
            <div class="card" style="padding: 1.25rem 1.5rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem;">
                <div>
                    <div style="font-weight: 600; font-size: 0.95rem; color: #1d1d1f;">Evidently AI Interactive Report</div>
                    <div style="font-size: 0.82rem; color: #86868b; margin-top: 0.2rem;">Generated interactive HTML report: <code>{report_path.name}</code></div>
                </div>
                <a href="/api/drift/report" target="_blank" class="button-primary">
                    Open Interactive Suite ↗
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )