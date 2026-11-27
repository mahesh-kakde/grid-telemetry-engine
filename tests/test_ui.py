import os
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ui", "app.py"))


def test_streamlit_app_renders_and_handles_region_and_shock():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    markdown_text = "\n".join(el.value for el in at.markdown)
    # verify calibrated simulation disclosure and no stale hardcoded 215,500 MW
    assert "Simulation Disclosure:" in markdown_text
    assert "215,500 MW" not in markdown_text

    # switch sidebar selectbox to Northern Region (NR)
    at.selectbox[0].set_value("NR").run()
    assert not at.exception
    nr_text = "\n".join(el.value for el in at.markdown)
    assert "NR — Northern Region" in nr_text

    # trigger severe grid shock radio option and verify CRITICAL GRID FAULT
    shock_radio = next(r for r in at.radio if "Normal Grid Operations (Standard Baseline)" in r.options)
    shock_radio.set_value("Simulate Severe Grid Shock (Extreme Load Surge)").run()
    assert not at.exception
    shock_markdown = "\n".join(el.value for el in at.markdown)
    assert "CRITICAL GRID FAULT" in shock_markdown
    assert "Envelope Breached" in shock_markdown
    assert "Increases fault surprise by" in shock_markdown
    assert 'role="button"' in shock_markdown
    assert 'aria-label=' in shock_markdown


def test_ui_helpers_freshness_accessibility_and_report_mobile_css():
    from datetime import datetime, timedelta, timezone
    from src.ui.helpers import evaluate_snapshot_freshness, format_info_tip
    from src.monitoring.drift_check import build_html_drift_report

    tip_html = format_info_tip('Check <script> "quote"', tip_right=True)
    assert 'role="button"' in tip_html
    assert 'tabindex="0"' in tip_html
    assert "tip-right" in tip_html
    assert "<script>" not in tip_html

    now_utc = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    fresh = evaluate_snapshot_freshness(
        {"snapshot_source": "supabase_live", "generated_at": (now_utc - timedelta(hours=3)).isoformat()},
        stale_after_hours=12.0,
        db_configured=True,
        now_utc=now_utc,
    )
    assert fresh["is_stale"] is False
    assert fresh["is_degraded_fallback"] is False
    assert fresh["warning_message"] is None

    stale_degraded = evaluate_snapshot_freshness(
        {"snapshot_source": "local_snapshot", "generated_at": (now_utc - timedelta(hours=18)).isoformat()},
        stale_after_hours=12.0,
        db_configured=True,
        now_utc=now_utc,
    )
    assert stale_degraded["is_stale"] is True
    assert stale_degraded["is_degraded_fallback"] is True
    assert "Degraded Snapshot Fallback" in (stale_degraded["warning_message"] or "")

    with open(APP_PATH, "r", encoding="utf-8") as f:
        app_src = f.read()
    assert ".info-tip:focus-visible::after" in app_src
    assert ".info-tip:focus::after" in app_src
    assert "y=-0.24" in app_src
    assert "rgba(248, 250, 252, 0.92)" in app_src
    assert ".app-footer" in app_src
    assert "left: 300px;" not in app_src
    assert "color: #475569;" in app_src
    assert '<a href="/" class="btn-home"' in app_src

    report_html = build_html_drift_report(
        {
            "observation_hours": 168,
            "window_source": "supabase_live",
            "drift_alarm": False,
            "drift_ratio": 0.0,
            "drifted_features": 0,
            "total_features": 1,
            "recommended_quantile_alpha": [0.1, 0.9],
            "feature_metrics": {},
        },
        region="NR",
    )
    assert ".report-container" in report_html
    assert ".metric-grid, .meta-bar" in report_html
    assert "\n            .container {{" not in report_html and ".container {" not in report_html