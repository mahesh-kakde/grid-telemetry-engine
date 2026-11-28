import os
from datetime import datetime, timedelta, timezone
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ui", "app.py"))


def test_streamlit_app_renders_and_handles_region_and_shock():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    markdown_text = "\n".join(el.value for el in at.markdown)
    assert "Simulation Note:" in markdown_text
    assert "215,500 MW" not in markdown_text
    assert "National Power Grid Telemetry" in markdown_text
    assert "Demand Forecast (All India)" in markdown_text
    assert "Regional Overview:" not in markdown_text
    assert "National Grid" not in markdown_text
    assert "Created by" in markdown_text
    assert "GitHub" in markdown_text
    assert "v2.0.0" not in markdown_text
    assert "·" not in markdown_text
    assert "notice-pill" not in markdown_text

    # switch region to NER on Forecast and set horizon to 48h to test per-region state restoration
    at.segmented_control(key="region_control").set_value("NER - Northeast").run()
    assert not at.exception
    assert not at.warning
    ner_text = "\n".join(el.value for el in at.markdown)
    assert "Baseline Note:" in ner_text
    assert "0 of 12 features shifted" in ner_text
    at.segmented_control(key="horizon_ctrl").set_value(48).run()
    assert not at.exception
    assert "Hourly Forecast Table (Next 48h)" in "\n".join(el.value for el in at.markdown)

    # switch region via top region control to Northern Region (NR) and toggle horizon
    at.segmented_control(key="region_control").set_value("NR - North").run()
    assert not at.exception
    assert not at.warning
    nr_text = "\n".join(el.value for el in at.markdown)
    assert "NR — Northern Region" in nr_text
    assert "December 2024 winter" in nr_text
    assert at.segmented_control(key="horizon_ctrl").value == 24

    at.segmented_control(key="horizon_ctrl").set_value(72).run()
    assert not at.exception
    assert len(at.expander) == 0
    assert "Hourly Forecast Table (Next 72h)" in "\n".join(el.value for el in at.markdown)

    # switch back to NER and verify 48h selection persisted without rotating widget keys
    at.segmented_control(key="region_control").set_value("NER - Northeast").run()
    assert not at.exception
    assert at.segmented_control(key="horizon_ctrl").value == 48

    # return to NR before Anomaly Check checks
    at.segmented_control(key="region_control").set_value("NR - North").run()
    assert not at.exception

    # navigate via global header to Anomaly Check and test surge + custom load
    at.segmented_control(key="nav_control").set_value("Anomaly Check").run()
    assert not at.exception
    assert "Anomaly Check (NR — Northern Region)" in "\n".join(el.value for el in at.markdown)

    at.segmented_control(key="scenario_ctrl").set_value("Surge Test").run()
    assert not at.exception
    shock_markdown = "\n".join(el.value for el in at.markdown)
    assert "Anomaly Alert" in shock_markdown
    assert "Breached" in shock_markdown
    assert "Increases anomaly risk" in shock_markdown
    assert 'role="button"' in shock_markdown
    assert "aria-label=" in shock_markdown

    at.segmented_control(key="scenario_ctrl").set_value("Custom Load").run()
    assert not at.exception
    assert not at.warning
    assert len(at.number_input) == 1
    assert at.number_input[0].max == 350_000.0
    nr_custom_default = at.number_input(key="custom_mw_ctrl").value

    # switching to NER updates custom load input to NER's regional scale using the same stable key
    at.segmented_control(key="region_control").set_value("NER - Northeast").run()
    assert not at.exception
    at.segmented_control(key="scenario_ctrl").set_value("Custom Load").run()
    assert not at.exception
    assert not at.warning
    ner_custom_default = at.number_input(key="custom_mw_ctrl").value
    assert ner_custom_default < nr_custom_default * 0.25

    at.segmented_control(key="region_control").set_value("NR - North").run()
    assert not at.exception

    # navigate to Model Health and Info for Nerds via header nav
    at.segmented_control(key="nav_control").set_value("Model Health").run()
    assert not at.exception
    assert len(at.expander) == 0
    mh_text = "\n".join(el.value for el in at.markdown)
    assert "Model Health (NR — Northern Region)" in mh_text
    assert "Feature Drift Breakdown" in mh_text
    assert "Statistical Drift Criteria" in mh_text

    at.segmented_control(key="nav_control").set_value("Info for Nerds").run()
    assert not at.exception
    assert len(at.expander) == 0
    nerds_text = "\n".join(el.value for el in at.markdown)
    assert "Info for Nerds" in nerds_text
    assert "Precomputed DB Snapshots" in nerds_text
    assert "How the Pipeline Works" in nerds_text
    assert "Core Decision & Calibration Formulas" in nerds_text
    assert "Engineering Highlights" in nerds_text
    assert "System Stack Summary" in nerds_text
    assert "https://github.com/mahesh-kakde/grid-telemetry-engine/blob/main/README.md" in nerds_text
    assert "README.md" in nerds_text


def test_ui_helpers_notices_accessibility_and_report_mobile_css():
    from src.monitoring.drift_check import build_html_drift_report
    from src.ui.components import build_notices_html
    from src.ui.helpers import evaluate_snapshot_freshness, format_info_tip, format_region_option
    from src.ui.styles import get_app_css

    assert format_region_option("ALL_INDIA") == "All India"
    assert format_region_option("NR") == "NR - North"

    from src.ui.components import build_footer_html
    footer_html = build_footer_html("2.0.0")
    assert "Created by" in footer_html
    assert "GitHub" in footer_html
    assert "https://github.com/mahesh-kakde/grid-telemetry-engine" in footer_html
    assert "Repository" not in footer_html
    assert "v2.0.0" not in footer_html
    assert "https://github.com/mahesh-kakde'" not in footer_html
    assert "·" not in footer_html

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
    assert "Live Supabase cache unavailable" in (stale_degraded["warning_message"] or "")

    # verify seasonal drift advisory notice explains winter baseline widening clearly
    advisory_html = build_notices_html(
        fresh,
        {"drift_alarm": True, "drifted_features": 5, "total_features": 12},
        widened=True,
    )
    assert "Baseline Advisory:" in advisory_html
    assert "December 2024 winter baseline" in advisory_html
    assert "90% CQR" in advisory_html
    assert "notice-pill" not in advisory_html
    assert "·" not in advisory_html

    css_src = get_app_css()
    assert ".info-tip:focus-visible::after" in css_src
    assert ".info-tip:focus::after" in css_src
    assert "tabular-nums" in css_src
    assert ".app-footer" in css_src
    assert "left: 300px;" not in css_src

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
    assert "December 2024 winter" in report_html
    assert "--bg-canvas: #f5f5f7" in report_html
    assert "Statistical Drift Criteria" in report_html
    assert "https://github.com/mahesh-kakde/grid-telemetry-engine" in report_html
    assert "26.5,11.2" in report_html
    assert "v2.0.0" not in report_html
    assert "\n            .container {{" not in report_html and ".container {" not in report_html

    from src.ui.helpers import render_not_found_html

    not_found_html = render_not_found_html("/favicon.png")
    assert "<title>404 Not Found | grid-telemetry-engine</title>" in not_found_html
    assert "National Power Grid Telemetry" in not_found_html
    assert "Error 404" not in not_found_html
    assert "This Page Went Off the Grid" in not_found_html
    assert ">Home</a>" in not_found_html
    assert "Return to Dashboard" not in not_found_html
    assert "border-radius: var(--radius-card)" in not_found_html
    assert "999px" not in not_found_html
    assert "https://github.com/mahesh-kakde/grid-telemetry-engine" in not_found_html
    assert "data:image/svg+xml" in render_not_found_html()

    from src.models.precompute import load_all_snapshots_from_supabase
    from src.server import _inject_seo_head
    from src.ui.components import build_dashboard_skeleton_html, build_operation_loading_html

    skel_html = build_dashboard_skeleton_html("NR — Northern <script>Region")
    assert "role='status'" in skel_html
    assert "aria-live='polite'" in skel_html
    assert "<script>" not in skel_html

    op_html = build_operation_loading_html("Evaluating <Load>", "Running Isolation Forest")
    assert "role='status'" in op_html
    assert "Evaluating &lt;Load&gt;" in op_html

    assert "[data-testid=\"stSpinner\"]" in css_src
    assert "prefers-reduced-motion" in css_src
    assert "grid-template-columns: repeat(2, minmax(0, 1fr)) !important;" in css_src
    assert "grid-template-columns: repeat(3, minmax(0, 1fr)) !important;" in css_src
    assert "border-width: 1px !important;" in css_src
    assert 'button[data-variant="segmented_control"]' in css_src
    assert ".footer-author" in css_src
    assert ".st-key-global_header_shell" in css_src
    assert ".st-key-nav_control" in css_src
    assert ".block-container:has(.st-key-nav_control)" in css_src
    assert ".st-key-horizon_ctrl:not([data-retained=\"true\"])" in css_src
    assert '[data-retained="true"]' in css_src
    assert "st-key-horizon_ctrl_" not in css_src
    assert "st-key-scenario_ctrl_" not in css_src
    assert "pointer-events: none !important;" in css_src
    assert "flex-direction: column !important;" in css_src
    assert "stHeaderActionElements" in css_src
    assert ":has(.chart-mobile-hint)" in css_src
    assert "div[data-testid=\"stElementContainer\"]:has(h3)" in css_src
    assert "font-size: 17.5px !important;" in css_src
    assert ".details-card" in css_src
    assert ".formula-box" in css_src
    assert ".readme-cta" in css_src
    assert 'div.stPlotlyChart[data-testid="stPlotlyChart"]' in css_src

    injected = _inject_seo_head(b"<html><head><title>Streamlit</title></head><body><div id=\"root\"></div></body></html>").decode("utf-8")
    assert "telemetry-boot-shell" in injected
    assert 'id="telemetry-boot-style"' in injected
    from unittest.mock import patch
    from src.models.precompute import load_precomputed_data

    assert load_all_snapshots_from_supabase(db_url="") == {}
    with patch("src.models.precompute.load_snapshot_from_supabase") as spy_db:
        local_snap = load_precomputed_data("ALL_INDIA", prefer_supabase=False)
        assert local_snap is not None and local_snap["snapshot_source"] == "local_snapshot"
        spy_db.assert_not_called()


def test_streamlit_404_page_title_and_layout():
    from unittest.mock import patch
    import streamlit as st

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.query_params["page"] = "404"
    with patch("streamlit.set_page_config", wraps=st.set_page_config) as spy_page_config:
        at.run()

    assert not at.exception
    assert spy_page_config.call_args.kwargs["page_title"] == "404 Not Found | grid-telemetry-engine"
    rendered = "\n".join(el.value for el in at.markdown)
    assert "Error 404" not in rendered
    assert "This Page Went Off the Grid" in rendered
    assert ">Home</a>" in rendered
    assert "Return to Dashboard" not in rendered