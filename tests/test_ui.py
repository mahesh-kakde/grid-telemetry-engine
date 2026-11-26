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