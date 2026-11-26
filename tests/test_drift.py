import numpy as np
from src.models.load_artifacts import load_reference_baseline
from src.monitoring.drift_check import (
    evaluate_feature_drift,
    build_html_drift_report,
)


def test_nominal_baseline_tail_does_not_false_alarm():
    ref_df = load_reference_baseline("ALL_INDIA")
    curr_df = ref_df.tail(168).copy()
    summary = evaluate_feature_drift(ref_df, curr_df, feature_cols=None)

    assert summary["total_features"] == 12
    assert summary["drift_alarm"] is False
    assert summary["drift_ratio"] <= 0.30
    assert summary["recommended_quantile_alpha"] == [0.10, 0.50, 0.90]


def test_severe_regime_shift_triggers_drift_alarm_and_widening():
    ref_df = load_reference_baseline("ALL_INDIA")
    shifted = ref_df.tail(168).copy()
    shifted["demand_mw"] = shifted["demand_mw"] * 1.25
    shifted["demand_rolling_std_24h"] = shifted["demand_rolling_std_24h"] * 1.60
    shifted["temperature_2m"] = shifted["temperature_2m"] + 9.0
    shifted["relative_humidity_2m"] = np.clip(shifted["relative_humidity_2m"] + 25.0, 5.0, 100.0)
    shifted["temp_delta_24h"] = shifted["temp_delta_24h"] + 4.5

    summary = evaluate_feature_drift(ref_df, shifted, feature_cols=None)
    assert summary["drift_alarm"] is True
    assert summary["retraining_advisory"] is True
    assert summary["recommended_quantile_alpha"] == [0.05, 0.50, 0.95]


def test_html_drift_report_escapes_untrusted_strings():
    malicious_summary = {
        "total_features": 1,
        "drifted_features": 1,
        "drift_ratio": 1.0,
        "drift_alarm": True,
        "recommended_quantile_alpha": [0.05, 0.50, 0.95],
        "feature_metrics": {
            "<script>alert(1)</script>": {
                "test": "wasserstein_ks",
                "statistic": 1.25,
                "p_value": 0.0001,
                "threshold": 0.50,
                "drift_detected": True,
            }
        },
    }
    html_out = build_html_drift_report(
        malicious_summary,
        region='<img src=x onerror="alert(2)">',
        generated_at="<b>now</b>",
    )
    assert "<script>alert(1)</script>" not in html_out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out
    assert "&lt;img src=x onerror=&quot;alert(2)&quot;&gt;" in html_out


def test_partial_post_warmup_window_falls_back_cleanly(monkeypatch):
    import polars as pl
    from datetime import datetime, timezone, timedelta
    from src.monitoring.drift_check import (
        select_drift_observation_window,
        fetch_trailing_telemetry_window,
        build_html_drift_report,
    )

    ref_df = load_reference_baseline("ALL_INDIA")
    partial_df = ref_df.tail(3).copy()
    selected, source = select_drift_observation_window(partial_df, ref_df, window_hours=168)
    assert source == "baseline_fallback"
    assert len(selected) == 168

    # simulate db returning 171 raw hours (only 3 valid post-warmup rows)
    t0 = datetime(2026, 11, 19, 6, 0, tzinfo=timezone.utc)
    rows = [
        {
            "timestamp_utc": t0 + timedelta(hours=i),
            "region": "ALL_INDIA",
            "demand_mw": 145000.0 + (i % 24) * 500.0,
            "frequency_hz": 50.0,
            "temperature_2m": 25.0,
            "relative_humidity_2m": 60.0,
            "surface_solar_radiation": 200.0 if 6 <= ((t0 + timedelta(hours=i)).hour) <= 18 else 0.0,
            "wind_speed_100m": 4.0,
        }
        for i in range(171)
    ]
    partial_pl = pl.DataFrame(rows)
    monkeypatch.setattr(
        "src.models.infer_and_detect.fetch_telemetry_window_from_db",
        lambda **kwargs: partial_pl,
    )
    window_df = fetch_trailing_telemetry_window(region="ALL_INDIA", window_hours=168)
    assert len(window_df) == 168
    summary = evaluate_feature_drift(ref_df, window_df)
    assert summary["feature_metrics"]["sin_hour_k1"]["score"] < 0.15
    assert summary["drift_alarm"] is False

    html_custom = build_html_drift_report({**summary, "observation_hours": 72}, region="ALL_INDIA")
    assert "Trailing 72h Window" in html_custom