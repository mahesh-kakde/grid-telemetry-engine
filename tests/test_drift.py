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