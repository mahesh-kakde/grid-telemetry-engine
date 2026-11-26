from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.models.load_artifacts import (
    ModelArtifactsBundle,
    REGIONS,
    QUANTILES,
)
from src.models.infer_and_detect import (
    enforce_quantile_monotonicity,
    predict_quantiles,
    detect_structural_anomalies,
    generate_72h_multi_step_forecast,
    MODEL_VERSION,
)
from src.models.explainers import AnomalySHAPExplainer


@pytest.fixture(scope="module")
def artifacts_bundle():
    bundle = ModelArtifactsBundle()
    bundle.preload_all()
    return bundle


def test_quantile_monotonicity_clean_and_perturbed():
    # clean inputs where p10 <= p50 <= p90
    p10_clean = np.array([100.0, 200.0, 300.0])
    p50_clean = np.array([150.0, 250.0, 350.0])
    p90_clean = np.array([200.0, 300.0, 400.0])

    res_10, res_50, res_90 = enforce_quantile_monotonicity(p10_clean, p50_clean, p90_clean)
    np.testing.assert_array_equal(res_10, p10_clean)
    np.testing.assert_array_equal(res_50, p50_clean)
    np.testing.assert_array_equal(res_90, p90_clean)

    # perturbed inverted inputs with quantile crossings
    p10_inverted = np.array([220.0, 100.0, 450.0])
    p50_inverted = np.array([150.0, 250.0, 300.0])
    p90_inverted = np.array([120.0, 200.0, 250.0])

    res_10, res_50, res_90 = enforce_quantile_monotonicity(
        p10_inverted, p50_inverted, p90_inverted
    )

    # assert p10 <= p50 <= p90 holds for all elements
    assert np.all(res_10 <= res_50)
    assert np.all(res_50 <= res_90)
    # verify row 0: sorted([220, 150, 120]) -> [120, 150, 220]
    assert res_10[0] == 120.0
    assert res_50[0] == 150.0
    assert res_90[0] == 220.0


def test_model_artifacts_bundle_completeness(artifacts_bundle):
    # check that all regional models and baselines are loaded in memory
    for region in REGIONS:
        assert region in artifacts_bundle.lgb_models
        for q in QUANTILES:
            assert q in artifacts_bundle.lgb_models[region]
            assert artifacts_bundle.lgb_models[region][q] is not None

        assert region in artifacts_bundle.isolation_forests
        assert artifacts_bundle.isolation_forests[region] is not None

        assert region in artifacts_bundle.reference_baselines
        df_base = artifacts_bundle.reference_baselines[region]
        assert isinstance(df_base, pd.DataFrame)
        assert len(df_base) > 0

        assert region in artifacts_bundle.quantile_calibration
    assert len(artifacts_bundle.feature_columns) == 26
    assert len(artifacts_bundle.anomaly_feature_columns) == 9

    assert len(artifacts_bundle.feature_columns) > 0


def test_predict_quantiles_monotonicity_on_real_data(artifacts_bundle):
    # assert non-crossing quantiles across regional baseline records
    for region in ["ALL_INDIA", "NR", "WR", "SR"]:
        df_base = artifacts_bundle.reference_baselines[region].head(25)
        p10, p50, p90 = predict_quantiles(df_base, region, artifacts_bundle)

        assert len(p10) == len(df_base)
        assert len(p50) == len(df_base)
        assert len(p90) == len(df_base)

        # strict monotonicity assertion
        assert np.all(p10 <= p50)
        assert np.all(p50 <= p90)

        # assert physical bounds for grid demand in megawatts
        assert np.all(p10 > 0.0)
        assert np.all(p90 < 350000.0)


def test_detect_structural_anomalies_nominal(artifacts_bundle):
    # on nominal baseline where actual load equals p50 median, no anomaly should trip
    df_base = artifacts_bundle.reference_baselines["NR"].head(20)
    p10, p50, p90 = predict_quantiles(df_base, "NR", artifacts_bundle)

    scores, is_structural, trigger_b = detect_structural_anomalies(
        X=df_base,
        y_actual=p50,
        p10=p10,
        p50=p50,
        p90=p90,
        region="NR",
        bundle=artifacts_bundle,
    )

    assert len(scores) == len(df_base)
    # nominal median load should never breach predictive residual envelope
    assert np.all(~trigger_b)
    assert np.all(~is_structural)


def test_detect_structural_anomalies_dual_trigger_logic(artifacts_bundle):
    # tests dual-trigger requirement: is_structural = trigger_a and trigger_b
    df_base = artifacts_bundle.reference_baselines["NR"].head(5)
    p10, p50, p90 = predict_quantiles(df_base, "NR", artifacts_bundle)

    # case 1: extreme physical demand breach (+200,000 mw above envelope)
    extreme_spike = p90 + (1.5 * (p90 - p10)) + 200000.0
    scores, is_structural, trigger_b = detect_structural_anomalies(
        X=df_base,
        y_actual=extreme_spike,
        p10=p10,
        p50=p50,
        p90=p90,
        region="NR",
        bundle=artifacts_bundle,
    )

    # trigger b must be true for all rows with massive residual breach
    assert np.all(trigger_b)

    # case 2: verify trigger b is false when within envelope even if artificial trigger a is simulated
    safe_actual = (p10 + p90) / 2.0
    _, is_structural_safe, trigger_b_safe = detect_structural_anomalies(
        X=df_base,
        y_actual=safe_actual,
        p10=p10,
        p50=p50,
        p90=p90,
        region="NR",
        bundle=artifacts_bundle,
    )
    assert np.all(~trigger_b_safe)
    assert np.all(~is_structural_safe)


def test_generate_72h_multi_step_forecast(artifacts_bundle):
    # build synthetic 180-hour telemetry sequence for multi-step rollout
    base_ts = datetime(2023, 6, 1, 0, 0, tzinfo=timezone.utc)
    n_hours = 180
    records = []
    for i in range(n_hours):
        ts = base_ts + timedelta(hours=i)
        records.append({
            "timestamp_utc": ts,
            "region": "NR",
            "demand_mw": 45000.0 + 5000.0 * np.sin(2 * np.pi * i / 24.0),
            "frequency_hz": 50.0,
            "temperature_2m": 32.0,
            "relative_humidity_2m": 55.0,
            "surface_solar_radiation": 400.0 if 6 <= ts.hour <= 18 else 0.0,
            "wind_speed_100m": 4.5,
        })
    df_history = pl.DataFrame(records)

    forecasts = generate_72h_multi_step_forecast(df_history, "NR", artifacts_bundle)

    # assert exactly 72 steps produced
    assert len(forecasts) == 72
    assert forecasts[0]["model_version"] == MODEL_VERSION

    # check chronological continuity and monotonicity across all 72 hours
    last_ts = df_history["timestamp_utc"].max()
    for idx, fc in enumerate(forecasts):
        expected_ts = last_ts + timedelta(hours=idx + 1)
        assert fc["target_timestamp_utc"] == expected_ts
        assert fc["region"] == "NR"
        # assert quantile monotonicity: p10 <= p50 <= p90
        assert fc["forecast_p10"] <= fc["forecast_p50"]
        assert fc["forecast_p50"] <= fc["forecast_p90"]


def test_shap_tree_explainer_attribution(artifacts_bundle):
    # verify shap explainer generates top-k physical feature drivers
    explainer = AnomalySHAPExplainer(artifacts_bundle)
    row = artifacts_bundle.reference_baselines["NR"].iloc[0]

    top_attrs = explainer.explain_anomaly_point(row, "NR", top_k=3)
    assert len(top_attrs) == 3

    # check dictionary schema and magnitude order
    for item in top_attrs:
        assert "feature" in item
        assert "shap_value" in item
        assert "actual_value" in item
        assert isinstance(item["feature"], str)
        assert isinstance(item["shap_value"], (float, int))
        assert isinstance(item["actual_value"], (float, int))

    # verify top-k attributions are ordered by absolute contribution magnitude
    abs_magnitudes = [abs(x["shap_value"]) for x in top_attrs]
    assert abs_magnitudes[0] >= abs_magnitudes[1]
    assert abs_magnitudes[1] >= abs_magnitudes[2]

    # test batch explanation
    batch_rows = artifacts_bundle.reference_baselines["NR"].head(3)
    batch_attrs = explainer.explain_anomalies_batch(batch_rows, "NR", top_k=3)
    assert len(batch_attrs) == 3
    assert len(batch_attrs[0]) == 3


def test_extreme_shock_fires_both_dual_triggers(artifacts_bundle):
    from src.models.infer_and_detect import predict_quantiles, detect_structural_anomalies

    ref_df = artifacts_bundle.reference_baselines["ALL_INDIA"].tail(5).copy()
    X = ref_df[artifacts_bundle.feature_columns]
    p10, p50, p90 = predict_quantiles(X, "ALL_INDIA", artifacts_bundle)

    extreme_shock = p90 + 150000.0
    scores, is_structural, trigger_b = detect_structural_anomalies(
        X=X,
        y_actual=extreme_shock,
        p10=p10,
        p50=p50,
        p90=p90,
        region="ALL_INDIA",
        bundle=artifacts_bundle,
    )
    assert np.all(trigger_b), "Extreme load shock must breach 1.5xIQR envelope (Trigger B)"
    assert np.all(is_structural), "Observation-aware Isolation Forest + IQR must flag CRITICAL GRID FAULT"


def test_cqr_calibration_widening_increases_band(artifacts_bundle):
    from src.models.infer_and_detect import predict_quantiles

    ref_df = artifacts_bundle.reference_baselines["ALL_INDIA"].tail(5).copy()
    X = ref_df[artifacts_bundle.feature_columns]
    p10_std, p50_std, p90_std = predict_quantiles(X, "ALL_INDIA", artifacts_bundle, widened=False)
    p10_wide, p50_wide, p90_wide = predict_quantiles(X, "ALL_INDIA", artifacts_bundle, widened=True)

    assert np.all(p10_std <= p50_std) and np.all(p50_std <= p90_std)
    assert np.all(p10_wide <= p50_wide) and np.all(p50_wide <= p90_wide)
    assert np.all((p90_wide - p10_wide) > (p90_std - p10_std))


def test_anomaly_shap_responds_to_demand_shock(artifacts_bundle):
    explainer = AnomalySHAPExplainer(artifacts_bundle)
    row = artifacts_bundle.reference_baselines["ALL_INDIA"].iloc[-1]

    norm_drivers = explainer.explain_anomaly_score(row, "ALL_INDIA", actual_demand_mw=float(row["demand_mw"]), top_k=3)
    shock_drivers = explainer.explain_anomaly_score(row, "ALL_INDIA", actual_demand_mw=310000.0, top_k=3)

    assert len(norm_drivers) == 3
    assert len(shock_drivers) == 3
    assert shock_drivers != norm_drivers

    shock_features = {item["feature"] for item in shock_drivers}
    assert shock_features & {"demand_mw", "demand_delta_1h", "demand_delta_24h", "demand_zscore_24h"}
    assert all(item["shap_value"] > 0 for item in shock_drivers)


def test_generate_72h_forecast_aligns_gaps_without_aliasing(artifacts_bundle):
    base_ts = datetime(2024, 6, 1, 0, 0, tzinfo=timezone.utc)
    clean_rows = []
    gapped_rows = []
    for i in range(200):
        ts = base_ts + timedelta(hours=i)
        rec = {
            "timestamp_utc": ts,
            "region": "NR",
            "demand_mw": 45000.0 + 4000.0 * np.sin(2 * np.pi * i / 24.0),
            "frequency_hz": 50.0,
            "temperature_2m": 30.0 + 4.0 * np.sin(2 * np.pi * i / 24.0),
            "relative_humidity_2m": 55.0,
            "surface_solar_radiation": 400.0 if 6 <= ts.hour <= 18 else 0.0,
            "wind_speed_100m": 4.5,
        }
        clean_rows.append(rec)
        if not (150 <= i <= 159):
            gapped_rows.append(rec)

    fc_clean = generate_72h_multi_step_forecast(pl.DataFrame(clean_rows), "NR", artifacts_bundle)
    fc_gapped = generate_72h_multi_step_forecast(pl.DataFrame(gapped_rows), "NR", artifacts_bundle)

    assert fc_gapped[0]["target_timestamp_utc"] == fc_clean[0]["target_timestamp_utc"]
    assert abs(fc_gapped[0]["forecast_p50"] - fc_clean[0]["forecast_p50"]) < 1e-3


def test_fetch_live_or_baseline_history_rejects_stale_db_window(artifacts_bundle, monkeypatch):
    from src.models import infer_and_detect as inf

    stale_end = datetime.now(timezone.utc) - timedelta(hours=100)
    stale_rows = pl.DataFrame([
        {
            "timestamp_utc": stale_end - timedelta(hours=200 - i),
            "region": "NR",
            "demand_mw": 42000.0,
            "frequency_hz": 50.0,
            "temperature_2m": 25.0,
            "relative_humidity_2m": 50.0,
            "surface_solar_radiation": 200.0,
            "wind_speed_100m": 4.0,
        }
        for i in range(200)
    ])
    monkeypatch.setattr(inf, "fetch_telemetry_window_from_db", lambda **kwargs: stale_rows)

    df_out, source = inf.fetch_live_or_baseline_history("NR", lookback_hours=168, bundle=artifacts_bundle)
    assert source == "baseline_fallback"
    assert len(df_out) == 168


def test_lru_bounded_model_and_shap_cache_eviction():
    bounded_bundle = ModelArtifactsBundle(max_cached_regions=2)
    bounded_explainer = AnomalySHAPExplainer(bounded_bundle)

    # cycle across 3 regions: NR -> WR -> SR; NR must be evicted once SR loads
    for reg in ["NR", "WR", "SR"]:
        row = bounded_bundle.reference_baselines[reg].iloc[-1]
        _ = bounded_bundle.lgb_models[reg]
        _ = bounded_explainer.explain_anomaly_point(row, reg, top_k=3)
        _ = bounded_explainer.explain_anomaly_score(row, reg, top_k=3)

    assert len(bounded_bundle.lgb_models) == 2
    assert set(bounded_bundle.lgb_models.keys()) == {"WR", "SR"}
    assert len(bounded_bundle.isolation_forests) == 2
    assert set(bounded_bundle.isolation_forests.keys()) == {"WR", "SR"}
    assert len(bounded_explainer.explainers) == 2
    assert set(bounded_explainer.explainers.keys()) == {"WR", "SR"}
    assert len(bounded_explainer.anomaly_explainers) == 2
    assert set(bounded_explainer.anomaly_explainers.keys()) == {"WR", "SR"}

    # touching WR makes it most-recently-used so loading ER evicts SR instead of WR
    _ = bounded_bundle.lgb_models["WR"]
    _ = bounded_bundle.lgb_models["ER"]
    assert set(bounded_bundle.lgb_models.keys()) == {"WR", "ER"}


def test_horizon_dependent_cqr_margins_and_ist_calendar_contract(artifacts_bundle):
    from src.models.infer_and_detect import IST_OFFSET
    from src.models.load_artifacts import resolve_calendar_timezone

    # clone bundle metadata safely to test horizon-wise CQR margins and IST calendar alignment
    class ContractBundle:
        feature_columns = artifacts_bundle.feature_columns
        anomaly_feature_columns = artifacts_bundle.anomaly_feature_columns
        lgb_models = artifacts_bundle.lgb_models
        isolation_forests = artifacts_bundle.isolation_forests
        reference_baselines = artifacts_bundle.reference_baselines
        quantile_calibration = {
            "NR": {
                "calendar_timezone": "Asia/Kolkata",
                "cqr_margin_80": 100.0,
                "cqr_margin_90": 200.0,
                "horizon_margins_80": [float(100 + h * 25) for h in range(72)],
                "horizon_margins_90": [float(200 + h * 40) for h in range(72)],
            }
        }

    contract_bundle = ContractBundle()
    assert resolve_calendar_timezone(contract_bundle.quantile_calibration, "NR") == "Asia/Kolkata"
    assert resolve_calendar_timezone({"NR": {"cqr_margin_80": 50.0}}, "NR") == "UTC"
    assert resolve_calendar_timezone(artifacts_bundle.quantile_calibration, "NR") == "Asia/Kolkata"
    assert len(artifacts_bundle.quantile_calibration["NR"]["horizon_margins_80"]) == 72
    assert len(artifacts_bundle.quantile_calibration["NR"]["horizon_margins_90"]) == 72

    df_sample = artifacts_bundle.reference_baselines["NR"].head(1)
    p10_h1, p50_h1, p90_h1 = predict_quantiles(df_sample, "NR", contract_bundle, widened=False, horizon_step=1)
    p10_h72, p50_h72, p90_h72 = predict_quantiles(df_sample, "NR", contract_bundle, widened=False, horizon_step=72)
    p10_w72, _, p90_w72 = predict_quantiles(df_sample, "NR", contract_bundle, widened=True, horizon_step=72)

    assert p50_h1[0] == pytest.approx(p50_h72[0])
    assert (p90_h72[0] - p10_h72[0]) > (p90_h1[0] - p10_h1[0])
    assert (p90_w72[0] - p10_w72[0]) > (p90_h72[0] - p10_h72[0])

    # rollout with IST contract runs cleanly without skew errors
    base_ts = datetime(2024, 6, 1, 18, 0, tzinfo=timezone.utc)
    hist = pl.DataFrame([
        {
            "timestamp_utc": base_ts + timedelta(hours=i),
            "region": "NR",
            "demand_mw": 45000.0 + 4000.0 * np.sin(2 * np.pi * ((i + 5.5) % 24.0) / 24.0),
            "frequency_hz": 50.0,
            "temperature_2m": 30.0,
            "relative_humidity_2m": 55.0,
            "surface_solar_radiation": 200.0 if 0 <= ((base_ts + timedelta(hours=i) + IST_OFFSET).hour) <= 12 else 0.0,
            "wind_speed_100m": 4.5,
        }
        for i in range(180)
    ])
    fc_ist = generate_72h_multi_step_forecast(hist, "NR", contract_bundle, widened=False)
    assert len(fc_ist) == 72
    assert (fc_ist[71]["forecast_p90"] - fc_ist[71]["forecast_p10"]) > (fc_ist[0]["forecast_p90"] - fc_ist[0]["forecast_p10"])