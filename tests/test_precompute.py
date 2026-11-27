import os
import json
import pandas as pd

from src.models.load_artifacts import ModelArtifactsBundle, REGIONS
from src.models.explainers import AnomalySHAPExplainer
from src.models.precompute import (
    load_precomputed_data,
    load_precomputed_forecasts,
    load_precomputed_anomaly,
    load_precomputed_drift,
    precompute_region,
    DEFAULT_PRECOMPUTED_DIR,
)


def test_precomputed_manifest_index():
    # asserts manifest index exists and lists all 6 regions
    manifest_path = os.path.join(DEFAULT_PRECOMPUTED_DIR, "manifest.json")
    assert os.path.exists(manifest_path), "manifest.json must be present in precomputed directory"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["file_count"] == len(REGIONS)
    assert set(manifest["regions"]) == set(REGIONS)
    assert "generated_at" in manifest
    assert "model_version" in manifest


def test_precomputed_artifacts_integrity_all_regions():
    # asserts all 6 regions have valid precomputed json files with monotonicity
    for region in REGIONS:
        data = load_precomputed_data(region)
        assert data is not None, f"precomputed data missing for {region}"
        assert data["region"] == region

        # verify 72h forecasts
        forecasts = load_precomputed_forecasts(region)
        assert forecasts is not None, f"forecasts missing for {region}"
        assert len(forecasts) == 72, f"expected 72 forecast hours, got {len(forecasts)}"

        for fc in forecasts:
            assert isinstance(fc["target_timestamp_utc"], pd.Timestamp)
            assert fc["region"] == region
            # quantile monotonicity check
            assert fc["forecast_p10"] <= fc["forecast_p50"] <= fc["forecast_p90"], (
                f"quantile monotonicity breached at {fc['target_timestamp_utc']}: "
                f"p10={fc['forecast_p10']}, p50={fc['forecast_p50']}, p90={fc['forecast_p90']}"
            )

        # verify nominal anomaly snapshot
        anomaly = load_precomputed_anomaly(region)
        assert anomaly is not None, f"anomaly snapshot missing for {region}"
        assert "actual_demand_mw" in anomaly
        assert "predicted_p50_mw" in anomaly
        assert "anomaly_score" in anomaly
        assert "is_structural" in anomaly
        assert "top_shap_features" in anomaly
        assert len(anomaly["top_shap_features"]) == 3

        # verify drift metrics summary
        drift = load_precomputed_drift(region)
        assert drift is not None, f"drift metrics missing for {region}"
        assert "drift_alarm" in drift
        assert "drift_ratio" in drift
        assert "feature_metrics" in drift
        assert len(drift["feature_metrics"]) > 0


def test_precompute_single_region_runner(tmp_path):
    # verifies on-demand execution of precompute_region into temporary path
    bundle = ModelArtifactsBundle()
    explainer = AnomalySHAPExplainer(bundle)
    payload = precompute_region("NER", bundle, explainer)

    assert payload["region"] == "NER"
    assert payload["data_source"] == "baseline_fallback"
    assert len(payload["forecasts"]) == 72
    assert "nominal_anomaly" in payload
    assert len(payload["nominal_anomaly"]["anomaly_shap_features"]) == 3
    assert "drift_summary" in payload
    assert payload["drift_summary"]["window_source"] == "baseline_fallback"

    # write to temp path and test loader with custom output_dir
    temp_dir = str(tmp_path)
    file_path = os.path.join(temp_dir, "NER.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loaded_fc = load_precomputed_forecasts("NER", output_dir=temp_dir)
    assert loaded_fc is not None
    assert len(loaded_fc) == 72


def test_precomputed_all_india_conserves_regional_forecast_sum():
    from src.models.infer_and_detect import SUB_REGIONS

    ai_fc = load_precomputed_forecasts("ALL_INDIA")
    sub_fcs = {reg: load_precomputed_forecasts(reg) for reg in SUB_REGIONS}
    assert ai_fc is not None and len(ai_fc) == 72
    for h in range(72):
        sub_sum = sum(sub_fcs[reg][h]["forecast_p50"] for reg in SUB_REGIONS)
        assert abs(ai_fc[h]["forecast_p50"] - sub_sum) <= 0.10


def test_precomputed_loader_nonexistent_region():
    # asserts graceful None return for invalid or nonexistent region
    assert load_precomputed_data("NONEXISTENT_GRID") is None
    assert load_precomputed_forecasts("NONEXISTENT_GRID") is None
    assert load_precomputed_anomaly("NONEXISTENT_GRID") is None
    assert load_precomputed_drift("NONEXISTENT_GRID") is None


def test_precompute_region_guards_partial_post_warmup_drift(monkeypatch):
    import polars as pl
    from datetime import datetime, timezone, timedelta

    t0 = datetime(2026, 11, 19, 6, 0, tzinfo=timezone.utc)
    partial_live = pl.DataFrame([
        {
            "timestamp_utc": t0 + timedelta(hours=i),
            "region": "NER",
            "demand_mw": 2400.0 + (i % 24) * 30.0,
            "frequency_hz": 50.0,
            "temperature_2m": 22.0,
            "relative_humidity_2m": 70.0,
            "surface_solar_radiation": 150.0 if 6 <= ((t0 + timedelta(hours=i)).hour) <= 18 else 0.0,
            "wind_speed_100m": 3.5,
        }
        for i in range(171)
    ])
    monkeypatch.setattr(
        "src.models.infer_and_detect.fetch_live_or_baseline_history",
        lambda *args, **kwargs: (partial_live, "supabase_live"),
    )

    bundle = ModelArtifactsBundle()
    explainer = AnomalySHAPExplainer(bundle)
    payload = precompute_region("NER", bundle, explainer, use_live_weather=False)
    drift = payload["drift_summary"]
    assert drift["observation_hours"] == 168
    assert drift["window_source"] == "baseline_fallback"
    assert drift["feature_metrics"]["sin_hour_k1"]["score"] == 0.0
    assert drift["drift_alarm"] is False