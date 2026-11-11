import math
import time
from datetime import datetime, timedelta, timezone
import polars as pl
import pytest

from src.features.lag_features import add_lag_features
from src.features.pipeline import FeaturePipeline
from src.features.rolling_features import add_rolling_features
from src.features.time_features import add_time_features


def test_fourier_cyclical_properties():
    # generate 24 hours of timestamps
    start = datetime(2026, 11, 10, 0, 0, 0, tzinfo=timezone.utc)
    ts_list = [start + timedelta(hours=i) for i in range(24)]
    df = pl.DataFrame({"timestamp": ts_list})

    transformed = add_time_features(df)

    # verify bounds in [-1, 1]
    for col_name in ["sin_24h", "cos_24h", "sin_12h", "cos_12h", "sin_168h", "cos_168h"]:
        vals = transformed[col_name].to_list()
        assert all(-1.0 <= v <= 1.0 for v in vals)

    # at 06:00, 24h sin must be 1.0 and cos must be ~0.0
    row_6 = transformed.filter(pl.col("hour") == 6).to_dicts()[0]
    assert math.isclose(row_6["sin_24h"], 1.0, abs_tol=1e-5)
    assert math.isclose(row_6["cos_24h"], 0.0, abs_tol=1e-5)

    # at 12:00, 24h sin must be ~0.0 and cos must be -1.0
    row_12 = transformed.filter(pl.col("hour") == 12).to_dicts()[0]
    assert math.isclose(row_12["sin_24h"], 0.0, abs_tol=1e-5)
    assert math.isclose(row_12["cos_24h"], -1.0, abs_tol=1e-5)


def test_zero_temporal_leakage():
    # construct 200 hours of deterministic telemetry
    start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=i) for i in range(200)]
    values = [40000.0 + i * 50.0 for i in range(200)]

    df_original = pl.DataFrame(
        {
            "timestamp": timestamps,
            "region": ["NR"] * 200,
            "metric_value": values,
        }
    )

    pipeline = FeaturePipeline(drop_nulls=False)
    features_orig = pipeline.transform(df_original)

    # pick target row index 175
    target_idx = 175
    target_row_orig = features_orig.row(target_idx, named=True)

    # create perturbed dataset where value at target_idx is drastically spiked (+50,000 mw)
    values_perturbed = list(values)
    values_perturbed[target_idx] = 999999.0
    df_perturbed = pl.DataFrame(
        {
            "timestamp": timestamps,
            "region": ["NR"] * 200,
            "metric_value": values_perturbed,
        }
    )

    features_perturbed = pipeline.transform(df_perturbed)
    target_row_perturbed = features_perturbed.row(target_idx, named=True)

    # all lag and rolling features at target_idx must be 100% invariant to perturbation at target_idx
    feature_names = pipeline.get_feature_names(include_target=False)
    for feat in feature_names:
        assert (
            target_row_orig[feat] == target_row_perturbed[feat]
        ), f"temporal leak detected in feature '{feat}' at row {target_idx}"


def test_regional_isolation():
    # test that lags and rolling stats do not leak across regional boundaries
    start = datetime(2026, 11, 1, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=i) for i in range(20)]

    df = pl.DataFrame(
        {
            "timestamp": timestamps + timestamps,
            "region": ["NR"] * 20 + ["WR"] * 20,
            "metric_value": [100.0] * 20 + [5000.0] * 20,
        }
    )

    pipeline = FeaturePipeline(lag_hours=[1, 2], rolling_windows=[6], drop_nulls=False)
    transformed = pipeline.transform(df)

    # the first row of WR (index 20) must have null for lag_1h, not 100.0 from NR
    wr_rows = transformed.filter(pl.col("region") == "WR")
    assert wr_rows["metric_value_lag_1h"][0] is None
    assert wr_rows["metric_value_roll_mean_6h"][0] is None


def test_lag_values_and_rolling_accuracy():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    n = 30
    df = pl.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in range(n)],
            "region": ["NR"] * n,
            "metric_value": [float(i * 10) for i in range(n)],  # 0.0, 10.0, 20.0, ...
        }
    )

    pipeline = FeaturePipeline(
        lag_hours=[1, 3], rolling_windows=[6], drop_nulls=False
    )
    transformed = pipeline.transform(df)

    # at index 10: metric_value is 100.0, lag_1h must be 90.0, lag_3h must be 70.0
    row_10 = transformed.row(10, named=True)
    assert row_10["metric_value_lag_1h"] == 90.0
    assert row_10["metric_value_lag_3h"] == 70.0

    # rolling 6h mean at index 10 uses rows 4 to 9 (40, 50, 60, 70, 80, 90) -> mean 65.0
    expected_mean = sum(range(40, 100, 10)) / 6.0
    assert math.isclose(row_10["metric_value_roll_mean_6h"], expected_mean)


def test_pipeline_performance_100k_rows():
    n = 100_000
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=i % 20000) for i in range(n)]
    regions = ["NR", "WR", "SR", "ER", "NER"] * (n // 5)
    values = [45000.0 + 1000.0 * math.sin(i * 0.1) for i in range(n)]

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "region": regions,
            "metric_value": values,
        }
    )

    pipeline = FeaturePipeline(drop_nulls=True)

    t0 = time.perf_counter()
    result = pipeline.transform(df)
    elapsed_seconds = time.perf_counter() - t0

    # acceptance criteria: < 500 ms (0.500 seconds)
    assert (
        elapsed_seconds < 0.500
    ), f"pipeline took {elapsed_seconds*1000:.1f}ms, exceeding 500ms budget"
    assert result.height > 0
    # verify zero nulls when drop_nulls is True
    assert result.null_count().sum_horizontal()[0] == 0


def test_pipeline_string_timestamp_support():
    df = pl.DataFrame(
        {
            "timestamp": [
                "2026-11-10T12:00:00+00:00",
                "2026-11-10T13:00:00+00:00",
                "2026-11-10T14:00:00+00:00",
            ],
            "region": ["NR", "NR", "NR"],
            "metric_value": [45000.0, 46000.0, 47000.0],
        }
    )

    pipeline = FeaturePipeline(drop_nulls=False)
    result = pipeline.transform(df)
    assert result["hour"].to_list() == [12, 13, 14]