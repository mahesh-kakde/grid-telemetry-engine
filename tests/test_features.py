import pytest
import numpy as np
import polars as pl
from datetime import datetime, timezone, timedelta

from src.features.build_features import compute_telemetry_features


def generate_mock_telemetry(hours: int = 200, region: str = "NR") -> pl.DataFrame:
    # generates synthetic chronological telemetry series for feature assertion
    t0 = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)
    rows = []
    for h in range(hours):
        rows.append({
            "timestamp_utc": t0 + timedelta(hours=h),
            "region": region,
            "demand_mw": 50000.0 + (h % 24) * 500.0,
            "temperature_2m": 20.0 + (h % 24) * 0.5,
            "relative_humidity_2m": 60.0 + (h % 24) * 0.2,
            "surface_solar_radiation": 200.0 if 6 <= (h % 24) <= 18 else 0.0,
            "wind_speed_100m": 4.5,
        })
    return pl.DataFrame(rows)


def test_zero_lookahead_leakage_causality():
    # asserts changing demand at timestamp t does not alter rolling moments at timestamp t
    df1 = generate_mock_telemetry(hours=200, region="NR")
    features1 = compute_telemetry_features(df1, drop_nulls_flag=False)

    # create copy and alter the demand at index 175
    df2 = df1.clone()
    target_idx = 175
    original_demand = df2["demand_mw"][target_idx]
    modified_demand = original_demand + 100000.0
    df2 = df2.with_columns(
        pl.when(pl.col("timestamp_utc") == df2["timestamp_utc"][target_idx])
        .then(modified_demand)
        .otherwise(pl.col("demand_mw"))
        .alias("demand_mw")
    )
    features2 = compute_telemetry_features(df2, drop_nulls_flag=False)

    # at target_idx, rolling mean and std must remain completely identical
    # because they strictly evaluate over records up to t - 1
    assert features1["demand_rolling_mean_6h"][target_idx] == features2["demand_rolling_mean_6h"][target_idx]
    assert features1["demand_rolling_mean_24h"][target_idx] == features2["demand_rolling_mean_24h"][target_idx]
    assert features1["demand_rolling_mean_168h"][target_idx] == features2["demand_rolling_mean_168h"][target_idx]
    assert features1["demand_rolling_std_6h"][target_idx] == features2["demand_rolling_std_6h"][target_idx]
    assert features1["demand_rolling_std_24h"][target_idx] == features2["demand_rolling_std_24h"][target_idx]
    assert features1["demand_rolling_std_168h"][target_idx] == features2["demand_rolling_std_168h"][target_idx]

    # conversely, target_idx + 1 must reflect the shock
    assert features1["demand_rolling_mean_6h"][target_idx + 1] != features2["demand_rolling_mean_6h"][target_idx + 1]


def test_autoregressive_lags_precision():
    # asserts exact alignment of lagged autoregressive values
    df = generate_mock_telemetry(hours=200, region="NR")
    features = compute_telemetry_features(df, drop_nulls_flag=False)

    idx = 180
    assert features["demand_lag_1h"][idx] == df["demand_mw"][idx - 1]
    assert features["demand_lag_2h"][idx] == df["demand_mw"][idx - 2]
    assert features["demand_lag_3h"][idx] == df["demand_mw"][idx - 3]
    assert features["demand_lag_24h"][idx] == df["demand_mw"][idx - 24]
    assert features["demand_lag_48h"][idx] == df["demand_mw"][idx - 48]
    assert features["demand_lag_168h"][idx] == df["demand_mw"][idx - 168]


def test_region_isolation_fences():
    # asserts lags and moments do not leak across distinct regional grids
    df_nr = generate_mock_telemetry(hours=200, region="NR")
    df_wr = generate_mock_telemetry(hours=200, region="WR")
    # artificially scale wr demand by 2x
    df_wr = df_wr.with_columns(pl.col("demand_mw") * 2.0)

    combined_df = pl.concat([df_nr, df_wr])
    features = compute_telemetry_features(combined_df, drop_nulls_flag=False)

    nr_features = features.filter(pl.col("region") == "NR")
    wr_features = features.filter(pl.col("region") == "WR")

    # verify 1h lag of first valid row in wr is not pulling from the end of nr
    idx = 10
    assert wr_features["demand_lag_1h"][idx] == df_wr["demand_mw"][idx - 1]
    assert nr_features["demand_lag_1h"][idx] == df_nr["demand_mw"][idx - 1]


def test_cyclical_fourier_properties():
    # asserts mathematical properties of diurnal and annual fourier harmonics
    df = generate_mock_telemetry(hours=48, region="SR")
    features = compute_telemetry_features(df, drop_nulls_flag=False)

    sin_k1 = features["sin_hour_k1"].to_numpy()
    cos_k1 = features["cos_hour_k1"].to_numpy()
    sin_k2 = features["sin_hour_k2"].to_numpy()
    cos_k2 = features["cos_hour_k2"].to_numpy()
    sin_doy = features["sin_doy"].to_numpy()
    cos_doy = features["cos_doy"].to_numpy()

    # bounds within [-1, 1]
    assert np.all(sin_k1 >= -1.0) and np.all(sin_k1 <= 1.0)
    assert np.all(cos_k1 >= -1.0) and np.all(cos_k1 <= 1.0)

    # pythagorean identity sin^2 + cos^2 == 1.0
    np.testing.assert_allclose(sin_k1**2 + cos_k1**2, 1.0, atol=1e-5)
    np.testing.assert_allclose(sin_k2**2 + cos_k2**2, 1.0, atol=1e-5)
    np.testing.assert_allclose(sin_doy**2 + cos_doy**2, 1.0, atol=1e-5)


def test_exogenous_weather_deltas():
    # asserts 24h temperature and relative humidity differentials
    df = generate_mock_telemetry(hours=50, region="ER")
    features = compute_telemetry_features(df, drop_nulls_flag=False)

    idx = 30
    expected_temp_delta = df["temperature_2m"][idx] - df["temperature_2m"][idx - 24]
    expected_rh_delta = df["relative_humidity_2m"][idx] - df["relative_humidity_2m"][idx - 24]

    assert pytest.approx(features["temp_delta_24h"][idx]) == expected_temp_delta
    assert pytest.approx(features["rh_delta_24h"][idx]) == expected_rh_delta


def test_statutory_holiday_encoding():
    # asserts national holidays are flagged correctly in 2023
    t_republic_day = datetime(2023, 1, 26, 12, 0, tzinfo=timezone.utc)
    t_normal_day = datetime(2023, 1, 27, 12, 0, tzinfo=timezone.utc)

    df = pl.DataFrame([
        {
            "timestamp_utc": t_republic_day,
            "region": "ALL_INDIA",
            "demand_mw": 150000.0,
            "temperature_2m": 18.0,
            "relative_humidity_2m": 60.0,
            "surface_solar_radiation": 100.0,
            "wind_speed_100m": 4.0,
        },
        {
            "timestamp_utc": t_normal_day,
            "region": "ALL_INDIA",
            "demand_mw": 155000.0,
            "temperature_2m": 19.0,
            "relative_humidity_2m": 58.0,
            "surface_solar_radiation": 110.0,
            "wind_speed_100m": 4.2,
        },
    ])

    features = compute_telemetry_features(df, drop_nulls_flag=False)
    assert features["is_holiday"][0] == 1
    assert features["is_holiday"][1] == 0


def test_drop_nulls_warmup_filtering():
    # asserts drop_nulls_flag strips uninitialized warmup rows
    df = generate_mock_telemetry(hours=200, region="NR")
    features_dropped = compute_telemetry_features(df, drop_nulls_flag=True)
    features_retained = compute_telemetry_features(df, drop_nulls_flag=False)

    assert len(features_retained) == 200
    # 200 - 168 warmup hours = 32 valid rows
    assert len(features_dropped) == 32
    assert features_dropped["demand_lag_168h"].null_count() == 0


def test_upsample_prevents_gap_shift_aliasing():
    import polars as pl
    from datetime import datetime, timezone
    from src.features.build_features import compute_telemetry_features

    # t0 = 00:00, missing 01:00..23:00 (23h hole), next observed = t0 + 24h
    df_gap = pl.DataFrame({
        "timestamp_utc": [
            datetime(2024, 2, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 2, 2, 0, 0, tzinfo=timezone.utc),
        ],
        "region": ["NR", "NR"],
        "demand_mw": [41000.0, 42500.0],
        "temperature_2m": [16.0, 17.5],
        "relative_humidity_2m": [60.0, 58.0],
        "surface_solar_radiation": [0.0, 0.0],
        "wind_speed_100m": [3.5, 4.0],
    })
    res = compute_telemetry_features(df_gap, drop_nulls_flag=False)
    assert len(res) == 2
    # t0 + 24h must see exact 24h lag while 1h lag remains null due to missing 23h slot
    assert res["demand_lag_1h"][1] is None
    assert res["demand_lag_24h"][1] == 41000.0


def test_ist_midnight_holiday_conversion():
    import polars as pl
    from datetime import datetime, timezone
    from src.features.build_features import compute_telemetry_features

    # 2024-01-25 19:00 UTC == 2024-01-26 00:30 IST (Republic Day in India)
    df_ist = pl.DataFrame({
        "timestamp_utc": [
            datetime(2024, 1, 25, 17, 0, tzinfo=timezone.utc),  # 22:30 IST on Jan 25 (non-holiday)
            datetime(2024, 1, 25, 19, 0, tzinfo=timezone.utc),  # 00:30 IST on Jan 26 (Republic Day)
        ],
        "region": ["ALL_INDIA", "ALL_INDIA"],
        "demand_mw": [140000.0, 130000.0],
        "temperature_2m": [20.0, 19.0],
        "relative_humidity_2m": [65.0, 66.0],
        "surface_solar_radiation": [0.0, 0.0],
        "wind_speed_100m": [3.0, 3.0],
    })
    res = compute_telemetry_features(df_ist, drop_nulls_flag=False)
    assert res["is_holiday"].to_list() == [0, 1]