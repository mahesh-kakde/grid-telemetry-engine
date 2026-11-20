import numpy as np
import polars as pl
import holidays


_HOLIDAY_CACHE = {}


def get_indian_holidays(years: list[int]) -> set:
    # retrieves set of statutory indian national holiday dates with in-memory caching
    if not years:
        years = [2023, 2024, 2025, 2026]
    key = tuple(sorted(years))
    if key not in _HOLIDAY_CACHE:
        ind_holidays = holidays.India(years=years)
        _HOLIDAY_CACHE[key] = set(ind_holidays.keys())
    return _HOLIDAY_CACHE[key]


def compute_telemetry_features(
    df_raw: pl.DataFrame,
    drop_nulls_flag: bool = True,
) -> pl.DataFrame:
    """
    Transforms raw telemetry into a zero-leakage feature matrix.
    Guaranteed deterministic across historical and live streaming batches.
    """
    years = df_raw["timestamp_utc"].dt.year().unique().to_list()
    holidays_set = get_indian_holidays(years)

    lazy_df = (
        df_raw.lazy()
        .sort(["region", "timestamp_utc"])
        .with_columns([
            pl.col("timestamp_utc").dt.hour().alias("hour"),
            pl.col("timestamp_utc").dt.weekday().alias("day_of_week"),
            pl.col("timestamp_utc").dt.ordinal_day().alias("day_of_year"),
            pl.col("timestamp_utc").dt.date().is_in(holidays_set).cast(pl.Int8).alias("is_holiday"),
        ])
        .with_columns([
            (2 * np.pi * pl.col("hour") / 24.0).sin().alias("sin_hour_k1"),
            (2 * np.pi * pl.col("hour") / 24.0).cos().alias("cos_hour_k1"),
            (4 * np.pi * pl.col("hour") / 24.0).sin().alias("sin_hour_k2"),
            (4 * np.pi * pl.col("hour") / 24.0).cos().alias("cos_hour_k2"),
            (2 * np.pi * pl.col("day_of_year") / 365.25).sin().alias("sin_doy"),
            (2 * np.pi * pl.col("day_of_year") / 365.25).cos().alias("cos_doy"),
        ])
        .with_columns([
            pl.col("demand_mw").shift(lag).over("region").alias(f"demand_lag_{lag}h")
            for lag in [1, 2, 3, 24, 48, 168]
        ])
        .with_columns([
            pl.col("demand_mw").shift(1).rolling_mean(window_size=w).over("region").alias(f"demand_rolling_mean_{w}h")
            for w in [6, 24, 168]
        ] + [
            pl.col("demand_mw").shift(1).rolling_std(window_size=w).over("region").alias(f"demand_rolling_std_{w}h")
            for w in [6, 24, 168]
        ])
        .with_columns([
            (pl.col("temperature_2m") - pl.col("temperature_2m").shift(24).over("region")).alias("temp_delta_24h"),
            (pl.col("relative_humidity_2m") - pl.col("relative_humidity_2m").shift(24).over("region")).alias("rh_delta_24h"),
        ])
    )

    if drop_nulls_flag:
        lazy_df = lazy_df.drop_nulls(subset=["demand_lag_168h", "demand_rolling_std_168h"])

    return lazy_df.collect()


# canonical alias matching blueprint specification
build_features = compute_telemetry_features