from typing import Optional
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


def _resolve_active_calendar_tz(df_raw: pl.DataFrame, calendar_tz: Optional[str]) -> str:
    if calendar_tz:
        return calendar_tz
    try:
        from src.models.load_artifacts import resolve_calendar_timezone

        region_hint: Optional[str] = None
        if "region" in df_raw.columns and len(df_raw) > 0:
            region_hint = str(df_raw["region"][0])
        return resolve_calendar_timezone(region=region_hint)
    except Exception:
        return "UTC"


def compute_telemetry_features(
    df_raw: pl.DataFrame,
    drop_nulls_flag: bool = True,
    calendar_tz: Optional[str] = None,
) -> pl.DataFrame:
    """
    Transforms raw telemetry into a zero-leakage feature matrix.
    Upsamples each region to a uniform 1h grid before computing lags and IST holidays
    so temporal gaps cannot silently alias lagged columns.
    """
    if df_raw.is_empty():
        return df_raw

    resolved_cal_tz = _resolve_active_calendar_tz(df_raw, calendar_tz)

    ts_dtype = df_raw.schema["timestamp_utc"]
    if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
        ts_expr = pl.col("timestamp_utc").dt.convert_time_zone("UTC").dt.cast_time_unit("us")
    elif isinstance(ts_dtype, pl.Datetime):
        ts_expr = pl.col("timestamp_utc").dt.replace_time_zone("UTC").dt.cast_time_unit("us")
    else:
        ts_expr = pl.col("timestamp_utc").cast(pl.Datetime(time_unit="us", time_zone="UTC"))

    df_norm = (
        df_raw.with_columns([
            ts_expr.alias("timestamp_utc"),
            pl.lit(True).alias("_is_observed"),
        ])
        .unique(subset=["region", "timestamp_utc"], keep="last")
        .sort(["region", "timestamp_utc"])
    )

    df_grid = (
        df_norm.upsample(time_column="timestamp_utc", every="1h", group_by="region")
        .with_columns(pl.col("region").forward_fill())
    )

    ist_years = (
        df_grid["timestamp_utc"]
        .dt.convert_time_zone("Asia/Kolkata")
        .dt.year()
        .unique()
        .drop_nulls()
        .to_list()
    )
    holidays_list = list(get_indian_holidays(ist_years))

    cal_ts = (
        pl.col("timestamp_utc")
        if resolved_cal_tz == "UTC"
        else pl.col("timestamp_utc").dt.convert_time_zone(resolved_cal_tz)
    )

    lazy_df = (
        df_grid.lazy()
        .sort(["region", "timestamp_utc"])
        .with_columns([
            cal_ts.dt.hour().alias("hour"),
            cal_ts.dt.weekday().alias("day_of_week"),
            cal_ts.dt.ordinal_day().alias("day_of_year"),
            pl.col("timestamp_utc")
            .dt.convert_time_zone("Asia/Kolkata")
            .dt.date()
            .is_in(holidays_list)
            .cast(pl.Int8)
            .alias("is_holiday"),
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
            (pl.col("demand_mw") - pl.col("demand_mw").shift(1).over("region")).alias("demand_delta_1h"),
            (pl.col("demand_mw") - pl.col("demand_mw").shift(24).over("region")).alias("demand_delta_24h"),
            (
                (pl.col("demand_mw") - pl.col("demand_rolling_mean_24h"))
                / pl.col("demand_rolling_std_24h").clip(lower_bound=1.0)
            ).alias("demand_zscore_24h"),
        ])
        .filter(pl.col("_is_observed") == True)
        .drop("_is_observed")
    )

    if drop_nulls_flag:
        lazy_df = lazy_df.drop_nulls(
            subset=[
                "demand_lag_168h",
                "demand_rolling_std_168h",
                "temp_delta_24h",
                "rh_delta_24h",
                "demand_zscore_24h",
            ]
        )

    return lazy_df.collect()


# canonical alias matching blueprint specification
build_features = compute_telemetry_features