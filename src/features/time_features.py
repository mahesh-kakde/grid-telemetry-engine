import math
from typing import TypeVar
import polars as pl

FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)


def get_time_expressions(timestamp_col: str = "timestamp") -> list[pl.Expr]:
    ts = pl.col(timestamp_col)
    hour = ts.dt.hour()
    weekday = ts.dt.weekday()
    hour_of_week = (weekday - 1) * 24 + hour
    day_of_year = ts.dt.ordinal_day()
    month = ts.dt.month()

    # cyclical fourier harmonic encodings and calendar parts
    return [
        hour.alias("hour"),
        weekday.alias("day_of_week"),
        (weekday >= 6).cast(pl.Int8).alias("is_weekend"),
        month.alias("month"),
        ((month - 1) // 3 + 1).cast(pl.Int8).alias("quarter"),
        # 24-hour diurnal cycle
        (2.0 * math.pi * hour / 24.0).sin().alias("sin_24h"),
        (2.0 * math.pi * hour / 24.0).cos().alias("cos_24h"),
        # 12-hour harmonic capturing morning and evening peaks
        (4.0 * math.pi * hour / 24.0).sin().alias("sin_12h"),
        (4.0 * math.pi * hour / 24.0).cos().alias("cos_12h"),
        # 168-hour weekly cycle
        (2.0 * math.pi * hour_of_week / 168.0).sin().alias("sin_168h"),
        (2.0 * math.pi * hour_of_week / 168.0).cos().alias("cos_168h"),
        # annual seasonal transition
        (2.0 * math.pi * day_of_year / 365.25).sin().alias("sin_365d"),
        (2.0 * math.pi * day_of_year / 365.25).cos().alias("cos_365d"),
    ]


def add_time_features(df: FrameT, timestamp_col: str = "timestamp") -> FrameT:
    is_eager = isinstance(df, pl.DataFrame)
    lf = df.lazy() if is_eager else df

    exprs = get_time_expressions(timestamp_col)
    result = lf.with_columns(exprs)

    return result.collect() if is_eager else result  # type: ignore[return-value]