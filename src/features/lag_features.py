from typing import TypeVar
import polars as pl

FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)

DEFAULT_LAGS: list[int] = [1, 2, 3, 24, 48, 168]


def get_lag_expressions(
    target_col: str = "metric_value",
    group_col: str = "region",
    lags: list[int] | None = None,
) -> list[pl.Expr]:
    active_lags = lags or DEFAULT_LAGS
    exprs: list[pl.Expr] = []

    # shifted lags partitioned strictly by region
    for lag in active_lags:
        exprs.append(
            pl.col(target_col)
            .shift(lag)
            .over(group_col)
            .alias(f"{target_col}_lag_{lag}h")
        )

    # momentum differences using strictly historical observations
    exprs.extend(
        [
            (
                pl.col(target_col).shift(1).over(group_col)
                - pl.col(target_col).shift(2).over(group_col)
            ).alias(f"{target_col}_diff_1h"),
            (
                pl.col(target_col).shift(1).over(group_col)
                - pl.col(target_col).shift(25).over(group_col)
            ).alias(f"{target_col}_diff_24h"),
            (
                pl.col(target_col).shift(1).over(group_col)
                - pl.col(target_col).shift(169).over(group_col)
            ).alias(f"{target_col}_diff_168h"),
        ]
    )

    return exprs


def add_lag_features(
    df: FrameT,
    target_col: str = "metric_value",
    group_col: str = "region",
    lags: list[int] | None = None,
) -> FrameT:
    is_eager = isinstance(df, pl.DataFrame)
    lf = df.lazy() if is_eager else df

    exprs = get_lag_expressions(target_col=target_col, group_col=group_col, lags=lags)
    result = lf.with_columns(exprs)

    return result.collect() if is_eager else result  # type: ignore[return-value]