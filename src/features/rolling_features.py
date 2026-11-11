from typing import TypeVar
import polars as pl

FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)

DEFAULT_WINDOWS: list[int] = [6, 24, 168]


def get_base_rolling_expressions(
    target_col: str = "metric_value",
    group_col: str = "region",
    windows: list[int] | None = None,
) -> list[pl.Expr]:
    active_windows = windows or DEFAULT_WINDOWS
    exprs: list[pl.Expr] = []

    # apply shift(1) prior to all rolling aggregations to prevent target leakage
    shifted_target = pl.col(target_col).shift(1)

    for w in active_windows:
        exprs.extend(
            [
                shifted_target.rolling_mean(w, min_samples=1)
                .over(group_col)
                .alias(f"{target_col}_roll_mean_{w}h"),
                shifted_target.rolling_std(w, min_samples=2)
                .over(group_col)
                .alias(f"{target_col}_roll_std_{w}h"),
                shifted_target.rolling_min(w, min_samples=1)
                .over(group_col)
                .alias(f"{target_col}_roll_min_{w}h"),
                shifted_target.rolling_max(w, min_samples=1)
                .over(group_col)
                .alias(f"{target_col}_roll_max_{w}h"),
            ]
        )

    return exprs


def get_derived_rolling_expressions(
    target_col: str = "metric_value",
    windows: list[int] | None = None,
) -> list[pl.Expr]:
    active_windows = windows or DEFAULT_WINDOWS
    if 24 not in active_windows:
        return []

    shifted_target = pl.col(target_col).shift(1)
    # 24-hour baseline ratio and anomaly z-score indicators
    return [
        (
            shifted_target
            / (pl.col(f"{target_col}_roll_mean_24h") + 1e-6)
        ).alias(f"{target_col}_roll_ratio_24h"),
        (
            (shifted_target - pl.col(f"{target_col}_roll_mean_24h"))
            / (pl.col(f"{target_col}_roll_std_24h") + 1e-6)
        ).alias(f"{target_col}_roll_zscore_24h"),
    ]


def add_rolling_features(
    df: FrameT,
    target_col: str = "metric_value",
    group_col: str = "region",
    windows: list[int] | None = None,
) -> FrameT:
    is_eager = isinstance(df, pl.DataFrame)
    lf = df.lazy() if is_eager else df

    active_windows = windows or DEFAULT_WINDOWS
    base_exprs = get_base_rolling_expressions(
        target_col=target_col,
        group_col=group_col,
        windows=active_windows,
    )
    lf = lf.with_columns(base_exprs)

    derived_exprs = get_derived_rolling_expressions(
        target_col=target_col,
        windows=active_windows,
    )
    if derived_exprs:
        lf = lf.with_columns(derived_exprs)

    return lf.collect() if is_eager else lf  # type: ignore[return-value]