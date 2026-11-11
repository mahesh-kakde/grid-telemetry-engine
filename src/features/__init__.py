from src.features.lag_features import add_lag_features
from src.features.rolling_features import add_rolling_features
from src.features.time_features import add_time_features


def __getattr__(name: str):
    if name == "FeaturePipeline":
        from src.features.pipeline import FeaturePipeline

        return FeaturePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FeaturePipeline",
    "add_lag_features",
    "add_rolling_features",
    "add_time_features",
]