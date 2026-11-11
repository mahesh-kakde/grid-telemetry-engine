def __getattr__(name: str):
    if name == "QuantileRegressor":
        from src.models.quantile_regressor import QuantileRegressor

        return QuantileRegressor
    if name == "ForecastEngine":
        from src.models.forecast_inference import ForecastEngine

        return ForecastEngine
    if name == "ForecastDataset":
        from src.models.forecast_dataset import ForecastDataset

        return ForecastDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ForecastDataset", "ForecastEngine", "QuantileRegressor"]