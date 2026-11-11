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
    if name == "AnomalyDetector":
        from src.models.anomaly_detector import AnomalyDetector

        return AnomalyDetector
    if name == "ExplainabilityEngine":
        from src.models.explainability import ExplainabilityEngine

        return ExplainabilityEngine
    if name == "AnomalyRunner":
        from src.models.anomaly_runner import AnomalyRunner

        return AnomalyRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnomalyDetector",
    "AnomalyRunner",
    "ExplainabilityEngine",
    "ForecastDataset",
    "ForecastEngine",
    "QuantileRegressor",
]