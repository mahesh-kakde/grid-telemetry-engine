from typing import Any

__all__ = ["DriftSuite", "DriftSuiteResult", "DriftRunner", "export_drift_report"]


def __getattr__(name: str) -> Any:
    if name in ("DriftSuite", "DriftSuiteResult"):
        from src.monitoring.drift_suite import DriftSuite, DriftSuiteResult

        return locals()[name]
    if name == "DriftRunner":
        from src.monitoring.drift_runner import DriftRunner

        return DriftRunner
    if name == "export_drift_report":
        from src.monitoring.report_exporter import export_drift_report

        return export_drift_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")