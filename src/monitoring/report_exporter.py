from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from src.monitoring.drift_suite import DriftSuiteResult


def export_drift_report(
    report: Any,
    output_path: str | Path | None = None,
    reports_dir: str | Path = "reports/drift",
) -> Path:
    if output_path is not None:
        target_path = Path(output_path).resolve()
    else:
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        target_path = (Path(reports_dir) / f"drift_report_{timestamp_str}.html").resolve()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(target_path))

    if not target_path.exists() or target_path.stat().st_size == 0:
        raise RuntimeError(f"failed to generate html drift report at {target_path}")

    return target_path


def extract_drift_summary(result: DriftSuiteResult) -> dict[str, Any]:
    return {
        "drift_share": result.drift_share,
        "dataset_drift": result.dataset_drift,
        "number_of_columns": result.number_of_columns,
        "number_of_drifted_columns": result.number_of_drifted_columns,
        "drifted_features": result.drifted_features,
    }