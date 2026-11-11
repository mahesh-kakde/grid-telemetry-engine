import warnings
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
import polars as pl

try:
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset
except ImportError:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset


@dataclass
class DriftSuiteResult:
    drift_share: float
    dataset_drift: bool
    number_of_columns: int
    number_of_drifted_columns: int
    drifted_features: dict[str, Any]
    report: Report


class DriftSuite:
    def __init__(
        self,
        drift_share_threshold: float = 0.5,
        stattest_threshold: float = 0.05,
    ) -> None:
        self.drift_share_threshold = float(drift_share_threshold)
        self.stattest_threshold = float(stattest_threshold)

    def _prepare_dataframe(
        self,
        data: pl.DataFrame | pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if isinstance(data, pl.DataFrame):
            df = data.to_pandas()
        else:
            df = data.copy()

        # filter columns to numeric features if not explicitly specified
        if feature_columns is not None:
            available = [c for c in feature_columns if c in df.columns]
            df = df[available]
        else:
            excluded = {"timestamp", "region", "id", "created_at"}
            numeric_cols = [
                c
                for c in df.columns
                if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
            ]
            df = df[numeric_cols]

        return df

    def run(
        self,
        reference_data: pl.DataFrame | pd.DataFrame,
        current_data: pl.DataFrame | pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> DriftSuiteResult:
        ref_df = self._prepare_dataframe(reference_data, feature_columns=feature_columns)
        curr_df = self._prepare_dataframe(current_data, feature_columns=feature_columns)

        if ref_df.empty or curr_df.empty:
            raise ValueError("reference and current dataframes must contain at least one column and row")

        common_cols = [c for c in ref_df.columns if c in curr_df.columns]
        if not common_cols:
            raise ValueError("no overlapping feature columns between reference and current data")

        ref_df = ref_df[common_cols]
        curr_df = curr_df[common_cols]

        # identify and isolate degenerate constant features to prevent histogram binning errors
        constant_features: dict[str, Any] = {}
        varying_cols: list[str] = []

        for c in common_cols:
            r_min, r_max = float(ref_df[c].min()), float(ref_df[c].max())
            c_min, c_max = float(curr_df[c].min()), float(curr_df[c].max())
            r_const = np.isclose(r_min, r_max)
            c_const = np.isclose(c_min, c_max)

            if r_const and c_const:
                is_same = np.isclose(r_min, c_min)
                constant_features[c] = {
                    "drift_detected": not is_same,
                    "drift_score": 1.0 if is_same else 0.0,
                    "stattest_name": "constant_value",
                    "stattest_threshold": self.stattest_threshold,
                    "column_type": "num",
                }
            elif r_const or c_const:
                constant_features[c] = {
                    "drift_detected": True,
                    "drift_score": 0.0,
                    "stattest_name": "constant_value",
                    "stattest_threshold": self.stattest_threshold,
                    "column_type": "num",
                }
            else:
                varying_cols.append(c)

        preset = DataDriftPreset(
            drift_share=self.drift_share_threshold,
            stattest_threshold=self.stattest_threshold,
        )
        report = Report(metrics=[preset])

        drifted_features: dict[str, Any] = {}

        if varying_cols:
            # suppress benign asymptotic ks test runtime warnings from scipy
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                report.run(reference_data=ref_df[varying_cols], current_data=curr_df[varying_cols])

            raw_dict = report.as_dict()
            metrics = raw_dict.get("metrics", [])

            for metric in metrics:
                m_name = metric.get("metric")
                result = metric.get("result", {})
                if m_name == "DataDriftTable":
                    drift_by_cols = result.get("drift_by_columns", {})
                    for col_name, col_data in drift_by_cols.items():
                        raw_score = col_data.get("drift_score", 0.0)
                        if isinstance(raw_score, np.generic):
                            score_val = float(raw_score.item())
                        else:
                            score_val = float(raw_score)

                        drifted_features[col_name] = {
                            "drift_detected": bool(col_data.get("drift_detected", False)),
                            "drift_score": score_val,
                            "stattest_name": str(col_data.get("stattest_name", "")),
                            "stattest_threshold": float(col_data.get("stattest_threshold", self.stattest_threshold)),
                            "column_type": str(col_data.get("column_type", "num")),
                        }
        else:
            # if all columns are constant, run report on a single dummy varying column for html generation
            dummy_ref = pd.DataFrame({"baseline": [1.0, 2.0, 3.0, 4.0, 5.0]})
            dummy_curr = pd.DataFrame({"baseline": [1.0, 2.0, 3.0, 4.0, 5.0]})
            report.run(reference_data=dummy_ref, current_data=dummy_curr)

        # merge constant features with varying features
        drifted_features.update(constant_features)

        number_of_columns = len(common_cols)
        number_of_drifted_columns = sum(
            1 for f in drifted_features.values() if f.get("drift_detected")
        )
        drift_share = float(number_of_drifted_columns / max(number_of_columns, 1))
        dataset_drift = drift_share >= self.drift_share_threshold

        return DriftSuiteResult(
            drift_share=drift_share,
            dataset_drift=dataset_drift,
            number_of_columns=number_of_columns,
            number_of_drifted_columns=number_of_drifted_columns,
            drifted_features=drifted_features,
            report=report,
        )