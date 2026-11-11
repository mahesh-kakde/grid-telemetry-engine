import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest

from src.features.pipeline import FeaturePipeline


class AnomalyDetector:
    def __init__(
        self,
        contamination: float = 0.02,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_names: list[str] = []
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> "AnomalyDetector":
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        self.model.fit(X)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("anomaly detector is not fitted")

        preds = self.model.predict(X)
        raw_decisions = self.model.decision_function(X)

        # map -1 to True (anomaly) and 1 to False (normal)
        is_anomaly = preds == -1

        # normalize decision function into continuous anomaly score in [0.0, 1.0]
        # raw decisions: negative for outliers, positive for inliers
        anomaly_scores = np.clip(0.5 - raw_decisions, 0.0, 1.0)

        return is_anomaly, np.round(anomaly_scores, 4)

    def save(self, dir_path: Path | str) -> Path:
        out_dir = Path(dir_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out_dir / "isolation_forest.joblib")

        meta = {
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }
        with open(out_dir / "anomaly_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return out_dir

    @classmethod
    def load(cls, dir_path: Path | str) -> "AnomalyDetector":
        in_dir = Path(dir_path)
        with open(in_dir / "anomaly_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        detector = cls(
            contamination=meta["contamination"],
            n_estimators=meta["n_estimators"],
        )
        detector.feature_names = meta["feature_names"]
        detector.is_fitted = meta["is_fitted"]
        detector.model = joblib.load(in_dir / "isolation_forest.joblib")

        return detector