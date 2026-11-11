from typing import Any
import numpy as np
import shap
from sklearn.ensemble import IsolationForest


class ExplainabilityEngine:
    def __init__(
        self,
        model: IsolationForest,
        feature_names: list[str],
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        # initialize tree explainer for tree-based anomaly models
        self.explainer = shap.TreeExplainer(model)

    def explain_instance(self, x: np.ndarray, top_k: int = 3) -> dict[str, float]:
        sample = x.reshape(1, -1)
        raw_shap = self.explainer.shap_values(sample)

        # handle 2d or list outputs from tree explainer
        vals = np.array(raw_shap).reshape(-1)
        top_indices = np.argsort(np.abs(vals))[::-1][:top_k]

        return {
            self.feature_names[idx]: round(float(vals[idx]), 4)
            for idx in top_indices
        }

    def batch_explain(
        self,
        X: np.ndarray,
        anomaly_mask: np.ndarray,
        top_k: int = 3,
    ) -> list[dict[str, float]]:
        # compute shap values solely for flagged anomalous timestamps
        explanations: list[dict[str, float]] = []

        for i in range(X.shape[0]):
            if not anomaly_mask[i]:
                explanations.append({})
            else:
                top_features = self.explain_instance(X[i], top_k=top_k)
                explanations.append(top_features)

        return explanations