import json
from pathlib import Path
from typing import Any
import joblib
import lightgbm as lgb
import numpy as np


class QuantileRegressor:
    def __init__(
        self,
        alphas: list[float] | None = None,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.alphas = alphas or [0.10, 0.50, 0.90]
        self.params = {
            "objective": "quantile",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "verbosity": -1,
            "n_jobs": -1,
        }
        self.models: dict[float, lgb.LGBMRegressor] = {}
        self.feature_names: list[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "QuantileRegressor":
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        for alpha in self.alphas:
            model = lgb.LGBMRegressor(alpha=alpha, **self.params)
            model.fit(X, y)
            self.models[alpha] = model

        return self

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if not self.models:
            raise RuntimeError("models have not been fitted yet")

        raw_preds = {alpha: self.models[alpha].predict(X) for alpha in self.alphas}

        # extract standard quantile arrays
        p10 = raw_preds.get(0.10, np.zeros(X.shape[0]))
        p50 = raw_preds.get(0.50, np.zeros(X.shape[0]))
        p90 = raw_preds.get(0.90, np.zeros(X.shape[0]))

        # enforce non-negativity and monotonic quantile order p10 <= p50 <= p90
        p10_adj = np.maximum(0.0, p10)
        p50_adj = np.maximum(p10_adj, p50)
        p90_adj = np.maximum(p50_adj, p90)

        return {
            "p10": np.round(p10_adj, 2),
            "p50": np.round(p50_adj, 2),
            "p90": np.round(p90_adj, 2),
        }

    def save(self, dir_path: Path | str) -> Path:
        out_dir = Path(dir_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        for alpha, model in self.models.items():
            alpha_tag = int(alpha * 100)
            joblib.dump(model, out_dir / f"lgb_q{alpha_tag}.joblib")

        meta = {
            "alphas": self.alphas,
            "params": self.params,
            "feature_names": self.feature_names,
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return out_dir

    @classmethod
    def load(cls, dir_path: Path | str) -> "QuantileRegressor":
        in_dir = Path(dir_path)
        with open(in_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        regressor = cls(alphas=meta["alphas"])
        regressor.params = meta["params"]
        regressor.feature_names = meta["feature_names"]

        for alpha in regressor.alphas:
            alpha_tag = int(alpha * 100)
            model_file = in_dir / f"lgb_q{alpha_tag}.joblib"
            regressor.models[alpha] = joblib.load(model_file)

        return regressor