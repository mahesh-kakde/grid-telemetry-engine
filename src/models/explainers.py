import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import shap

try:
    import psycopg2
    from psycopg2.extras import execute_values, Json
except ImportError:
    psycopg2 = None
    execute_values = None
    Json = None

from src.models.load_artifacts import (
    ModelArtifactsBundle,
    REGIONS,
    LazyDict,
)

logger = logging.getLogger(__name__)


class AnomalySHAPExplainer:
    # wraps path-dependent treeexplainers for both p50 demand and isolation forest anomaly models
    def __init__(self, bundle: Optional[ModelArtifactsBundle] = None):
        if bundle is None:
            self.bundle = ModelArtifactsBundle()
        else:
            self.bundle = bundle

        self.feature_columns = self.bundle.feature_columns
        self.anomaly_feature_columns = getattr(
            self.bundle, "anomaly_feature_columns", self.bundle.feature_columns
        )
        max_cached = getattr(self.bundle, "max_cached_regions", None)
        self.explainers: Dict[str, shap.TreeExplainer] = LazyDict(
            self._build_tree_explainer, max_entries=max_cached
        )
        self.anomaly_explainers: Dict[str, shap.TreeExplainer] = LazyDict(
            self._build_isolation_explainer, max_entries=max_cached
        )

    def _build_tree_explainer(self, region: str) -> shap.TreeExplainer:
        # constructs path-dependent treeexplainer for target region using median p50 booster
        if region not in REGIONS and region not in self.bundle.model_registry:
            raise KeyError(f"region '{region}' not configured in shap explainer")
        booster = self.bundle.lgb_models[region][0.50]
        # tree path-dependent explainer provides exact shapley values with minimal memory footprint
        return shap.TreeExplainer(booster)

    def _build_isolation_explainer(self, region: str) -> shap.TreeExplainer:
        # constructs treeexplainer for observation-aware isolation forest sentinel
        if region not in REGIONS and region not in self.bundle.isolation_forests:
            raise KeyError(f"region '{region}' not configured in isolation shap explainer")
        iso_model = self.bundle.isolation_forests[region]
        return shap.TreeExplainer(iso_model)

    def _initialize_tree_explainers(self) -> None:
        # eagerly warm up explainers across all regions when explicitly requested
        for region in REGIONS:
            _ = self.explainers[region]
            _ = self.anomaly_explainers[region]

    def get_explainer(self, region: str) -> shap.TreeExplainer:
        # returns or lazily creates treeexplainer for specified region
        return self.explainers[region]

    @staticmethod
    def _rank_top_attributions(
        cols: List[str],
        shap_vec: np.ndarray,
        actual_values: List[float],
        top_k: int = 3,
        precision: int = 2,
    ) -> List[Dict[str, Any]]:
        feature_attributions = []
        for i, col in enumerate(cols):
            phi = float(shap_vec[i])
            act_val = float(actual_values[i])
            feature_attributions.append({
                "feature": col,
                "shap_value": round(phi, precision),
                "actual_value": round(act_val, 2),
                "abs_magnitude": abs(phi),
            })

        feature_attributions.sort(key=lambda item: item["abs_magnitude"], reverse=True)
        return [
            {
                "feature": item["feature"],
                "shap_value": item["shap_value"],
                "actual_value": item["actual_value"],
            }
            for item in feature_attributions[:top_k]
        ]

    def explain_anomaly_point(
        self,
        feature_row: pd.Series,
        region: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        # calculates shap feature attributions for expected p50 demand baseline (mw)
        explainer = self.get_explainer(region)
        X_point = pd.DataFrame([feature_row[self.feature_columns]]).astype(float)

        shap_vals = explainer.shap_values(X_point)
        if isinstance(shap_vals, list):
            shap_vec = np.array(shap_vals[0]).flatten()
        else:
            shap_vec = np.array(shap_vals).flatten()

        actuals = [float(feature_row[col]) for col in self.feature_columns]
        return self._rank_top_attributions(
            self.feature_columns, shap_vec, actuals, top_k=top_k, precision=2
        )

    def explain_anomaly_score(
        self,
        feature_row: pd.Series,
        region: str,
        actual_demand_mw: Optional[float] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        # calculates observation-aware isolation forest shap drivers (+ = pushes toward anomaly)
        from src.models.infer_and_detect import _build_anomaly_feature_matrix

        actual_val = float(actual_demand_mw) if actual_demand_mw is not None else float(feature_row["demand_mw"])
        X_base = pd.DataFrame([feature_row])
        X_anom = _build_anomaly_feature_matrix(
            X_base, np.array([actual_val]), self.anomaly_feature_columns
        )

        explainer = self.anomaly_explainers[region]
        shap_vals = explainer.shap_values(X_anom)
        if isinstance(shap_vals, list):
            depth_vec = np.array(shap_vals[0]).flatten()
        else:
            depth_vec = np.array(shap_vals).flatten()

        # negate isolation depth shap so positive values indicate increased anomaly surprise
        risk_vec = -depth_vec
        actuals = [float(X_anom.iloc[0][col]) for col in self.anomaly_feature_columns]
        return self._rank_top_attributions(
            self.anomaly_feature_columns, risk_vec, actuals, top_k=top_k, precision=3
        )

    def explain_anomalies_batch(
        self,
        df_anomalies: pd.DataFrame,
        region: str,
        top_k: int = 3,
    ) -> List[List[Dict[str, Any]]]:
        # computes top-k attributions for multiple anomaly records
        results = []
        for _, row in df_anomalies.iterrows():
            top_attrs = self.explain_anomaly_point(row, region, top_k=top_k)
            results.append(top_attrs)
        return results


def persist_anomaly_events_to_supabase(
    anomaly_events: List[Dict[str, Any]],
    db_url: str,
) -> int:
    # idempotent upsert of anomaly events and shap attributions into telemetry_anomalies
    if not anomaly_events:
        return 0
    if psycopg2 is None:
        raise ImportError("psycopg2 is required for database persistence. Install psycopg2-binary.")

    rows = [
        (
            ev["timestamp_utc"],
            ev["region"],
            ev["actual_demand_mw"],
            ev.get("predicted_p10_mw"),
            ev.get("predicted_p50_mw"),
            ev.get("predicted_p90_mw"),
            ev["anomaly_score"],
            bool(ev.get("is_structural", False)),
            bool(ev.get("residual_breach", False)),
            Json(ev["top_shap_features"]),
        )
        for ev in anomaly_events
    ]

    upsert_sql = """
        INSERT INTO telemetry_anomalies (
            timestamp_utc,
            region,
            actual_demand_mw,
            predicted_p10_mw,
            predicted_p50_mw,
            predicted_p90_mw,
            anomaly_score,
            is_structural,
            residual_breach,
            top_shap_features
        ) VALUES %s
        ON CONFLICT (region, timestamp_utc) DO UPDATE SET
            actual_demand_mw = EXCLUDED.actual_demand_mw,
            predicted_p10_mw = EXCLUDED.predicted_p10_mw,
            predicted_p50_mw = EXCLUDED.predicted_p50_mw,
            predicted_p90_mw = EXCLUDED.predicted_p90_mw,
            anomaly_score = EXCLUDED.anomaly_score,
            is_structural = EXCLUDED.is_structural,
            residual_breach = EXCLUDED.residual_breach,
            top_shap_features = EXCLUDED.top_shap_features;
    """

    conn = psycopg2.connect(db_url, connect_timeout=10)
    try:
        with conn.cursor() as cur:
            execute_values(cur, upsert_sql, rows)
        conn.commit()
        logger.info(f"successfully upserted {len(rows)} anomaly events to supabase")
        return len(rows)
    finally:
        conn.close()