import os
import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb

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
    get_artifacts_dir,
    LazyDict,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class AnomalySHAPExplainer:
    # wraps shap treeexplainer with on-demand lazy initialization
    def __init__(self, bundle: Optional[ModelArtifactsBundle] = None):
        if bundle is None:
            self.bundle = ModelArtifactsBundle()
        else:
            self.bundle = bundle

        self.feature_columns = self.bundle.feature_columns
        self.explainers: Dict[str, shap.TreeExplainer] = LazyDict(self._build_tree_explainer)

    def _build_tree_explainer(self, region: str) -> shap.TreeExplainer:
        # constructs path-dependent treeexplainer for target region using median p50 booster
        if region not in REGIONS and region not in self.bundle.model_registry:
            raise KeyError(f"region '{region}' not configured in shap explainer")
        booster = self.bundle.lgb_models[region][0.50]
        # tree path-dependent explainer provides exact shapley values with minimal memory footprint
        return shap.TreeExplainer(booster)

    def _initialize_tree_explainers(self) -> None:
        # eagerly warm up explainers across all regions when explicitly requested
        for region in REGIONS:
            _ = self.explainers[region]

    def get_explainer(self, region: str) -> shap.TreeExplainer:
        # returns or lazily creates treeexplainer for specified region
        return self.explainers[region]

    def explain_anomaly_point(
        self,
        feature_row: pd.Series,
        region: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        # calculates shap feature attributions and formats top-k physical drivers
        explainer = self.get_explainer(region)
        X_point = pd.DataFrame([feature_row[self.feature_columns]])

        shap_vals = explainer.shap_values(X_point)
        if isinstance(shap_vals, list):
            shap_vec = np.array(shap_vals[0]).flatten()
        else:
            shap_vec = np.array(shap_vals).flatten()

        # rank features by absolute attribution magnitude
        feature_attributions = []
        for i, col in enumerate(self.feature_columns):
            phi = float(shap_vec[i])
            act_val = float(feature_row[col])
            feature_attributions.append({
                "feature": col,
                "shap_value": round(phi, 2),
                "actual_value": round(act_val, 2),
                "abs_magnitude": abs(phi),
            })

        feature_attributions.sort(key=lambda item: item["abs_magnitude"], reverse=True)

        top_attributions = [
            {
                "feature": item["feature"],
                "shap_value": item["shap_value"],
                "actual_value": item["actual_value"],
            }
            for item in feature_attributions[:top_k]
        ]
        return top_attributions

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
    # persists verified anomaly events and top shap feature jsonb to telemetry_anomalies table
    if not anomaly_events:
        return 0
    if psycopg2 is None:
        raise ImportError("psycopg2 is required for database persistence. Install psycopg2-binary.")

    rows = [
        (
            ev["timestamp_utc"],
            ev["region"],
            ev["actual_demand_mw"],
            ev.get("predicted_p50_mw"),
            ev["anomaly_score"],
            ev.get("is_structural", True),
            Json(ev["top_shap_features"]),
        )
        for ev in anomaly_events
    ]

    insert_sql = """
        INSERT INTO telemetry_anomalies (
            timestamp_utc,
            region,
            actual_demand_mw,
            predicted_p50_mw,
            anomaly_score,
            is_structural,
            top_shap_features
        ) VALUES %s;
    """

    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, rows)
        conn.commit()
        logger.info(f"successfully persisted {len(rows)} anomaly events to supabase")
        return len(rows)
    finally:
        conn.close()