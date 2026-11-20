import os
import json
import logging
from typing import List, Dict, Any, Optional
import lightgbm as lgb
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REGIONS = ["ALL_INDIA", "NR", "WR", "SR", "ER", "NER"]
QUANTILES = [0.10, 0.50, 0.90]


def get_artifacts_dir(custom_path: Optional[str] = None) -> str:
    # resolves artifacts root directory from argument, environment, or default root
    if custom_path:
        return custom_path
    env_path = os.environ.get("ARTIFACTS_PATH")
    if env_path:
        return env_path
    # default to artifacts folder relative to project root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_dir = os.path.join(base_dir, "artifacts")
    return default_dir


def load_feature_columns(artifacts_dir: Optional[str] = None) -> List[str]:
    # loads model feature contract from json
    path = os.path.join(get_artifacts_dir(artifacts_dir), "feature_columns.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"feature columns contract not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_registry(artifacts_dir: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    # loads region-to-booster mapping metadata
    path = os.path.join(get_artifacts_dir(artifacts_dir), "models", "model_registry.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"model registry not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_lgb_booster(
    region: str,
    quantile: float,
    artifacts_dir: Optional[str] = None,
) -> lgb.Booster:
    # loads LightGBM quantile regression booster for specified region and quantile
    q_key = str(quantile)
    reg_dir = get_artifacts_dir(artifacts_dir)
    registry = load_model_registry(reg_dir)

    if region not in registry:
        raise KeyError(f"region '{region}' not present in model registry")
    if q_key not in registry[region]:
        raise KeyError(f"quantile '{q_key}' not present for region '{region}' in registry")

    model_filename = registry[region][q_key]
    model_path = os.path.join(reg_dir, "models", model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"booster file not found: {model_path}")

    booster = lgb.Booster(model_file=model_path)
    return booster


def load_isolation_forest(
    region: str,
    artifacts_dir: Optional[str] = None,
) -> Any:
    # deserializes calibrated isolation forest model for specified region
    reg_dir = get_artifacts_dir(artifacts_dir)
    model_path = os.path.join(reg_dir, "models", f"isolation_forest_{region}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"isolation forest model not found: {model_path}")
    return joblib.load(model_path)


def load_reference_baseline(
    region: str,
    artifacts_dir: Optional[str] = None,
) -> pd.DataFrame:
    # loads 30-day reference baseline parquet dataframe for drift monitoring
    reg_dir = get_artifacts_dir(artifacts_dir)
    parquet_path = os.path.join(reg_dir, "baselines", f"reference_baseline_{region}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"reference baseline parquet not found: {parquet_path}")
    return pd.read_parquet(parquet_path)


def load_shap_background(
    region: str,
    artifacts_dir: Optional[str] = None,
) -> Any:
    # deserializes pre-computed 150 k-means background cluster centers
    reg_dir = get_artifacts_dir(artifacts_dir)
    shap_path = os.path.join(reg_dir, "baselines", f"shap_background_{region}.joblib")
    if not os.path.exists(shap_path):
        raise FileNotFoundError(f"shap background centroid file not found: {shap_path}")
    return joblib.load(shap_path)


class LazyDict(dict):
    # dictionary with on-demand lazy loader callback for missing keys
    def __init__(self, loader_fn):
        super().__init__()
        self._loader_fn = loader_fn

    def __getitem__(self, key):
        if not super().__contains__(key):
            val = self._loader_fn(key)
            self[key] = val
            return val
        return super().__getitem__(key)

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        return key in REGIONS

    def get(self, key, default=None):
        if not super().__contains__(key):
            try:
                val = self._loader_fn(key)
                self[key] = val
                return val
            except Exception:
                return default
        return super().get(key, default)


class ModelArtifactsBundle:
    # in-memory registry bundle holding deserialized models with lazy on-demand loading
    def __init__(self, artifacts_dir: Optional[str] = None):
        self.artifacts_dir = get_artifacts_dir(artifacts_dir)
        self.feature_columns = load_feature_columns(self.artifacts_dir)
        self.model_registry = load_model_registry(self.artifacts_dir)
        self.lgb_models: Dict[str, Dict[float, lgb.Booster]] = LazyDict(self._load_region_boosters)
        self.isolation_forests: Dict[str, Any] = LazyDict(
            lambda r: load_isolation_forest(r, self.artifacts_dir)
        )
        self.shap_backgrounds: Dict[str, Any] = LazyDict(
            lambda r: load_shap_background(r, self.artifacts_dir)
        )
        self.reference_baselines: Dict[str, pd.DataFrame] = LazyDict(
            lambda r: load_reference_baseline(r, self.artifacts_dir)
        )

    def _load_region_boosters(self, region: str) -> Dict[float, lgb.Booster]:
        # deserializes lightgbm boosters for target region
        return {
            q: load_lgb_booster(region, q, self.artifacts_dir)
            for q in QUANTILES
        }

    def preload_all(self) -> None:
        # loads all boosters, isolation forests, and baselines into memory
        logger.info("preloading all regional model artifacts into memory...")
        for region in REGIONS:
            _ = self.lgb_models[region]
            _ = self.isolation_forests[region]
            _ = self.shap_backgrounds[region]
            _ = self.reference_baselines[region]
        logger.info("all regional model artifacts successfully preloaded")


def validate_artifacts_presence(artifacts_dir: Optional[str] = None) -> bool:
    # checks availability and integrity of all required artifact files
    reg_dir = get_artifacts_dir(artifacts_dir)
    cols = load_feature_columns(reg_dir)
    if not cols:
        return False
    registry = load_model_registry(reg_dir)
    for region in REGIONS:
        if region not in registry:
            return False
        for q in ["0.1", "0.5", "0.9"]:
            if q not in registry[region]:
                return False
            m_path = os.path.join(reg_dir, "models", registry[region][q])
            if not os.path.exists(m_path):
                return False
        if not os.path.exists(os.path.join(reg_dir, "models", f"isolation_forest_{region}.joblib")):
            return False
        if not os.path.exists(os.path.join(reg_dir, "baselines", f"reference_baseline_{region}.parquet")):
            return False
        if not os.path.exists(os.path.join(reg_dir, "baselines", f"shap_background_{region}.joblib")):
            return False
    return True