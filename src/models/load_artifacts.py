import gc
import os
import json
import logging
import threading
from typing import List, Dict, Any, Optional
import lightgbm as lgb
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REGIONS = ["ALL_INDIA", "NR", "WR", "SR", "ER", "NER"]
QUANTILES = [0.10, 0.50, 0.90]
DEFAULT_ANOMALY_FEATURES = [
    "demand_mw",
    "demand_delta_1h",
    "demand_delta_24h",
    "demand_zscore_24h",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_solar_radiation",
    "sin_hour_k1",
    "cos_hour_k1",
]


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
    # loads forecast model feature contract from json
    path = os.path.join(get_artifacts_dir(artifacts_dir), "feature_columns.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"feature columns contract not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_anomaly_feature_columns(artifacts_dir: Optional[str] = None) -> List[str]:
    # loads isolation forest feature contract from json with safe default fallback
    path = os.path.join(get_artifacts_dir(artifacts_dir), "anomaly_feature_columns.json")
    if not os.path.exists(path):
        return list(DEFAULT_ANOMALY_FEATURES)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_quantile_calibration(artifacts_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    # loads conformal quantile regression margins and contract metadata per region
    path = os.path.join(get_artifacts_dir(artifacts_dir), "models", "quantile_calibration.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_calendar_timezone(
    quantile_calibration: Optional[Dict[str, Any]] = None,
    region: Optional[str] = None,
    artifacts_dir: Optional[str] = None,
) -> str:
    # resolves calendar feature timezone from calibration metadata while preserving legacy UTC contracts
    cal_map = quantile_calibration if quantile_calibration is not None else load_quantile_calibration(artifacts_dir)
    if not isinstance(cal_map, dict) or not cal_map:
        return "UTC"
    if region and isinstance(cal_map.get(region), dict):
        tz_val = cal_map[region].get("calendar_timezone")
        if tz_val:
            return str(tz_val)
    for entry in cal_map.values():
        if isinstance(entry, dict) and entry.get("calendar_timezone"):
            return str(entry["calendar_timezone"])
    return "UTC"


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


class LazyDict(dict):
    # thread-safe lazy loader dictionary with optional lru eviction for memory-constrained runtimes
    def __init__(self, loader_fn, max_entries: Optional[int] = None):
        super().__init__()
        self._loader_fn = loader_fn
        self._max_entries = max_entries if (max_entries is None or max_entries > 0) else 1
        self._lru_keys: List[Any] = []
        self._lock = threading.RLock()

    def _touch(self, key: Any) -> None:
        if key in self._lru_keys:
            self._lru_keys.remove(key)
        self._lru_keys.append(key)

    def _evict_if_needed(self) -> bool:
        evicted = False
        if self._max_entries is not None:
            while len(self._lru_keys) >= self._max_entries:
                oldest = self._lru_keys.pop(0)
                if super().__contains__(oldest):
                    super().__delitem__(oldest)
                    evicted = True
        return evicted

    def __getitem__(self, key):
        with self._lock:
            if super().__contains__(key):
                self._touch(key)
                return super().__getitem__(key)
            val = self._loader_fn(key)
            evicted = self._evict_if_needed()
            super().__setitem__(key, val)
            self._touch(key)
            if evicted:
                gc.collect()
            return val

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        return key in REGIONS

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
        except Exception as exc:
            logger.warning(f"lazy artifact load failed for '{key}': {exc}")
            return default


def _resolve_max_cached_regions(max_cached_regions: Optional[int]) -> Optional[int]:
    if max_cached_regions is not None:
        return max(1, int(max_cached_regions))
    raw_env = os.environ.get("MODEL_CACHE_MAX_REGIONS", "").strip()
    if raw_env:
        try:
            val = int(raw_env)
            return max(1, val) if val > 0 else None
        except ValueError:
            logger.warning(f"ignoring invalid MODEL_CACHE_MAX_REGIONS='{raw_env}'")
    return None


class ModelArtifactsBundle:
    # in-memory registry bundle holding deserialized models with optional lru region cap
    def __init__(
        self,
        artifacts_dir: Optional[str] = None,
        max_cached_regions: Optional[int] = None,
    ):
        self.artifacts_dir = get_artifacts_dir(artifacts_dir)
        self.max_cached_regions = _resolve_max_cached_regions(max_cached_regions)
        self.feature_columns = load_feature_columns(self.artifacts_dir)
        self.anomaly_feature_columns = load_anomaly_feature_columns(self.artifacts_dir)
        self.quantile_calibration = load_quantile_calibration(self.artifacts_dir)
        self.model_registry = load_model_registry(self.artifacts_dir)
        self.lgb_models: Dict[str, Dict[float, lgb.Booster]] = LazyDict(
            self._load_region_boosters, max_entries=self.max_cached_regions
        )
        self.isolation_forests: Dict[str, Any] = LazyDict(
            lambda r: load_isolation_forest(r, self.artifacts_dir),
            max_entries=self.max_cached_regions,
        )
        self.reference_baselines: Dict[str, pd.DataFrame] = LazyDict(
            lambda r: load_reference_baseline(r, self.artifacts_dir),
            max_entries=self.max_cached_regions,
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
            _ = self.reference_baselines[region]
        logger.info("all regional model artifacts successfully preloaded")


def validate_artifacts_presence(artifacts_dir: Optional[str] = None) -> bool:
    # checks availability and integrity of all required artifact files
    reg_dir = get_artifacts_dir(artifacts_dir)
    cols = load_feature_columns(reg_dir)
    if not cols:
        return False
    anom_path = os.path.join(reg_dir, "anomaly_feature_columns.json")
    if not os.path.exists(anom_path):
        return False
    cal_path = os.path.join(reg_dir, "models", "quantile_calibration.json")
    if not os.path.exists(cal_path):
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
    return True