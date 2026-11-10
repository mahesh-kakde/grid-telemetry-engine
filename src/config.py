from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    supabase_url: str | None = None
    supabase_key: str | None = None
    telemetry_api_key: str | None = None
    telemetry_region: str = "NR"
    sqlite_db_path: Path = Path("data/local_telemetry.db")
    model_store_path: Path = Path("./models")
    evidently_report_dir: Path = Path("./reports/drift")
    batch_interval_hours: int = 6
    forecast_horizon_hours: int = 72

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def has_supabase_credentials(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()