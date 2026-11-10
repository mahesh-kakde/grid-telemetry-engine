from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RegionEnum(str, Enum):
    NR = "NR"
    WR = "WR"
    SR = "SR"
    ER = "ER"
    NER = "NER"
    ALL = "ALL"


def _normalize_datetime(value: datetime) -> datetime:
    # ensure datetimes are timezone aware in utc
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class TelemetryRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime
    region: str
    metric_value: float
    metric_type: str = "DEMAND_MW"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)

    @field_validator("timestamp", "created_at", mode="before")
    @classmethod
    def parse_datetime(cls, value: Any) -> datetime:
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if isinstance(value, datetime):
            return _normalize_datetime(value)
        raise ValueError("invalid datetime format")

    @field_validator("region")
    @classmethod
    def validate_region(cls, value: str) -> str:
        cleaned = value.strip().upper()
        valid_regions = {r.value for r in RegionEnum}
        if cleaned not in valid_regions:
            raise ValueError(f"invalid region: {value}; must be one of {sorted(valid_regions)}")
        return cleaned

    @field_validator("metric_value")
    @classmethod
    def validate_metric_value(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("metric_value must be strictly positive")
        return round(float(value), 3)


class ForecastRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    forecast_origin: datetime
    target_timestamp: datetime
    region: str
    p10: float
    p50: float
    p90: float
    model_version: str = "v1.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)

    @field_validator("forecast_origin", "target_timestamp", "created_at", mode="before")
    @classmethod
    def parse_datetime(cls, value: Any) -> datetime:
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if isinstance(value, datetime):
            return _normalize_datetime(value)
        raise ValueError("invalid datetime format")

    @field_validator("region")
    @classmethod
    def validate_region(cls, value: str) -> str:
        cleaned = value.strip().upper()
        valid_regions = {r.value for r in RegionEnum}
        if cleaned not in valid_regions:
            raise ValueError(f"invalid region: {value}; must be one of {sorted(valid_regions)}")
        return cleaned

    @model_validator(mode="after")
    def validate_forecast_bounds(self) -> "ForecastRecord":
        if self.target_timestamp <= self.forecast_origin:
            raise ValueError("target_timestamp must be strictly after forecast_origin")
        if self.p10 < 0 or self.p50 < 0 or self.p90 < 0:
            raise ValueError("quantile predictions must be non-negative")
        if not (self.p10 <= self.p50 <= self.p90):
            raise ValueError(f"quantile crossing detected: p10={self.p10}, p50={self.p50}, p90={self.p90}")
        return self


class AnomalyRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime
    region: str
    metric_value: float
    anomaly_score: float
    is_anomaly: bool
    top_shap_features: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)

    @field_validator("timestamp", "created_at", mode="before")
    @classmethod
    def parse_datetime(cls, value: Any) -> datetime:
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if isinstance(value, datetime):
            return _normalize_datetime(value)
        raise ValueError("invalid datetime format")

    @field_validator("region")
    @classmethod
    def validate_region(cls, value: str) -> str:
        cleaned = value.strip().upper()
        valid_regions = {r.value for r in RegionEnum}
        if cleaned not in valid_regions:
            raise ValueError(f"invalid region: {value}; must be one of {sorted(valid_regions)}")
        return cleaned


class DriftRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reference_window_start: datetime
    reference_window_end: datetime
    current_window_start: datetime
    current_window_end: datetime
    drift_share: float = Field(ge=0.0, le=1.0)
    drifted_features: dict[str, Any] = Field(default_factory=dict)
    report_url: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)

    @field_validator(
        "run_timestamp",
        "reference_window_start",
        "reference_window_end",
        "current_window_start",
        "current_window_end",
        "created_at",
        mode="before",
    )
    @classmethod
    def parse_datetime(cls, value: Any) -> datetime:
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if isinstance(value, datetime):
            return _normalize_datetime(value)
        raise ValueError("invalid datetime format")

    @model_validator(mode="after")
    def validate_windows(self) -> "DriftRecord":
        if self.reference_window_end < self.reference_window_start:
            raise ValueError("reference_window_end cannot precede reference_window_start")
        if self.current_window_end < self.current_window_start:
            raise ValueError("current_window_end cannot precede current_window_start")
        return self