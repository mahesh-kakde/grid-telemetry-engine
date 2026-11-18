from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class GridRegion(str, Enum):
    NR = "NR"
    WR = "WR"
    SR = "SR"
    ER = "ER"
    NER = "NER"
    ALL_INDIA = "ALL_INDIA"


class RawGridRecord(BaseModel):
    timestamp_utc: datetime = Field(..., description="Timestamp normalized to UTC")
    region: GridRegion
    demand_mw: float = Field(..., gt=0.0, lt=350_000.0, description="Active demand in MW")
    frequency_hz: Optional[float] = Field(None, ge=47.5, le=52.5, description="Frequency in Hz")

    @field_validator("timestamp_utc", mode="before")
    def parse_and_normalize_tz(cls, v):
        if isinstance(v, str):
            # handle iso format and zulu timezone strings
            clean_str = v.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_str)
        elif isinstance(v, datetime):
            dt = v
        else:
            raise ValueError("Unsupported datetime input format")

        if dt.tzinfo is None:
            # coerce unlocalized ist timestamps to utc
            dt = dt.replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
        return dt.astimezone(timezone.utc)


class RawWeatherRecord(BaseModel):
    timestamp_utc: datetime
    region: GridRegion
    temperature_2m: float = Field(..., ge=-20.0, le=60.0)
    relative_humidity_2m: float = Field(..., ge=0.0, le=100.0)
    surface_solar_radiation: float = Field(..., ge=0.0, le=1600.0)
    wind_speed_100m: float = Field(..., ge=0.0, le=150.0)

    @field_validator("timestamp_utc", mode="before")
    def parse_weather_tz(cls, v):
        if isinstance(v, str):
            clean_str = v.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_str)
        elif isinstance(v, datetime):
            dt = v
        else:
            raise ValueError("Unsupported datetime input format")

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)


class TelemetryPayload(BaseModel):
    ingestion_batch_id: str
    grid_records: List[RawGridRecord]
    weather_records: List[RawWeatherRecord]