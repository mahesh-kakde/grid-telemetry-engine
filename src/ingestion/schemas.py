from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


IST_TIMEZONE = timezone(timedelta(hours=5, minutes=30))


def _normalize_timestamp_to_utc(value: object, naive_tz: timezone) -> datetime:
    """Parses ISO-8601 strings or datetimes, interpreting naive values with naive_tz before converting to UTC."""
    if isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        dt = value
    else:
        raise ValueError("Unsupported datetime input format")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=naive_tz)
    return dt.astimezone(timezone.utc)


class GridRegion(str, Enum):
    NR = "NR"
    WR = "WR"
    SR = "SR"
    ER = "ER"
    NER = "NER"
    ALL_INDIA = "ALL_INDIA"


class RawGridRecord(BaseModel):
    timestamp_utc: datetime = Field(..., description="Timestamp normalized to UTC (naive grid inputs assumed IST)")
    region: GridRegion
    demand_mw: float = Field(..., gt=0.0, lt=350_000.0, description="Active demand in MW")
    frequency_hz: Optional[float] = Field(None, ge=47.5, le=52.5, description="Frequency in Hz")

    @field_validator("timestamp_utc", mode="before")
    @classmethod
    def parse_and_normalize_tz(cls, v):
        return _normalize_timestamp_to_utc(v, IST_TIMEZONE)


class RawWeatherRecord(BaseModel):
    timestamp_utc: datetime = Field(..., description="Timestamp normalized to UTC (naive Open-Meteo inputs assumed UTC)")
    region: GridRegion
    temperature_2m: float = Field(..., ge=-20.0, le=60.0)
    relative_humidity_2m: float = Field(..., ge=0.0, le=100.0)
    surface_solar_radiation: float = Field(..., ge=0.0, le=1600.0)
    wind_speed_100m: float = Field(..., ge=0.0, le=150.0)

    @field_validator("timestamp_utc", mode="before")
    @classmethod
    def parse_weather_tz(cls, v):
        return _normalize_timestamp_to_utc(v, timezone.utc)


class TelemetryPayload(BaseModel):
    ingestion_batch_id: str
    grid_records: List[RawGridRecord]
    weather_records: List[RawWeatherRecord]