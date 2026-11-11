from datetime import datetime
from typing import Any
from fastapi import APIRouter, Depends, Query
from src.api.dependencies import get_db, validate_region_param
from src.db.client import DatabaseClient

router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])


@router.get("/recent")
def get_recent_telemetry(
    region: str = Query(default="ALL", description="grid region identifier"),
    limit: int = Query(default=168, ge=1, le=1000, description="number of recent records"),
    db: DatabaseClient = Depends(get_db),
) -> list[dict[str, Any]]:
    cleaned_region = validate_region_param(region)
    return db.get_recent_telemetry(region=cleaned_region, limit=limit)


@router.get("/historical")
def get_historical_telemetry(
    region: str = Query(default="ALL", description="grid region identifier"),
    start_time: datetime | None = Query(default=None, description="iso start timestamp"),
    end_time: datetime | None = Query(default=None, description="iso end timestamp"),
    limit: int = Query(default=720, ge=1, le=5000, description="maximum records to retrieve"),
    db: DatabaseClient = Depends(get_db),
) -> list[dict[str, Any]]:
    cleaned_region = validate_region_param(region)
    return db.get_historical_telemetry(
        region=cleaned_region,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )