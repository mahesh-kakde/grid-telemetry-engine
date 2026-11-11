from typing import Any
from fastapi import APIRouter, Depends, Query
from src.api.dependencies import get_db, validate_region_param
from src.db.client import DatabaseClient

router = APIRouter(prefix="/api/forecasts", tags=["forecasts"])


@router.get("/latest")
def get_latest_forecasts(
    region: str = Query(default="ALL", description="grid region identifier"),
    limit: int = Query(default=72, ge=1, le=168, description="forecast horizon steps"),
    db: DatabaseClient = Depends(get_db),
) -> list[dict[str, Any]]:
    cleaned_region = validate_region_param(region)
    return db.get_latest_forecasts(region=cleaned_region, limit=limit)