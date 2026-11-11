from typing import Any
from fastapi import APIRouter, Depends, Query
from src.api.dependencies import get_db, validate_region_param
from src.db.client import DatabaseClient

router = APIRouter(prefix="/api/anomalies", tags=["anomalies"])


@router.get("/recent")
def get_recent_anomalies(
    region: str = Query(default="ALL", description="grid region identifier"),
    limit: int = Query(default=50, ge=1, le=200, description="number of records to retrieve"),
    anomalies_only: bool = Query(default=False, description="filter exclusively for flagged anomalies"),
    db: DatabaseClient = Depends(get_db),
) -> list[dict[str, Any]]:
    cleaned_region = validate_region_param(region)
    records = db.get_recent_anomalies(region=cleaned_region, limit=limit)
    if anomalies_only:
        records = [r for r in records if r.get("is_anomaly")]
    return records