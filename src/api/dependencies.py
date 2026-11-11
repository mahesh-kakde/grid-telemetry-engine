from fastapi import HTTPException, Request, status
from src.config import Settings, get_settings
from src.db.client import DatabaseClient, get_db_client
from src.schemas.telemetry import RegionEnum

VALID_REGIONS = {r.value for r in RegionEnum}


def get_db(request: Request) -> DatabaseClient:
    db = getattr(request.app.state, "db", None)
    if db is None:
        settings = getattr(request.app.state, "settings", None) or get_settings()
        db = get_db_client(settings)
        request.app.state.db = db
    return db


def validate_region_param(region: str) -> str:
    cleaned = region.strip().upper()
    if cleaned == "NATIONAL":
        cleaned = "ALL"
    if cleaned not in VALID_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"invalid region '{region}'; must be one of {sorted(VALID_REGIONS | {'NATIONAL'})}",
        )
    return cleaned