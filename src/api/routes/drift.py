from pathlib import Path
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from src.api.dependencies import get_db
from src.db.client import DatabaseClient

router = APIRouter(prefix="/api/drift", tags=["drift"])


@router.get("/status")
def get_drift_status(
    db: DatabaseClient = Depends(get_db),
) -> dict[str, Any]:
    latest = db.get_latest_drift()
    if latest is None:
        return {
            "status": "no_drift_recorded",
            "message": "no evidently drift monitoring evaluations recorded yet",
        }
    return {
        "status": "ok",
        "data": latest,
    }


@router.get("/report")
def get_drift_report_html(
    report_path: str | None = Query(default=None, description="optional explicit path to html report"),
    db: DatabaseClient = Depends(get_db),
) -> FileResponse:
    target_file: Path | None = None

    if report_path:
        p = Path(report_path).resolve()
        if p.exists() and p.is_file():
            target_file = p
    else:
        latest = db.get_latest_drift()
        if latest and latest.get("report_url"):
            candidate = Path(latest["report_url"]).resolve()
            if candidate.exists() and candidate.is_file():
                target_file = candidate

        # fallback: search reports/drift directory for the most recent html report
        if target_file is None:
            reports_dir = Path("reports/drift").resolve()
            if reports_dir.exists():
                html_files = sorted(reports_dir.glob("*.html"), key=lambda f: f.stat().st_mtime, reverse=True)
                if html_files:
                    target_file = html_files[0]

    if target_file is None or not target_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no evidently drift report html found on disk",
        )

    return FileResponse(
        path=target_file,
        media_type="text/html",
        filename=target_file.name,
    )