from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import anomalies, drift, forecasts, telemetry
from src.config import Settings, get_settings
from src.db.client import DatabaseClient, get_db_client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    if not hasattr(app.state, "settings") or app.state.settings is None:
        app.state.settings = get_settings()
    if not hasattr(app.state, "db") or app.state.db is None:
        app.state.db = get_db_client(app.state.settings)
    yield


def create_app(settings: Settings | None = None, db_client: DatabaseClient | None = None) -> FastAPI:
    app = FastAPI(
        title="Grid Telemetry Engine API",
        description="REST backend serving 72-hour power grid forecasts, anomaly attributions, and drift monitoring.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    if settings is not None:
        app.state.settings = settings
    if db_client is not None:
        app.state.db = db_client

    # configure cors for web access and personal domain
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    # mount api routers
    app.include_router(telemetry.router)
    app.include_router(forecasts.router)
    app.include_router(anomalies.router)
    app.include_router(drift.router)

    @app.get("/", tags=["system"])
    def root() -> dict[str, Any]:
        return {
            "app": "Grid Telemetry Engine API",
            "version": "1.0.0",
            "docs": "/docs",
            "domain": "https://grid.maheshkakde.com",
            "status": "healthy",
        }

    @app.get("/health", tags=["system"])
    def health(request: Request) -> dict[str, Any]:
        db = getattr(request.app.state, "db", None)
        mode = db.mode if db else "unknown"
        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "database_mode": mode,
        }

    return app


app = create_app()