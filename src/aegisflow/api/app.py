"""FastAPI application factory.

Serves the REST API, the WebSocket alert channel, and the dashboard itself. One process
runs the whole system, which is what makes the "single box on the factory floor, no cloud"
deployment story true rather than aspirational.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from aegisflow import __version__
from aegisflow.api import routes, ws
from aegisflow.core.logging import configure_logging, get_logger
from aegisflow.core.settings import Settings, get_settings
from aegisflow.db.session import dispose_engine, init_db

log = get_logger(__name__)

DESCRIPTION = """
Policy-grounded factory compliance and alert escalation.

* **Module 1** detection engine - YOLOv8n plus colour/geometry analysis of the policy's
  observable indicators
* **Module 2** severity matrix - tiers derived from the policy's own callout keywords and
  hazard-context language, never hard-coded
* **Module 3** escalation - LOW/MEDIUM to the database log, HIGH/CRITICAL to the log **and**
  the `/ws/alerts` WebSocket
* **Module 4** reports - append-only JSON, CSV and PDF audit records
* **Module 5** dashboard - live feed, alert timeline, historical log and export
""".strip()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    configure_logging(settings.log_level, settings.tuning.logging.json_output)
    await init_db(settings)
    log.info(
        "AegisFlow EHS %s ready | provider=%s | db=%s",
        __version__,
        settings.llm_provider.value,
        settings.db_url,
    )
    try:
        yield
    finally:
        await dispose_engine()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application."""
    settings = settings or get_settings()

    app = FastAPI(
        title="AegisFlow EHS",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    # The dashboard is served from the same origin, so CORS is only needed for a separately
    # hosted frontend during development.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    app.include_router(routes.router)
    app.include_router(ws.router)

    _mount_dashboard(app, settings)
    return app


def _mount_dashboard(app: FastAPI, settings: Settings) -> None:
    """Serve the dashboard, if it has been built or shipped."""
    frontend = settings.path("frontend")
    index = frontend / "index.html"

    if not index.exists():

        @app.get("/", include_in_schema=False)
        async def _no_dashboard() -> RedirectResponse:
            return RedirectResponse(url="/docs")

        log.debug("no frontend/index.html; / redirects to /docs")
        return

    assets = frontend / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/", include_in_schema=False)
    async def dashboard() -> FileResponse:
        return FileResponse(index, media_type="text/html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        icon = frontend / "favicon.svg"
        if icon.exists():
            return FileResponse(icon, media_type="image/svg+xml")
        return FileResponse(index, media_type="text/html")

    log.debug("dashboard mounted from %s", frontend)


def serve(settings: Settings | None = None) -> None:
    """Run the app with uvicorn. Entry point for ``aegisflow serve``.

    Uvicorn is pointed at :func:`create_app` in factory mode rather than at a module-level
    instance, so importing this module never builds an app or touches the database -
    ``aegisflow --help`` must not need either.
    """
    import uvicorn

    settings = settings or get_settings()
    uvicorn.run(
        APP_FACTORY,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        factory=True,
    )


APP_FACTORY = "aegisflow.api.app:create_app"
"""Uvicorn import string. Used by ``serve()`` and by ``aegisflow serve``."""
