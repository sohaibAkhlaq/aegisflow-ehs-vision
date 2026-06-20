"""
main.py — FastAPI application entry point.

Mounts all five modules into a single unified API + dashboard server:

  GET  /                        → serve dashboard SPA
  GET  /api/health              → health check
  POST /api/process             → trigger video processing pipeline
  GET  /api/events              → query violation events (with filters)
  GET  /api/events/stats        → summary statistics
  GET  /api/export/csv          → download filtered CSV
  GET  /api/export/json         → download filtered JSON
  GET  /api/rules               → return parsed compliance rules
  WS   /ws                      → WebSocket stream for live alerts
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import settings, BEHAVIOR_CLASSES, SEVERITY_COLORS
from .detection.policy_parser import load_compliance_rules
from .detection.detector import ComplianceDetector
from .detection.video_processor import VideoProcessor, process_all_clips, ClipViolation
from .detection.demo_generator import generate_demo_violations
from .severity.classifier import SeverityClassifier, SeverityResult
from .escalation.pipeline import escalation_pipeline
from .escalation.websocket_manager import ws_manager
from .reports.models import init_db, make_async_engine, make_async_session_factory
from .reports.report_generator import ReportGenerator

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Application state
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    engine = None
    session_factory = None
    report_gen: Optional[ReportGenerator] = None
    detector: Optional[ComplianceDetector] = None
    classifier: SeverityClassifier = SeverityClassifier()
    compliance_rules: list = []
    processing: bool = False

app_state = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all subsystems on startup."""
    logger.info("═══ Factory Compliance & Alert Escalation System — Starting ═══")

    # Database
    app_state.engine = make_async_engine(str(settings.db_path))
    app_state.session_factory = make_async_session_factory(app_state.engine)
    await init_db(app_state.engine)
    logger.info(f"Database initialised at {settings.db_path}")

    # Report generator
    app_state.report_gen = ReportGenerator(app_state.session_factory, settings.outputs_dir)

    # Policy parser
    app_state.compliance_rules = load_compliance_rules(
        pdf_path=settings.policy_pdf,
        api_key=settings.gemini_api_key,
    )
    logger.info(f"Loaded {len(app_state.compliance_rules)} compliance rules")

    # Detection engine
    app_state.detector = ComplianceDetector(
        yolo_model_name=settings.yolo_model,
        yolo_conf=settings.yolo_conf,
        vlm_fallback_conf=settings.vlm_fallback_conf,
        api_key=settings.gemini_api_key,
    )

    # Auto-load demo violations if data/ is empty
    video_files = list(settings.data_dir.glob("*.mp4")) + \
                  list(settings.data_dir.glob("*.avi")) + \
                  list(settings.data_dir.glob("*.mov"))

    if not video_files:
        logger.info("No video clips in data/ — loading demo violations")
        await _ingest_demo_violations()
    else:
        logger.info(f"Found {len(video_files)} video clip(s) in data/")

    logger.info("═══ System Ready — Dashboard at http://localhost:8000 ═══")
    yield

    # Shutdown
    if app_state.engine:
        await app_state.engine.dispose()
    logger.info("═══ System Shutdown ═══")


async def _ingest_demo_violations():
    """Ingest synthetic demo violations so the dashboard is pre-populated."""
    demo_violations = generate_demo_violations(count_per_class=3)
    classifier = app_state.classifier

    for v in demo_violations:
        sev = classifier.classify(v.behavior_class_id, v.confidence)
        evt = await app_state.report_gen.record_violation(v, sev)
        await escalation_pipeline.route(
            v.event_id,
            {
                "clip_id": v.clip_id,
                "behavior_name": v.behavior_name,
                "behavior_class_id": v.behavior_class_id,
                "zone": v.zone,
                "description": v.description,
                "policy_ref": v.policy_ref,
                "confidence": v.confidence,
                "timestamp_in_clip": v.timestamp_sec,
                "thumbnail_b64": v.thumbnail_b64,
            },
            sev,
        )


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AegisFlow EHS Vision Platform",
    description="KMP-OHS-POL-001 automated monitoring -- AegisFlow Intelligent Safety Operations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static dashboard files
_STATIC_DIR = Path(__file__).parent / "dashboard" / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    """Serve the main dashboard SPA."""
    html_path = _STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "system": "Factory Compliance & Alert Escalation System",
        "policy": "KMP-OHS-POL-001",
        "rules_loaded": len(app_state.compliance_rules),
        "processing": app_state.processing,
        "websocket_clients": ws_manager.connection_count,
    }


@app.get("/api/rules")
async def get_rules():
    """Return the parsed compliance rules from the policy document."""
    return {
        "source": "KMP-OHS-POL-001 -- Occupational Health & Safety Compliance Policy Manual",
        "rules": [r.to_dict() for r in app_state.compliance_rules],
    }


@app.get("/api/rules/config")
async def get_rules_config():
    """Return the dynamic rules overrides config."""
    config_file = settings.rules_config_path
    if config_file.exists():
        try:
            return json.loads(config_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Error reading rules config: {e}")
    return {}


@app.post("/api/rules/config")
async def save_rules_config(config: dict):
    """Save the dynamic rules config overrides and reload the rules registry."""
    config_file = settings.rules_config_path
    try:
        config_file.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        # Reload rules in app state to apply immediately
        app_state.compliance_rules = load_compliance_rules(
            pdf_path=settings.policy_pdf,
            api_key=settings.gemini_api_key,
        )
        # Broadcast status update so frontend can reload stats/display
        await ws_manager.broadcast_status("rules_updated", {"message": "Policy configurations updated and reloaded"})
        return {"status": "success", "message": "Policy configuration saved successfully"}
    except Exception as e:
        logger.error(f"Error saving rules config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")


# ── Processing ─────────────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    use_demo: bool = False
    sample_every: int = 15


@app.post("/api/process")
async def trigger_processing(req: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Trigger video processing pipeline.
    Runs in background; progress streams via WebSocket.
    """
    if app_state.processing:
        raise HTTPException(status_code=409, detail="Processing already running")

    background_tasks.add_task(_run_processing_pipeline, req)
    return {"status": "started", "message": "Processing pipeline triggered"}


async def _run_processing_pipeline(req: ProcessRequest):
    """Background task: process all clips or inject demo violations."""
    app_state.processing = True
    await ws_manager.broadcast_status("processing_started", {"message": "Video analysis started"})

    try:
        if req.use_demo:
            await _ingest_demo_violations()
            await ws_manager.broadcast_status("processing_complete", {
                "message": "Demo violations loaded",
                "violations_found": 12,
            })
            return

        total = 0

        async def on_violation(v: ClipViolation):
            nonlocal total
            sev = app_state.classifier.classify(v.behavior_class_id, v.confidence)
            await app_state.report_gen.record_violation(v, sev)
            await escalation_pipeline.route(
                v.event_id,
                {
                    "clip_id": v.clip_id,
                    "behavior_name": v.behavior_name,
                    "behavior_class_id": v.behavior_class_id,
                    "zone": v.zone,
                    "description": v.description,
                    "policy_ref": v.policy_ref,
                    "confidence": v.confidence,
                    "timestamp_in_clip": v.timestamp_sec,
                    "thumbnail_b64": v.thumbnail_b64,
                },
                sev,
            )
            total += 1

        await process_all_clips(
            data_dir=settings.data_dir,
            detector=app_state.detector,
            on_violation=on_violation,
            sample_every=req.sample_every,
        )

        await ws_manager.broadcast_status("processing_complete", {
            "message": f"Processing complete — {total} violations found",
            "violations_found": total,
        })

    except Exception as exc:
        logger.error(f"Processing pipeline error: {exc}", exc_info=True)
        await ws_manager.broadcast_status("processing_error", {"error": str(exc)})
    finally:
        app_state.processing = False


# ── Events & Reports ────────────────────────────────────────────────────────

@app.get("/api/events")
async def get_events(
    severity: Optional[str] = Query(None, description="Filter by severity tier"),
    behavior_class_id: Optional[int] = Query(None, description="Filter by class ID (0-3)"),
    date_from: Optional[str] = Query(None, description="ISO date (e.g. 2024-01-01)"),
    date_to: Optional[str] = Query(None, description="ISO date (e.g. 2024-12-31)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    include_thumbnails: bool = Query(False),
):
    """Query violation events with optional filters."""
    dt_from = datetime.fromisoformat(date_from) if date_from else None
    dt_to = datetime.fromisoformat(date_to) if date_to else None

    records, total = await app_state.report_gen.get_events(
        severity=severity,
        behavior_class_id=behavior_class_id,
        date_from=dt_from,
        date_to=dt_to,
        limit=limit,
        offset=offset,
        include_thumbnails=include_thumbnails,
    )
    return {"total": total, "offset": offset, "limit": limit, "events": records}


@app.get("/api/events/stats")
async def get_stats():
    """Return aggregate statistics for the dashboard summary panel."""
    stats = await app_state.report_gen.get_summary_stats()
    stats["behavior_classes"] = BEHAVIOR_CLASSES
    stats["severity_colors"] = SEVERITY_COLORS
    return stats


@app.get("/api/export/csv")
async def export_csv(
    severity: Optional[str] = Query(None),
    behavior_class_id: Optional[int] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
):
    """Download violation log as CSV."""
    dt_from = datetime.fromisoformat(date_from) if date_from else None
    dt_to = datetime.fromisoformat(date_to) if date_to else None
    csv_data = await app_state.report_gen.export_csv(severity, behavior_class_id, dt_from, dt_to)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=compliance_audit_log.csv"},
    )


@app.get("/api/export/json")
async def export_json(
    severity: Optional[str] = Query(None),
    behavior_class_id: Optional[int] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
):
    """Download violation log as JSON."""
    dt_from = datetime.fromisoformat(date_from) if date_from else None
    dt_to = datetime.fromisoformat(date_to) if date_to else None
    json_data = await app_state.report_gen.export_json(severity, behavior_class_id, dt_from, dt_to)
    return Response(
        content=json_data,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=compliance_audit_log.json"},
    )


# ── WebSocket ───────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time alert streaming.
    Connected clients receive:
      - type="alert"  → HIGH/CRITICAL violations (triggers dashboard strobe)
      - type="event"  → LOW/MEDIUM violations (updates timeline quietly)
      - type="status" → processing status updates
    """
    await ws_manager.connect(websocket)
    try:
        # Send current stats on connect
        stats = await app_state.report_gen.get_summary_stats()
        await websocket.send_json({"type": "connected", "stats": stats})

        # Keep alive — listen for client pings
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_text("pong")  # heartbeat
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(websocket)


# ─────────────────────────────────────────────────────────────────────────────
# Dev server entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
