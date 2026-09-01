"""REST routes backing the three dashboard views.

=========================================  ====  ======================================
Route                                      View  Purpose
=========================================  ====  ======================================
``GET  /api/health``                        -    liveness plus provider/policy state
``GET  /api/stats``                         A    header tiles
``GET  /api/clips``                         A    processed clips for playback
``GET  /api/clips/{clip_id}/video``         A    stream an annotated clip
``GET  /api/events``                        C    filtered, paginated history
``GET  /api/events/export``                 C    CSV / JSON download
``GET  /api/events/{event_id}``            A,B   single record
``GET  /api/policy``                        -    parsed rules + derived severity matrix
``WS   /ws/alerts``                        A,B   live HIGH/CRITICAL alerts
=========================================  ====  ======================================

``/api/events/export`` is registered before ``/api/events/{event_id}`` because otherwise
"export" would be captured as an event id.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from aegisflow import __version__
from aegisflow.api.schemas import (
    ClipOut,
    EventOut,
    EventPage,
    HealthOut,
    PolicyOut,
    PolicyRuleOut,
    StatsOut,
)
from aegisflow.core.enums import REPORT_SEVERITIES, BehaviorClass, Severity
from aegisflow.core.logging import get_logger
from aegisflow.core.schemas import REPORT_FIELDS, ViolationEvent
from aegisflow.core.settings import Settings, get_settings
from aegisflow.db import crud
from aegisflow.db.session import get_db_session
from aegisflow.escalation.bus import get_alert_bus
from aegisflow.policy import load_rule_set
from aegisflow.severity import SeverityMatrix

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["compliance"])

SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


def _to_out(event: ViolationEvent) -> EventOut:
    return EventOut(
        event_id=str(event.event_id),
        timestamp=event.timestamp,
        clip_id=event.clip_id,
        zone=event.zone,
        behavior_class=event.behavior_class,
        policy_rule_ref=event.policy_rule_ref,
        event_description=event.event_description,
        severity=event.severity,
        escalation_action=event.escalation_action,
        confidence=event.confidence,
        detection_method=event.detection_method,
        severity_rationale=event.severity_rationale,
        clip_timestamp_s=event.clip_timestamp_s,
        frame_index=event.frame_index,
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthOut)
async def health(session: SessionDep) -> HealthOut:
    settings = get_settings()
    policy_loaded = True
    try:
        load_rule_set(settings)
    except Exception:
        policy_loaded = False

    database_ready = True
    try:
        await crud.run_summary(session)
    except Exception:
        database_ready = False

    return HealthOut(
        version=__version__,
        llm_provider=settings.llm_provider.value,
        llm_available=settings.llm_available,
        policy_loaded=policy_loaded,
        database_ready=database_ready,
        alert_subscribers=get_alert_bus().subscriber_count,
    )


# ---------------------------------------------------------------------------
# View A - stats and clips
# ---------------------------------------------------------------------------


@router.get("/stats", response_model=StatsOut)
async def stats(session: SessionDep) -> StatsOut:
    summary = await crud.run_summary(session)
    by_severity = await crud.severity_counts(session)
    return StatsOut(
        events_recorded=int(summary["events_recorded"]),
        clips_processed=int(summary["clips_processed"]),
        frames_analysed=int(summary["frames_analysed"]),
        processing_seconds=float(summary["processing_seconds"]),
        vlm_calls=int(summary["vlm_calls"]),
        by_severity=by_severity,
        by_behavior=await crud.behavior_counts(session),
        by_zone=await crud.zone_counts(session),
        alerts_total=sum(count for tier, count in by_severity.items() if tier in REPORT_SEVERITIES),
    )


@router.get("/clips", response_model=list[ClipOut])
async def clips(session: SessionDep, limit: int = Query(200, ge=1, le=1000)) -> list[ClipOut]:
    rows = await crud.list_annotated_clips(session, limit=limit)
    out: list[ClipOut] = []
    for row in rows:
        out.append(
            ClipOut(
                clip_id=row.clip_id,
                zone=row.zone,
                worst_severity=Severity(row.worst_severity) if row.worst_severity else None,
                violation_count=row.violation_count,
                has_video=Path(row.path).exists(),
            )
        )
    return out


@router.get("/clips/{clip_id}/video")
async def clip_video(clip_id: str, session: SessionDep) -> FileResponse:
    """Serve the annotated clip for View A playback."""
    row = await crud.get_annotated_clip(session, clip_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"clip {clip_id!r} has not been processed")
    path = Path(row.path)
    if not path.is_absolute():
        path = get_settings().path(path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="annotated video file is missing on disk")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


# ---------------------------------------------------------------------------
# View C - history and export
# ---------------------------------------------------------------------------


def _parse_filters(
    date_from: datetime | None,
    date_to: datetime | None,
    severity: list[str] | None,
    behavior_class: list[str] | None,
) -> tuple[list[Severity] | None, list[BehaviorClass] | None]:
    try:
        tiers = [Severity(s.upper()) for s in severity] if severity else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid severity: {exc}") from exc
    try:
        behaviors = [BehaviorClass(b.lower()) for b in behavior_class] if behavior_class else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid behavior_class: {exc}") from exc
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from is after date_to")
    return tiers, behaviors


@router.get("/events", response_model=EventPage)
async def list_events(
    session: SessionDep,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    severity: Annotated[list[str] | None, Query()] = None,
    behavior_class: Annotated[list[str] | None, Query()] = None,
    zone: str | None = None,
    clip_id: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> EventPage:
    """Filtered history. The three filters View C requires, plus zone and clip."""
    tiers, behaviors = _parse_filters(date_from, date_to, severity, behavior_class)
    filters = {
        "date_from": date_from,
        "date_to": date_to,
        "severity": tiers,
        "behavior_class": behaviors,
        "zone": zone,
        "clip_id": clip_id,
    }
    events = await crud.list_events(session, limit=limit, offset=offset, **filters)
    total = await crud.count_events(session, **filters)
    return EventPage(total=total, limit=limit, offset=offset, items=[_to_out(e) for e in events])


@router.get("/events/export")
async def export_events(
    session: SessionDep,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    severity: Annotated[list[str] | None, Query()] = None,
    behavior_class: Annotated[list[str] | None, Query()] = None,
    zone: str | None = None,
    fmt: Annotated[str, Query(alias="format", pattern="^(csv|json)$")] = "csv",
    limit: int = Query(10000, ge=1, le=100000),
) -> Response:
    """Download the filtered log. Same filters as ``/api/events``."""
    tiers, behaviors = _parse_filters(date_from, date_to, severity, behavior_class)
    events = await crud.list_events(
        session,
        date_from=date_from,
        date_to=date_to,
        severity=tiers,
        behavior_class=behaviors,
        zone=zone,
        limit=limit,
        newest_first=False,
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if fmt == "json":
        payload = json.dumps([e.to_report_row() for e in events], indent=2)
        return Response(
            content=payload,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="aegisflow-audit-{stamp}.json"'},
        )

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(REPORT_FIELDS)
    for event in events:
        row = event.to_report_row()
        writer.writerow([row.get(field, "") for field in REPORT_FIELDS])
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="aegisflow-audit-{stamp}.csv"'},
    )


@router.get("/events/{event_id}", response_model=EventOut)
async def get_event(event_id: str, session: SessionDep) -> EventOut:
    event = await crud.get_event(session, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"event {event_id!r} not found")
    return _to_out(event)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@router.get("/policy", response_model=PolicyOut)
async def policy(settings: Annotated[Settings, Depends(get_settings)]) -> PolicyOut:
    """The parsed rules and the severity matrix derived from them.

    Exposed so the dashboard can show *why* a tier was assigned, straight from the policy,
    rather than asking the viewer to trust a colour.
    """
    try:
        rule_set = load_rule_set(settings)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"policy rules unavailable ({exc}); run 'aegisflow policy parse'",
        ) from exc

    matrix = SeverityMatrix(rule_set)
    rules: list[PolicyRuleOut] = []
    for rule in rule_set.rules:
        base, signals = matrix.base_severity(rule)
        rules.append(
            PolicyRuleOut(
                behavior_class=rule.behavior_class,
                domain=rule.domain,
                section_ref=rule.section_ref,
                observable_indicator=rule.observable_indicator,
                callout=rule.callout.value,
                base_severity=base,
                numeric_threshold=rule.numeric_threshold,
                source_quote=rule.source_quote,
                derivation=" | ".join(signals),
                validated=rule.validated,
            )
        )

    return PolicyOut(
        document_id=rule_set.document_id,
        source_sha256=rule_set.source_sha256,
        parsed_at=rule_set.parsed_at,
        extraction_method=rule_set.extraction_method,
        rules=rules,
        warnings=list(rule_set.warnings),
    )
