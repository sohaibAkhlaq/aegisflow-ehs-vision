"""Append-only data access.

**There is deliberately no update or delete path for violation events.** Compliance records
are evidence: the assignment requires an immutable audit trail, so immutability is enforced
by the absence of the operation, not by a convention someone can forget. A correction is a
new row.

The query helpers exist to serve the dashboard's Historical Log (assignment Module 5,
View C), whose filters are date range, severity tier and behaviour class - the three columns
the schema indexes.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aegisflow.core.enums import BehaviorClass, DetectionMethod, EscalationAction, Severity
from aegisflow.core.logging import get_logger
from aegisflow.core.schemas import ClipResult, PolicyRuleSet, ViolationEvent
from aegisflow.db.models import AnnotatedClipRow, ClipRunRow, PolicyRuleRow, ViolationEventRow

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


def to_row(event: ViolationEvent) -> ViolationEventRow:
    return ViolationEventRow(
        event_id=str(event.event_id),
        timestamp=event.timestamp,
        clip_id=event.clip_id,
        zone=event.zone,
        behavior_class=event.behavior_class.value,
        policy_rule_ref=event.policy_rule_ref,
        event_description=event.event_description,
        severity=event.severity.value,
        escalation_action=event.escalation_action.value,
        confidence=event.confidence,
        detection_method=event.detection_method.value,
        severity_rationale=event.severity_rationale,
        clip_timestamp_s=event.clip_timestamp_s,
        frame_index=event.frame_index,
    )


def from_row(row: ViolationEventRow) -> ViolationEvent:
    timestamp = row.timestamp
    if timestamp.tzinfo is None:
        # SQLite does not preserve tzinfo; the column is written as UTC.
        timestamp = timestamp.replace(tzinfo=UTC)
    return ViolationEvent(
        event_id=uuid.UUID(row.event_id),
        timestamp=timestamp,
        clip_id=row.clip_id,
        zone=row.zone,
        behavior_class=BehaviorClass(row.behavior_class),
        policy_rule_ref=row.policy_rule_ref,
        event_description=row.event_description,
        severity=Severity(row.severity),
        escalation_action=EscalationAction(row.escalation_action),
        confidence=row.confidence,
        detection_method=DetectionMethod(row.detection_method),
        severity_rationale=row.severity_rationale,
        clip_timestamp_s=row.clip_timestamp_s,
        frame_index=row.frame_index,
    )


# ---------------------------------------------------------------------------
# Writes (insert only)
# ---------------------------------------------------------------------------


async def insert_event(session: AsyncSession, event: ViolationEvent) -> ViolationEvent:
    """Append one compliance record."""
    session.add(to_row(event))
    await session.flush()
    return event


async def insert_events(
    session: AsyncSession, events: Sequence[ViolationEvent]
) -> list[ViolationEvent]:
    if not events:
        return []
    session.add_all([to_row(e) for e in events])
    await session.flush()
    return list(events)


async def insert_clip_run(
    session: AsyncSession,
    result: ClipResult,
    ground_truth: BehaviorClass | None = None,
    policy_sha256: str = "",
) -> int:
    """Record that a clip was processed, violations or not."""
    row = ClipRunRow(
        clip_id=result.clip_id,
        clip_path=result.clip_path,
        zone=result.zone,
        ground_truth_class=ground_truth.value if ground_truth else None,
        frames_analysed=result.frames_analysed,
        duration_s=result.duration_s,
        processing_s=result.processing_s,
        violation_count=len(result.events),
        vlm_calls=result.vlm_calls,
        policy_sha256=policy_sha256,
    )
    session.add(row)
    await session.flush()
    return int(row.id)


async def upsert_policy_rules(session: AsyncSession, rule_set: PolicyRuleSet) -> int:
    """Snapshot the rules in force, keyed by the PDF digest.

    Re-running the parser on an unchanged document is a no-op, so this is safe to call at
    the start of every run.
    """
    existing = await session.scalars(
        select(PolicyRuleRow.behavior_class).where(
            PolicyRuleRow.source_sha256 == rule_set.source_sha256
        )
    )
    known = set(existing.all())
    added = 0
    for rule in rule_set.rules:
        if rule.behavior_class.value in known:
            continue
        session.add(
            PolicyRuleRow(
                source_sha256=rule_set.source_sha256,
                document_id=rule_set.document_id,
                behavior_class=rule.behavior_class.value,
                domain=rule.domain,
                section_ref=rule.section_ref,
                observable_indicator=rule.observable_indicator,
                callout=rule.callout.value,
                source_quote=rule.source_quote,
                numeric_threshold=rule.numeric_threshold,
                validated=rule.validated,
            )
        )
        added += 1
    if added:
        await session.flush()
    return added


async def record_annotated_clip(
    session: AsyncSession,
    clip_id: str,
    path: str,
    zone: str,
    worst_severity: Severity | None,
    violation_count: int,
    run_id: int | None = None,
) -> None:
    """Register a rendered playback asset for View A.

    This is the one table with replace semantics: re-processing a clip produces a new
    rendering of the same source, which is a cache entry rather than an audit record.
    """
    existing = await session.get(AnnotatedClipRow, clip_id)
    if existing is not None:
        existing.path = path
        existing.zone = zone
        existing.worst_severity = worst_severity.value if worst_severity else None
        existing.violation_count = violation_count
        existing.run_id = run_id
    else:
        session.add(
            AnnotatedClipRow(
                clip_id=clip_id,
                path=path,
                zone=zone,
                worst_severity=worst_severity.value if worst_severity else None,
                violation_count=violation_count,
                run_id=run_id,
            )
        )
    await session.flush()


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def _apply_filters(
    statement: Select[tuple[ViolationEventRow]],
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    severity: Sequence[Severity] | None = None,
    behavior_class: Sequence[BehaviorClass] | None = None,
    zone: str | None = None,
    clip_id: str | None = None,
) -> Select[tuple[ViolationEventRow]]:
    """The three filters View C requires, plus zone and clip for drill-down."""
    if date_from is not None:
        statement = statement.where(ViolationEventRow.timestamp >= date_from)
    if date_to is not None:
        statement = statement.where(ViolationEventRow.timestamp <= date_to)
    if severity:
        statement = statement.where(ViolationEventRow.severity.in_([s.value for s in severity]))
    if behavior_class:
        statement = statement.where(
            ViolationEventRow.behavior_class.in_([b.value for b in behavior_class])
        )
    if zone:
        statement = statement.where(ViolationEventRow.zone == zone)
    if clip_id:
        statement = statement.where(ViolationEventRow.clip_id == clip_id)
    return statement


async def list_events(
    session: AsyncSession,
    *,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    severity: Sequence[Severity] | None = None,
    behavior_class: Sequence[BehaviorClass] | None = None,
    zone: str | None = None,
    clip_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
    newest_first: bool = True,
) -> list[ViolationEvent]:
    """Filtered, paginated history."""
    statement = _apply_filters(
        select(ViolationEventRow), date_from, date_to, severity, behavior_class, zone, clip_id
    )
    order = (
        ViolationEventRow.timestamp.desc() if newest_first else ViolationEventRow.timestamp.asc()
    )
    statement = statement.order_by(order).limit(limit).offset(offset)
    rows = await session.scalars(statement)
    return [from_row(row) for row in rows.all()]


async def count_events(
    session: AsyncSession,
    *,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    severity: Sequence[Severity] | None = None,
    behavior_class: Sequence[BehaviorClass] | None = None,
    zone: str | None = None,
    clip_id: str | None = None,
) -> int:
    statement = _apply_filters(
        select(ViolationEventRow), date_from, date_to, severity, behavior_class, zone, clip_id
    )
    total = await session.scalar(select(func.count()).select_from(statement.subquery()))
    return int(total or 0)


async def get_event(session: AsyncSession, event_id: str) -> ViolationEvent | None:
    row = await session.get(ViolationEventRow, event_id)
    return from_row(row) if row else None


async def severity_counts(session: AsyncSession) -> dict[str, int]:
    rows = await session.execute(
        select(ViolationEventRow.severity, func.count()).group_by(ViolationEventRow.severity)
    )
    return {severity: int(count) for severity, count in rows.all()}


async def behavior_counts(session: AsyncSession) -> dict[str, int]:
    rows = await session.execute(
        select(ViolationEventRow.behavior_class, func.count()).group_by(
            ViolationEventRow.behavior_class
        )
    )
    return {behavior: int(count) for behavior, count in rows.all()}


async def zone_counts(session: AsyncSession) -> dict[str, int]:
    rows = await session.execute(
        select(ViolationEventRow.zone, func.count()).group_by(ViolationEventRow.zone)
    )
    return {zone: int(count) for zone, count in rows.all()}


async def list_annotated_clips(session: AsyncSession, limit: int = 200) -> list[AnnotatedClipRow]:
    rows = await session.scalars(
        select(AnnotatedClipRow).order_by(AnnotatedClipRow.created_at.desc()).limit(limit)
    )
    return list(rows.all())


async def get_annotated_clip(session: AsyncSession, clip_id: str) -> AnnotatedClipRow | None:
    return await session.get(AnnotatedClipRow, clip_id)


async def latest_policy_rules(session: AsyncSession) -> list[PolicyRuleRow]:
    """Rules from the most recently parsed policy version."""
    newest = await session.scalar(
        select(PolicyRuleRow.source_sha256).order_by(PolicyRuleRow.parsed_at.desc()).limit(1)
    )
    if newest is None:
        return []
    rows = await session.scalars(select(PolicyRuleRow).where(PolicyRuleRow.source_sha256 == newest))
    return list(rows.all())


async def run_summary(session: AsyncSession) -> dict[str, float | int]:
    """Aggregate processing stats for the dashboard header."""
    clips = await session.scalar(select(func.count()).select_from(ClipRunRow)) or 0
    events = await session.scalar(select(func.count()).select_from(ViolationEventRow)) or 0
    frames = await session.scalar(select(func.sum(ClipRunRow.frames_analysed))) or 0
    seconds = await session.scalar(select(func.sum(ClipRunRow.processing_s))) or 0.0
    vlm = await session.scalar(select(func.sum(ClipRunRow.vlm_calls))) or 0
    return {
        "clips_processed": int(clips),
        "events_recorded": int(events),
        "frames_analysed": int(frames),
        "processing_seconds": round(float(seconds), 2),
        "vlm_calls": int(vlm),
    }
