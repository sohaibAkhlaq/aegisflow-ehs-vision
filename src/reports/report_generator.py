"""
report_generator.py — Module 4: Automated compliance report generation.

Responsibilities:
  1. Accept a ClipViolation + SeverityResult and persist it to SQLite
  2. Append the record to outputs/audit_log.csv (append-only)
  3. Append the record to outputs/audit_log.json (append-only JSONL)
  4. Expose query functions for the dashboard (filter, paginate, export)

All writes are immutable (append-only). Records are never modified after
initial write, ensuring audit integrity per Module 4 spec.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select, and_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..detection.video_processor import ClipViolation
from ..severity.classifier import SeverityResult
from .models import ViolationEvent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CSV column order (matches assessment required fields)
# ─────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "event_id", "timestamp", "clip_id", "zone",
    "behavior_class", "policy_rule_ref", "event_description",
    "severity", "escalation_action",
    "confidence", "detector_source", "frame_index",
    "timestamp_in_clip", "alert_triggered", "logged_at",
]


class ReportGenerator:
    """
    Module 4 — auto-generates compliance records for every detected violation.

    Args:
        session_factory: Async SQLAlchemy session factory.
        outputs_dir: Directory for CSV/JSON audit logs.
    """

    def __init__(self, session_factory, outputs_dir: Path) -> None:
        self._session_factory = session_factory
        self._outputs_dir = outputs_dir
        self._csv_path = outputs_dir / "audit_log.csv"
        self._json_path = outputs_dir / "audit_log.json"
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        """Write CSV header row if the file doesn't exist yet."""
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()

    # ─────────────────────────────────────────────────────────────────────────
    # Write
    # ─────────────────────────────────────────────────────────────────────────

    async def record_violation(
        self,
        violation: ClipViolation,
        severity_result: SeverityResult,
    ) -> ViolationEvent:
        """
        Persist a violation to all report targets atomically.

        Returns the created ViolationEvent ORM object.
        """
        now = datetime.now(timezone.utc)

        # Build ORM object
        evt = ViolationEvent(
            event_id=violation.event_id,
            timestamp=now,
            clip_id=violation.clip_id,
            zone=violation.zone,
            behavior_class=violation.behavior_name,
            behavior_class_id=violation.behavior_class_id,
            policy_rule_ref=violation.policy_ref,
            event_description=violation.description,
            severity=severity_result.tier,
            escalation_action=severity_result.escalation_action,
            confidence=violation.confidence,
            detector_source=violation.detector_source,
            frame_index=violation.frame_index,
            timestamp_in_clip=violation.timestamp_sec,
            clip_path=violation.clip_path,
            alert_triggered=severity_result.requires_realtime_alert,
            logged_at=now,
            thumbnail_b64=violation.thumbnail_b64,
        )

        # 1. Persist to SQLite
        async with self._session_factory() as session:
            session.add(evt)
            await session.commit()
            await session.refresh(evt)

        # 2. Append to CSV (append-only)
        self._append_csv(evt)

        # 3. Append to JSON log (JSONL format)
        self._append_json(evt)

        logger.info(
            f"Report generated — event_id={evt.event_id} "
            f"class={evt.behavior_class} severity={evt.severity}"
        )
        return evt

    def _append_csv(self, evt: ViolationEvent) -> None:
        """Append one row to the audit CSV (no-lock; single-process assumption)."""
        try:
            row = {k: evt.to_csv_row().get(k, "") for k in CSV_FIELDS}
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writerow(row)
        except Exception as exc:
            logger.error(f"CSV append failed: {exc}")

    def _append_json(self, evt: ViolationEvent) -> None:
        """Append one JSON line to the JSONL audit log."""
        try:
            record = evt.to_csv_row()  # no thumbnail in JSON log
            with open(self._json_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.error(f"JSON append failed: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────────────────

    async def get_events(
        self,
        severity: Optional[str] = None,
        behavior_class_id: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
        include_thumbnails: bool = False,
    ) -> tuple[list[dict], int]:
        """
        Query events with optional filters. Returns (records, total_count).
        """
        async with self._session_factory() as session:
            stmt = select(ViolationEvent)
            conditions = []

            if severity:
                conditions.append(ViolationEvent.severity == severity.upper())
            if behavior_class_id is not None:
                conditions.append(ViolationEvent.behavior_class_id == behavior_class_id)
            if date_from:
                conditions.append(ViolationEvent.timestamp >= date_from)
            if date_to:
                conditions.append(ViolationEvent.timestamp <= date_to)

            if conditions:
                stmt = stmt.where(and_(*conditions))

            # Count
            count_stmt = select(func.count()).select_from(
                stmt.subquery()
            )
            total = (await session.execute(count_stmt)).scalar() or 0

            # Paginate
            stmt = stmt.order_by(desc(ViolationEvent.timestamp)).limit(limit).offset(offset)
            rows = (await session.execute(stmt)).scalars().all()

        records = []
        for row in rows:
            d = row.to_dict()
            if not include_thumbnails:
                d.pop("thumbnail_b64", None)
            records.append(d)

        return records, total

    async def export_csv(
        self,
        severity: Optional[str] = None,
        behavior_class_id: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> str:
        """Export filtered records as a CSV string."""
        records, _ = await self.get_events(
            severity=severity,
            behavior_class_id=behavior_class_id,
            date_from=date_from,
            date_to=date_to,
            limit=10000,
            include_thumbnails=False,
        )
        output = io.StringIO()
        if records:
            writer = csv.DictWriter(output, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        return output.getvalue()

    async def export_json(
        self,
        severity: Optional[str] = None,
        behavior_class_id: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> str:
        """Export filtered records as a JSON string."""
        records, _ = await self.get_events(
            severity=severity,
            behavior_class_id=behavior_class_id,
            date_from=date_from,
            date_to=date_to,
            limit=10000,
            include_thumbnails=False,
        )
        return json.dumps(records, default=str, indent=2)

    async def get_summary_stats(self) -> dict:
        """Return aggregate stats for the dashboard summary panel."""
        async with self._session_factory() as session:
            total = (await session.execute(
                select(func.count()).select_from(ViolationEvent)
            )).scalar() or 0

            by_severity = {}
            for tier in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
                count = (await session.execute(
                    select(func.count()).where(ViolationEvent.severity == tier)
                )).scalar() or 0
                by_severity[tier] = count

            by_class = {}
            for cid in range(4):
                count = (await session.execute(
                    select(func.count()).where(ViolationEvent.behavior_class_id == cid)
                )).scalar() or 0
                by_class[cid] = count

            alerts_triggered = (await session.execute(
                select(func.count()).where(ViolationEvent.alert_triggered == True)
            )).scalar() or 0

        return {
            "total_events": total,
            "by_severity": by_severity,
            "by_class": by_class,
            "alerts_triggered": alerts_triggered,
        }
