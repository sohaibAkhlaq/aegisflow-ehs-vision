"""SQLAlchemy 2.0 models.

Three tables:

* ``violation_events`` - the audit trail. **Append-only.**
* ``policy_rules`` - a snapshot of the rules that were in force for a run, so an old event
  can be explained even after the policy PDF changes.
* ``clip_runs`` - per-clip processing record, including clips where nothing was found. A
  compliance system has to be able to show it *looked* at a clip, not just that it found
  something.

Indexes on ``timestamp``, ``severity`` and ``behavior_class`` exist because those are the
three filters the dashboard's Historical Log offers (assignment Module 5, View C). Building
them now means the query layer never needs a schema migration to stay fast.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for every model."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ViolationEventRow(Base):
    """One immutable compliance record.

    Column set mirrors :class:`aegisflow.core.schemas.ViolationEvent`: the nine fields the
    assignment mandates, plus confidence, detection method and severity rationale.
    """

    __tablename__ = "violation_events"

    # --- the nine required fields ---
    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    clip_id: Mapped[str] = mapped_column(String(255), nullable=False)
    zone: Mapped[str] = mapped_column(String(64), nullable=False)
    behavior_class: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_rule_ref: Mapped[str] = mapped_column(String(32), nullable=False)
    event_description: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    escalation_action: Mapped[str] = mapped_column(String(64), nullable=False)

    # --- additions for defensibility ---
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    detection_method: Mapped[str] = mapped_column(String(32), nullable=False)
    severity_rationale: Mapped[str] = mapped_column(Text, default="")

    # --- provenance ---
    clip_timestamp_s: Mapped[float] = mapped_column(Float, default=0.0)
    frame_index: Mapped[int] = mapped_column(Integer, default=0)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        # Exactly the three filters View C exposes, plus a composite for the common
        # "recent high-severity events" query the timeline opens with.
        Index("ix_events_timestamp", "timestamp"),
        Index("ix_events_severity", "severity"),
        Index("ix_events_behavior_class", "behavior_class"),
        Index("ix_events_severity_timestamp", "severity", "timestamp"),
        Index("ix_events_clip", "clip_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<ViolationEventRow {self.event_id[:8]} {self.behavior_class} "
            f"{self.severity} {self.clip_id}>"
        )


class PolicyRuleRow(Base):
    """Snapshot of one parsed policy rule.

    Keyed by the PDF digest so several policy versions can coexist and every event stays
    explainable against the rules that actually applied when it was raised.
    """

    __tablename__ = "policy_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    document_id: Mapped[str] = mapped_column(String(64), nullable=False)
    behavior_class: Mapped[str] = mapped_column(String(64), nullable=False)
    domain: Mapped[str] = mapped_column(String(128), default="")
    section_ref: Mapped[str] = mapped_column(String(32), nullable=False)
    observable_indicator: Mapped[str] = mapped_column(Text, default="")
    callout: Mapped[str] = mapped_column(String(32), default="NONE")
    source_quote: Mapped[str] = mapped_column(Text, default="")
    numeric_threshold: Mapped[int | None] = mapped_column(Integer, nullable=True)
    validated: Mapped[bool] = mapped_column(Boolean, default=False)
    parsed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("source_sha256", "behavior_class", name="uq_policy_rule_version"),
        Index("ix_policy_rules_sha", "source_sha256"),
    )


class ClipRunRow(Base):
    """One clip processed - whether or not anything was found."""

    __tablename__ = "clip_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    clip_id: Mapped[str] = mapped_column(String(255), nullable=False)
    clip_path: Mapped[str] = mapped_column(Text, nullable=False)
    zone: Mapped[str] = mapped_column(String(64), default="")
    ground_truth_class: Mapped[str | None] = mapped_column(String(64), nullable=True)
    frames_analysed: Mapped[int] = mapped_column(Integer, default=0)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    processing_s: Mapped[float] = mapped_column(Float, default=0.0)
    violation_count: Mapped[int] = mapped_column(Integer, default=0)
    vlm_calls: Mapped[int] = mapped_column(Integer, default=0)
    policy_sha256: Mapped[str] = mapped_column(String(64), default="")
    run_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_clip_runs_clip", "clip_id"),
        Index("ix_clip_runs_run_at", "run_at"),
    )


class AnnotatedClipRow(Base):
    """Rendered playback asset for the dashboard's Live Feed Monitor (View A)."""

    __tablename__ = "annotated_clips"

    clip_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    zone: Mapped[str] = mapped_column(String(64), default="")
    worst_severity: Mapped[str | None] = mapped_column(String(16), nullable=True)
    violation_count: Mapped[int] = mapped_column(Integer, default=0)
    run_id: Mapped[int | None] = mapped_column(
        ForeignKey("clip_runs.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
