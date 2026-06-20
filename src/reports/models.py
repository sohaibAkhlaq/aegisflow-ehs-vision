"""
models.py — Module 4: SQLAlchemy ORM models for compliance event storage.

Schema is designed to capture every required field from the assessment spec
plus additional audit metadata. The database is SQLite-backed via aiosqlite
for async compatibility with FastAPI.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Violation Event model
# ─────────────────────────────────────────────────────────────────────────────

class ViolationEvent(Base):
    """
    Immutable compliance event record.
    One row per detected violation — the primary audit trail entity.

    Fields map directly to the Module 4 required report fields plus
    supplementary detection metadata.
    """

    __tablename__ = "violation_events"

    # ── Primary key & identity ────────────────────────────────────────────────
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(36), nullable=False, unique=True, index=True)  # UUID

    # ── Module 4 required fields ──────────────────────────────────────────────
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    clip_id = Column(String(255), nullable=False, index=True)
    zone = Column(String(100), nullable=False)
    behavior_class = Column(String(100), nullable=False)       # behavior name
    behavior_class_id = Column(Integer, nullable=False, index=True)
    policy_rule_ref = Column(String(50), nullable=False)       # e.g. "Section 3.3.2"
    event_description = Column(Text, nullable=False)
    severity = Column(String(20), nullable=False, index=True)  # LOW|MEDIUM|HIGH|CRITICAL
    escalation_action = Column(Text, nullable=False)

    # ── Detection metadata ────────────────────────────────────────────────────
    confidence = Column(Float, nullable=False, default=0.0)
    detector_source = Column(String(50), nullable=False, default="yolo")
    frame_index = Column(Integer, nullable=True)
    timestamp_in_clip = Column(Float, nullable=True)           # seconds from start
    clip_path = Column(String(512), nullable=True)

    # ── Escalation metadata ───────────────────────────────────────────────────
    alert_triggered = Column(Boolean, nullable=False, default=False)
    logged_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # ── Thumbnail ─────────────────────────────────────────────────────────────
    thumbnail_b64 = Column(Text, nullable=True)  # base64 JPEG for dashboard

    def to_dict(self) -> dict:
        """Serialize to a dict matching Module 4 report field schema."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat() + "Z" if self.timestamp else None,
            "clip_id": self.clip_id,
            "zone": self.zone,
            "behavior_class": self.behavior_class,
            "behavior_class_id": self.behavior_class_id,
            "policy_rule_ref": self.policy_rule_ref,
            "event_description": self.event_description,
            "severity": self.severity,
            "escalation_action": self.escalation_action,
            "confidence": round(self.confidence, 4),
            "detector_source": self.detector_source,
            "frame_index": self.frame_index,
            "timestamp_in_clip": self.timestamp_in_clip,
            "clip_path": self.clip_path,
            "alert_triggered": self.alert_triggered,
            "logged_at": self.logged_at.isoformat() + "Z" if self.logged_at else None,
            "thumbnail_b64": self.thumbnail_b64,
        }

    def to_csv_row(self) -> dict:
        """Return a flat dict suitable for CSV export (no thumbnail)."""
        d = self.to_dict()
        d.pop("thumbnail_b64", None)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Database engine factory
# ─────────────────────────────────────────────────────────────────────────────

def make_async_engine(db_path: str):
    """Create an async SQLite engine for use with FastAPI."""
    url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_async_engine(url, echo=False, future=True)
    return engine


def make_async_session_factory(engine):
    """Create an async session factory bound to the given engine."""
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db(engine) -> None:
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
