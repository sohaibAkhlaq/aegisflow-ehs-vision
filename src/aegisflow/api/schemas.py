"""Request and response models for the HTTP/WebSocket API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from aegisflow.core.enums import BehaviorClass, DetectionMethod, EscalationAction, Severity


class EventOut(BaseModel):
    """A compliance record as the dashboard sees it."""

    event_id: str
    timestamp: datetime
    clip_id: str
    zone: str
    behavior_class: BehaviorClass
    policy_rule_ref: str
    event_description: str
    severity: Severity
    escalation_action: EscalationAction
    confidence: float
    detection_method: DetectionMethod
    severity_rationale: str
    clip_timestamp_s: float = 0.0
    frame_index: int = 0


class EventPage(BaseModel):
    """Paginated history for View C."""

    total: int = Field(description="Matching records, ignoring limit/offset")
    limit: int
    offset: int
    items: list[EventOut]


class StatsOut(BaseModel):
    """Header tiles for View A."""

    events_recorded: int = 0
    clips_processed: int = 0
    frames_analysed: int = 0
    processing_seconds: float = 0.0
    vlm_calls: int = 0
    by_severity: dict[str, int] = Field(default_factory=dict)
    by_behavior: dict[str, int] = Field(default_factory=dict)
    by_zone: dict[str, int] = Field(default_factory=dict)
    alerts_total: int = 0


class ClipOut(BaseModel):
    """A processed clip available for playback in View A."""

    clip_id: str
    zone: str
    worst_severity: Severity | None = None
    violation_count: int = 0
    has_video: bool = True


class PolicyRuleOut(BaseModel):
    """One parsed rule, for the dashboard's policy panel."""

    behavior_class: BehaviorClass
    domain: str
    section_ref: str
    observable_indicator: str
    callout: str
    base_severity: Severity
    numeric_threshold: int | None = None
    source_quote: str = ""
    derivation: str = ""
    validated: bool = False


class PolicyOut(BaseModel):
    """The parsed policy, with provenance."""

    document_id: str
    source_sha256: str
    parsed_at: datetime
    extraction_method: str
    rules: list[PolicyRuleOut]
    warnings: list[str] = Field(default_factory=list)


class HealthOut(BaseModel):
    status: str = "ok"
    version: str
    llm_provider: str
    llm_available: bool
    policy_loaded: bool
    database_ready: bool
    alert_subscribers: int = 0


class AlertEnvelope(BaseModel):
    """WebSocket message shape. Mirrors ``escalation.bus.AlertMessage``."""

    type: str
    sent_at: str
    payload: dict[str, Any] = Field(default_factory=dict)
