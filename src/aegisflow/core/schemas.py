"""Pydantic contracts that cross module boundaries.

``ViolationEvent`` is the seam of the whole system: Modules 1-2 produce it, Modules 3-5
consume it. It carries the nine fields the assignment mandates plus three we add for
defensibility (``confidence``, ``detection_method``, ``severity_rationale``).

Compliance records are evidence, so ``ViolationEvent`` and the policy models are **frozen**.
Immutability is enforced by the type, not by convention.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer

from aegisflow.core.enums import (
    BehaviorClass,
    DetectionMethod,
    EscalationAction,
    PolicyCallout,
    Severity,
)

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
"""A probability-like score in [0, 1]."""

BBox = tuple[float, float, float, float]
"""Axis-aligned box in pixels, ``(x1, y1, x2, y2)``."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Policy (Module 2a output)
# ---------------------------------------------------------------------------


class PolicyRule(BaseModel):
    """One behavioural rule extracted from the compliance policy PDF.

    Every field here is lifted from the document, not authored by us. ``source_quote`` is
    the literal sentence the rule came from and is what makes the rule auditable - the
    faithfulness gate in ``policy/validate.py`` checks it appears verbatim in the PDF.
    """

    model_config = ConfigDict(frozen=True)

    behavior_class: BehaviorClass
    domain: str = Field(description="Behavioural domain, e.g. 'Pedestrian Movement'")
    section_ref: str = Field(pattern=r"^Section \d+(\.\d+)*$", description="e.g. 'Section 3.3.2'")
    observable_indicator: str = Field(min_length=3)
    unsafe_description: str = Field(min_length=3)
    safe_description: str = ""
    callout: PolicyCallout = PolicyCallout.NONE
    callout_text: str = ""
    source_quote: str = Field(default="", description="Literal PDF sentence backing this rule")

    # Severity signals mined from the policy prose (see severity/matrix.py).
    high_frequency: bool = Field(
        default=False, description="Policy calls this the most/highest-frequency behaviour"
    )
    unambiguous: bool = Field(
        default=False, description="Policy states the threshold or classification is unambiguous"
    )
    standalone_condition: bool = Field(
        default=False,
        description="Policy says the condition counts regardless of personnel proximity",
    )
    numeric_threshold: int | None = Field(
        default=None, description="e.g. 2 blocks, parsed from Section 6.2"
    )

    # Provenance
    extraction_method: str = "deterministic"
    validated: bool = False
    validation_notes: tuple[str, ...] = ()


class PolicyRuleSet(BaseModel):
    """The full parsed policy: the machine-readable form of KMP-OHS-POL-001."""

    model_config = ConfigDict(frozen=True)

    document_id: str = "KMP-OHS-POL-001"
    source_path: str
    source_sha256: str = Field(description="Digest of the PDF that produced these rules")
    parsed_at: datetime = Field(default_factory=_utcnow)
    extraction_method: str = "deterministic"
    rules: tuple[PolicyRule, ...] = ()
    sections: dict[str, str] = Field(
        default_factory=dict, description="section number -> heading title"
    )
    warnings: tuple[str, ...] = Field(
        default=(), description="Rules dropped or flagged by the faithfulness gate"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unsafe_rule_count(self) -> int:
        return sum(1 for r in self.rules if r.behavior_class.is_unsafe)

    def rule_for(self, behavior: BehaviorClass) -> PolicyRule | None:
        for rule in self.rules:
            if rule.behavior_class is behavior:
                return rule
        return None

    def require_rule(self, behavior: BehaviorClass) -> PolicyRule:
        rule = self.rule_for(behavior)
        if rule is None:
            raise KeyError(f"no parsed policy rule for {behavior.value}")
        return rule

    @property
    def unsafe_rules(self) -> tuple[PolicyRule, ...]:
        return tuple(r for r in self.rules if r.behavior_class.is_unsafe)

    @field_serializer("parsed_at")
    def _ser_dt(self, value: datetime) -> str:
        return value.isoformat()


# ---------------------------------------------------------------------------
# Detection (Module 1)
# ---------------------------------------------------------------------------


class ObjectBox(BaseModel):
    """A single localised object from the detector."""

    model_config = ConfigDict(frozen=True)

    label: str
    confidence: Confidence
    bbox: BBox

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre of the box.

        Walkway containment is judged on where a person's *feet* are, not the box centre
        (policy Section 3.3.2 - position on the floor relative to the green markings).
        """
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class FrameObservation(BaseModel):
    """Everything the detectors saw in one sampled frame."""

    model_config = ConfigDict(frozen=True)

    frame_index: int = Field(ge=0, description="Index in the source video")
    timestamp_s: float = Field(ge=0.0, description="Offset into the clip, seconds")
    width: int
    height: int
    persons: tuple[ObjectBox, ...] = ()
    vehicles: tuple[ObjectBox, ...] = ()

    @property
    def person_count(self) -> int:
        return len(self.persons)

    @property
    def forklift_present(self) -> bool:
        return bool(self.vehicles)


class FrameContext(BaseModel):
    """Clip-level context the severity matrix uses for its third signal.

    Summarised across sampled frames so Module 2b never needs the frames themselves.
    """

    model_config = ConfigDict(frozen=True)

    max_person_count: int = 0
    forklift_present: bool = False
    person_near_panel: bool = False
    multiple_unauthorized_persons: bool = False
    frames_analysed: int = 0


class DetectionRecord(BaseModel):
    """Module 1's output: one detected violation within one clip.

    Corresponds to the assignment's "structured detection record": clip identifier,
    timestamp within the clip, rule breached, description, and zone.
    """

    model_config = ConfigDict(frozen=True)

    clip_id: str
    behavior_class: BehaviorClass
    confidence: Confidence
    detection_method: DetectionMethod
    first_frame_index: int = Field(ge=0)
    first_timestamp_s: float = Field(ge=0.0)
    frame_count: int = Field(ge=1, description="Sampled frames supporting this detection")
    bboxes: tuple[BBox, ...] = ()
    description: str = ""
    zone: str = "Zone-Unassigned"
    context: FrameContext = FrameContext()
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Detector-specific numbers, e.g. {'block_count': 3, 'vest': 'red_black'}",
    )
    ambiguous: bool = Field(
        default=False, description="Confidence margin fell inside the VLM tie-break band"
    )


# ---------------------------------------------------------------------------
# Severity (Module 2b)
# ---------------------------------------------------------------------------


class SeverityAssessment(BaseModel):
    """Module 2b's output: a tier plus the policy justification for it."""

    model_config = ConfigDict(frozen=True)

    behavior_class: BehaviorClass
    severity: Severity
    base_severity: Severity
    policy_rule_ref: str
    callout: PolicyCallout
    rationale: str = Field(description="Quoted policy sentence plus the context adjustment applied")
    signals: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# The seam (consumed by Modules 3, 4, 5)
# ---------------------------------------------------------------------------


class ViolationEvent(BaseModel):
    """An immutable compliance record.

    The nine assignment-mandated fields, plus ``confidence``, ``detection_method`` and
    ``severity_rationale``.

    Frozen: a correction is a new record, never an edit. ``escalation_action`` is set by
    Module 3 via :meth:`with_escalation`.
    """

    model_config = ConfigDict(frozen=True)

    # --- the nine required fields ---
    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    timestamp: datetime = Field(default_factory=_utcnow)
    clip_id: str
    zone: str
    behavior_class: BehaviorClass
    policy_rule_ref: str
    event_description: str = Field(min_length=1)
    severity: Severity
    escalation_action: EscalationAction = EscalationAction.PENDING

    # --- additions that make the record defensible in review ---
    confidence: Confidence = 1.0
    detection_method: DetectionMethod = DetectionMethod.YOLO
    severity_rationale: str = ""

    # --- provenance, not part of the audit contract ---
    clip_timestamp_s: float = 0.0
    frame_index: int = 0

    def with_escalation(self, action: EscalationAction) -> ViolationEvent:
        """Return a copy with the routing decision recorded (frozen models never mutate)."""
        return self.model_copy(update={"escalation_action": action})

    @field_serializer("timestamp")
    def _ser_timestamp(self, value: datetime) -> str:
        """ISO 8601, as the assignment's report-field table requires."""
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    @field_serializer("event_id")
    def _ser_event_id(self, value: uuid.UUID) -> str:
        return str(value)

    def to_report_row(self) -> dict[str, Any]:
        """Flat dict in the assignment's field order - used by the CSV/JSON writers."""
        return {
            "event_id": str(self.event_id),
            "timestamp": self._ser_timestamp(self.timestamp),
            "clip_id": self.clip_id,
            "zone": self.zone,
            "behavior_class": self.behavior_class.value,
            "policy_rule_ref": self.policy_rule_ref,
            "event_description": self.event_description,
            "severity": self.severity.value,
            "escalation_action": self.escalation_action.value,
            "confidence": round(self.confidence, 4),
            "detection_method": self.detection_method.value,
            "severity_rationale": self.severity_rationale,
        }


REPORT_FIELDS: tuple[str, ...] = (
    "event_id",
    "timestamp",
    "clip_id",
    "zone",
    "behavior_class",
    "policy_rule_ref",
    "event_description",
    "severity",
    "escalation_action",
    "confidence",
    "detection_method",
    "severity_rationale",
)
"""Column order for CSV export. The first nine are the assignment's mandated fields."""

REQUIRED_REPORT_FIELDS: tuple[str, ...] = REPORT_FIELDS[:9]


class ClipResult(BaseModel):
    """Outcome of running the pipeline over one clip."""

    model_config = ConfigDict(frozen=True)

    clip_id: str
    clip_path: str
    zone: str
    frames_analysed: int
    duration_s: float
    processing_s: float
    detections: tuple[DetectionRecord, ...] = ()
    events: tuple[ViolationEvent, ...] = ()
    vlm_calls: int = 0

    @property
    def compliant(self) -> bool:
        """True when no violation was found - the policy's expected default state."""
        return not self.events
