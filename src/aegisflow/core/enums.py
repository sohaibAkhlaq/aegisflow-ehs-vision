"""Closed vocabularies shared across every module.

A note on the policy-grounding requirement
------------------------------------------
The assignment requires that behaviour classes be *derived from the policy document*, not
hard-coded. ``BehaviorClass`` below is a **typed registry**, not the source of truth:

* The policy parser reads the behaviour names out of the PDF (Section 8 quick-reference
  table plus the Section 3-6 behavioural-standards headings) and slugifies them.
* :func:`BehaviorClass.from_policy_name` resolves those derived slugs onto this registry.
* If the PDF yields a behaviour this registry does not know, parsing **fails loudly** rather
  than silently ignoring it (see ``policy/validate.py``).

So the enum constrains and type-checks what the PDF produced; it never substitutes for it.
Nothing about a behaviour beyond its identity - indicator, section reference, callout,
severity - is defined here. All of that is parsed.
"""

from __future__ import annotations

import re
from enum import StrEnum

# ``enum.StrEnum`` (3.11+) is what we want throughout: members compare equal to their string
# value and ``str(member)`` yields the bare value, so JSON and CSV export never leak
# ``ClassName.MEMBER``. Re-exported here so every module imports its vocabulary from one place.
__all__ = [
    "REPORT_SEVERITIES",
    "SAFE_BEHAVIORS",
    "UNSAFE_BEHAVIORS",
    "BehaviorClass",
    "DetectionMethod",
    "EscalationAction",
    "LLMProviderName",
    "PolicyCallout",
    "Severity",
    "StrEnum",
    "slugify",
]


class BehaviorClass(StrEnum):
    """The eight behaviours defined by KMP-OHS-POL-001 (four unsafe, four safe)."""

    # --- unsafe (the four the detection system monitors) ---
    SAFE_WALKWAY_VIOLATION = "safe_walkway_violation"
    UNAUTHORIZED_INTERVENTION = "unauthorized_intervention"
    OPENED_PANEL_COVER = "opened_panel_cover"
    CARRYING_OVERLOAD_WITH_FORKLIFT = "carrying_overload_with_forklift"

    # --- safe counterparts (the required default state; never alerted on) ---
    SAFE_WALKWAY = "safe_walkway"
    AUTHORIZED_INTERVENTION = "authorized_intervention"
    CLOSED_PANEL_COVER = "closed_panel_cover"
    SAFE_CARRYING = "safe_carrying"

    @property
    def is_unsafe(self) -> bool:
        return self in UNSAFE_BEHAVIORS

    @property
    def safe_counterpart(self) -> BehaviorClass:
        """The compliant pair for this behaviour (policy Section 8)."""
        return _COUNTERPARTS[self]

    @property
    def display_name(self) -> str:
        return self.value.replace("_", " ").title()

    @classmethod
    def from_policy_name(cls, name: str) -> BehaviorClass:
        """Resolve a behaviour name lifted from the policy PDF onto this registry.

        Accepts the prose spellings the manual uses - "Carrying Overload with Forklift",
        "Safe Walkway Violation", "Opened Panel Cover" - as well as dataset folder names
        such as ``3_carrying_overload_with_forklift``.

        Raises:
            KeyError: if the PDF named a behaviour this registry does not know. Callers
                treat that as a parse failure rather than skipping the rule.
        """
        slug = slugify(name)
        # Drop a leading dataset class-id ("3_carrying_overload..." -> "carrying_overload...")
        slug = re.sub(r"^\d+_", "", slug)
        for member in cls:
            if member.value == slug:
                return member
        raise KeyError(f"policy named an unknown behaviour: {name!r} (slug {slug!r})")


UNSAFE_BEHAVIORS: frozenset[BehaviorClass] = frozenset(
    {
        BehaviorClass.SAFE_WALKWAY_VIOLATION,
        BehaviorClass.UNAUTHORIZED_INTERVENTION,
        BehaviorClass.OPENED_PANEL_COVER,
        BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT,
    }
)

SAFE_BEHAVIORS: frozenset[BehaviorClass] = frozenset(set(BehaviorClass) - UNSAFE_BEHAVIORS)

_COUNTERPARTS: dict[BehaviorClass, BehaviorClass] = {
    BehaviorClass.SAFE_WALKWAY_VIOLATION: BehaviorClass.SAFE_WALKWAY,
    BehaviorClass.SAFE_WALKWAY: BehaviorClass.SAFE_WALKWAY_VIOLATION,
    BehaviorClass.UNAUTHORIZED_INTERVENTION: BehaviorClass.AUTHORIZED_INTERVENTION,
    BehaviorClass.AUTHORIZED_INTERVENTION: BehaviorClass.UNAUTHORIZED_INTERVENTION,
    BehaviorClass.OPENED_PANEL_COVER: BehaviorClass.CLOSED_PANEL_COVER,
    BehaviorClass.CLOSED_PANEL_COVER: BehaviorClass.OPENED_PANEL_COVER,
    BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT: BehaviorClass.SAFE_CARRYING,
    BehaviorClass.SAFE_CARRYING: BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT,
}


class Severity(StrEnum):
    """Risk tiers required by the assignment's Module 2 schema."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]

    @property
    def requires_realtime_alert(self) -> bool:
        """HIGH and CRITICAL route to a real-time alert *and* the DB log (Module 3)."""
        return self.rank >= Severity.HIGH.rank

    def escalated(self, steps: int = 1) -> Severity:
        """Move up the ladder, clamped at CRITICAL."""
        order = list(_SEVERITY_ORDER)
        return order[min(order.index(self) + steps, len(order) - 1)]

    def de_escalated(self, steps: int = 1) -> Severity:
        order = list(_SEVERITY_ORDER)
        return order[max(order.index(self) - steps, 0)]


_SEVERITY_ORDER: tuple[Severity, ...] = (
    Severity.LOW,
    Severity.MEDIUM,
    Severity.HIGH,
    Severity.CRITICAL,
)
_SEVERITY_RANK: dict[Severity, int] = {s: i for i, s in enumerate(_SEVERITY_ORDER)}


REPORT_SEVERITIES: frozenset[str] = frozenset({Severity.HIGH.value, Severity.CRITICAL.value})
"""Tiers that trigger a real-time alert as well as a log (assignment Module 3)."""


class PolicyCallout(StrEnum):
    """Callout boxes used by KMP-OHS-POL-001.

    The primary severity signal: the assignment notes that two behaviour categories sit
    under a WARNING callout and two under a CRITICAL SAFETY NOTICE callout.
    """

    CRITICAL_SAFETY_NOTICE = "CRITICAL SAFETY NOTICE"
    WARNING = "WARNING"
    IMPORTANT = "IMPORTANT"
    NOTE = "NOTE"
    NONE = "NONE"

    @property
    def base_severity(self) -> Severity:
        """Base tier implied by the callout keyword alone, before context adjustment."""
        return _CALLOUT_BASE[self]


_CALLOUT_BASE: dict[PolicyCallout, Severity] = {
    PolicyCallout.CRITICAL_SAFETY_NOTICE: Severity.HIGH,
    PolicyCallout.WARNING: Severity.MEDIUM,
    PolicyCallout.IMPORTANT: Severity.LOW,
    PolicyCallout.NOTE: Severity.LOW,
    PolicyCallout.NONE: Severity.LOW,
}


class DetectionMethod(StrEnum):
    """How a detection was reached - recorded on every event for auditability."""

    YOLO = "yolo"
    HSV = "hsv"
    CONTOUR = "contour"
    GEOMETRY = "geometry"
    VLM_TIEBREAK = "vlm_tiebreak"


class EscalationAction(StrEnum):
    """The routing action taken by Module 3."""

    PENDING = "Pending"
    LOGGED = "Logged to DB"
    ALERTED = "Real-time alert triggered + DB log"


class LLMProviderName(StrEnum):
    GROQ = "groq"
    GEMINI = "gemini"
    OFFLINE = "offline"


def slugify(text: str) -> str:
    """Normalise prose lifted from a PDF into a stable snake_case key."""
    cleaned = re.sub(r"[^\w\s-]", " ", text.strip().lower())
    cleaned = re.sub(r"\bwith\s+forklift\b", "with_forklift", cleaned)
    return re.sub(r"[\s-]+", "_", cleaned).strip("_")
