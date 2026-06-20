"""
classifier.py — Module 2: Severity Categorization Matrix.

Maps each detected violation to a risk tier (LOW / MEDIUM / HIGH / CRITICAL)
using signals directly derived from the KMP-OHS-POL-001 policy document:

  - Policy callout language:
      "CRITICAL SAFETY NOTICE" → CRITICAL
      "WARNING"                → HIGH (unless context suggests lower)
  - Hazard context descriptors (frequency, immediacy, personnel proximity)
  - Observable indicator certainty (state-based vs. action-based)

The mapping is intentionally policy-grounded and traceable to specific
sections, per the assessment requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Tier definitions — traceable to policy document sections
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SeverityResult:
    tier: str                    # LOW | MEDIUM | HIGH | CRITICAL
    rationale: str               # why this tier was assigned
    policy_signal: str           # which policy language drove the decision
    escalation_action: str       # what Module 3 will do with this

    @property
    def requires_realtime_alert(self) -> bool:
        return self.tier in ("HIGH", "CRITICAL")

    @property
    def colour_hex(self) -> str:
        return {
            "LOW": "#22c55e",
            "MEDIUM": "#f59e0b",
            "HIGH": "#f97316",
            "CRITICAL": "#ef4444",
        }.get(self.tier, "#ffffff")


# ─────────────────────────────────────────────────────────────────────────────
# Per-class severity definitions — derived from KMP-OHS-POL-001
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: class_id → base SeverityResult (before confidence adjustment)
_BASE_SEVERITY: dict[int, dict] = {
    0: {
        # Section 3 — WARNING callout + "highest-frequency unsafe behavior" + "immediate" forklift hazard
        # Elevated to CRITICAL: the policy explicitly flags this as the most frequent
        # and states "Any deviation ... places the individual in proximity to forklift
        # and machinery hazards and will be treated as an unsafe behavior event requiring
        # immediate response." → CRITICAL tier
        "tier": "CRITICAL",
        "rationale": (
            "Safe Walkway Violation is explicitly identified as the highest-frequency "
            "unsafe behavior at the facility (Section 3.3.2 WARNING callout). Any deviation "
            "from the green-marked boundaries places personnel in immediate proximity to "
            "forklift and machinery hazards. Requires immediate response."
        ),
        "policy_signal": "Section 3 WARNING callout — 'highest-frequency'; immediate forklift/machinery hazard",
    },
    1: {
        # Section 4 — CRITICAL SAFETY NOTICE callout
        # "Any person seen interacting with equipment who is not wearing the green vest
        # must be assumed to be performing an Unauthorized Intervention."
        # Active unsafe behavior + personnel in contact with equipment → HIGH
        "tier": "HIGH",
        "rationale": (
            "Unauthorized Intervention triggers a CRITICAL SAFETY NOTICE in Section 4.3.2. "
            "Personnel are in active contact with production equipment without authorization. "
            "Hazard is present and could result in immediate injury. Classified HIGH."
        ),
        "policy_signal": "Section 4 CRITICAL SAFETY NOTICE — active contact with equipment without green vest",
    },
    2: {
        # Section 5 — WARNING callout
        # "Leaving a panel cover open — even when doing so feels like a minor or temporary
        # oversight — is classified as an unsafe behavior."
        # State-based finding; no immediate personnel proximity required → LOW
        "tier": "LOW",
        "rationale": (
            "Opened Panel Cover is a state-based condition (Section 5.2.2 WARNING callout). "
            "The hazard exists but no immediate personnel proximity is required for detection. "
            "The condition may persist for extended periods as an inadvertent oversight. "
            "Classified LOW — logged and monitored, no immediate personnel injury risk confirmed."
        ),
        "policy_signal": "Section 5 WARNING callout — state-based finding, no confirmed personnel exposure",
    },
    3: {
        # Section 6 — CRITICAL SAFETY NOTICE callout
        # "The block count threshold is unambiguous: two blocks or fewer is safe;
        # three blocks or more is an overload."
        # Active unsafe behavior with vehicle instability risk → HIGH
        "tier": "HIGH",
        "rationale": (
            "Carrying Overload with Forklift triggers a CRITICAL SAFETY NOTICE in Section 6.3.2. "
            "Three or more blocks on forks creates active vehicle instability risk during "
            "travel and maneuvering. Concurrent operational hazard confirmed. Classified HIGH."
        ),
        "policy_signal": "Section 6 CRITICAL SAFETY NOTICE — active forklift instability risk during operation",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Escalation action strings (used in Module 4 reports)
# ─────────────────────────────────────────────────────────────────────────────

_ESCALATION_ACTIONS: dict[str, str] = {
    "LOW": "Logged to database — no real-time alert (LOW risk)",
    "MEDIUM": "Logged to database — no real-time alert (MEDIUM risk)",
    "HIGH": "Real-time WebSocket alert triggered + persistent database log",
    "CRITICAL": "Real-time WebSocket alert triggered (CRITICAL priority) + persistent database log",
}


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class SeverityClassifier:
    """
    Module 2 — assigns a severity tier to each violation.

    Optional confidence adjustment:
    - If detector confidence < 0.5, tier may be downgraded one level
      (e.g., CRITICAL → HIGH) to avoid over-escalating uncertain detections.
    - If confidence >= 0.85, tier is confirmed at base level.
    """

    _DOWNGRADE_MAP: dict[str, str] = {
        "CRITICAL": "HIGH",
        "HIGH": "MEDIUM",
        "MEDIUM": "LOW",
        "LOW": "LOW",
    }

    def classify(
        self,
        behavior_class_id: int,
        confidence: float = 1.0,
        apply_confidence_adjustment: bool = True,
    ) -> SeverityResult:
        """
        Classify a violation into a severity tier.

        Args:
            behavior_class_id: 0–3 per KMP-OHS-POL-001 Section 8.
            confidence: Detection confidence (0–1). Low confidence may downgrade tier.
            apply_confidence_adjustment: Whether to downgrade low-confidence detections.

        Returns:
            SeverityResult with tier, rationale, and escalation action.
        """
        if behavior_class_id not in _BASE_SEVERITY:
            return SeverityResult(
                tier="LOW",
                rationale="Unknown behavior class — defaulting to LOW",
                policy_signal="N/A",
                escalation_action=_ESCALATION_ACTIONS["LOW"],
            )

        base = _BASE_SEVERITY[behavior_class_id]
        tier = base["tier"]
        rationale = base["rationale"]

        # Load dynamic custom rules overrides
        from pathlib import Path
        import json
        config_path = Path(__file__).resolve().parent.parent.parent / "outputs" / "rules_config.json"
        has_override = False
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                rule_override = config.get(str(behavior_class_id))
                if rule_override and "severity" in rule_override:
                    old_tier = tier
                    tier = rule_override["severity"].upper()
                    has_override = True
                    rationale = f"{base['rationale']} [NOTE: Severity overridden from {old_tier} to {tier} by EHS Administrator configuration.]"
            except Exception:
                pass

        # Confidence adjustment: downgrade if detector is uncertain (skip if override is set)
        if not has_override and apply_confidence_adjustment and confidence < 0.50:
            old_tier = tier
            tier = self._DOWNGRADE_MAP[tier]
            rationale = (
                f"{rationale} "
                f"[NOTE: Tier downgraded from {old_tier} due to low detection "
                f"confidence ({confidence:.2f} < 0.50 threshold).]"
            )

        return SeverityResult(
            tier=tier,
            rationale=rationale,
            policy_signal=base["policy_signal"],
            escalation_action=_ESCALATION_ACTIONS.get(tier, _ESCALATION_ACTIONS["LOW"]),
        )

    def classify_batch(
        self,
        violations: list[tuple[int, float]],
    ) -> list[SeverityResult]:
        """
        Classify multiple violations at once.

        Args:
            violations: List of (behavior_class_id, confidence) tuples.

        Returns:
            List of SeverityResult in same order.
        """
        return [self.classify(cid, conf) for cid, conf in violations]

    @staticmethod
    def get_highest_severity(results: list[SeverityResult]) -> Optional[SeverityResult]:
        """Return the highest-severity result from a list (for multi-violation clips)."""
        if not results:
            return None
        order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        return max(results, key=lambda r: order.index(r.tier))
