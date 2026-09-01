"""Module 2 - Severity Categorization Matrix.

Turns a detected violation into a LOW / MEDIUM / HIGH / CRITICAL tier, using three signals.
Two come from the policy document; one comes from the frame.

**Signal 1 - callout keyword.** The manual attaches a callout box to each behavioural
section. ``CRITICAL SAFETY NOTICE`` -> HIGH base, ``WARNING`` -> MEDIUM base, otherwise LOW.
The assignment confirms this is the intended reading: two categories sit under a WARNING and
two under a CRITICAL SAFETY NOTICE.

**Signal 2 - hazard-context language.** Phrases mined from the section prose by the policy
parser:

* ``high_frequency`` - *"the most frequently occurring unsafe behavior"* (3.3.2). Recurrence
  is an explicit severity criterion in the assignment's CRIT tier definition, so it raises
  the tier.
* ``unambiguous`` - *"the block count threshold is unambiguous"* (6.3.2), *"must be assumed
  to be performing an Unauthorized Intervention"* (4.3.2). The policy removes discretion, so
  the tier is not softened by weak detector confidence.
* ``standalone_condition`` - *"regardless of how long the panel has been open or whether
  personnel are in the immediate vicinity"* (5.2.2). The policy says personnel exposure is
  **not** a precondition, which by the assignment's own LOW definition ("a state-based
  finding with no concurrent personnel exposure") makes the base tier LOW.

**Signal 3 - frame context.** Person count, forklift presence, person-near-panel proximity.
This is what lifts a state-based finding to an exposure-based one.

Nothing here hard-codes a tier per behaviour: feed the module a different policy and the
matrix moves. The tables in the docs describe what *this* manual produces.
"""

from __future__ import annotations

from aegisflow.core.enums import BehaviorClass, PolicyCallout, Severity
from aegisflow.core.logging import get_logger
from aegisflow.core.schemas import (
    DetectionRecord,
    FrameContext,
    PolicyRule,
    PolicyRuleSet,
    SeverityAssessment,
)

log = get_logger(__name__)

# Below this, a detection is treated as weak evidence and the tier is softened by one step -
# unless the policy called the classification unambiguous.
LOW_CONFIDENCE_THRESHOLD = 0.45


class SeverityMatrix:
    """Policy-grounded severity classifier.

    Construct once per run with the parsed rule set, then call :meth:`assess` per detection.
    """

    def __init__(self, rule_set: PolicyRuleSet) -> None:
        self._rule_set = rule_set
        # Keyed on the rule, not on its behaviour class: two rules for the same behaviour
        # (a re-parse, or a test exercising several callout variants) must not collide.
        # PolicyRule is frozen, so it hashes by value.
        self._base_cache: dict[PolicyRule, tuple[Severity, list[str]]] = {}

    # ------------------------------------------------------------------ public

    def assess(self, detection: DetectionRecord) -> SeverityAssessment:
        """Assign a tier to one detection."""
        rule = self._rule_set.rule_for(detection.behavior_class)
        if rule is None:
            # Never invent a tier for a behaviour the policy did not define.
            raise KeyError(
                f"no policy rule for {detection.behavior_class.value}; "
                "re-run 'aegisflow policy parse'"
            )

        base, base_signals = self.base_severity(rule)
        final, context_signals = self._apply_context(base, rule, detection)

        return SeverityAssessment(
            behavior_class=detection.behavior_class,
            severity=final,
            base_severity=base,
            policy_rule_ref=rule.section_ref,
            callout=rule.callout,
            rationale=self._rationale(rule, base, final, base_signals + context_signals),
            signals=tuple(base_signals + context_signals),
        )

    def base_severity(self, rule: PolicyRule) -> tuple[Severity, list[str]]:
        """Tier implied by the policy text alone, before any frame context."""
        cached = self._base_cache.get(rule)
        if cached is not None:
            return cached

        signals: list[str] = []
        severity = rule.callout.base_severity
        signals.append(f"callout={rule.callout.value} -> base {severity.value}")

        # The policy explicitly decouples this condition from personnel exposure, which is
        # the assignment's definition of a LOW, state-based finding.
        if rule.standalone_condition:
            severity = Severity.LOW
            signals.append(
                "policy states the condition applies regardless of personnel proximity "
                "-> state-based finding, base lowered to LOW"
            )

        # Recurrence is named in the assignment's CRIT criteria and called out in 3.3.2.
        if rule.high_frequency:
            severity = severity.escalated()
            signals.append(
                f"policy identifies this as the highest-frequency unsafe behaviour "
                f"-> raised to {severity.value}"
            )

        # A policy that says "must be assumed" or "unambiguous" is removing discretion.
        if rule.unambiguous and rule.callout is PolicyCallout.CRITICAL_SAFETY_NOTICE:
            severity = severity.escalated()
            signals.append(
                f"policy declares the classification unambiguous under a CRITICAL SAFETY "
                f"NOTICE -> raised to {severity.value}"
            )

        result = (severity, signals)
        self._base_cache[rule] = result
        log.debug("base severity for %s: %s", rule.behavior_class.value, severity.value)
        return result

    # ----------------------------------------------------------------- internal

    def _apply_context(
        self,
        base: Severity,
        rule: PolicyRule,
        detection: DetectionRecord,
    ) -> tuple[Severity, list[str]]:
        """Adjust the policy-derived base by what was actually in the frame."""
        context = detection.context
        severity = base
        signals: list[str] = []

        if rule.standalone_condition:
            severity, extra = self._escalate_state_based(severity, context)
            signals.extend(extra)
        else:
            severity, extra = self._escalate_exposure_based(severity, rule, context)
            signals.extend(extra)

        # Weak evidence softens the tier, but only where the policy left room for judgement.
        if detection.confidence < LOW_CONFIDENCE_THRESHOLD and not rule.unambiguous:
            softened = severity.de_escalated()
            if softened is not severity:
                signals.append(
                    f"detector confidence {detection.confidence:.2f} below "
                    f"{LOW_CONFIDENCE_THRESHOLD:.2f} and policy language is not absolute "
                    f"-> softened to {softened.value}"
                )
                severity = softened

        return severity, signals

    def _escalate_state_based(
        self, severity: Severity, context: FrameContext
    ) -> tuple[Severity, list[str]]:
        """A condition finding becomes an exposure finding when people are present."""
        signals: list[str] = []
        if context.person_near_panel:
            severity = severity.escalated(2)
            signals.append(
                f"personnel detected at the affected equipment -> concurrent exposure, "
                f"raised to {severity.value}"
            )
        elif context.max_person_count > 0:
            severity = severity.escalated()
            signals.append(
                f"{context.max_person_count} person(s) in frame -> personnel present, "
                f"raised to {severity.value}"
            )
        return severity, signals

    def _escalate_exposure_based(
        self,
        severity: Severity,
        rule: PolicyRule,
        context: FrameContext,
    ) -> tuple[Severity, list[str]]:
        """Behaviour findings escalate on concurrent hazards."""
        signals: list[str] = []

        # Section 3.1 names forklift traffic as the hazard the walkway separates people
        # from, so a forklift in frame during a walkway breach is the acute case.
        if rule.behavior_class is BehaviorClass.SAFE_WALKWAY_VIOLATION and context.forklift_present:
            severity = severity.escalated()
            signals.append(
                f"forklift concurrently in frame during a walkway breach "
                f"(hazard named in {rule.section_ref.replace('3.3.2', '3.1')}) "
                f"-> raised to {severity.value}"
            )

        if context.multiple_unauthorized_persons:
            severity = severity.escalated()
            signals.append(
                f"multiple unauthorised personnel involved -> raised to {severity.value}"
            )

        return severity, signals

    def _rationale(
        self,
        rule: PolicyRule,
        base: Severity,
        final: Severity,
        signals: list[str],
    ) -> str:
        """Human-readable justification, anchored to a quoted policy sentence.

        This string is exported in the audit record and shown in the dashboard, so a
        reviewer can trace any tier back to a line of the manual.
        """
        anchor = rule.callout_text or rule.source_quote or rule.unsafe_description
        anchor = anchor.strip()
        if len(anchor) > 260:
            anchor = anchor[:257].rstrip() + "..."

        movement = f"base {base.value}" if base is final else f"base {base.value} -> {final.value}"
        detail = "; ".join(signals) if signals else "no adjustments applied"
        return f'{rule.section_ref} ({rule.callout.value}): "{anchor}" | {movement} | {detail}'


def describe_matrix(rule_set: PolicyRuleSet) -> list[dict[str, str]]:
    """The derived matrix, for the CLI, the README and the dashboard.

    Shows what the *current* policy produces rather than a hard-coded table - which is the
    point of deriving it.
    """
    matrix = SeverityMatrix(rule_set)
    rows: list[dict[str, str]] = []
    for rule in rule_set.unsafe_rules:
        base, signals = matrix.base_severity(rule)
        rows.append(
            {
                "behavior_class": rule.behavior_class.value,
                "section": rule.section_ref,
                "callout": rule.callout.value,
                "base_severity": base.value,
                "derivation": " | ".join(signals),
            }
        )
    return rows
