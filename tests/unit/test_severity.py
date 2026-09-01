"""Severity matrix tests.

The graded requirement is that tiers are *derived from the policy*, not hard-coded. These
tests assert the derivation behaves as a function of the policy signals: change the callout
or the hazard language in the rule and the tier must move.
"""

from __future__ import annotations

import pytest

from aegisflow.core.enums import BehaviorClass, PolicyCallout, Severity
from aegisflow.core.schemas import PolicyRule, PolicyRuleSet
from aegisflow.severity import LOW_CONFIDENCE_THRESHOLD, SeverityMatrix, describe_matrix


def rule(**kwargs) -> PolicyRule:
    defaults = {
        "behavior_class": BehaviorClass.SAFE_WALKWAY_VIOLATION,
        "domain": "Pedestrian Movement",
        "section_ref": "Section 3.3.2",
        "observable_indicator": "person outside green markings",
        "unsafe_description": "movement outside the marked walkway",
        "callout": PolicyCallout.WARNING,
    }
    return PolicyRule(**{**defaults, **kwargs})


def matrix_for(*rules: PolicyRule) -> SeverityMatrix:
    return SeverityMatrix(
        PolicyRuleSet(source_path="p.pdf", source_sha256="0" * 64, rules=tuple(rules))
    )


class TestBaseSeverityDerivation:
    """Signals 1 and 2: callout keyword and hazard-context language."""

    def test_callout_alone_sets_the_base(self):
        matrix = matrix_for(rule())
        assert matrix.base_severity(rule(callout=PolicyCallout.NOTE))[0] is Severity.LOW
        assert matrix.base_severity(rule(callout=PolicyCallout.WARNING))[0] is Severity.MEDIUM
        assert (
            matrix.base_severity(rule(callout=PolicyCallout.CRITICAL_SAFETY_NOTICE))[0]
            is Severity.HIGH
        )

    def test_high_frequency_language_raises_the_tier(self):
        """Section 3.3.2 calls walkway violation the highest-frequency unsafe behaviour."""
        matrix = matrix_for(rule())
        plain, _ = matrix.base_severity(rule(callout=PolicyCallout.WARNING))
        frequent, signals = matrix.base_severity(
            rule(
                callout=PolicyCallout.WARNING,
                high_frequency=True,
                behavior_class=BehaviorClass.SAFE_WALKWAY,
            )
        )
        assert frequent.rank == plain.rank + 1
        assert any("frequency" in s for s in signals)

    def test_standalone_condition_lowers_the_base_to_low(self):
        """5.2.2 counts 'regardless of whether personnel are in the immediate vicinity'."""
        matrix = matrix_for(rule())
        severity, signals = matrix.base_severity(
            rule(
                behavior_class=BehaviorClass.OPENED_PANEL_COVER,
                section_ref="Section 5.2.2",
                callout=PolicyCallout.WARNING,
                standalone_condition=True,
            )
        )
        assert severity is Severity.LOW
        assert any("regardless" in s or "state-based" in s for s in signals)

    def test_unambiguous_under_critical_notice_reaches_critical(self):
        matrix = matrix_for(rule())
        severity, signals = matrix.base_severity(
            rule(
                behavior_class=BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT,
                section_ref="Section 6.3.2",
                callout=PolicyCallout.CRITICAL_SAFETY_NOTICE,
                unambiguous=True,
            )
        )
        assert severity is Severity.CRITICAL
        assert any("unambiguous" in s for s in signals)

    def test_unambiguous_alone_does_not_escalate_a_warning(self):
        """The escalation is conditioned on the CRITICAL SAFETY NOTICE callout."""
        matrix = matrix_for(rule())
        severity, _ = matrix.base_severity(rule(callout=PolicyCallout.WARNING, unambiguous=True))
        assert severity is Severity.MEDIUM


class TestContextEscalation:
    """Signal 3: what else was in the frame."""

    def test_panel_alone_is_low(self, rule_set, make_detection):
        matrix = SeverityMatrix(rule_set)
        assessment = matrix.assess(make_detection(BehaviorClass.OPENED_PANEL_COVER))
        assert assessment.severity is Severity.LOW
        assert not assessment.severity.requires_realtime_alert

    def test_panel_with_person_in_frame_becomes_medium(self, rule_set, make_detection):
        matrix = SeverityMatrix(rule_set)
        assessment = matrix.assess(
            make_detection(BehaviorClass.OPENED_PANEL_COVER, max_person_count=1)
        )
        assert assessment.severity is Severity.MEDIUM

    def test_panel_with_person_at_the_panel_becomes_high_and_alerts(self, rule_set, make_detection):
        matrix = SeverityMatrix(rule_set)
        assessment = matrix.assess(
            make_detection(
                BehaviorClass.OPENED_PANEL_COVER,
                max_person_count=1,
                person_near_panel=True,
            )
        )
        assert assessment.severity is Severity.HIGH
        assert assessment.severity.requires_realtime_alert

    def test_walkway_with_forklift_escalates(self, rule_set, make_detection):
        """Section 3.1 names forklift traffic as the hazard the walkway separates from."""
        matrix = SeverityMatrix(rule_set)
        without = matrix.assess(
            make_detection(BehaviorClass.SAFE_WALKWAY_VIOLATION, max_person_count=1)
        )
        with_forklift = matrix.assess(
            make_detection(
                BehaviorClass.SAFE_WALKWAY_VIOLATION,
                max_person_count=1,
                forklift_present=True,
            )
        )
        assert with_forklift.severity.rank > without.severity.rank

    def test_forklift_overload_is_critical(self, rule_set, make_detection):
        matrix = SeverityMatrix(rule_set)
        assessment = matrix.assess(
            make_detection(BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT, forklift_present=True)
        )
        assert assessment.severity is Severity.CRITICAL


class TestConfidenceSoftening:
    def test_weak_evidence_softens_a_discretionary_tier(self, rule_set, make_detection):
        matrix = SeverityMatrix(rule_set)
        strong = matrix.assess(
            make_detection(BehaviorClass.SAFE_WALKWAY_VIOLATION, confidence=0.9, max_person_count=1)
        )
        weak = matrix.assess(
            make_detection(
                BehaviorClass.SAFE_WALKWAY_VIOLATION,
                confidence=LOW_CONFIDENCE_THRESHOLD - 0.1,
                max_person_count=1,
            )
        )
        assert weak.severity.rank < strong.severity.rank

    def test_weak_evidence_does_not_soften_an_unambiguous_rule(self, rule_set, make_detection):
        """The policy removes discretion for the forklift threshold, so we do not add any."""
        matrix = SeverityMatrix(rule_set)
        weak = matrix.assess(
            make_detection(
                BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT,
                confidence=LOW_CONFIDENCE_THRESHOLD - 0.2,
            )
        )
        assert weak.severity is Severity.CRITICAL


class TestTraceability:
    def test_rationale_cites_the_policy_section(self, rule_set, make_detection):
        matrix = SeverityMatrix(rule_set)
        assessment = matrix.assess(make_detection(BehaviorClass.UNAUTHORIZED_INTERVENTION))
        assert assessment.policy_rule_ref == "Section 4.3.2"
        assert "Section 4.3.2" in assessment.rationale
        assert assessment.callout is PolicyCallout.CRITICAL_SAFETY_NOTICE

    def test_rationale_records_the_movement_and_signals(self, rule_set, make_detection):
        matrix = SeverityMatrix(rule_set)
        assessment = matrix.assess(
            make_detection(
                BehaviorClass.OPENED_PANEL_COVER, max_person_count=2, person_near_panel=True
            )
        )
        assert "base LOW" in assessment.rationale
        assert assessment.base_severity is Severity.LOW
        assert assessment.severity is Severity.HIGH
        assert assessment.signals

    def test_unknown_behaviour_raises_rather_than_defaulting(self, rule_set, make_detection):
        """Never invent a tier for a behaviour the policy did not define."""
        matrix = SeverityMatrix(rule_set)
        with pytest.raises(KeyError):
            matrix.assess(make_detection(BehaviorClass.SAFE_CARRYING))


class TestDerivedMatrixShape:
    def test_describe_matrix_covers_every_unsafe_rule(self, rule_set):
        rows = describe_matrix(rule_set)
        assert len(rows) == 4
        assert {row["behavior_class"] for row in rows} == {
            rule.behavior_class.value for rule in rule_set.unsafe_rules
        }
        for row in rows:
            assert row["derivation"], "every tier must record how it was derived"

    def test_two_warnings_and_two_critical_notices(self, rule_set):
        """The assignment's own hint, used as a self-check on the matrix inputs."""
        callouts = [r.callout for r in rule_set.unsafe_rules]
        assert callouts.count(PolicyCallout.WARNING) == 2
        assert callouts.count(PolicyCallout.CRITICAL_SAFETY_NOTICE) == 2
