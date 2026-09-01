"""Contract tests for the shared schemas.

These guard the seam between the modules. If something here breaks, every downstream module
is affected, so they are the strictest tests in the suite.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from aegisflow.core.enums import (
    REPORT_SEVERITIES,
    SAFE_BEHAVIORS,
    UNSAFE_BEHAVIORS,
    BehaviorClass,
    DetectionMethod,
    EscalationAction,
    PolicyCallout,
    Severity,
    slugify,
)
from aegisflow.core.schemas import (
    REPORT_FIELDS,
    REQUIRED_REPORT_FIELDS,
    ObjectBox,
    ViolationEvent,
)


class TestBehaviorClass:
    def test_four_unsafe_four_safe(self):
        assert len(UNSAFE_BEHAVIORS) == 4
        assert len(SAFE_BEHAVIORS) == 4
        assert not (UNSAFE_BEHAVIORS & SAFE_BEHAVIORS)

    def test_every_unsafe_has_a_safe_counterpart(self):
        """Policy Section 8: each unsafe behaviour pairs with exactly one safe behaviour."""
        for behavior in UNSAFE_BEHAVIORS:
            counterpart = behavior.safe_counterpart
            assert counterpart in SAFE_BEHAVIORS
            assert counterpart.safe_counterpart is behavior

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Safe Walkway Violation", BehaviorClass.SAFE_WALKWAY_VIOLATION),
            ("Carrying Overload with Forklift", BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT),
            ("Opened Panel Cover", BehaviorClass.OPENED_PANEL_COVER),
            ("Unauthorized Intervention", BehaviorClass.UNAUTHORIZED_INTERVENTION),
            # dataset folder spellings
            ("3_carrying_overload_with_forklift", BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT),
            ("2_opened_panel_cover", BehaviorClass.OPENED_PANEL_COVER),
        ],
    )
    def test_resolves_policy_and_folder_spellings(self, name, expected):
        assert BehaviorClass.from_policy_name(name) is expected

    def test_unknown_behaviour_raises_rather_than_guessing(self):
        """A behaviour the policy names but we do not know must fail loudly."""
        with pytest.raises(KeyError):
            BehaviorClass.from_policy_name("Blocked Fire Exit")

    def test_slugify_keeps_with_forklift_together(self):
        assert slugify("Carrying Overload with Forklift") == "carrying_overload_with_forklift"


class TestSeverity:
    def test_ordering(self):
        assert (
            Severity.LOW.rank < Severity.MEDIUM.rank < Severity.HIGH.rank < Severity.CRITICAL.rank
        )

    def test_only_high_and_critical_alert(self):
        """Assignment Module 3: LOW/MED log only, HIGH/CRIT alert + log."""
        assert not Severity.LOW.requires_realtime_alert
        assert not Severity.MEDIUM.requires_realtime_alert
        assert Severity.HIGH.requires_realtime_alert
        assert Severity.CRITICAL.requires_realtime_alert

    def test_report_severities_matches_the_property(self):
        assert {s.value for s in Severity if s.requires_realtime_alert} == REPORT_SEVERITIES

    def test_escalation_clamps_at_both_ends(self):
        assert Severity.CRITICAL.escalated() is Severity.CRITICAL
        assert Severity.LOW.de_escalated() is Severity.LOW
        assert Severity.LOW.escalated(2) is Severity.HIGH
        assert Severity.CRITICAL.de_escalated(3) is Severity.LOW


class TestPolicyCallout:
    def test_callout_ranking_drives_base_tiers(self):
        assert PolicyCallout.CRITICAL_SAFETY_NOTICE.base_severity is Severity.HIGH
        assert PolicyCallout.WARNING.base_severity is Severity.MEDIUM
        assert PolicyCallout.NOTE.base_severity is Severity.LOW
        assert PolicyCallout.NONE.base_severity is Severity.LOW


class TestViolationEvent:
    def test_carries_all_nine_mandated_fields(self, make_event):
        row = make_event().to_report_row()
        for field in REQUIRED_REPORT_FIELDS:
            assert field in row, f"mandated report field {field} missing"
        assert len(REQUIRED_REPORT_FIELDS) == 9

    def test_extra_fields_come_after_the_mandated_nine(self):
        assert REPORT_FIELDS[:9] == REQUIRED_REPORT_FIELDS
        assert set(REPORT_FIELDS[9:]) == {
            "confidence",
            "detection_method",
            "severity_rationale",
        }

    def test_is_immutable(self, make_event):
        """Compliance records are evidence: enforced by the type, not by convention."""
        event = make_event()
        with pytest.raises(ValidationError):
            event.severity = Severity.LOW  # type: ignore[misc]
        with pytest.raises(ValidationError):
            event.event_description = "edited"  # type: ignore[misc]

    def test_with_escalation_returns_a_copy(self, make_event):
        event = make_event()
        assert event.escalation_action is EscalationAction.PENDING
        routed = event.with_escalation(EscalationAction.ALERTED)
        assert routed.escalation_action is EscalationAction.ALERTED
        assert event.escalation_action is EscalationAction.PENDING
        assert routed.event_id == event.event_id

    def test_timestamp_serialises_as_iso8601_zulu(self, make_event):
        event = make_event(timestamp=datetime(2022, 11, 25, 10, 45, tzinfo=UTC))
        assert event.to_report_row()["timestamp"] == "2022-11-25T10:45:00Z"

    def test_event_ids_are_unique(self, make_event):
        ids = {make_event().event_id for _ in range(50)}
        assert len(ids) == 50

    def test_report_row_is_json_serialisable(self, make_event):
        """The JSONL writer depends on this."""
        payload = json.dumps(make_event().to_report_row())
        assert json.loads(payload)["behavior_class"] == "safe_walkway_violation"

    def test_rejects_empty_description(self):
        with pytest.raises(ValidationError):
            ViolationEvent(
                clip_id="c.mp4",
                zone="Zone-1",
                behavior_class=BehaviorClass.SAFE_WALKWAY,
                policy_rule_ref="Section 3.3.1",
                event_description="",
                severity=Severity.LOW,
            )

    def test_rejects_out_of_range_confidence(self, make_event):
        with pytest.raises(ValidationError):
            make_event(confidence=1.4)

    def test_enum_values_serialise_bare(self, make_event):
        row = make_event(severity=Severity.CRITICAL).to_report_row()
        assert row["severity"] == "CRITICAL"
        assert row["detection_method"] == DetectionMethod.HSV.value
        assert uuid.UUID(row["event_id"])


class TestObjectBox:
    def test_foot_point_is_bottom_centre(self):
        """Walkway containment tests feet, not box centres."""
        box = ObjectBox(label="person", confidence=0.9, bbox=(10.0, 20.0, 30.0, 60.0))
        assert box.foot_point == (20.0, 60.0)
        assert box.centroid == (20.0, 40.0)

    def test_area(self):
        box = ObjectBox(label="person", confidence=0.5, bbox=(0.0, 0.0, 10.0, 4.0))
        assert box.area == 40.0

    def test_degenerate_box_has_zero_area(self):
        box = ObjectBox(label="person", confidence=0.5, bbox=(10.0, 10.0, 5.0, 5.0))
        assert box.area == 0.0


class TestPolicyRuleSet:
    def test_lookup_by_behaviour(self, rule_set):
        rule = rule_set.require_rule(BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT)
        assert rule.section_ref == "Section 6.3.2"
        assert rule.numeric_threshold == 2

    def test_missing_rule_raises(self, rule_set):
        with pytest.raises(KeyError):
            rule_set.require_rule(BehaviorClass.SAFE_CARRYING)

    def test_counts_only_unsafe_rules(self, rule_set):
        assert rule_set.unsafe_rule_count == 4
        assert len(rule_set.unsafe_rules) == 4

    def test_section_ref_format_is_enforced(self, rule_set):
        """policy_rule_ref must be a citation, not free text."""
        from aegisflow.core.schemas import PolicyRule

        with pytest.raises(ValidationError):
            PolicyRule(
                behavior_class=BehaviorClass.OPENED_PANEL_COVER,
                domain="Electrical",
                section_ref="somewhere in the manual",
                observable_indicator="panel open",
                unsafe_description="panel left open",
            )

    def test_rules_are_immutable(self, rule_set):
        with pytest.raises(ValidationError):
            rule_set.rules[0].section_ref = "Section 9.9"  # type: ignore[misc]
