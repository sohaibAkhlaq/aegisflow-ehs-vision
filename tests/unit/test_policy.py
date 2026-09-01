"""Policy parsing and the faithfulness gate.

The assignment asks how we would verify that extracted rules are faithful to the source
document. ``validate.py`` is the answer, and these are the tests that prove it works -
including deliberately planting a hallucinated rule and asserting it is rejected.
"""

from __future__ import annotations

import pytest

from aegisflow.core.enums import BehaviorClass, PolicyCallout
from aegisflow.core.errors import PolicyValidationError
from aegisflow.policy.extract import Section, extract_document, normalise, squash
from aegisflow.policy.parse import build_rules
from aegisflow.policy.validate import validate_rules


@pytest.fixture(scope="module")
def document(request):
    from aegisflow.core.settings import get_settings

    settings = get_settings()
    path = settings.path(settings.policy_pdf)
    if not path.exists():
        pytest.skip("compliance_policy.pdf not present")
    return extract_document(path)


class TestTextHygiene:
    def test_replacement_char_becomes_a_dash(self):
        """The manual's em-dashes decode as U+FFFD; left alone they break substring checks."""
        assert "-" in normalise("Non-Compliant Behavior � Safe Walkway Violation")
        assert "�" not in normalise("a � b")

    def test_squash_is_whitespace_and_case_insensitive(self):
        assert squash("  Safe   Walkway\nViolation ") == "safe walkway violation"

    def test_section_ref_format(self):
        assert Section(number="3.3.2", title="x").ref == "Section 3.3.2"


class TestExtraction:
    def test_finds_the_numbered_section_tree(self, document):
        for number in ("3.3.2", "4.3.2", "5.2.2", "6.3.2"):
            assert document.section(number) is not None, f"missing section {number}"

    def test_body_lines_are_not_mistaken_for_headings(self, document):
        """'2 blocks or fewer' and '3 or more blocks' must not parse as sections."""
        assert document.section("2") is None or "block" not in document.section("2").title.lower()
        for number in document.sections:
            assert not number.startswith("0")

    def test_binds_callouts_to_the_right_sections(self, document):
        kinds = {c.section_number: c.kind for c in document.callouts}
        assert kinds.get("3.3.2") is PolicyCallout.WARNING
        assert kinds.get("4.3.2") is PolicyCallout.CRITICAL_SAFETY_NOTICE
        assert kinds.get("5.2.2") is PolicyCallout.WARNING
        assert kinds.get("6.3.2") is PolicyCallout.CRITICAL_SAFETY_NOTICE

    def test_multiline_callout_keyword_is_recognised(self, document):
        """'CRITICAL SAFETY NOTICE' is rendered one word per line in the source PDF."""
        critical = [c for c in document.callouts if c.kind is PolicyCallout.CRITICAL_SAFETY_NOTICE]
        assert len(critical) == 2
        assert all(len(c.text) > 40 for c in critical)

    def test_records_a_digest_of_the_source(self, document):
        assert len(document.sha256) == 64

    def test_secondary_extractor_ran(self, document):
        """pypdf provides the independent second opinion used for cross-validation."""
        assert len(document.secondary_text) > 1000


class TestDerivation:
    def test_derives_exactly_four_unsafe_behaviours(self, document):
        rules, warnings = build_rules(document)
        assert len(rules) == 4
        assert not warnings
        assert {r.behavior_class for r in rules} == {
            BehaviorClass.SAFE_WALKWAY_VIOLATION,
            BehaviorClass.UNAUTHORIZED_INTERVENTION,
            BehaviorClass.OPENED_PANEL_COVER,
            BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT,
        }

    def test_each_rule_cites_its_governing_section(self, document):
        rules, _ = build_rules(document)
        expected = {
            BehaviorClass.SAFE_WALKWAY_VIOLATION: "Section 3.3.2",
            BehaviorClass.UNAUTHORIZED_INTERVENTION: "Section 4.3.2",
            BehaviorClass.OPENED_PANEL_COVER: "Section 5.2.2",
            BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT: "Section 6.3.2",
        }
        for rule in rules:
            assert rule.section_ref == expected[rule.behavior_class]

    def test_two_warnings_and_two_critical_notices(self, document):
        """The assignment's own hint, used as a self-check."""
        rules, _ = build_rules(document)
        callouts = [r.callout for r in rules]
        assert callouts.count(PolicyCallout.WARNING) == 2
        assert callouts.count(PolicyCallout.CRITICAL_SAFETY_NOTICE) == 2

    def test_reads_the_numeric_threshold_from_section_6(self, document):
        rules, _ = build_rules(document)
        forklift = next(
            r for r in rules if r.behavior_class is BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT
        )
        assert forklift.numeric_threshold == 2

    def test_mines_the_severity_signals(self, document):
        rules, _ = build_rules(document)
        by_class = {r.behavior_class: r for r in rules}
        assert by_class[BehaviorClass.SAFE_WALKWAY_VIOLATION].high_frequency
        assert by_class[BehaviorClass.OPENED_PANEL_COVER].standalone_condition
        assert by_class[BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT].unambiguous

    def test_safe_counterpart_is_not_confused_with_the_unsafe_one(self, document):
        """'Safe Walkway' is a substring of 'Safe Walkway Violation'."""
        rules, _ = build_rules(document)
        walkway = next(r for r in rules if r.behavior_class is BehaviorClass.SAFE_WALKWAY_VIOLATION)
        assert walkway.safe_description
        assert "violation" not in walkway.safe_description.lower()

    def test_table_cells_are_collapsed_to_one_line(self, document):
        rules, _ = build_rules(document)
        for rule in rules:
            assert "\n" not in rule.domain
            assert "\n" not in rule.observable_indicator


class TestFaithfulnessGate:
    def test_accepts_genuinely_derived_rules(self, document):
        rules, _ = build_rules(document)
        report = validate_rules(rules, document)
        assert len(report.accepted) == 4
        assert not report.rejected
        assert all(r.validated for r in report.accepted)

    def test_rejects_an_invented_indicator(self, document):
        """The hallucination case the assignment asks about."""
        rules, _ = build_rules(document)
        poisoned = list(rules)
        poisoned[0] = poisoned[0].model_copy(
            update={
                "observable_indicator": "worker not wearing a blue hard hat near the "
                "hydraulic press",
            }
        )
        report = validate_rules(poisoned, document)
        assert len(report.rejected) == 1
        reason = report.rejected[0][1]
        assert "observable_indicator" in reason
        assert "not traceable" in reason

    def test_rejects_a_fabricated_citation(self, document):
        rules, _ = build_rules(document)
        poisoned = list(rules)
        poisoned[1] = poisoned[1].model_copy(update={"section_ref": "Section 12.4.9"})
        report = validate_rules(poisoned, document)
        assert any("does not exist" in reason for _, reason in report.rejected)

    def test_rejected_rules_are_excluded_not_repaired(self, document):
        rules, _ = build_rules(document)
        poisoned = list(rules)
        poisoned[0] = poisoned[0].model_copy(
            update={"source_quote": "The facility mandates weekly fire drills."}
        )
        report = validate_rules(poisoned, document)
        assert len(report.accepted) == 3
        assert report.warnings

    def test_strict_mode_raises_on_a_short_rule_set(self, document):
        rules, _ = build_rules(document)
        with pytest.raises(PolicyValidationError):
            validate_rules(rules[:2], document, strict=True)

    def test_non_strict_mode_warns_instead(self, document):
        rules, _ = build_rules(document)
        report = validate_rules(rules[:2], document, strict=False)
        assert report.warnings
        assert len(report.accepted) == 2

    def test_paraphrase_is_rejected_even_when_plausible(self, document):
        """Failing closed: a missing rule is visible, a hallucinated one is not."""
        rules, _ = build_rules(document)
        poisoned = list(rules)
        poisoned[2] = poisoned[2].model_copy(
            update={"unsafe_description": "A panel cover which has been left ajar."}
        )
        report = validate_rules(poisoned, document)
        assert len(report.rejected) == 1
