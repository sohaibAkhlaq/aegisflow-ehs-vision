"""
tests/test_severity.py — Unit tests for Module 2: Severity Categorization Matrix.
"""

from __future__ import annotations

import pytest
from src.severity.classifier import SeverityClassifier, SeverityResult


class TestSeverityClassifier:
    """Full coverage of the severity classification logic."""

    @pytest.fixture(autouse=True)
    def classifier(self):
        self.clf = SeverityClassifier()

    # ── Base tier mapping ─────────────────────────────────────────────────

    def test_class_0_walkway_is_critical(self):
        result = self.clf.classify(0, confidence=0.9)
        assert result.tier == "CRITICAL"

    def test_class_1_intervention_is_high(self):
        result = self.clf.classify(1, confidence=0.9)
        assert result.tier == "HIGH"

    def test_class_2_panel_is_low(self):
        result = self.clf.classify(2, confidence=0.9)
        assert result.tier == "LOW"

    def test_class_3_forklift_is_high(self):
        result = self.clf.classify(3, confidence=0.9)
        assert result.tier == "HIGH"

    # ── Result structure ──────────────────────────────────────────────────

    def test_result_has_all_fields(self):
        result = self.clf.classify(0, confidence=0.9)
        assert result.tier
        assert result.rationale
        assert result.policy_signal
        assert result.escalation_action

    def test_critical_requires_alert(self):
        result = self.clf.classify(0, confidence=0.9)
        assert result.requires_realtime_alert is True

    def test_high_requires_alert(self):
        result = self.clf.classify(1, confidence=0.9)
        assert result.requires_realtime_alert is True

    def test_low_does_not_require_alert(self):
        result = self.clf.classify(2, confidence=0.9)
        assert result.requires_realtime_alert is False

    def test_colour_hex_is_valid(self):
        for cid in range(4):
            result = self.clf.classify(cid, confidence=0.9)
            assert result.colour_hex.startswith("#")
            assert len(result.colour_hex) == 7

    # ── Confidence adjustment ─────────────────────────────────────────────

    def test_low_confidence_downgrades_critical_to_high(self):
        result = self.clf.classify(0, confidence=0.3, apply_confidence_adjustment=True)
        assert result.tier == "HIGH"

    def test_low_confidence_downgrades_high_to_medium(self):
        result = self.clf.classify(1, confidence=0.3, apply_confidence_adjustment=True)
        assert result.tier == "MEDIUM"

    def test_low_confidence_downgrades_low_stays_low(self):
        result = self.clf.classify(2, confidence=0.3, apply_confidence_adjustment=True)
        assert result.tier == "LOW"

    def test_high_confidence_no_downgrade(self):
        result = self.clf.classify(0, confidence=0.9, apply_confidence_adjustment=True)
        assert result.tier == "CRITICAL"

    def test_no_confidence_adjustment_flag(self):
        """When apply_confidence_adjustment=False, low confidence must not downgrade."""
        result = self.clf.classify(0, confidence=0.1, apply_confidence_adjustment=False)
        assert result.tier == "CRITICAL"

    def test_boundary_confidence_exactly_50(self):
        """Confidence of exactly 0.50 should NOT trigger downgrade (< 0.50 rule)."""
        result = self.clf.classify(0, confidence=0.50, apply_confidence_adjustment=True)
        assert result.tier == "CRITICAL"

    def test_boundary_confidence_just_below_50(self):
        """Confidence of 0.499 SHOULD trigger downgrade."""
        result = self.clf.classify(0, confidence=0.499, apply_confidence_adjustment=True)
        assert result.tier == "HIGH"

    # ── Unknown class ─────────────────────────────────────────────────────

    def test_unknown_class_returns_low(self):
        result = self.clf.classify(99, confidence=0.9)
        assert result.tier == "LOW"

    # ── Batch ─────────────────────────────────────────────────────────────

    def test_batch_classify(self):
        pairs  = [(0, 0.9), (1, 0.9), (2, 0.9), (3, 0.9)]
        results = self.clf.classify_batch(pairs)
        assert len(results) == 4
        assert results[0].tier == "CRITICAL"
        assert results[1].tier == "HIGH"
        assert results[2].tier == "LOW"
        assert results[3].tier == "HIGH"

    # ── Highest severity ──────────────────────────────────────────────────

    def test_get_highest_severity(self):
        results = [
            self.clf.classify(2, 0.9),   # LOW
            self.clf.classify(1, 0.9),   # HIGH
            self.clf.classify(0, 0.9),   # CRITICAL
        ]
        highest = SeverityClassifier.get_highest_severity(results)
        assert highest.tier == "CRITICAL"

    def test_get_highest_severity_empty(self):
        highest = SeverityClassifier.get_highest_severity([])
        assert highest is None

    # ── Policy signal content ─────────────────────────────────────────────

    def test_critical_signal_references_walkway(self):
        result = self.clf.classify(0, confidence=0.9)
        assert "walkway" in result.policy_signal.lower() or \
               "section 3" in result.policy_signal.lower()

    def test_escalation_actions_are_descriptive(self):
        for cid in range(4):
            result = self.clf.classify(cid, confidence=0.9)
            assert len(result.escalation_action) > 10

    def test_dynamic_severity_override(self):
        """Test that classifier respects dynamic custom severity configuration overrides."""
        import json
        from pathlib import Path
        config_path = Path(__file__).resolve().parent.parent / "outputs" / "rules_config.json"
        
        # Back up existing config if present
        backup = None
        if config_path.exists():
            backup = config_path.read_text(encoding="utf-8")
        
        try:
            # Set custom config override
            override = {
                "2": {
                    "severity": "CRITICAL",
                    "confidence_threshold": 0.40,
                    "active": True
                }
            }
            config_path.parent.mkdir(exist_ok=True)
            config_path.write_text(json.dumps(override), encoding="utf-8")
            
            # Re-classify class 2 (default is LOW, but overridden to CRITICAL)
            result = self.clf.classify(2, confidence=0.9)
            assert result.tier == "CRITICAL"
            assert "overridden" in result.rationale.lower()
        finally:
            # Restore backup
            if backup is not None:
                config_path.write_text(backup, encoding="utf-8")
            elif config_path.exists():
                config_path.unlink()
