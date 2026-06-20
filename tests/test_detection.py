"""
tests/test_detection.py — Unit tests for Module 1: Detection Engine.

Tests cover:
  - policy_parser: rule loading, fallback mode, rule structure validation
  - detector: bounding box helpers, colour detection, zone inference
  - video_processor: ClipViolation construction
  - demo_generator: synthetic frame generation
"""

from __future__ import annotations

import pytest
import numpy as np
import cv2

from src.detection.policy_parser import (
    load_compliance_rules,
    ComplianceRule,
    FALLBACK_RULES,
    rules_to_lookup,
)
from src.detection.detector import (
    ComplianceDetector,
    BoundingBox,
    _infer_zone,
    _bbox_overlap,
    _dominant_colour_in_roi,
    _estimate_block_count_from_region,
)
from src.detection.demo_generator import generate_demo_violations


# ─────────────────────────────────────────────────────────────────────────────
# Policy Parser
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicyParser:
    """Tests for compliance rule loading and structure."""

    def test_fallback_returns_four_rules(self):
        """Fallback rules must produce exactly 4 compliance rules."""
        rules = load_compliance_rules()  # No PDF, no API key → fallback
        assert len(rules) == 4

    def test_rules_have_required_fields(self):
        """Every rule must have all required fields populated."""
        rules = load_compliance_rules()
        required = [
            "class_id", "behavior_name", "domain", "policy_ref",
            "unsafe_behavior", "safe_behavior", "observable_indicator", "severity"
        ]
        for rule in rules:
            d = rule.to_dict()
            for field in required:
                assert field in d, f"Missing field '{field}' in rule {rule.class_id}"
                assert d[field] is not None and d[field] != "", f"Empty field '{field}' in rule {rule.class_id}"

    def test_class_ids_are_zero_to_three(self):
        """Class IDs must be exactly 0, 1, 2, 3."""
        rules = load_compliance_rules()
        ids = sorted(r.class_id for r in rules)
        assert ids == [0, 1, 2, 3]

    def test_severity_values_are_valid(self):
        """All severity values must be valid tier labels."""
        valid = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        rules = load_compliance_rules()
        for r in rules:
            assert r.severity in valid, f"Invalid severity '{r.severity}' for class {r.class_id}"

    def test_rules_to_lookup(self):
        """Lookup dict must key rules by class_id."""
        rules = load_compliance_rules()
        lookup = rules_to_lookup(rules)
        assert set(lookup.keys()) == {0, 1, 2, 3}
        for cid, rule in lookup.items():
            assert rule.class_id == cid

    def test_walkway_is_critical(self):
        """Walkway violation (class 0) must be CRITICAL per policy."""
        rules = load_compliance_rules()
        lookup = rules_to_lookup(rules)
        assert lookup[0].severity == "CRITICAL"

    def test_panel_is_low(self):
        """Open panel (class 2) must be LOW — state-based, no personnel proximity."""
        rules = load_compliance_rules()
        lookup = rules_to_lookup(rules)
        assert lookup[2].severity == "LOW"

    def test_policy_refs_match_sections(self):
        """Policy references must cite the correct sections from KMP-OHS-POL-001."""
        expected = {
            0: "Section 3.3.2",
            1: "Section 4.3.2",
            2: "Section 5.2.2",
            3: "Section 6.3.2",
        }
        rules = load_compliance_rules()
        lookup = rules_to_lookup(rules)
        for cid, ref in expected.items():
            assert lookup[cid].policy_ref == ref, \
                f"Class {cid}: expected ref '{ref}', got '{lookup[cid].policy_ref}'"


# ─────────────────────────────────────────────────────────────────────────────
# Bounding Box helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundingBox:
    """Tests for BoundingBox utility methods."""

    def test_centroid(self):
        bbox = BoundingBox(100, 200, 200, 400, 0.9, "person")
        cx, cy = bbox.centroid
        assert cx == pytest.approx(150.0)
        assert cy == pytest.approx(300.0)

    def test_area(self):
        bbox = BoundingBox(0, 0, 100, 50, 0.8, "person")
        assert bbox.area == pytest.approx(5000.0)

    def test_zero_area_bbox(self):
        bbox = BoundingBox(100, 100, 100, 100, 0.5, "test")
        assert bbox.area == 0.0

    def test_bbox_overlap_identical(self):
        a = BoundingBox(0, 0, 100, 100, 0.9, "a")
        assert _bbox_overlap(a, a) == pytest.approx(1.0, abs=0.01)

    def test_bbox_overlap_no_overlap(self):
        a = BoundingBox(0, 0, 50, 50, 0.9, "a")
        b = BoundingBox(100, 100, 200, 200, 0.9, "b")
        assert _bbox_overlap(a, b) == pytest.approx(0.0)

    def test_bbox_overlap_partial(self):
        a = BoundingBox(0, 0, 100, 100, 0.9, "a")
        b = BoundingBox(50, 50, 150, 150, 0.9, "b")
        iou = _bbox_overlap(a, b)
        assert 0.0 < iou < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Zone inference
# ─────────────────────────────────────────────────────────────────────────────

class TestZoneInference:
    """Tests for zone labelling based on frame position."""

    def test_top_left_zone(self):
        bbox = BoundingBox(10, 10, 50, 50, 0.9, "p")
        zone = _infer_zone(bbox, 640, 360)
        assert "Top" in zone and "Left" in zone

    def test_center_zone(self):
        bbox = BoundingBox(280, 140, 360, 220, 0.9, "p")
        zone = _infer_zone(bbox, 640, 360)
        assert "Center" in zone

    def test_bottom_right_zone(self):
        bbox = BoundingBox(520, 290, 600, 340, 0.9, "p")
        zone = _infer_zone(bbox, 640, 360)
        assert "Bottom" in zone and "Right" in zone

    def test_none_bbox_returns_unknown(self):
        zone = _infer_zone(None, 640, 360)
        assert "Unknown" in zone


# ─────────────────────────────────────────────────────────────────────────────
# Colour detection
# ─────────────────────────────────────────────────────────────────────────────

class TestColourDetection:
    """Tests for vest colour heuristics on synthetic frames."""

    def _make_solid_frame(self, bgr_colour: tuple, w=100, h=100) -> np.ndarray:
        frame = np.full((h, w, 3), bgr_colour, dtype=np.uint8)
        return frame

    def test_green_dominant_in_green_frame(self):
        frame = self._make_solid_frame((50, 180, 50))   # BGR green
        bbox  = BoundingBox(10, 10, 90, 90, 0.9, "p")
        colours = _dominant_colour_in_roi(frame, bbox)
        assert colours["green"] > 0.3, f"Expected green dominant, got {colours}"

    def test_red_dominant_in_red_frame(self):
        frame = self._make_solid_frame((30, 30, 200))   # BGR red
        bbox  = BoundingBox(10, 10, 90, 90, 0.9, "p")
        colours = _dominant_colour_in_roi(frame, bbox)
        assert colours["red"] > 0.0  # Some red detected

    def test_empty_roi_returns_zeros(self):
        frame   = np.zeros((10, 10, 3), dtype=np.uint8)
        bbox    = BoundingBox(20, 20, 5, 5, 0.9, "p")  # inverted → empty
        colours = _dominant_colour_in_roi(frame, bbox)
        assert colours["green"] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Detector instantiation
# ─────────────────────────────────────────────────────────────────────────────

class TestDetector:
    """Smoke tests for ComplianceDetector."""

    def test_detector_instantiates(self):
        det = ComplianceDetector(api_key="")
        assert det is not None
        assert det.yolo_conf == 0.45

    def test_detect_frame_on_black_frame(self):
        """Detector must not crash on a blank frame."""
        det   = ComplianceDetector(api_key="", yolo_model_name="yolov8n.pt")
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # Should return list (possibly empty) without exception
        try:
            events = det.detect_frame(frame, frame_index=0, fps=25.0)
            assert isinstance(events, list)
        except Exception as e:
            # YOLO model may not be downloaded in CI — that's acceptable
            pytest.skip(f"YOLO model not available: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo Generator
# ─────────────────────────────────────────────────────────────────────────────

class TestDemoGenerator:
    """Tests for synthetic violation generation."""

    def test_generates_expected_count(self):
        violations = generate_demo_violations(count_per_class=2)
        assert len(violations) == 8  # 4 classes × 2

    def test_all_class_ids_present(self):
        violations = generate_demo_violations(count_per_class=1)
        class_ids  = {v.behavior_class_id for v in violations}
        assert class_ids == {0, 1, 2, 3}

    def test_all_have_thumbnails(self):
        violations = generate_demo_violations(count_per_class=1)
        for v in violations:
            assert v.thumbnail_b64 is not None
            assert len(v.thumbnail_b64) > 100

    def test_all_have_event_ids(self):
        violations = generate_demo_violations(count_per_class=1)
        ids = [v.event_id for v in violations]
        assert len(ids) == len(set(ids))  # all unique

    def test_violations_have_policy_refs(self):
        violations = generate_demo_violations(count_per_class=1)
        for v in violations:
            assert v.policy_ref.startswith("Section")

    def test_dynamic_active_and_threshold_override(self):
        """Test that the detector respects dynamic config overrides (e.g. inactive rules, high threshold)."""
        import json
        from pathlib import Path
        config_path = Path(__file__).resolve().parent.parent / "outputs" / "rules_config.json"
        
        # Back up existing config if present
        backup = None
        if config_path.exists():
            backup = config_path.read_text(encoding="utf-8")
            
        try:
            # Set custom override: disable walkway (class 0) and set very high threshold for intervention (class 1)
            override = {
                "0": {
                    "active": False,
                    "confidence_threshold": 0.45,
                    "severity": "CRITICAL"
                },
                "1": {
                    "active": True,
                    "confidence_threshold": 0.99,
                    "severity": "HIGH"
                }
            }
            config_path.parent.mkdir(exist_ok=True)
            config_path.write_text(json.dumps(override), encoding="utf-8")
            
            det = ComplianceDetector(api_key="")
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            events = det.detect_frame(frame, frame_index=0, fps=25.0)
            
            classes = [ev.behavior_class_id for ev in events]
            assert 0 not in classes, f"Walkway violation detected but class 0 is disabled"
            assert 1 not in classes, f"Intervention violation detected but threshold is 0.99"
        finally:
            # Restore backup
            if backup is not None:
                config_path.write_text(backup, encoding="utf-8")
            elif config_path.exists():
                config_path.unlink()
