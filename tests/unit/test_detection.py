"""Detection helpers: geometry, colour, temporal smoothing, camera identity.

These are the pieces that read the policy's observable indicators off a frame, so they are
tested on synthetic images where the right answer is known exactly - no video required.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from aegisflow.core.enums import BehaviorClass, DetectionMethod
from aegisflow.core.settings import get_settings
from aegisflow.detection import geometry as geo
from aegisflow.detection.temporal import FrameVerdict, PersistenceTracker
from aegisflow.detection.video import _resize


@pytest.fixture(scope="module")
def colours():
    return get_settings().tuning.color


class TestBoxGeometry:
    def test_iou_of_identical_boxes_is_one(self):
        box = (0.0, 0.0, 10.0, 10.0)
        assert geo.iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes_have_zero_iou(self):
        assert geo.iou((0, 0, 5, 5), (10, 10, 15, 15)) == 0.0

    def test_half_overlap(self):
        assert geo.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)

    def test_contains_fraction(self):
        assert geo.contains_fraction((2, 2, 4, 4), (0, 0, 10, 10)) == pytest.approx(1.0)
        assert geo.contains_fraction((-5, 0, 5, 10), (0, 0, 10, 10)) == pytest.approx(0.5)

    def test_gap_is_zero_when_touching(self):
        assert geo.gap((0, 0, 5, 5), (5, 0, 10, 5)) == 0.0
        assert geo.gap((0, 0, 5, 5), (8, 0, 10, 5)) == pytest.approx(3.0)

    def test_torso_roi_excludes_head_and_legs(self):
        person = (100.0, 0.0, 200.0, 200.0)
        x1, y1, x2, y2 = geo.torso_roi(person, (400, 400, 3))
        assert y1 > 0 and y2 < 200
        assert x1 > 100 and x2 < 200

    def test_clip_box_clamps_to_the_frame(self):
        box = geo.clip_box((-50.0, -50.0, 900.0, 900.0), (360, 640, 3))
        assert box == (0.0, 0.0, 640.0, 360.0)

    def test_crop_of_an_inverted_box_is_empty(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        assert geo.crop(image, (60.0, 60.0, 20.0, 20.0)).size == 0


class TestColour:
    def test_green_patch_reads_as_green(self, green_patch, colours):
        assert geo.colour_ratio(green_patch, colours.green_vest) > 0.9

    def test_green_patch_does_not_read_as_red(self, green_patch, colours):
        ratio = geo.red_ratio(green_patch, colours.red_vest_low, colours.red_vest_high)
        assert ratio < 0.05

    def test_red_patch_reads_as_red(self, red_patch, colours):
        ratio = geo.red_ratio(red_patch, colours.red_vest_low, colours.red_vest_high)
        assert ratio > 0.9

    def test_red_patch_does_not_read_as_green(self, red_patch, colours):
        assert geo.colour_ratio(red_patch, colours.green_vest) < 0.05

    def test_vest_threshold_separates_the_two(self, green_patch, red_patch, colours):
        """The 0.12 threshold sits in the measured gap between the classes."""
        green = geo.colour_ratio(green_patch, colours.green_vest)
        red_as_green = geo.colour_ratio(red_patch, colours.green_vest)
        assert red_as_green < colours.min_vest_pixel_ratio < green

    def test_empty_image_is_safe(self, colours):
        empty = np.empty((0, 0, 3), dtype=np.uint8)
        assert geo.colour_ratio(empty, colours.green_vest) == 0.0

    def test_red_wraps_around_the_hue_origin(self, colours):
        """Red spans H=179->0, so it needs two bands; one alone misses half of it."""
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[:, :] = (30, 30, 200)
        both = geo.red_ratio(image, colours.red_vest_low, colours.red_vest_high)
        assert both > 0.9


class TestPolygons:
    def test_point_in_polygon(self):
        square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        assert geo.point_in_polygon((5.0, 5.0), square)
        assert not geo.point_in_polygon((15.0, 5.0), square)

    def test_degenerate_polygon_contains_nothing(self):
        assert not geo.point_in_polygon((1.0, 1.0), [(0.0, 0.0), (1.0, 1.0)])

    def test_denormalise(self):
        pixels = geo.denormalise_polygon([(0.5, 0.5), (1.0, 1.0)], 640, 360)
        assert pixels == [(320.0, 180.0), (640.0, 360.0)]

    def test_largest_contour_respects_min_area(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (30, 30), 255, -1)
        assert geo.largest_contour(mask, min_area=10) is not None
        assert geo.largest_contour(mask, min_area=100000) is None

    def test_merge_overlapping_counts_each_block_once(self):
        """The safe/unsafe boundary is a count of 2 vs 3, so double-counting matters."""
        boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (50, 50, 60, 60)]
        assert len(geo.merge_overlapping(boxes)) == 2


class TestCameraFingerprint:
    def test_identical_scenes_are_close(self):
        image = np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8)
        a = geo.scene_fingerprint(image)
        b = geo.scene_fingerprint(image.copy())
        assert geo.fingerprint_distance(a, b) == pytest.approx(0.0)

    def test_different_scenes_are_far(self):
        dark = np.full((360, 640, 3), 20, dtype=np.uint8)
        bright = np.full((360, 640, 3), 200, dtype=np.uint8)
        distance = geo.fingerprint_distance(
            geo.scene_fingerprint(dark), geo.scene_fingerprint(bright)
        )
        assert distance > 100

    def test_fingerprint_ignores_small_moving_objects(self):
        """It must identify the camera, not the frame's contents."""
        base = np.full((360, 640, 3), 120, dtype=np.uint8)
        moved = base.copy()
        cv2.rectangle(moved, (300, 200), (330, 260), (10, 10, 10), -1)
        distance = geo.fingerprint_distance(
            geo.scene_fingerprint(base), geo.scene_fingerprint(moved)
        )
        assert distance < 12, "a person-sized object must not change camera identity"

    def test_mismatched_shapes_are_infinitely_far(self):
        a = np.zeros((9, 16), dtype=np.float32)
        b = np.zeros((4, 4), dtype=np.float32)
        assert geo.fingerprint_distance(a, b) == float("inf")


class TestFrameResize:
    def test_downscales_the_long_edge(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        resized, scale = _resize(frame, 640)
        assert max(resized.shape[:2]) == 640
        assert scale == pytest.approx(640 / 1920)

    def test_small_frames_are_left_alone(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        resized, scale = _resize(frame, 640)
        assert resized.shape == frame.shape
        assert scale == 1.0


def verdict(behavior, index, confidence=0.8, method=DetectionMethod.HSV, ambiguous=False):
    return FrameVerdict(
        behavior_class=behavior,
        frame_index=index,
        timestamp_s=index * 0.25,
        confidence=confidence,
        method=method,
        ambiguous=ambiguous,
    )


class TestPersistence:
    BEHAVIOR = BehaviorClass.SAFE_WALKWAY_VIOLATION

    def test_a_short_run_does_not_fire(self):
        tracker = PersistenceTracker({self.BEHAVIOR: 3})
        for index in range(2):
            tracker.update([verdict(self.BEHAVIOR, index)])
        assert tracker.events() == []

    def test_a_long_enough_run_fires(self):
        tracker = PersistenceTracker({self.BEHAVIOR: 3})
        for index in range(3):
            tracker.update([verdict(self.BEHAVIOR, index)])
        events = tracker.events()
        assert len(events) == 1
        assert events[0].first_frame_index == 0
        assert events[0].frame_count == 3

    def test_a_gap_breaks_the_run(self):
        """Single-frame flicker is the main false-positive source; a gap must reset."""
        tracker = PersistenceTracker({self.BEHAVIOR: 3})
        tracker.update([verdict(self.BEHAVIOR, 0)])
        tracker.update([verdict(self.BEHAVIOR, 1)])
        tracker.update([])  # nothing seen
        tracker.update([verdict(self.BEHAVIOR, 3)])
        tracker.update([verdict(self.BEHAVIOR, 4)])
        assert tracker.events() == []

    def test_classes_are_tracked_independently(self):
        """Multi-violation clips need each behaviour to persist on its own."""
        other = BehaviorClass.OPENED_PANEL_COVER
        tracker = PersistenceTracker({self.BEHAVIOR: 2, other: 4})
        for index in range(3):
            tracker.update([verdict(self.BEHAVIOR, index), verdict(other, index)])
        classes = {event.behavior_class for event in tracker.events()}
        assert classes == {self.BEHAVIOR}

    def test_both_classes_can_fire(self):
        other = BehaviorClass.OPENED_PANEL_COVER
        tracker = PersistenceTracker({self.BEHAVIOR: 2, other: 2})
        for index in range(3):
            tracker.update([verdict(self.BEHAVIOR, index), verdict(other, index)])
        assert len(tracker.events()) == 2

    def test_confidence_is_averaged_over_the_run(self):
        tracker = PersistenceTracker({self.BEHAVIOR: 3})
        for index, confidence in enumerate((0.6, 0.8, 1.0)):
            tracker.update([verdict(self.BEHAVIOR, index, confidence=confidence)])
        assert tracker.events()[0].confidence == pytest.approx(0.8)

    def test_a_vlm_consultation_is_recorded_in_the_method(self):
        """The audit trail must show when a model was consulted."""
        tracker = PersistenceTracker({self.BEHAVIOR: 3})
        tracker.update([verdict(self.BEHAVIOR, 0)])
        tracker.update([verdict(self.BEHAVIOR, 1, method=DetectionMethod.VLM_TIEBREAK)])
        tracker.update([verdict(self.BEHAVIOR, 2)])
        assert tracker.events()[0].method is DetectionMethod.VLM_TIEBREAK

    def test_ambiguous_only_when_the_majority_of_frames_were(self):
        tracker = PersistenceTracker({self.BEHAVIOR: 3})
        tracker.update([verdict(self.BEHAVIOR, 0, ambiguous=True)])
        tracker.update([verdict(self.BEHAVIOR, 1, ambiguous=False)])
        tracker.update([verdict(self.BEHAVIOR, 2, ambiguous=False)])
        assert tracker.events()[0].ambiguous is False

    def test_numeric_evidence_takes_the_worst_observed_value(self):
        """For a threshold rule, the worst observed state is the one that matters."""
        tracker = PersistenceTracker({self.BEHAVIOR: 2})
        for index, blocks in enumerate((3, 5)):
            item = verdict(self.BEHAVIOR, index)
            item.evidence["block_count"] = blocks
            tracker.update([item])
        assert tracker.events()[0].evidence["block_count"] == 5
