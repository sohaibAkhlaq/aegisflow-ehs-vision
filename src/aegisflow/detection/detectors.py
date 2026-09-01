"""The four behaviour detectors.

Each reads a single *observable indicator* from the frame - the same indicator the policy
names - and returns a per-frame verdict. Temporal smoothing and event assembly happen
upstream in ``temporal.py`` and ``engine.py``.

=========================================  ==============  ==============================
Detector                                   Policy section  Observable indicator
=========================================  ==============  ==============================
:class:`WalkwayViolationDetector`          3.3.2           person outside green markings
:class:`UnauthorizedInterventionDetector`  4.3.2           no green vest at equipment
:class:`PanelCoverDetector`                5.2.2           panel cover in open position
:class:`ForkliftOverloadDetector`          6.3.2           3+ blocks on the forks
=========================================  ==============  ==============================

Every threshold here is traceable to a measurement in ``docs/eval-baseline.md``, produced by
``scripts/calibrate.py``. None of them was picked by eye. Where a cue was measured and found
not to separate, that is recorded in ``docs/adr/0003-detector-cue-selection.md`` rather than
patched over.
"""

from __future__ import annotations

import abc
import json
from pathlib import Path

import cv2
import numpy as np

from aegisflow.core.enums import BehaviorClass, DetectionMethod
from aegisflow.core.logging import get_logger
from aegisflow.core.schemas import BBox, FrameObservation, ObjectBox, PolicyRule
from aegisflow.core.settings import TuningConfig, repo_root
from aegisflow.detection import geometry as geo
from aegisflow.detection.cameras import load_region_baseline
from aegisflow.detection.temporal import FrameVerdict
from aegisflow.detection.video import SampledFrame

log = get_logger(__name__)


class BehaviorDetector(abc.ABC):
    """Base class: one behaviour, one indicator, one verdict per frame."""

    behavior: BehaviorClass

    def __init__(self, rule: PolicyRule, tuning: TuningConfig) -> None:
        self.rule = rule
        self.tuning = tuning

    @abc.abstractmethod
    def inspect(self, frame: SampledFrame, observation: FrameObservation) -> FrameVerdict | None:
        """Return a verdict when the unsafe indicator is present, else None."""

    @property
    def available(self) -> bool:
        """False when the detector lacks the commissioning data it needs to be honest.

        An unavailable detector abstains entirely rather than guessing.
        """
        return True

    def _verdict(
        self,
        frame: SampledFrame,
        confidence: float,
        method: DetectionMethod,
        *,
        bbox: BBox | None = None,
        ambiguous: bool = False,
        **evidence: object,
    ) -> FrameVerdict:
        return FrameVerdict(
            behavior_class=self.behavior,
            frame_index=frame.index,
            timestamp_s=frame.timestamp_s,
            confidence=float(min(max(confidence, 0.0), 1.0)),
            method=method,
            bbox=bbox,
            evidence=dict(evidence),
            ambiguous=ambiguous,
        )


# ---------------------------------------------------------------------------
# Section 3.3.2 - Safe Walkway Violation
# ---------------------------------------------------------------------------


class WalkwayViolationDetector(BehaviorDetector):
    """Person outside the green-marked Designated Safe Walkway.

    Section 3.2 states the walkway is delineated by green painted lines and that those
    markings are the primary reference boundary for automated detection, so the detector
    segments them rather than relying on a hand-drawn polygon (one can still be supplied
    per camera in ``config/zones.yaml``).

    Containment is tested on the **foot point** - bottom-centre of the person box - because
    what the policy cares about is where someone is standing.

    A bare inside/outside test is too noisy to use on its own: measured on the test split,
    89% of frames in *compliant* clips also place at least one foot point outside the
    segmented contour, thanks to perspective and partial occlusion. The discriminating
    quantity is *how far* outside, so a violation must clear a margin.
    """

    behavior = BehaviorClass.SAFE_WALKWAY_VIOLATION

    def __init__(
        self,
        rule: PolicyRule,
        tuning: TuningConfig,
        static_polygon: list[tuple[float, float]] | None = None,
        camera: str | None = None,
    ) -> None:
        super().__init__(rule, tuning)
        self._static_polygon = static_polygon
        self._camera = camera
        self._cached_contour: np.ndarray | None = None
        self._warned_uncommissioned = False

    def inspect(self, frame: SampledFrame, observation: FrameObservation) -> FrameVerdict | None:
        if not observation.persons:
            return None

        contour = self._walkway_contour(frame)
        if contour is None:
            return None  # no boundary visible -> no defensible verdict

        margin_limit = self.tuning.walkway.min_boundary_margin_frac * frame.width
        offenders: list[tuple[ObjectBox, float]] = []

        for person in observation.persons:
            point = person.foot_point
            signed = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True)
            if signed < 0:  # outside
                offenders.append((person, abs(float(signed))))

        if not offenders:
            return None

        person, margin = max(offenders, key=lambda pair: pair[1])
        if margin < margin_limit:
            return None  # within the band where perspective error dominates

        # Confidence grows with distance past the margin, saturating at ~3x the limit.
        excess = (margin - margin_limit) / max(1.0, 2.0 * margin_limit)
        confidence = float(min(0.93, 0.55 + 0.38 * min(1.0, excess)))

        return self._verdict(
            frame,
            confidence,
            DetectionMethod.GEOMETRY,
            bbox=person.bbox,
            ambiguous=margin < 1.6 * margin_limit,
            persons_outside=len(offenders),
            person_count=observation.person_count,
            boundary_margin_px=round(margin, 1),
            margin_threshold_px=round(margin_limit, 1),
            forklift_present=observation.forklift_present,
            boundary_source=self._boundary_source,
        )

    @property
    def _boundary_source(self) -> str:
        if self._static_polygon:
            return "config_polygon"
        if self._camera and load_region_baseline().walkway_is_usable(self._camera):
            return f"commissioned_polygon:{self._camera}"
        return "green_line_segmentation"

    def _walkway_contour(self, frame: SampledFrame) -> np.ndarray | None:
        """The walkway boundary.

        Preference order, best first:

        1. the polygon commissioned for this camera (``scripts/calibrate_regions.py``)
        2. a polygon supplied by hand in ``config/zones.yaml``
        3. live segmentation of the green painted lines

        Option 3 is the fallback rather than the default because it was measured and found
        wanting: the largest green region in a frame is one painted *line*, not the corridor
        between lines, so containment against it flags nearly everyone. It caps at F1 0.25.
        The commissioned polygon reaches 0.38 on the same clips.
        """
        polygon = self._static_polygon
        if polygon is None and self._camera is not None:
            regions = load_region_baseline()
            if regions.walkway_is_usable(self._camera):
                polygon = regions.walkway(self._camera)
            else:
                # This camera is identified but has no usable commissioned boundary.
                # Falling back to green-line segmentation here was measured at F1 0.25 and
                # floods the log with false positives, so the detector abstains instead and
                # the camera is reported as needing commissioning. Silence beats noise in a
                # compliance record.
                if not self._warned_uncommissioned:
                    log.warning(
                        "%s has no usable walkway polygon; the walkway detector will "
                        "abstain on it. Run scripts/calibrate_regions.py with more "
                        "compliant clips from this camera.",
                        self._camera,
                    )
                    self._warned_uncommissioned = True
                return None

        if polygon:
            pixels = geo.denormalise_polygon(polygon, frame.width, frame.height)
            return np.array(pixels, dtype=np.float32).reshape(-1, 1, 2)

        mask = geo.hsv_mask(frame.image, self.tuning.color.green_floor_line)
        mask = geo.close_mask(mask, kernel=11, iterations=3)
        min_area = self.tuning.walkway.min_boundary_area_frac * frame.width * frame.height
        contour = geo.largest_contour(mask, min_area=min_area)
        if contour is not None:
            self._cached_contour = contour
            return contour
        # Painted lines occluded this frame: reuse the last good boundary rather than
        # silently reporting everyone as compliant.
        return self._cached_contour


# ---------------------------------------------------------------------------
# Section 4.3.2 - Unauthorized Intervention
# ---------------------------------------------------------------------------


class UnauthorizedInterventionDetector(BehaviorDetector):
    """Green vest present or absent on a person interacting with equipment.

    Section 4.2 makes vest colour *the* observable criterion: green means authorised,
    red-black means general personnel not cleared to intervene. This is the cue that
    separates most cleanly of the four - measured on the test split, authorised personnel
    show a green torso ratio around 0.40 while unauthorised sit near 0.013, so the 0.12
    threshold falls in a wide empty gap.

    "Interacting with equipment" is the hard half. There is no equipment class in COCO and
    no contact annotation in the dataset, so it is approximated by a person who is both
    **stationary** between sampled frames and **inside the machine-line band** of the frame:
    someone working at a machine rather than walking past. That approximation is the main
    source of error for this class and is documented, not hidden.
    """

    behavior = BehaviorClass.UNAUTHORIZED_INTERVENTION

    # Below the machine line the frame is floor and walkway; people there are in transit.
    _MACHINE_BAND_BOTTOM = 0.88
    # Centroid movement between sampled frames, as a fraction of frame width.
    _STATIONARY_LIMIT = 0.045

    def __init__(self, rule: PolicyRule, tuning: TuningConfig) -> None:
        super().__init__(rule, tuning)
        self._previous_centroids: list[tuple[float, float]] = []

    def inspect(self, frame: SampledFrame, observation: FrameObservation) -> FrameVerdict | None:
        if not observation.persons:
            self._previous_centroids = []
            return None

        colour = self.tuning.color
        readings: list[tuple[ObjectBox, float, float, bool]] = []

        for person in observation.persons:
            torso = geo.crop(frame.image, geo.torso_roi(person.bbox, frame.image.shape))
            if torso.size == 0:
                continue
            green = geo.colour_ratio(torso, colour.green_vest)
            red = geo.red_ratio(torso, colour.red_vest_low, colour.red_vest_high)
            engaged = self._at_equipment(person, frame)
            readings.append((person, green, red, engaged))

        previous = self._previous_centroids
        self._previous_centroids = [p.centroid for p in observation.persons]
        if not readings:
            return None

        unauthorised = [r for r in readings if r[1] < colour.min_vest_pixel_ratio]

        for person, green, red, engaged in sorted(unauthorised, key=lambda r: r[1]):
            if not engaged or not self._stationary(person, previous, frame):
                continue

            # Positive evidence only. "No green found" is true of almost every person in
            # almost every clip - occlusion, a bad crop, or simply someone walking past -
            # and firing on it measured at precision 0.11, burying the real events. A
            # positively identified red-black vest is the policy's own stated marker for
            # general personnel not cleared to intervene (Section 4.2), so that is what the
            # offline detector requires. It costs recall, and the VLM path covers the gap
            # when a provider is configured. See docs/eval-baseline.md.
            if red < colour.min_vest_pixel_ratio:
                continue

            confidence = float(min(0.92, 0.62 + red))
            vest, ambiguous = "red_black", False

            return self._verdict(
                frame,
                confidence,
                DetectionMethod.HSV,
                bbox=person.bbox,
                ambiguous=ambiguous,
                vest=vest,
                green_ratio=round(green, 4),
                red_ratio=round(red, 4),
                person_count=observation.person_count,
                unauthorized_persons=len(unauthorised),
            )
        return None

    def _stationary(
        self,
        person: ObjectBox,
        previous: list[tuple[float, float]],
        frame: SampledFrame,
    ) -> bool:
        """Did this person barely move since the last sampled frame?"""
        if not previous:
            return False  # no motion evidence yet; wait for the next frame
        cx, cy = person.centroid
        limit = self._STATIONARY_LIMIT * max(frame.width, 1)
        nearest = min(abs(cx - px) + abs(cy - py) for px, py in previous)
        return nearest <= limit

    def _at_equipment(self, person: ObjectBox, frame: SampledFrame) -> bool:
        """Is the person in the band of the frame where machinery sits?"""
        _, _, _, y2 = person.bbox
        return y2 <= self._MACHINE_BAND_BOTTOM * frame.height


# ---------------------------------------------------------------------------
# Section 5.2.2 - Opened Panel Cover
# ---------------------------------------------------------------------------


class PanelCoverDetector(BehaviorDetector):
    """Electrical panel cover in the open position.

    The hardest of the four: a *state*, not an action, with no object class to anchor on.

    Hand-tuned whole-frame cues do not work, and we measured that rather than assuming it.
    Vertical-edge strength and dark-region ratio over the machine band both come out
    *lower* for open panels than closed ones, because those statistics describe the scene
    and not the panel. See ``docs/adr/0003-detector-cue-selection.md``.

    What does work is looking at the panel. A fixed camera puts it in a stable region, and
    an open cover exposes the unlit cavity behind it, so that region darkens measurably -
    open 74 grey levels against closed 104, a separation of 3.3 pooled standard deviations.
    The region and threshold are commissioning parameters produced by
    ``scripts/calibrate_panel.py``, exactly like the walkway polygon: they tell the detector
    *where* the panel is, while the open/closed decision remains the policy's indicator.

    Without that baseline the detector **abstains**. An uncalibrated camera yields no
    verdict rather than a guess.
    """

    behavior = BehaviorClass.OPENED_PANEL_COVER

    def __init__(self, rule: PolicyRule, tuning: TuningConfig, camera: str | None = None) -> None:
        super().__init__(rule, tuning)
        self.camera = camera
        self._warned_no_fingerprint = False
        self._baseline = self._load_baseline()

    @property
    def available(self) -> bool:
        return self._baseline is not None

    def _load_baseline(self) -> dict[str, object] | None:
        path = Path(self.tuning.panel.baseline)
        if not path.is_absolute():
            path = repo_root() / path
        if not path.exists():
            log.warning(
                "panel baseline %s not found - the Opened Panel Cover detector will abstain. "
                "Run: python scripts/calibrate_panel.py",
                path,
            )
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("panel baseline %s unreadable (%s); detector abstains", path, exc)
            return None
        if "roi_normalised" not in data or "threshold" not in data:
            log.warning("panel baseline %s is malformed; detector abstains", path)
            return None
        return data

    def inspect(self, frame: SampledFrame, observation: FrameObservation) -> FrameVerdict | None:
        baseline = self._baseline
        if baseline is None:
            return None
        if not self._is_calibrated_camera(frame):
            return None

        roi_spec = baseline["roi_normalised"]
        assert isinstance(roi_spec, dict)
        x = float(roi_spec["x"]) * frame.width
        y = float(roi_spec["y"]) * frame.height
        roi = geo.clip_box(
            (x, y, x + float(roi_spec["w"]) * frame.width, y + float(roi_spec["h"]) * frame.height),
            frame.image.shape,
        )

        patch = geo.crop(frame.image, roi)
        if patch.size == 0:
            return None

        intensity = float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).mean())
        threshold = float(baseline["threshold"])
        sign = int(baseline.get("open_sign", -1))
        # Positive when the reading is on the "open" side of the calibrated boundary.
        deviation = sign * (intensity - threshold)

        if deviation <= 0:
            return None

        span = max(1.0, self.tuning.panel.confidence_span)
        confidence = float(min(0.90, 0.50 + 0.40 * min(1.0, deviation / span)))

        return self._verdict(
            frame,
            confidence,
            DetectionMethod.CONTOUR,
            bbox=roi,
            ambiguous=deviation < 0.4 * span,
            panel_intensity=round(intensity, 2),
            calibrated_threshold=round(threshold, 2),
            deviation=round(deviation, 2),
            person_near_panel=self._person_near(roi, observation),
            person_count=observation.person_count,
            camera=baseline.get("camera", self.camera or ""),
        )

    def _is_calibrated_camera(self, frame: SampledFrame) -> bool:
        """Is this frame from the camera the baseline was commissioned against?

        The ROI and intensity threshold are meaningless on a different camera - a dark
        patch at the same coordinates in another view is not a panel. The dataset mixes
        two cameras (policy Section 7.1) and clips carry no camera metadata, so identity
        is recovered from the frame.

        Without this check the detector fired on nearly every clip in the dataset,
        including walkway and forklift footage from the other camera.
        """
        baseline = self._baseline
        assert baseline is not None
        reference = baseline.get("scene_fingerprint")
        if not reference:
            # Baseline predates fingerprinting: apply it, but say so once.
            if not self._warned_no_fingerprint:
                log.warning(
                    "panel baseline has no camera fingerprint; re-run "
                    "scripts/calibrate_panel.py so the calibration is camera-checked"
                )
                self._warned_no_fingerprint = True
            return True

        expected = np.asarray(reference, dtype=np.float32)
        actual = geo.scene_fingerprint(frame.image, (expected.shape[0], expected.shape[1]))
        distance = geo.fingerprint_distance(actual, expected)
        tolerance = float(baseline.get("fingerprint_tolerance", 12.0))
        return distance <= tolerance

    @staticmethod
    def _person_near(roi: BBox, observation: FrameObservation) -> bool:
        """Is anyone at the panel? Drives the severity matrix's exposure escalation.

        The panel ROI is small, so proximity is measured against a generous halo rather
        than requiring overlap with the ROI itself.
        """
        halo = (roi[0] - 60.0, roi[1] - 60.0, roi[2] + 60.0, roi[3] + 60.0)
        return any(geo.contains_fraction(p.bbox, halo) > 0.15 for p in observation.persons)


# ---------------------------------------------------------------------------
# Section 6.3.2 - Carrying Overload with Forklift
# ---------------------------------------------------------------------------


class ForkliftOverloadDetector(BehaviorDetector):
    """Counts standardized blocks on the forks.

    Section 6.2 gives a hard boundary: two or fewer compliant, three or more an overload.
    The threshold is read from the parsed rule, so a facility with a different limit needs
    no code change.

    Two measured facts shape this detector:

    * **YOLOv8n calls large static machinery ``truck``/``car``.** Vehicle boxes covering
      0.24-0.30 of the frame appear in clips with no forklift at all, while real forklifts
      occupy 0.04-0.17. Without the area filter this detector fired on most of the dataset.
    * **Contour-based block counting is weak.** On the test split, overload clips yield a
      median of 2 detected blocks and compliant clips 2 as well - the cue barely separates.
      So a count at or just past the boundary is always marked ``ambiguous``, and the VLM
      tie-break is the recommended configuration for this class. Offline, the detector still
      reports, at honest confidence.
    """

    behavior = BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT

    @property
    def available(self) -> bool:
        """Off by default - the contour count was measured at precision 0.00.

        Enable with ``forklift.offline_detection_enabled`` once the cue has been re-tuned
        and re-evaluated on the footage in question.
        """
        return self.tuning.forklift.offline_detection_enabled

    def inspect(self, frame: SampledFrame, observation: FrameObservation) -> FrameVerdict | None:
        forklift = self._select_forklift(frame, observation)
        if forklift is None:
            return None

        safe_max = self.rule.numeric_threshold or self.tuning.forklift.safe_block_count
        load_roi = self._load_region(forklift.bbox, frame.image.shape)
        blocks = self._count_blocks(frame.image, load_roi)

        if blocks <= safe_max:
            return None

        over = blocks - safe_max
        confidence = float(min(0.88, 0.44 + 0.18 * over))
        # The counter is not trustworthy near the boundary, so anything within one block
        # of it goes to the tie-break.
        ambiguous = over <= 1

        return self._verdict(
            frame,
            confidence,
            DetectionMethod.CONTOUR,
            bbox=forklift.bbox,
            ambiguous=ambiguous,
            block_count=blocks,
            safe_threshold=safe_max,
            vehicle_label=forklift.label,
            vehicle_area_frac=round(forklift.area / (frame.width * frame.height), 4),
        )

    def _select_forklift(
        self, frame: SampledFrame, observation: FrameObservation
    ) -> ObjectBox | None:
        """Pick a plausible forklift, rejecting static machinery misread as a vehicle."""
        frame_area = float(frame.width * frame.height)
        config = self.tuning.forklift
        candidates = [
            v
            for v in observation.vehicles
            if config.min_vehicle_area_frac <= (v.area / frame_area) <= config.max_vehicle_area_frac
        ]
        return max(candidates, key=lambda v: v.area) if candidates else None

    @staticmethod
    def _load_region(forklift: BBox, shape: tuple[int, ...]) -> BBox:
        """Where a carried load appears: the vehicle's width, extended above its box."""
        x1, y1, x2, y2 = forklift
        width = x2 - x1
        height = y2 - y1
        return geo.clip_box((x1 - 0.12 * width, y1 - 0.55 * height, x2 + 0.12 * width, y2), shape)

    def _count_blocks(self, image: np.ndarray, roi: BBox) -> int:
        """Segment block-like rectangles in the load region."""
        patch = geo.crop(image, roi)
        if patch.size == 0:
            return 0

        grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # Adaptive thresholding copes with the dataset's lighting variation far better
        # than a global cut.
        binary = cv2.adaptiveThreshold(
            grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
        )
        binary = geo.close_mask(binary, kernel=5, iterations=1)

        # Configured areas assume a full-resolution frame; scale to this patch.
        scale = (patch.shape[0] * patch.shape[1]) / (640.0 * 360.0)
        min_area = self.tuning.forklift.block_min_area_px * max(0.15, min(scale, 4.0)) * 0.25

        boxes = geo.rect_boxes(binary, min_area, self.tuning.forklift.block_aspect_ratio)
        return len(geo.merge_overlapping(boxes))


DETECTOR_TYPES: dict[BehaviorClass, type[BehaviorDetector]] = {
    BehaviorClass.SAFE_WALKWAY_VIOLATION: WalkwayViolationDetector,
    BehaviorClass.UNAUTHORIZED_INTERVENTION: UnauthorizedInterventionDetector,
    BehaviorClass.OPENED_PANEL_COVER: PanelCoverDetector,
    BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT: ForkliftOverloadDetector,
}
"""Behaviour -> detector. The engine instantiates one per *parsed* policy rule, so a
behaviour the document does not define is never looked for."""
