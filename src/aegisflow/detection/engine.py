"""Module 1 - Detection Engine.

Composes the pieces into ``clip -> list[DetectionRecord]``:

1. sample frames (``video.py``)
2. locate people and vehicles, batched (``yolo.py``)
3. run each detector the *parsed policy* defines (``detectors.py``)
4. keep only verdicts that persist (``temporal.py``)
5. adjudicate borderline cases with the VLM, if a provider is configured
6. emit one record per sustained violation, with clip-level context attached

Step 3 is where policy grounding is enforced: detectors are instantiated from
``PolicyRuleSet.unsafe_rules``, so a behaviour the document does not define is not looked
for, and the indicator text and section reference travel with the record.
"""

from __future__ import annotations

import re

from aegisflow.core.enums import BehaviorClass, DetectionMethod
from aegisflow.core.errors import LLMProviderError
from aegisflow.core.logging import get_logger
from aegisflow.core.schemas import (
    DetectionRecord,
    FrameContext,
    FrameObservation,
    PolicyRuleSet,
)
from aegisflow.core.settings import Settings, get_settings
from aegisflow.core.zoning import camera_for, clip_id_for, resolve_zone, walkway_polygon
from aegisflow.detection.cameras import load_camera_registry, zone_for_camera
from aegisflow.detection.detectors import (
    DETECTOR_TYPES,
    BehaviorDetector,
    PanelCoverDetector,
    WalkwayViolationDetector,
)
from aegisflow.detection.temporal import FrameVerdict, PersistenceTracker, SustainedEvent
from aegisflow.detection.video import SampledFrame, VideoInfo, encode_png, iter_frames, probe
from aegisflow.detection.vlm import VlmIndicatorDetector
from aegisflow.detection.yolo import YoloDetector
from aegisflow.llm.base import LLMProvider

log = get_logger(__name__)

_INT_RE = re.compile(r"-?\d+")


class DetectionEngine:
    """Runs the detection pipeline over clips.

    Construct once per run: the YOLO model and the detector set are reused across clips.
    """

    def __init__(
        self,
        rule_set: PolicyRuleSet,
        settings: Settings | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.rule_set = rule_set
        self.provider = provider
        self._yolo = YoloDetector(self.settings.tuning.detection)
        self._vlm_calls = 0
        self._abstain_logged: set[BehaviorClass] = set()
        self._vlm = (
            VlmIndicatorDetector(
                rule_set,
                provider,
                safe_block_count=self.settings.tuning.forklift.safe_block_count,
            )
            if provider is not None and not provider.is_offline
            else None
        )

    # ------------------------------------------------------------------ public

    def warmup(self) -> None:
        self._yolo.warmup()

    @property
    def vlm_calls(self) -> int:
        """Total VLM consultations this run - reported in the evaluation summary."""
        return self._vlm_calls

    async def analyse(self, clip_path: str) -> tuple[list[DetectionRecord], VideoInfo, int]:
        """Analyse one clip.

        Returns ``(records, video_info, frames_analysed)``.
        """
        tuning = self.settings.tuning
        info = probe(clip_path)
        clip_id = clip_id_for(clip_path)

        frames = list(
            iter_frames(
                clip_path,
                sample_fps=tuning.video.sample_fps,
                imgsz=tuning.video.infer_imgsz,
                max_frames=tuning.video.max_frames_per_clip,
            )
        )
        observations = self._yolo.observe_batch(frames)

        # Camera identity comes from the image, never from the clip's folder name - the
        # folder is the ground-truth label, and using it here would leak it into inference.
        camera = load_camera_registry().identify(frames[0].image) if frames else None
        zone = zone_for_camera(camera, self.settings) or resolve_zone(
            clip_path, settings=self.settings
        )

        detectors = self._build_detectors(zone, camera)
        tracker = PersistenceTracker(
            required_frames={
                behavior: tuning.detection.persistence_for(behavior.value) for behavior in detectors
            }
        )

        for frame, observation in zip(frames, observations, strict=False):
            verdicts: list[FrameVerdict] = []
            for detector in detectors.values():
                verdict = detector.inspect(frame, observation)
                if verdict is not None:
                    verdicts.append(verdict)
            tracker.update(verdicts)

        # A configured vision model answers the policy's indicator questions directly.
        # Its verdicts are merged as sustained events rather than run through the frame
        # tracker, because it inspects a couple of representative frames, not every frame.
        vlm_events = await self._vlm_events(frames)

        context = self._clip_context(observations, tracker)
        records: list[DetectionRecord] = []

        classical = {e.behavior_class: e for e in tracker.events()}
        merged: dict[BehaviorClass, SustainedEvent] = dict(classical)
        # The model's answer wins where both spoke: on this footage it is the more reliable
        # reader of every indicator except the panel, which has a commissioned baseline.
        for behavior, event in vlm_events.items():
            if behavior is BehaviorClass.OPENED_PANEL_COVER and behavior in classical:
                continue
            merged[behavior] = event

        for behavior in sorted(merged, key=lambda b: b.value):
            event = merged[behavior]
            if event.method is not DetectionMethod.VLM_TIEBREAK:
                event = await self._maybe_tiebreak(event, frames, observations)
                if event is None:
                    continue  # tie-break overturned the detection
            records.append(self._to_record(event, clip_id, zone, context))

        log.debug("%s: %d frames, %d violation(s)", clip_id, tracker.frames_seen, len(records))
        return records, info, tracker.frames_seen

    # ----------------------------------------------------------------- internal

    async def _vlm_events(self, frames: list[SampledFrame]) -> dict[BehaviorClass, SustainedEvent]:
        """Run the vision model over a few frames, if one is configured."""
        if self._vlm is None or not self._vlm.enabled or not frames:
            return {}

        budget = self.settings.tuning.vlm_tiebreak.max_calls_per_clip
        before = self._vlm.calls
        verdicts = await self._vlm.inspect_clip(frames, max_frames=budget)
        self._vlm_calls += self._vlm.calls - before

        grouped: dict[BehaviorClass, list[FrameVerdict]] = {}
        for verdict in verdicts:
            grouped.setdefault(verdict.behavior_class, []).append(verdict)

        events: dict[BehaviorClass, SustainedEvent] = {}
        for behavior, group in grouped.items():
            first = min(group, key=lambda v: v.frame_index)
            evidence: dict[str, object] = {}
            for verdict in group:
                evidence.update(verdict.evidence)
            evidence["vlm_frames_agreeing"] = len(group)
            events[behavior] = SustainedEvent(
                behavior_class=behavior,
                first_frame_index=first.frame_index,
                first_timestamp_s=first.timestamp_s,
                frame_count=len(group),
                confidence=max(v.confidence for v in group),
                method=DetectionMethod.VLM_TIEBREAK,
                bboxes=(),
                evidence=evidence,
                ambiguous=False,
            )
        return events

    def _build_detectors(
        self, zone: str, camera: str | None = None
    ) -> dict[BehaviorClass, BehaviorDetector]:
        """One detector per unsafe rule the policy actually defines."""
        detectors: dict[BehaviorClass, BehaviorDetector] = {}
        for rule in self.rule_set.unsafe_rules:
            detector_type = DETECTOR_TYPES.get(rule.behavior_class)
            if detector_type is None:
                log.warning(
                    "policy defines %s but no detector implements it", rule.behavior_class.value
                )
                continue
            if detector_type is WalkwayViolationDetector:
                detector: BehaviorDetector = WalkwayViolationDetector(
                    rule,
                    self.settings.tuning,
                    static_polygon=walkway_polygon(zone, self.settings),
                    camera=camera,
                )
            elif detector_type is PanelCoverDetector:
                detector = PanelCoverDetector(
                    rule,
                    self.settings.tuning,
                    camera=camera or camera_for(zone, self.settings),
                )
            else:
                detector = detector_type(rule, self.settings.tuning)

            if not detector.available:
                # Either missing commissioning data or a cue measured as unreliable.
                # Abstaining is deliberate: see docs/adr/0003. The VLM path covers these
                # classes when a provider is configured.
                if rule.behavior_class not in self._abstain_logged:
                    self._abstain_logged.add(rule.behavior_class)
                    log.info(
                        "%s detector abstains offline (not commissioned, or cue disabled "
                        "in config); configure an LLM provider to cover this class",
                        rule.behavior_class.value,
                    )
                continue
            detectors[rule.behavior_class] = detector
        return detectors

    @staticmethod
    def _clip_context(
        observations: list[FrameObservation], tracker: PersistenceTracker
    ) -> FrameContext:
        """Clip-level summary for the severity matrix's third signal."""
        max_persons = max((o.person_count for o in observations), default=0)
        forklift = any(o.forklift_present for o in observations)

        person_near_panel = False
        multiple_unauthorized = False
        for event in tracker.events():
            if event.behavior_class is BehaviorClass.OPENED_PANEL_COVER:
                person_near_panel = bool(event.evidence.get("person_near_panel", False))
            if event.behavior_class is BehaviorClass.UNAUTHORIZED_INTERVENTION:
                unauthorized = event.evidence.get("unauthorized_persons", 0)
                multiple_unauthorized = isinstance(unauthorized, int) and unauthorized > 1

        return FrameContext(
            max_person_count=max_persons,
            forklift_present=forklift,
            person_near_panel=person_near_panel,
            multiple_unauthorized_persons=multiple_unauthorized,
            frames_analysed=len(observations),
        )

    def _to_record(
        self,
        event: SustainedEvent,
        clip_id: str,
        zone: str,
        context: FrameContext,
    ) -> DetectionRecord:
        rule = self.rule_set.require_rule(event.behavior_class)
        return DetectionRecord(
            clip_id=clip_id,
            behavior_class=event.behavior_class,
            confidence=event.confidence,
            detection_method=event.method,
            first_frame_index=event.first_frame_index,
            first_timestamp_s=event.first_timestamp_s,
            frame_count=event.frame_count,
            bboxes=event.bboxes,
            description=self._describe(event, rule.observable_indicator),
            zone=zone,
            context=context,
            evidence=event.evidence,
            ambiguous=event.ambiguous,
        )

    @staticmethod
    def _describe(event: SustainedEvent, indicator: str) -> str:
        """Human-readable ``event_description``, phrased around the policy's indicator."""
        detail = ""
        if (blocks := event.evidence.get("block_count")) is not None:
            detail = f" Counted {blocks} blocks on the forks."
        elif (vest := event.evidence.get("vest")) is not None:
            detail = f" Vest classified as {vest}."
        elif (margin := event.evidence.get("boundary_margin_px")) is not None:
            detail = f" Foot position {margin} px beyond the green boundary."
        elif event.evidence.get("cavity_found") is not None:
            detail = " Open-cover geometry observed on the panel face."

        return (
            f"{event.behavior_class.display_name} observed from "
            f"{event.first_timestamp_s:.1f}s, sustained over {event.frame_count} sampled "
            f"frame(s). Policy indicator: {indicator}.{detail}"
        )

    # ------------------------------------------------------------- VLM tie-break

    async def _maybe_tiebreak(
        self,
        event: SustainedEvent,
        frames: list[SampledFrame],
        observations: list[FrameObservation],
    ) -> SustainedEvent | None:
        """Ask a vision model about a borderline detection.

        Only the forklift block count is adjudicated: it is the one indicator that reduces
        to a closed numeric question a VLM answers reliably, and it is the case the
        assignment names. Returns ``None`` when the model overturns the detection.
        """
        config = self.settings.tuning.vlm_tiebreak
        if not (
            config.enabled
            and event.ambiguous
            and self.provider is not None
            and not self.provider.is_offline
            and self._vlm_calls < config.max_calls_per_clip
            and event.behavior_class is BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT
        ):
            return event

        frame = next((f for f in frames if f.index == event.first_frame_index), None)
        if frame is None:
            return event

        safe_max = event.evidence.get("safe_threshold")
        safe_max = safe_max if isinstance(safe_max, int) else 2
        question = (
            "This is a frame from a factory security camera showing a forklift. "
            "How many standardized rectangular blocks are being carried on the forks? "
            "Reply with a single integer and nothing else."
        )

        try:
            self._vlm_calls += 1
            answer = await self.provider.answer_about_image(encode_png(frame.image), question)
        except LLMProviderError as exc:
            # Expected: keep the classical verdict rather than failing the clip.
            log.info("VLM tie-break unavailable (%s); keeping CV verdict", exc)
            return event

        match = _INT_RE.search(answer or "")
        if match is None:
            log.debug("VLM tie-break gave no parseable count (%r); keeping CV verdict", answer)
            return event

        counted = int(match.group())
        evidence = {
            **event.evidence,
            "vlm_block_count": counted,
            "cv_block_count": event.evidence.get("block_count"),
            "vlm_answer": (answer or "").strip()[:80],
        }

        if counted <= safe_max:
            log.debug(
                "VLM overturned forklift detection: counted %d (<= %d safe)", counted, safe_max
            )
            return None

        return SustainedEvent(
            behavior_class=event.behavior_class,
            first_frame_index=event.first_frame_index,
            first_timestamp_s=event.first_timestamp_s,
            frame_count=event.frame_count,
            # The model resolved the ambiguity, so this is no longer a borderline call.
            confidence=min(0.95, max(event.confidence, 0.80)),
            method=DetectionMethod.VLM_TIEBREAK,
            bboxes=event.bboxes,
            evidence=evidence,
            ambiguous=False,
        )
