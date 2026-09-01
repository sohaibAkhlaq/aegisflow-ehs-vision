"""Temporal persistence and event de-duplication.

The single biggest false-positive reducer available to us. A per-frame verdict is noisy:
a person's foot lands on a walkway boundary for one frame, a contour fragments, YOLO loses a
box to occlusion. Requiring a verdict to *hold* across several consecutive sampled frames
removes almost all of that without touching recall, because every behaviour the policy
defines lasts far longer than the sampling interval.

The required run length is per behaviour and lives in ``config/settings.yaml``:

* **State-based** conditions (an open panel cover) get the longest window. The condition
  holds for the whole clip, so demanding six frames of agreement costs nothing.
* **Action-based** events (a walkway breach) get the shortest, because a brief real event
  must not be smoothed away.

One clip yields at most one event per behaviour class - the first sustained occurrence, with
the supporting frame count attached as evidence. Different classes in the same clip stay
independent, which is what the assignment's multi-violation requirement needs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from aegisflow.core.enums import BehaviorClass, DetectionMethod
from aegisflow.core.schemas import BBox


@dataclass
class FrameVerdict:
    """One detector's opinion about one frame."""

    behavior_class: BehaviorClass
    frame_index: int
    timestamp_s: float
    confidence: float
    method: DetectionMethod
    bbox: BBox | None = None
    evidence: dict[str, object] = field(default_factory=dict)
    ambiguous: bool = False


@dataclass
class SustainedEvent:
    """A verdict that held long enough to be reported."""

    behavior_class: BehaviorClass
    first_frame_index: int
    first_timestamp_s: float
    frame_count: int
    confidence: float
    method: DetectionMethod
    bboxes: tuple[BBox, ...]
    evidence: dict[str, object]
    ambiguous: bool


class PersistenceTracker:
    """Accumulates per-frame verdicts and reports the ones that persist.

    Frames must be fed in ascending order. A gap - a sampled frame with no verdict for a
    class - breaks that class's run, because the behaviour stopped being observable.
    """

    def __init__(
        self, required_frames: dict[BehaviorClass, int], default_required: int = 3
    ) -> None:
        self._required = required_frames
        self._default = default_required
        self._runs: dict[BehaviorClass, list[FrameVerdict]] = defaultdict(list)
        self._completed: dict[BehaviorClass, SustainedEvent] = {}
        self._seen_frames = 0

    def required_for(self, behavior: BehaviorClass) -> int:
        return max(1, self._required.get(behavior, self._default))

    def update(self, frame_verdicts: list[FrameVerdict]) -> None:
        """Feed every verdict produced for one frame."""
        self._seen_frames += 1
        present = {verdict.behavior_class for verdict in frame_verdicts}

        for verdict in frame_verdicts:
            self._runs[verdict.behavior_class].append(verdict)
            self._maybe_complete(verdict.behavior_class)

        # Any class not seen this frame has its run broken.
        for behavior in list(self._runs):
            if behavior not in present:
                self._runs[behavior].clear()

    def _maybe_complete(self, behavior: BehaviorClass) -> None:
        run = self._runs[behavior]
        if len(run) < self.required_for(behavior):
            return
        if behavior in self._completed:
            # Already reported; extend the frame count so evidence reflects duration.
            existing = self._completed[behavior]
            self._completed[behavior] = SustainedEvent(
                behavior_class=existing.behavior_class,
                first_frame_index=existing.first_frame_index,
                first_timestamp_s=existing.first_timestamp_s,
                frame_count=len(run),
                confidence=max(existing.confidence, _mean_confidence(run)),
                method=existing.method,
                bboxes=existing.bboxes,
                evidence=existing.evidence,
                ambiguous=existing.ambiguous and any(v.ambiguous for v in run),
            )
            return

        first = run[0]
        self._completed[behavior] = SustainedEvent(
            behavior_class=behavior,
            first_frame_index=first.frame_index,
            first_timestamp_s=first.timestamp_s,
            frame_count=len(run),
            confidence=_mean_confidence(run),
            method=_dominant_method(run),
            bboxes=tuple(v.bbox for v in run if v.bbox is not None)[:8],
            evidence=_merge_evidence(run),
            # Ambiguous only if the *majority* of supporting frames were borderline.
            ambiguous=sum(1 for v in run if v.ambiguous) > len(run) / 2,
        )

    def events(self) -> list[SustainedEvent]:
        """Sustained events, in order of first occurrence."""
        return sorted(
            self._completed.values(), key=lambda e: (e.first_frame_index, e.behavior_class.value)
        )

    @property
    def frames_seen(self) -> int:
        return self._seen_frames


def _mean_confidence(run: list[FrameVerdict]) -> float:
    if not run:
        return 0.0
    return float(sum(v.confidence for v in run) / len(run))


def _dominant_method(run: list[FrameVerdict]) -> DetectionMethod:
    """The method that produced most of the supporting frames.

    A VLM tie-break anywhere in the run wins, because that is the fact worth recording in
    the audit trail: a model was consulted for this decision.
    """
    if any(v.method is DetectionMethod.VLM_TIEBREAK for v in run):
        return DetectionMethod.VLM_TIEBREAK
    counts: dict[DetectionMethod, int] = defaultdict(int)
    for verdict in run:
        counts[verdict.method] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _merge_evidence(run: list[FrameVerdict]) -> dict[str, object]:
    """Evidence from the highest-confidence frame, plus run-level aggregates."""
    if not run:
        return {}
    best = max(run, key=lambda v: v.confidence)
    merged = dict(best.evidence)
    merged["supporting_frames"] = len(run)
    merged["peak_confidence"] = round(best.confidence, 4)

    # Numeric evidence gets a max across the run: the worst observed state is the one that
    # matters for a threshold rule like the forklift block count.
    for key in ("block_count", "person_count"):
        values = [v.evidence.get(key) for v in run if isinstance(v.evidence.get(key), int)]
        if values:
            merged[key] = max(values)  # type: ignore[assignment]
    return merged
