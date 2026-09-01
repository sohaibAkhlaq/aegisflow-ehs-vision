"""The integrated pipeline: video in, compliance record out.

    clip -> DetectionEngine        (Module 1)  -> DetectionRecord[]
         -> SeverityMatrix         (Module 2)  -> SeverityAssessment
         -> ViolationEvent
         -> EscalationRouter       (Module 3)  -> DB log, and alert when HIGH/CRITICAL
         -> ReportWriter           (Module 4)  -> append-only JSON / CSV / per-event files

**Multi-violation handling is structural, not special-cased.** One
:class:`ViolationEvent` is emitted per detected violation per clip, each carrying its own
severity and its own routing decision. A clip with an open panel and a walkway breach yields
two records, two severities and one alert. Nothing is merged, maxed or de-duplicated across
behaviour classes - which is what the assignment asks for.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from aegisflow.core.enums import BehaviorClass, Severity
from aegisflow.core.errors import AegisFlowError
from aegisflow.core.logging import get_logger
from aegisflow.core.schemas import (
    ClipResult,
    DetectionRecord,
    PolicyRuleSet,
    SeverityAssessment,
    ViolationEvent,
)
from aegisflow.core.settings import Settings, get_settings
from aegisflow.core.zoning import behavior_from_clip_path, clip_id_for
from aegisflow.detection import DetectionEngine
from aegisflow.llm.base import LLMProvider
from aegisflow.severity import SeverityMatrix

log = get_logger(__name__)


@dataclass
class RunStats:
    """Aggregate outcome of a pipeline run."""

    clips: int = 0
    clips_with_violations: int = 0
    events: int = 0
    alerts: int = 0
    frames: int = 0
    vlm_calls: int = 0
    seconds: float = 0.0
    failures: list[tuple[str, str]] = field(default_factory=list)
    by_severity: dict[str, int] = field(default_factory=dict)
    by_behavior: dict[str, int] = field(default_factory=dict)

    def record(self, result: ClipResult) -> None:
        self.clips += 1
        self.frames += result.frames_analysed
        self.seconds += result.processing_s
        self.vlm_calls += result.vlm_calls
        if result.events:
            self.clips_with_violations += 1
        for event in result.events:
            self.events += 1
            if event.severity.requires_realtime_alert:
                self.alerts += 1
            self.by_severity[event.severity.value] = (
                self.by_severity.get(event.severity.value, 0) + 1
            )
            key = event.behavior_class.value
            self.by_behavior[key] = self.by_behavior.get(key, 0) + 1

    @property
    def mean_clip_seconds(self) -> float:
        return self.seconds / self.clips if self.clips else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "clips": self.clips,
            "clips_with_violations": self.clips_with_violations,
            "events": self.events,
            "alerts": self.alerts,
            "frames": self.frames,
            "vlm_calls": self.vlm_calls,
            "seconds": round(self.seconds, 2),
            "mean_clip_seconds": round(self.mean_clip_seconds, 3),
            "by_severity": dict(self.by_severity),
            "by_behavior": dict(self.by_behavior),
            "failures": len(self.failures),
        }


class CompliancePipeline:
    """Composes Modules 1-4 over one or many clips."""

    def __init__(
        self,
        rule_set: PolicyRuleSet,
        settings: Settings | None = None,
        provider: LLMProvider | None = None,
        sink: object | None = None,
        writer: object | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.rule_set = rule_set
        self.engine = DetectionEngine(rule_set, self.settings, provider)
        self.matrix = SeverityMatrix(rule_set)
        self.sink = sink
        self.writer = writer
        self.stats = RunStats()

    def warmup(self) -> None:
        self.engine.warmup()

    # ------------------------------------------------------------------ one clip

    async def process_clip(self, clip_path: str | Path) -> ClipResult:
        """Run the full pipeline over one clip."""
        path = str(clip_path)
        started = time.perf_counter()
        vlm_before = self.engine.vlm_calls

        records, info, frames = await self.engine.analyse(path)
        events: list[ViolationEvent] = []

        for record in records:
            assessment = self.matrix.assess(record)
            event = self._to_event(record, assessment)

            # Module 3: each event routes on its own merit.
            if self.sink is not None:
                event = await self.sink.route(event)  # type: ignore[attr-defined]

            # Module 4: the audit record.
            if self.writer is not None:
                await self.writer.write(event)  # type: ignore[attr-defined]

            events.append(event)

        result = ClipResult(
            clip_id=clip_id_for(path),
            clip_path=path,
            zone=records[0].zone if records else _zone_for(path, self.settings),
            frames_analysed=frames,
            duration_s=info.duration_s,
            processing_s=time.perf_counter() - started,
            detections=tuple(records),
            events=tuple(events),
            vlm_calls=self.engine.vlm_calls - vlm_before,
        )
        self.stats.record(result)
        return result

    # ----------------------------------------------------------------- many clips

    async def process_clips(
        self,
        clip_paths: Iterable[str | Path],
        *,
        on_result: object | None = None,
        stop_on_error: bool = False,
    ) -> list[ClipResult]:
        """Run over a batch, continuing past individual clip failures.

        One unreadable clip must not abandon a 691-clip regression run, so failures are
        collected in ``stats.failures`` and reported at the end.
        """
        results: list[ClipResult] = []
        for clip_path in clip_paths:
            try:
                result = await self.process_clip(clip_path)
            except AegisFlowError as exc:
                log.warning("skipping %s: %s", clip_path, exc)
                self.stats.failures.append((str(clip_path), str(exc)))
                if stop_on_error:
                    raise
                continue
            except Exception as exc:
                log.exception("unexpected failure on %s", clip_path)
                self.stats.failures.append((str(clip_path), repr(exc)))
                if stop_on_error:
                    raise
                continue

            results.append(result)
            if on_result is not None:
                maybe = on_result(result)
                if hasattr(maybe, "__await__"):
                    await maybe
        return results

    # ----------------------------------------------------------------- internal

    def _to_event(self, record: DetectionRecord, assessment: SeverityAssessment) -> ViolationEvent:
        return ViolationEvent(
            clip_id=record.clip_id,
            zone=record.zone,
            behavior_class=record.behavior_class,
            policy_rule_ref=assessment.policy_rule_ref,
            event_description=record.description,
            severity=assessment.severity,
            confidence=record.confidence,
            detection_method=record.detection_method,
            severity_rationale=assessment.rationale,
            clip_timestamp_s=record.first_timestamp_s,
            frame_index=record.first_frame_index,
        )


def _zone_for(path: str, settings: Settings) -> str:
    from aegisflow.core.zoning import resolve_zone

    return resolve_zone(path, settings=settings)


# ---------------------------------------------------------------------------
# Clip discovery
# ---------------------------------------------------------------------------


def discover_clips(
    settings: Settings | None = None,
    split: str | None = "test",
    behavior: BehaviorClass | None = None,
    limit: int | None = None,
    per_class: int | None = None,
) -> list[Path]:
    """Find dataset clips.

    Layout is ``data/raw/<split>/<class_folder>/*.mp4``. ``per_class`` takes an even sample
    across classes, which matters because the dataset is ~9:1 imbalanced and an unbalanced
    smoke run would tell you almost nothing about the rare classes.
    """
    settings = settings or get_settings()
    root = settings.path(settings.data_root)
    if split:
        root = root / split
    if not root.exists():
        return []

    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not class_dirs:
        return sorted(root.glob("*.mp4"))[:limit]

    selected: list[Path] = []
    for class_dir in class_dirs:
        if behavior is not None:
            folder_class = behavior_from_clip_path(class_dir / "x.mp4")
            if folder_class is not behavior:
                continue
        clips = sorted(class_dir.glob("*.mp4"))
        selected.extend(clips[:per_class] if per_class else clips)

    return selected[:limit] if limit else selected


def worst_severity(events: Sequence[ViolationEvent]) -> Severity | None:
    """Highest tier among a clip's events - used for the dashboard status indicator."""
    if not events:
        return None
    return max((e.severity for e in events), key=lambda s: s.rank)
