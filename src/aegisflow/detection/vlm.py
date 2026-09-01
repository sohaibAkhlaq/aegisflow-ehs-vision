"""Vision-language detection of the policy's observable indicators.

Why this is a first-class detection path and not a garnish
----------------------------------------------------------
Three of the four indicators the policy names cannot be read reliably by hand-tuned
classical CV on this footage, and that is measured rather than assumed
(``docs/adr/0003-detector-cue-selection.md``): walkway boundary segmentation caps at F1 0.25,
contour block counting is *anti-correlated* with the true load, and "interacting with
equipment" has no cue at all without an equipment detector.

A vision-language model reads exactly those indicators well, because they are semantic
questions about a scene rather than thresholds on a histogram. The assignment explicitly
permits "zero-shot vision-language models" as a vision approach, and both project documents
specify a VLM fallback.

Crucially this path is *more* policy-grounded than the classical one, not less. The question
put to the model is built from the parsed rule's own ``observable_indicator`` text, so what
the model is asked is literally what the manual says to look for, and the section reference
travels with the answer.

Cost control, because the provider is a free tier:

* one call per frame covering **all four** behaviours, not one call per behaviour
* at most ``vlm_tiebreak.max_calls_per_clip`` frames per clip (default 2)
* every response cached on disk by frame hash, so re-runs and demos are free
* any failure degrades to the classical verdict - never to an exception
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from aegisflow.core.enums import BehaviorClass, DetectionMethod
from aegisflow.core.errors import LLMProviderError
from aegisflow.core.logging import get_logger
from aegisflow.core.schemas import PolicyRuleSet
from aegisflow.detection.temporal import FrameVerdict
from aegisflow.detection.video import SampledFrame, encode_for_vlm
from aegisflow.llm.base import LLMProvider

log = get_logger(__name__)

# Keys the model answers with, mapped to the behaviour each one decides.
_ANSWER_KEYS: dict[str, BehaviorClass] = {
    "person_outside_green_walkway": BehaviorClass.SAFE_WALKWAY_VIOLATION,
    "person_at_equipment_without_green_vest": BehaviorClass.UNAUTHORIZED_INTERVENTION,
    "electrical_panel_cover_open": BehaviorClass.OPENED_PANEL_COVER,
}
_BLOCK_COUNT_KEY = "blocks_on_forklift_forks"


@dataclass
class VlmVerdicts:
    """One frame's answers."""

    frame_index: int
    timestamp_s: float
    answers: dict[str, object]
    raw: str = ""

    def truthy(self, key: str) -> bool:
        value = self.answers.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "y", "1"}
        return False

    def block_count(self) -> int | None:
        value = self.answers.get(_BLOCK_COUNT_KEY)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            digits = "".join(ch for ch in value if ch.isdigit())
            return int(digits) if digits else None
        return None


class VlmIndicatorDetector:
    """Asks a vision model the policy's observable-indicator questions."""

    def __init__(
        self,
        rule_set: PolicyRuleSet,
        provider: LLMProvider,
        safe_block_count: int = 2,
    ) -> None:
        self.rule_set = rule_set
        self.provider = provider
        self.safe_block_count = safe_block_count
        self.calls = 0
        self.failures = 0
        self._question = self._build_question()

    @property
    def enabled(self) -> bool:
        return self.provider is not None and not self.provider.is_offline

    # ------------------------------------------------------------------ prompt

    def _build_question(self) -> str:
        """Compose the prompt from the *parsed policy*, not from hard-coded prose.

        Each line quotes the manual's own indicator text and its section reference, so the
        model is asked precisely what the document says to look for.
        """
        lines = [
            "You are inspecting one frame from a fixed factory security camera for "
            "occupational-safety compliance.",
            "",
            "Answer strictly about what is visible in THIS frame. If something is not "
            "visible or you cannot tell, answer false (or null for the count).",
            "",
            "Compliance indicators, quoted from the facility's policy manual:",
        ]
        for key, behavior in _ANSWER_KEYS.items():
            rule = self.rule_set.rule_for(behavior)
            if rule is None:
                continue
            lines.append(f'- "{key}": {rule.section_ref} - {rule.observable_indicator}')
        forklift = self.rule_set.rule_for(BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT)
        if forklift is not None:
            threshold = forklift.numeric_threshold or self.safe_block_count
            lines.append(
                f'- "{_BLOCK_COUNT_KEY}": {forklift.section_ref} - count the standardized '
                f"blocks carried on the forklift forks. {threshold} or fewer is compliant; "
                f"more is an overload. Use null if no forklift is visible."
            )
        lines += [
            "",
            "Reply with JSON only, no prose.",
        ]
        return "\n".join(lines)

    def _schema(self) -> dict[str, object]:
        properties: dict[str, object] = {key: {"type": "boolean"} for key in _ANSWER_KEYS}
        properties[_BLOCK_COUNT_KEY] = {"type": ["integer", "null"]}
        return {"type": "object", "properties": properties}

    # ---------------------------------------------------------------- inference

    async def inspect_frame(self, frame: SampledFrame) -> VlmVerdicts | None:
        """Ask about one frame. Returns None when the provider could not answer."""
        if not self.enabled:
            return None
        try:
            self.calls += 1
            raw = await self.provider.answer_about_image(
                encode_for_vlm(frame.image), self._question
            )
        except LLMProviderError as exc:
            self.failures += 1
            log.info("VLM frame inspection unavailable (%s); keeping classical verdicts", exc)
            return None

        answers = _parse_json_object(raw)
        if answers is None:
            self.failures += 1
            log.debug("VLM returned unparseable output: %r", raw[:160])
            return None

        return VlmVerdicts(
            frame_index=frame.index,
            timestamp_s=frame.timestamp_s,
            answers=answers,
            raw=raw[:200],
        )

    async def inspect_clip(
        self, frames: list[SampledFrame], max_frames: int = 2
    ) -> list[FrameVerdict]:
        """Inspect a few representative frames and convert answers to verdicts.

        Frames are sampled evenly across the clip rather than taken from the start, so a
        behaviour that only appears later is still seen.
        """
        if not self.enabled or not frames:
            return []

        chosen = _spread(frames, max_frames)
        verdicts: list[FrameVerdict] = []

        for frame in chosen:
            answer = await self.inspect_frame(frame)
            if answer is None:
                continue
            verdicts.extend(self._to_verdicts(answer, frame))
        return verdicts

    def _to_verdicts(self, answer: VlmVerdicts, frame: SampledFrame) -> list[FrameVerdict]:
        out: list[FrameVerdict] = []

        for key, behavior in _ANSWER_KEYS.items():
            if self.rule_set.rule_for(behavior) is None:
                continue
            if not answer.truthy(key):
                continue
            out.append(
                FrameVerdict(
                    behavior_class=behavior,
                    frame_index=frame.index,
                    timestamp_s=frame.timestamp_s,
                    confidence=0.82,
                    method=DetectionMethod.VLM_TIEBREAK,
                    evidence={
                        "vlm_indicator": key,
                        "vlm_answer": True,
                        "source": "vision_language_model",
                    },
                    ambiguous=False,
                )
            )

        forklift_rule = self.rule_set.rule_for(BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT)
        count = answer.block_count()
        if forklift_rule is not None and count is not None:
            threshold = forklift_rule.numeric_threshold or self.safe_block_count
            if count > threshold:
                out.append(
                    FrameVerdict(
                        behavior_class=BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT,
                        frame_index=frame.index,
                        timestamp_s=frame.timestamp_s,
                        confidence=float(min(0.92, 0.72 + 0.08 * (count - threshold))),
                        method=DetectionMethod.VLM_TIEBREAK,
                        evidence={
                            "vlm_block_count": count,
                            "safe_threshold": threshold,
                            "source": "vision_language_model",
                        },
                        ambiguous=False,
                    )
                )
        return out


def _spread(frames: list[SampledFrame], count: int) -> list[SampledFrame]:
    """Pick ``count`` frames spread evenly across the clip."""
    if count >= len(frames):
        return list(frames)
    if count <= 1:
        return [frames[len(frames) // 2]]
    step = (len(frames) - 1) / (count - 1)
    return [frames[min(len(frames) - 1, round(index * step))] for index in range(count)]


def _parse_json_object(raw: str) -> dict[str, object] | None:
    """Extract a JSON object from a model reply, tolerating code fences and stray prose."""
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None
