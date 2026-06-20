"""
video_processor.py — Module 1 orchestrator: ingests video clips and drives
the per-frame detection pipeline, producing a stream of DetectionEvents.

Design choices:
- Samples every N-th frame to avoid redundant detections on static scenes
- Deduplicates events: if the same class is detected in consecutive sampled
  frames, only the first occurrence is reported (avoids flooding the log)
- Emits a thumbnail JPEG (base64) for each event for dashboard display
- Supports both file-based processing and async generator (streaming) mode
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Optional

import cv2
import numpy as np

from .detector import ComplianceDetector, DetectionEvent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClipViolation:
    """One violation record produced from a video clip."""
    event_id: str
    clip_id: str
    clip_path: str
    frame_index: int
    timestamp_sec: float
    behavior_class_id: int
    behavior_name: str
    confidence: float
    zone: str
    description: str
    policy_ref: str
    detector_source: str
    thumbnail_b64: Optional[str] = None   # base64-encoded JPEG thumbnail


def _encode_thumbnail(frame: np.ndarray, max_w: int = 640) -> str:
    """Encode a frame as a base64 JPEG, resized for dashboard display."""
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (max_w, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _annotate_frame(frame: np.ndarray, events: list[DetectionEvent]) -> np.ndarray:
    """Draw bounding boxes and labels on the frame for the thumbnail."""
    COLOUR_MAP = {
        0: (0, 0, 255),    # CRITICAL → red
        1: (0, 128, 255),  # HIGH → orange
        2: (0, 255, 0),    # LOW → green
        3: (0, 165, 255),  # HIGH → orange-ish
    }
    annotated = frame.copy()
    for ev in events:
        colour = COLOUR_MAP.get(ev.behavior_class_id, (255, 255, 255))
        if ev.bbox is not None:
            cv2.rectangle(
                annotated,
                (int(ev.bbox.x1), int(ev.bbox.y1)),
                (int(ev.bbox.x2), int(ev.bbox.y2)),
                colour, 2,
            )
            label = f"{ev.behavior_name[:25]} {ev.confidence:.2f}"
            cv2.putText(
                annotated, label,
                (int(ev.bbox.x1), int(ev.bbox.y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2,
            )
        else:
            # No bbox — write text in top-left
            cv2.putText(
                annotated,
                f"[VLM] {ev.behavior_name[:30]}",
                (10, 30 + events.index(ev) * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
            )
    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# Processor
# ─────────────────────────────────────────────────────────────────────────────

class VideoProcessor:
    """
    Ingests a video clip and yields ClipViolation events frame-by-frame.

    Args:
        detector: Configured ComplianceDetector instance.
        sample_every: Process 1 out of every N frames (default = every 15th).
        dedup_window: Frames within which same class is considered duplicate.
    """

    def __init__(
        self,
        detector: ComplianceDetector,
        sample_every: int = 15,
        dedup_window: int = 45,
    ) -> None:
        self.detector = detector
        self.sample_every = sample_every
        self.dedup_window = dedup_window

    def process_clip(self, clip_path: Path) -> list[ClipViolation]:
        """
        Synchronous: process a full clip and return all violations.
        Suitable for batch / background processing.
        """
        violations: list[ClipViolation] = []
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {clip_path}")
            return violations

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        clip_id = clip_path.stem

        logger.info(
            f"Processing clip '{clip_id}' — {total_frames} frames @ {fps:.1f}fps"
        )

        last_detected: dict[int, int] = {}  # class_id → last frame with event
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.sample_every == 0:
                events = self.detector.detect_frame(frame, frame_idx, fps)
                annotated = _annotate_frame(frame, events) if events else frame

                for ev in events:
                    cid = ev.behavior_class_id
                    last_seen = last_detected.get(cid, -999)
                    if frame_idx - last_seen < self.dedup_window:
                        continue  # deduplicate
                    last_detected[cid] = frame_idx

                    thumb = _encode_thumbnail(annotated)
                    violations.append(ClipViolation(
                        event_id=str(uuid.uuid4()),
                        clip_id=clip_id,
                        clip_path=str(clip_path),
                        frame_index=frame_idx,
                        timestamp_sec=ev.timestamp_sec,
                        behavior_class_id=ev.behavior_class_id,
                        behavior_name=ev.behavior_name,
                        confidence=ev.confidence,
                        zone=ev.zone,
                        description=ev.description,
                        policy_ref=ev.policy_ref,
                        detector_source=ev.detector_source,
                        thumbnail_b64=thumb,
                    ))

            frame_idx += 1

        cap.release()
        logger.info(
            f"Clip '{clip_id}' complete — {len(violations)} violations found"
        )
        return violations

    async def process_clip_async(
        self, clip_path: Path
    ) -> AsyncGenerator[ClipViolation, None]:
        """
        Async generator: yields violations as they are found.
        Suitable for real-time streaming to the WebSocket pipeline.
        """
        loop = asyncio.get_event_loop()
        violations = await loop.run_in_executor(None, self.process_clip, clip_path)
        for v in violations:
            yield v
            await asyncio.sleep(0)  # yield control


# ─────────────────────────────────────────────────────────────────────────────
# Batch processor
# ─────────────────────────────────────────────────────────────────────────────

async def process_all_clips(
    data_dir: Path,
    detector: ComplianceDetector,
    on_violation,  # async callback(ClipViolation)
    sample_every: int = 15,
) -> int:
    """
    Scan data_dir for video files and process each one.
    Calls on_violation callback for each found violation.
    Returns total violation count.
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    clips = sorted([
        p for p in data_dir.iterdir()
        if p.suffix.lower() in video_extensions
    ])

    if not clips:
        logger.warning(f"No video clips found in {data_dir}")
        return 0

    processor = VideoProcessor(detector, sample_every=sample_every)
    total = 0

    for clip in clips:
        logger.info(f"Starting clip: {clip.name}")
        async for violation in processor.process_clip_async(clip):
            await on_violation(violation)
            total += 1

    return total
