"""Render annotated playback clips for the dashboard's Live Feed Monitor (View A).

The assignment asks View A to show processed clips with compliance status indicators and
severity colour coding. Rather than re-running inference in the browser, the pipeline burns
the overlay in once and the dashboard plays the result - which is what keeps the UI
responsive on a CPU-only machine.

Overlay contents:

* severity-coloured border and status banner (no violation / violation / alert active)
* a bounding box per detected violation, labelled with behaviour and tier
* the policy section reference, so the evidence and its justification are on the same frame
"""

from __future__ import annotations

from pathlib import Path

import cv2

from aegisflow.core.enums import Severity
from aegisflow.core.logging import get_logger
from aegisflow.core.schemas import ClipResult
from aegisflow.core.settings import Settings, get_settings
from aegisflow.detection.video import iter_frames, probe

log = get_logger(__name__)

# BGR, matching the dashboard's severity tokens.
SEVERITY_BGR: dict[str, tuple[int, int, int]] = {
    "LOW": (235, 99, 37),
    "MEDIUM": (105, 150, 5),
    "HIGH": (6, 119, 217),
    "CRITICAL": (38, 38, 220),
}
COMPLIANT_BGR = (105, 150, 5)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def render_annotated_clip(
    clip_path: str | Path,
    result: ClipResult,
    settings: Settings | None = None,
    output_dir: Path | None = None,
) -> Path | None:
    """Write an annotated MP4. Returns its path, or None if it could not be written."""
    settings = settings or get_settings()
    destination = output_dir or settings.path(settings.outputs_root) / "annotated"
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"{Path(clip_path).stem}_annotated.mp4"

    tuning = settings.tuning
    info = probe(clip_path)
    worst = _worst(result)

    # Boxes are recorded at inference scale, so the output is written at that scale too -
    # no coordinate conversion, and a much smaller file for the dashboard to stream.
    frames = list(
        iter_frames(
            clip_path,
            sample_fps=tuning.video.sample_fps,
            imgsz=tuning.video.infer_imgsz,
            max_frames=tuning.video.max_frames_per_clip,
        )
    )
    if not frames:
        return None

    height, width = frames[0].image.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, tuning.video.sample_fps),
        (width, height),
    )
    if not writer.isOpened():
        log.warning("could not open a video writer for %s", output_path)
        return None

    # frame index -> the detections active at that point in the clip
    by_frame: dict[int, list] = {}
    for record in result.detections:
        for frame in frames:
            if frame.index >= record.first_frame_index:
                by_frame.setdefault(frame.index, []).append(record)

    try:
        for frame in frames:
            canvas = frame.image.copy()
            active = by_frame.get(frame.index, [])
            _draw_boxes(canvas, active)
            _draw_banner(canvas, active, worst, frame.timestamp_s, info.clip_id)
            writer.write(canvas)
    finally:
        writer.release()

    log.debug("annotated clip written: %s", output_path)
    return output_path


def _worst(result: ClipResult) -> Severity | None:
    if not result.events:
        return None
    return max((e.severity for e in result.events), key=lambda s: s.rank)


def _severity_for(record, result_events) -> str:
    for event in result_events:
        if event.behavior_class is record.behavior_class:
            return event.severity.value
    return "MEDIUM"


def _draw_boxes(canvas, records) -> None:
    for record in records:
        colour = SEVERITY_BGR.get(_record_severity(record), SEVERITY_BGR["MEDIUM"])
        for box in record.bboxes[:3]:
            x1, y1, x2, y2 = (round(v) for v in box)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
            label = record.behavior_class.display_name
            _label(canvas, label, (x1, max(12, y1 - 6)), colour)


def _record_severity(record) -> str:
    """Severity carried on the detection's evidence, defaulting sensibly."""
    value = record.evidence.get("severity")
    return str(value) if isinstance(value, str) else "MEDIUM"


def _label(canvas, text: str, origin: tuple[int, int], colour) -> None:
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.42, 1)
    x, y = origin
    cv2.rectangle(canvas, (x, y - th - 4), (x + tw + 8, y + 3), colour, -1)
    cv2.putText(canvas, text, (x + 4, y - 2), _FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_banner(canvas, active, worst: Severity | None, timestamp: float, clip_id: str) -> None:
    height, width = canvas.shape[:2]

    if not active:
        status = "NO VIOLATION DETECTED"
        colour = COMPLIANT_BGR
    else:
        tier = worst.value if worst else "MEDIUM"
        colour = SEVERITY_BGR.get(tier, SEVERITY_BGR["MEDIUM"])
        alerting = worst is not None and worst.requires_realtime_alert
        status = f"{'ALERT ACTIVE - ' if alerting else ''}{tier} - {len(active)} VIOLATION(S)"

    # Severity border, so the state is readable even in a small dashboard tile.
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), colour, 4)

    bar_height = 26
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

    cv2.putText(canvas, status, (8, 18), _FONT, 0.46, colour, 1, cv2.LINE_AA)
    right = f"{clip_id}  t={timestamp:04.1f}s"
    (tw, _), _ = cv2.getTextSize(right, _FONT, 0.4, 1)
    cv2.putText(canvas, right, (width - tw - 8, 18), _FONT, 0.4, (210, 210, 210), 1, cv2.LINE_AA)
