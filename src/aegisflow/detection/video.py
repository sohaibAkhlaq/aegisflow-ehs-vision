"""Frame sampling and decoding.

The clips are 1920x1080 at ~25 fps, 5-14 s each, and the machine has no GPU and 8 GB of
RAM. Two decisions follow from that, and they are what make CPU-only inference viable:

* **Sample at 4 fps, not 25.** Every behaviour the policy defines persists for far longer
  than 250 ms - a walkway breach, an open panel, an overloaded forklift. Analysing every
  sixth frame loses nothing and cuts the work by ~6x.
* **Infer at 640 px, not 1920.** A further ~9x reduction in pixels. YOLOv8n is trained at
  640 anyway, so this is the model's native scale rather than a compromise.

Frames are yielded one at a time and never accumulated, so peak memory is one frame.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from aegisflow.core.errors import VideoReadError
from aegisflow.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class VideoInfo:
    """Properties of a source clip."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    @property
    def clip_id(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class SampledFrame:
    """One analysed frame, at inference scale.

    ``image`` is the downscaled BGR frame the detectors work on. ``scale`` maps a
    coordinate in ``image`` back to the source resolution, so boxes can be reported in
    original pixels when needed.
    """

    index: int
    timestamp_s: float
    image: np.ndarray
    scale: float
    source_width: int
    source_height: int

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


def probe(path: str | Path) -> VideoInfo:
    """Read a clip's properties without decoding it."""
    p = Path(path)
    if not p.exists():
        raise VideoReadError(f"clip not found: {p}")

    capture = cv2.VideoCapture(str(p))
    if not capture.isOpened():
        capture.release()
        raise VideoReadError(f"could not open clip: {p}")
    try:
        return VideoInfo(
            path=p,
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)) or 25.0,
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()


def iter_frames(
    path: str | Path,
    sample_fps: float = 4.0,
    imgsz: int = 640,
    max_frames: int = 64,
) -> Iterator[SampledFrame]:
    """Yield downscaled frames at approximately ``sample_fps``.

    Frames are skipped by decoding and discarding rather than by seeking: ``CAP_PROP_POS_
    FRAMES`` seeking is unreliable on these H.264 clips (it lands on the nearest keyframe),
    and sequential decode of a 7-second clip is fast enough that accuracy is worth more.
    """
    info = probe(path)
    capture = cv2.VideoCapture(str(info.path))
    if not capture.isOpened():
        capture.release()
        raise VideoReadError(f"could not open clip: {info.path}")

    stride = max(1, round(info.fps / sample_fps)) if sample_fps > 0 else 1
    emitted = 0
    index = 0

    try:
        while emitted < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                scaled, scale = _resize(frame, imgsz)
                yield SampledFrame(
                    index=index,
                    timestamp_s=index / info.fps if info.fps > 0 else 0.0,
                    image=scaled,
                    scale=scale,
                    source_width=info.width,
                    source_height=info.height,
                )
                emitted += 1
            index += 1
    finally:
        capture.release()

    if emitted == 0:
        raise VideoReadError(f"no frames decoded from {info.path}")
    log.debug("%s: %d frames sampled (stride %d)", info.clip_id, emitted, stride)


def _resize(frame: np.ndarray, imgsz: int) -> tuple[np.ndarray, float]:
    """Scale the longest edge to ``imgsz``, preserving aspect ratio."""
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= imgsz:
        return frame, 1.0
    scale = imgsz / longest
    resized = cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def encode_png(image: np.ndarray) -> bytes:
    """PNG-encode a frame. Lossless, for archival or local use."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise VideoReadError("failed to PNG-encode a frame")
    return bytes(buffer.tobytes())


def encode_for_vlm(image: np.ndarray, max_width: int = 512, quality: int = 78) -> bytes:
    """JPEG-encode a frame for a vision model, sized to stay inside a token budget.

    PNG is the wrong format for camera footage and the cost is not academic: a 640 px PNG
    of one frame is ~400 KB and bills at roughly 2,300 tokens, against a Groq free-tier
    budget of 8,000 tokens per minute - about three requests a minute before the API starts
    returning 429. The same frame as JPEG at 512 px is ~26 KB, a 15x reduction, with no
    visible loss of the things the model is being asked about (a vest colour, a block count,
    whether a panel is open).
    """
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        image = cv2.resize(
            image,
            (max_width, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise VideoReadError("failed to JPEG-encode a frame")
    return bytes(buffer.tobytes())


def middle_frame(path: str | Path, imgsz: int = 640) -> SampledFrame | None:
    """A single representative frame, used for thumbnails and quick probes."""
    info = probe(path)
    target = max(0, info.frame_count // 2)
    capture = cv2.VideoCapture(str(info.path))
    try:
        if not capture.isOpened():
            return None
        for index in range(target + 1):
            ok, frame = capture.read()
            if not ok:
                return None
            if index == target:
                scaled, scale = _resize(frame, imgsz)
                return SampledFrame(
                    index=index,
                    timestamp_s=index / info.fps if info.fps > 0 else 0.0,
                    image=scaled,
                    scale=scale,
                    source_width=info.width,
                    source_height=info.height,
                )
    finally:
        capture.release()
    return None
