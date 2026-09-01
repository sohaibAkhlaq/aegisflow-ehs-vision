"""Commission the electrical-panel detector for a camera.

Why this exists
---------------
Panel state (Section 5.2.2) is the one indicator with no object class to anchor on and no
hand-tunable colour cue: an open cover looks like a slightly different patch of machinery.
Whole-frame edge and darkness statistics do not separate open from closed at all - measured
on the test split, the *open* class scores marginally **lower** on both, because those
statistics describe the scene rather than the panel.

What does work is looking at the panel itself. A fixed camera means the panel occupies a
stable region of the frame, and an open cover exposes the unlit cavity behind it, so the
region gets darker. This script locates that region and records the intensity boundary
between the two states - the same commissioning step a real installation would perform
against its own known-closed panels.

Method
------
1. Take per-clip median frames from the *train* split for the open and closed classes.
   Medians suppress transient people and forklifts, leaving the static scene.
2. Score every candidate window by a t-like separation statistic,
   ``|mean_open - mean_closed| / pooled_sd``, and keep the best.
3. Record the window, the two class means, and the midpoint threshold.

The artefact is written to ``artifacts/models/panel_baseline.json`` and consumed by
``PanelCoverDetector``. Only the train split is used, so the test split stays honest.

    python scripts/calibrate_panel.py --camera CAM-02
    python scripts/calibrate_panel.py --camera CAM-02 --window 64 --clips 16
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aegisflow.core.logging import configure_logging, get_logger  # noqa: E402
from aegisflow.core.settings import get_settings  # noqa: E402
from aegisflow.detection import geometry as geo  # noqa: E402
from aegisflow.detection.video import iter_frames  # noqa: E402

log = get_logger("calibrate_panel")

OPEN_CLASS = "2_opened_panel_cover"
CLOSED_CLASS = "6_closed_panel_cover"
ARTEFACT = "artifacts/models/panel_baseline.json"


def clip_median(path: str, frames: int, imgsz: int) -> np.ndarray | None:
    """Median greyscale frame of one clip - the static scene, people removed."""
    images = [
        cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        for frame in iter_frames(path, sample_fps=2.0, imgsz=imgsz, max_frames=frames)
    ]
    if not images:
        return None
    return np.median(np.stack(images), axis=0)


def clip_fingerprint(path: str, frames: int, imgsz: int) -> np.ndarray | None:
    """Camera signature of one clip."""
    prints = [
        geo.scene_fingerprint(frame.image)
        for frame in iter_frames(path, sample_fps=2.0, imgsz=imgsz, max_frames=frames)
    ]
    if not prints:
        return None
    return np.median(np.stack(prints), axis=0)


def load_class(
    split: str, class_name: str, count: int, frames: int, imgsz: int, data_root: Path
) -> list[np.ndarray]:
    clips = sorted(glob.glob(str(data_root / split / class_name / "*.mp4")))
    if not clips:
        return []
    chosen = random.sample(clips, min(count, len(clips)))
    medians = [clip_median(c, frames, imgsz) for c in chosen]
    return [m for m in medians if m is not None]


def best_window(
    positives: np.ndarray, negatives: np.ndarray, size: int, stride: int
) -> tuple[int, int, float]:
    """Window with the strongest open-vs-closed separation."""
    mean_pos, mean_neg = positives.mean(0), negatives.mean(0)
    pooled_sd = np.sqrt((positives.var(0) + negatives.var(0)) / 2.0) + 1e-3
    separation = np.abs(mean_pos - mean_neg) / pooled_sd

    height, width = separation.shape
    best = (0, 0, -1.0)
    for y in range(0, max(1, height - size), stride):
        for x in range(0, max(1, width - size), stride):
            score = float(separation[y : y + size, x : x + size].mean())
            if score > best[2]:
                best = (x, y, score)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="CAM-02", help="camera id from config/zones.yaml")
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--clips", type=int, default=16, help="clips per class")
    parser.add_argument("--frames", type=int, default=4, help="frames per clip for the median")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    configure_logging("INFO")
    random.seed(args.seed)
    settings = get_settings()
    imgsz = settings.tuning.video.infer_imgsz
    data_root = settings.path(settings.data_root)

    if args.split == "test":
        log.warning(
            "calibrating on the test split leaks evaluation data; use --split train "
            "for any number you intend to report"
        )

    log.info("loading %s clips per class from the %s split", args.clips, args.split)
    opens = load_class(args.split, OPEN_CLASS, args.clips, args.frames, imgsz, data_root)
    closeds = load_class(args.split, CLOSED_CLASS, args.clips, args.frames, imgsz, data_root)

    if len(opens) < 4 or len(closeds) < 4:
        log.error(
            "need at least 4 clips per class (got %d open, %d closed). "
            "Is the dataset present under %s?",
            len(opens),
            len(closeds),
            data_root,
        )
        return 1

    # Camera signature, built from the compliant class: that is the camera's normal state.
    closed_clips = sorted(glob.glob(str(data_root / args.split / CLOSED_CLASS / "*.mp4")))
    fingerprints = [
        f
        for f in (
            clip_fingerprint(c, args.frames, imgsz)
            for c in random.sample(closed_clips, min(args.clips, len(closed_clips)))
        )
        if f is not None
    ]
    reference_fp = np.stack(fingerprints).mean(0)
    within = [geo.fingerprint_distance(f, reference_fp) for f in fingerprints]
    # 1.6x the worst same-camera distance, floored generously. Measured margin to the
    # other camera in this dataset is ~3x, so this is not a tight fit.
    tolerance = max(12.0, float(max(within)) * 1.6)
    log.info(
        "camera fingerprint: within-camera distance max %.2f -> tolerance %.1f",
        max(within),
        tolerance,
    )

    positives, negatives = np.stack(opens), np.stack(closeds)
    frame_h, frame_w = positives.shape[1:3]
    x, y, score = best_window(positives, negatives, args.window, args.stride)

    def feature(median: np.ndarray) -> float:
        return float(median[y : y + args.window, x : x + args.window].mean())

    open_values = [feature(m) for m in opens]
    closed_values = [feature(m) for m in closeds]
    open_mean = float(np.mean(open_values))
    closed_mean = float(np.mean(closed_values))
    threshold = (open_mean + closed_mean) / 2.0
    # Which side of the threshold means "open". Expected to be -1: an open cover exposes
    # the unlit cavity, so the region darkens.
    sign = 1 if open_mean > closed_mean else -1

    baseline = {
        "camera": args.camera,
        "roi_normalised": {
            "x": round(x / frame_w, 5),
            "y": round(y / frame_h, 5),
            "w": round(args.window / frame_w, 5),
            "h": round(args.window / frame_h, 5),
        },
        "feature": "mean_grey_intensity",
        "open_mean": round(open_mean, 2),
        "closed_mean": round(closed_mean, 2),
        "open_sd": round(float(np.std(open_values)), 2),
        "closed_sd": round(float(np.std(closed_values)), 2),
        "threshold": round(threshold, 2),
        "open_sign": sign,
        "separation_score": round(score, 3),
        "scene_fingerprint": [[round(float(v), 2) for v in row] for row in reference_fp],
        "fingerprint_grid": list(geo.FINGERPRINT_GRID),
        "fingerprint_tolerance": round(tolerance, 2),
        "calibrated_on": {
            "split": args.split,
            "open_clips": len(opens),
            "closed_clips": len(closeds),
            "imgsz": imgsz,
        },
        "policy_reference": "Section 5.2.2 - Opened Panel Cover",
        "note": (
            "ROI located by maximising open-vs-closed separation on the train split. "
            "This is a commissioning parameter for a fixed camera, equivalent to the "
            "walkway polygon: it tells the detector WHERE the panel is. The open/closed "
            "decision itself remains the policy's observable indicator."
        ),
    }

    path = settings.path(ARTEFACT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    log.info("panel ROI: x=%d y=%d size=%d (separation %.2f)", x, y, args.window, score)
    log.info(
        "open mean %.1f | closed mean %.1f | threshold %.1f (open_sign %+d)",
        open_mean,
        closed_mean,
        threshold,
        sign,
    )
    log.info("wrote %s", path)

    if score < 1.0:
        log.warning(
            "separation score %.2f is weak; the panel detector will be unreliable on this "
            "camera. Report it as a known limitation rather than tuning until it looks good.",
            score,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
