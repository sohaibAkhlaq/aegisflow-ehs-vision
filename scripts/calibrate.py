"""Measure raw detector signals per class, so thresholds come from data.

Every threshold in ``config/settings.yaml`` should be traceable to a separation observed
here, not to a guess. Run this after any change to camera, lighting, or a detector's cue set.

    python scripts/calibrate.py --per-class 6 --split test
    python scripts/calibrate.py --per-class 10 --signals panel,walkway

Output is a percentile table per behaviour class. A usable threshold sits in the gap between
the positive class's low percentiles and the negative classes' high percentiles. Where there
is no gap, the cue does not separate and the detector needs a different signal - that is a
finding, not a failure.
"""

from __future__ import annotations

import argparse
import glob
import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aegisflow.core.logging import configure_logging  # noqa: E402
from aegisflow.core.settings import get_settings  # noqa: E402
from aegisflow.detection import geometry as geo  # noqa: E402
from aegisflow.detection.video import iter_frames  # noqa: E402
from aegisflow.detection.yolo import YoloDetector  # noqa: E402

SIGNALS = ("panel", "walkway", "vest", "blocks")


def panel_signals(frame, observation, tuning) -> dict[str, float]:
    """Cues used by PanelCoverDetector."""
    roi = geo.clip_box(
        (0.0, 0.16 * frame.height, float(frame.width), 0.80 * frame.height), frame.image.shape
    )
    patch = geo.crop(frame.image, roi)
    if patch.size == 0:
        return {}
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    dark = cv2.inRange(hsv, np.array((0, 0, 0), np.uint8), np.array((179, 255, 62), np.uint8))
    dark = geo.close_mask(dark, kernel=7, iterations=2)
    contour = geo.largest_contour(dark, min_area=0.004 * patch.shape[0] * patch.shape[1])
    return {
        "panel_edge": geo.vertical_edge_strength(patch),
        "panel_dark": geo.dark_region_ratio(patch),
        "panel_cavity_area": (
            (cv2.contourArea(contour) / patch.size * 3.0) if contour is not None else 0.0
        ),
        "panel_cavity_aspect": geo.rect_aspect(contour) if contour is not None else 0.0,
    }


def walkway_signals(frame, observation, tuning) -> dict[str, float]:
    """Cues used by WalkwayViolationDetector."""
    mask = geo.hsv_mask(frame.image, tuning.color.green_floor_line)
    mask = geo.close_mask(mask, kernel=11, iterations=3)
    total = frame.width * frame.height
    contour = geo.largest_contour(mask, min_area=0.0)
    out: dict[str, float] = {
        "green_mask_ratio": geo.mask_ratio(mask),
        "walkway_area_ratio": (cv2.contourArea(contour) / total) if contour is not None else 0.0,
    }
    if contour is not None and observation.persons:
        outside = sum(
            1 for p in observation.persons if not geo.point_in_contour(p.foot_point, contour)
        )
        out["frac_persons_outside"] = outside / len(observation.persons)
        margins = [
            abs(
                cv2.pointPolygonTest(
                    contour, (float(p.foot_point[0]), float(p.foot_point[1])), True
                )
            )
            for p in observation.persons
        ]
        out["min_margin_px"] = min(margins)
    return out


def vest_signals(frame, observation, tuning) -> dict[str, float]:
    """Cues used by UnauthorizedInterventionDetector."""
    colour = tuning.color
    greens: list[float] = []
    reds: list[float] = []
    for person in observation.persons:
        torso = geo.crop(frame.image, geo.torso_roi(person.bbox, frame.image.shape))
        if torso.size == 0:
            continue
        greens.append(geo.colour_ratio(torso, colour.green_vest))
        reds.append(geo.red_ratio(torso, colour.red_vest_low, colour.red_vest_high))
    if not greens:
        return {}
    return {
        "torso_green_max": max(greens),
        "torso_green_mean": statistics.fmean(greens),
        "torso_red_max": max(reds),
        "person_count": float(len(observation.persons)),
    }


def block_signals(frame, observation, tuning) -> dict[str, float]:
    """Cues used by ForkliftOverloadDetector."""
    if not observation.vehicles:
        return {"vehicle_present": 0.0}
    forklift = max(observation.vehicles, key=lambda v: v.area)
    x1, y1, x2, y2 = forklift.bbox
    h = y2 - y1
    roi = geo.clip_box(
        (x1 - 0.12 * (x2 - x1), y1 - 0.55 * h, x2 + 0.12 * (x2 - x1), y2), frame.image.shape
    )
    patch = geo.crop(frame.image, roi)
    if patch.size == 0:
        return {"vehicle_present": 1.0}
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    binary = geo.close_mask(binary, kernel=5, iterations=1)
    scale = (patch.shape[0] * patch.shape[1]) / (640.0 * 360.0)
    min_area = tuning.forklift.block_min_area_px * max(0.15, min(scale, 4.0)) * 0.25
    boxes = geo.merge_overlapping(
        geo.rect_boxes(binary, min_area, tuning.forklift.block_aspect_ratio)
    )
    return {
        "vehicle_present": 1.0,
        "vehicle_area_ratio": forklift.area / (frame.width * frame.height),
        "block_count": float(len(boxes)),
    }


EXTRACTORS = {
    "panel": panel_signals,
    "walkway": walkway_signals,
    "vest": vest_signals,
    "blocks": block_signals,
}


def percentiles(values: list[float]) -> str:
    if not values:
        return "        (no samples)"
    ordered = sorted(values)

    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
        return ordered[idx]

    return (
        f"n={len(ordered):<5} p05={pct(5):>8.4f} p25={pct(25):>8.4f} "
        f"p50={pct(50):>8.4f} p75={pct(75):>8.4f} p95={pct(95):>8.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--signals", default=",".join(SIGNALS))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    configure_logging("WARNING")
    settings = get_settings()
    tuning = settings.tuning
    wanted = [s.strip() for s in args.signals.split(",") if s.strip() in EXTRACTORS]

    random.seed(args.seed)
    yolo = YoloDetector(tuning.detection)
    yolo.warmup()

    root = settings.path(settings.data_root) / args.split
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not class_dirs:
        print(f"no class directories under {root}", file=sys.stderr)
        return 1

    # signal name -> class name -> values
    collected: dict[str, dict[str, list[float]]] = {}

    for class_dir in class_dirs:
        clips = sorted(glob.glob(str(class_dir / "*.mp4")))
        if not clips:
            continue
        chosen = random.sample(clips, min(args.per_class, len(clips)))
        print(f"  {class_dir.name}: {len(chosen)} clips", file=sys.stderr)

        for clip in chosen:
            frames = list(
                iter_frames(clip, tuning.video.sample_fps, tuning.video.infer_imgsz, args.frames)
            )
            observations = yolo.observe_batch(frames)
            for frame, observation in zip(frames, observations, strict=False):
                for name in wanted:
                    for key, value in EXTRACTORS[name](frame, observation, tuning).items():
                        collected.setdefault(key, {}).setdefault(class_dir.name, []).append(
                            float(value)
                        )

    print(f"\n{'=' * 100}")
    print(f"CALIBRATION  split={args.split}  per_class={args.per_class}  frames/clip={args.frames}")
    print("=" * 100)
    for key in sorted(collected):
        print(f"\n### {key}")
        for class_name in sorted(collected[key]):
            print(f"  {class_name:<34} {percentiles(collected[key][class_name])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
