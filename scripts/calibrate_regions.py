"""Commission the Designated Safe Walkway polygon, per camera.

Why
---
Policy Section 3.2 says the walkway is delineated by green painted lines and that those
markings are "the primary reference boundary for the automated walkway compliance detection
system". Segmenting them live turns out not to work: the largest green region in a frame is
one painted *line*, not the corridor between the lines, so a containment test against it
reports almost everyone as outside. Measured on the test split, that cue caps at F1 0.25 at
every threshold - see docs/eval-baseline.md.

What works is commissioning the boundary once per camera, the same pattern that made the
panel detector reliable. In a real installation the polygon is drawn on screen during
commissioning. Here it is recovered from compliant traffic: the convex hull of foot points
observed in clips of people walking the route correctly *is* the permitted pedestrian zone.

Per camera matters. Pooling both views produces a hull covering 67% of the frame, which
excludes nobody; the CAM-01 hull alone is 9.4%.

    python scripts/calibrate_regions.py
    python scripts/calibrate_regions.py --clips 24 --shrink 0.04
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aegisflow.core.logging import configure_logging, get_logger  # noqa: E402
from aegisflow.core.settings import get_settings  # noqa: E402
from aegisflow.detection import geometry as geo  # noqa: E402
from aegisflow.detection.video import iter_frames  # noqa: E402
from aegisflow.detection.yolo import YoloDetector  # noqa: E402

log = get_logger("calibrate_regions")

COMPLIANT_WALKING_CLASS = "4_safe_walkway"
ARTEFACT = "artifacts/models/region_baseline.json"
CAMERA_REGISTRY = "artifacts/models/camera_registry.json"


def load_cameras(settings) -> dict[str, np.ndarray]:
    path = settings.path(CAMERA_REGISTRY)
    if not path.exists():
        log.error("%s missing - run scripts/calibrate_cameras.py first", path)
        return {}
    registry = json.loads(path.read_text(encoding="utf-8"))
    return {
        entry["camera"]: np.asarray(entry["fingerprint"], dtype=np.float32)
        for entry in registry.get("cameras", [])
    }


def identify(image, cameras: dict[str, np.ndarray]) -> str | None:
    if not cameras:
        return None
    fingerprint = geo.scene_fingerprint(image)
    return min(cameras, key=lambda name: geo.fingerprint_distance(fingerprint, cameras[name]))


def shrink_hull(hull: np.ndarray, factor: float) -> np.ndarray:
    """Pull the hull slightly toward its centroid.

    The hull is the *outer envelope* of compliant traffic, so its edge is where compliant
    people were still standing. Shrinking a little makes the boundary the walkway edge
    rather than its outermost observed point, which is what the margin test needs.
    """
    if factor <= 0:
        return hull
    points = hull.reshape(-1, 2).astype(np.float32)
    centre = points.mean(axis=0)
    pulled = centre + (points - centre) * (1.0 - factor)
    return pulled.reshape(-1, 1, 2).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--clips", type=int, default=24, help="compliant clips to sample")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--shrink", type=float, default=0.03)
    parser.add_argument("--min-points", type=int, default=15)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    configure_logging("INFO")
    random.seed(args.seed)
    settings = get_settings()
    tuning = settings.tuning
    data_root = settings.path(settings.data_root)

    if args.split == "test":
        log.warning("commissioning on the test split leaks evaluation data; prefer --split train")

    cameras = load_cameras(settings)
    if not cameras:
        return 1

    clips = sorted(glob.glob(str(data_root / args.split / COMPLIANT_WALKING_CLASS / "*.mp4")))
    if not clips:
        log.error("no clips under %s", data_root / args.split / COMPLIANT_WALKING_CLASS)
        return 1
    sample = random.sample(clips, min(args.clips, len(clips)))
    log.info("observing compliant walking in %d clips", len(sample))

    yolo = YoloDetector(tuning.detection)
    yolo.warmup()

    by_camera: dict[str, list[tuple[float, float]]] = defaultdict(list)
    frame_size: tuple[int, int] = (tuning.video.infer_imgsz, tuning.video.infer_imgsz)

    for clip in sample:
        frames = list(
            iter_frames(clip, tuning.video.sample_fps, tuning.video.infer_imgsz, args.frames)
        )
        if not frames:
            continue
        camera = identify(frames[0].image, cameras)
        if camera is None:
            continue
        frame_size = (frames[0].width, frames[0].height)
        for observation in yolo.observe_batch(frames):
            for person in observation.persons:
                by_camera[camera].append(person.foot_point)

    width, height = frame_size
    regions: dict[str, object] = {}

    for camera, points in by_camera.items():
        if len(points) < args.min_points:
            log.warning("%s: only %d foot points; skipping", camera, len(points))
            continue
        array = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        hull = shrink_hull(cv2.convexHull(array), args.shrink)
        area_frac = float(cv2.contourArea(hull) / (width * height))

        normalised = [
            [round(float(x) / width, 5), round(float(y) / height, 5)]
            for x, y in hull.reshape(-1, 2)
        ]
        regions[camera] = {
            "walkway_polygon": normalised,
            "observed_points": len(points),
            "area_fraction": round(area_frac, 4),
        }
        log.info(
            "%s: walkway polygon from %d foot points, %d vertices, %.1f%% of frame",
            camera,
            len(points),
            len(normalised),
            area_frac * 100,
        )
        if area_frac > 0.45:
            log.warning(
                "%s walkway covers %.0f%% of the frame - too permissive to exclude anyone. "
                "More compliant clips from this camera would tighten it.",
                camera,
                area_frac * 100,
            )

    if not regions:
        log.error("no regions commissioned")
        return 1

    payload = {
        "source": "convex hull of foot points observed during compliant walking",
        "policy_reference": "Section 3.2 - Designated Safe Walkway boundaries",
        "calibrated_on": {
            "split": args.split,
            "class": COMPLIANT_WALKING_CLASS,
            "clips": len(sample),
            "frame_size": [width, height],
            "shrink": args.shrink,
        },
        "cameras": regions,
        "note": (
            "Commissioning parameter, equivalent to drawing the walkway on screen at "
            "install time. It tells the detector WHERE the permitted pedestrian zone is; "
            "whether a person is inside it remains the policy's observable indicator."
        ),
    }

    path = settings.path(ARTEFACT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
