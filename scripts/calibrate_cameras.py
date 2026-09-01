"""Build the camera registry: which fixed camera a frame came from.

Why
---
The policy describes two IP cameras covering different parts of the floor (Section 7.1), and
``config/zones.yaml`` records which behavioural domains each zone can exhibit - pedestrian
movement and forklift load in Zone-1, equipment intervention and electrical panels in
Zone-2. That mapping is only usable if the system knows which camera it is looking at.

The dataset clips carry no camera metadata, and resolving the camera from the *class folder*
would leak the ground-truth label into inference. So camera identity is recovered from the
image itself.

Method (deliberately label-free)
--------------------------------
1. Sample clips from across the whole train split, ignoring folder names entirely.
2. Reduce each to a coarse 9x16 greyscale fingerprint (the static scene, not its contents).
3. k-means with k=2. Fixed cameras produce two tight, well-separated clusters; measured
   here, within-camera distance is ~6 and between-camera ~30.
4. Name the clusters by matching against the panel baseline, which commissioning already
   tied to CAM-02. The other cluster is CAM-01.

Step 4 is the only step that consumes prior knowledge, and it is commissioning knowledge -
equivalent to an installer noting which stream is which - not a dataset label.

    python scripts/calibrate_cameras.py
    python scripts/calibrate_cameras.py --clips 120 --k 2
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

import numpy as np  # noqa: E402

from aegisflow.core.logging import configure_logging, get_logger  # noqa: E402
from aegisflow.core.settings import get_settings  # noqa: E402
from aegisflow.detection import geometry as geo  # noqa: E402
from aegisflow.detection.video import iter_frames  # noqa: E402

log = get_logger("calibrate_cameras")

ARTEFACT = "artifacts/models/camera_registry.json"
PANEL_BASELINE = "artifacts/models/panel_baseline.json"


def clip_fingerprint(path: str, frames: int, imgsz: int) -> np.ndarray | None:
    prints = [
        geo.scene_fingerprint(frame.image)
        for frame in iter_frames(path, sample_fps=2.0, imgsz=imgsz, max_frames=frames)
    ]
    if not prints:
        return None
    return np.median(np.stack(prints), axis=0)


def kmeans(data: np.ndarray, k: int, iterations: int = 40, seed: int = 0) -> np.ndarray:
    """Minimal k-means with k-means++ seeding. Avoids pulling in scikit-learn at runtime."""
    rng = np.random.default_rng(seed)
    centroids = [data[rng.integers(len(data))]]
    for _ in range(k - 1):
        distances = np.min(
            np.stack([np.abs(data - c).mean(axis=(1, 2)) for c in centroids]), axis=0
        )
        probabilities = distances / max(distances.sum(), 1e-9)
        centroids.append(data[rng.choice(len(data), p=probabilities)])
    centres = np.stack(centroids)

    labels = np.zeros(len(data), dtype=int)
    for _ in range(iterations):
        distances = np.stack([np.abs(data - c).mean(axis=(1, 2)) for c in centres])
        new_labels = distances.argmin(axis=0)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for index in range(k):
            member = data[labels == index]
            if len(member):
                centres[index] = member.mean(0)
    return labels, centres


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--clips", type=int, default=120, help="clips sampled overall")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--k", type=int, default=2, help="number of cameras")
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    configure_logging("INFO")
    random.seed(args.seed)
    settings = get_settings()
    imgsz = settings.tuning.video.infer_imgsz
    data_root = settings.path(settings.data_root)

    # Every clip in the split, pooled. Folder names are used only to enumerate files.
    all_clips = sorted(glob.glob(str(data_root / args.split / "*" / "*.mp4")))
    if len(all_clips) < args.k * 4:
        log.error("not enough clips under %s", data_root / args.split)
        return 1
    sample = random.sample(all_clips, min(args.clips, len(all_clips)))
    log.info("fingerprinting %d clips (label-free sample)", len(sample))

    prints: list[np.ndarray] = []
    for clip in sample:
        fingerprint = clip_fingerprint(clip, args.frames, imgsz)
        if fingerprint is not None:
            prints.append(fingerprint)

    data = np.stack(prints)
    labels, centres = kmeans(data, args.k, seed=args.seed)

    # Cluster quality
    stats = []
    for index in range(args.k):
        members = data[labels == index]
        if not len(members):
            continue
        within = float(np.abs(members - centres[index]).mean())
        stats.append((index, len(members), within))
        log.info(
            "cluster %d: %d clips, mean within-cluster distance %.2f", index, len(members), within
        )

    between = float(np.abs(centres[0] - centres[1]).mean()) if args.k == 2 else 0.0
    log.info("between-cluster distance: %.2f", between)
    if between < 2 * max(s[2] for s in stats):
        log.warning(
            "clusters are not well separated (between %.1f vs within %.1f). The cameras may "
            "look alike, or this dataset may come from a single view.",
            between,
            max(s[2] for s in stats),
        )

    # Name clusters using commissioning knowledge: the panel baseline is CAM-02.
    names = {index: f"CAM-{index + 1:02d}" for index in range(args.k)}
    baseline_path = settings.path(PANEL_BASELINE)
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        reference = baseline.get("scene_fingerprint")
        if reference:
            expected = np.asarray(reference, dtype=np.float32)
            distances = [float(np.abs(centres[i] - expected).mean()) for i in range(args.k)]
            panel_cluster = int(np.argmin(distances))
            panel_camera = str(baseline.get("camera", "CAM-02"))
            names = {}
            others = [c for c in ("CAM-01", "CAM-02", "CAM-03") if c != panel_camera]
            for index in range(args.k):
                names[index] = panel_camera if index == panel_cluster else others.pop(0)
            log.info(
                "cluster %d matches the panel baseline (distance %.2f) -> %s",
                panel_cluster,
                distances[panel_cluster],
                panel_camera,
            )
    else:
        log.warning("no panel baseline found; clusters get generic names")

    tolerance = max(12.0, max(s[2] for s in stats) * 2.2)
    registry = {
        "fingerprint_grid": list(geo.FINGERPRINT_GRID),
        "match_tolerance": round(tolerance, 2),
        "between_cluster_distance": round(between, 2),
        "calibrated_on": {"split": args.split, "clips": len(prints), "imgsz": imgsz},
        "cameras": [
            {
                "camera": names[index],
                "clips": int((labels == index).sum()),
                "within_cluster_distance": round(within, 2),
                "fingerprint": [[round(float(v), 2) for v in row] for row in centres[index]],
            }
            for index, _, within in stats
        ],
        "note": (
            "Clusters derived by k-means over scene fingerprints of a label-free sample of "
            "the train split. Used to resolve which camera - and therefore which zone, and "
            "therefore which behavioural domains - a frame belongs to at inference time."
        ),
    }

    path = settings.path(ARTEFACT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    log.info("wrote %s (%d cameras, tolerance %.1f)", path, len(stats), tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
