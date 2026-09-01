"""Sweep detector thresholds against clip-level labels.

Running the full pipeline per parameter value would take hours, so YOLO observations are
computed once per clip and cached; the sweep then re-runs only the cheap decision logic.

Every threshold that ends up in ``config/settings.yaml`` should be justified by a curve from
here, and the chosen operating point recorded in ``docs/eval-baseline.md``.

    python scripts/sweep_thresholds.py --cache --per-class 12
    python scripts/sweep_thresholds.py --detector walkway
"""

from __future__ import annotations

import argparse
import glob
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from aegisflow.core.logging import configure_logging  # noqa: E402
from aegisflow.core.settings import get_settings  # noqa: E402
from aegisflow.detection import geometry as geo  # noqa: E402
from aegisflow.detection.video import iter_frames  # noqa: E402
from aegisflow.detection.yolo import YoloDetector  # noqa: E402

console = Console()
CACHE = "data/processed/sweep_cache.pkl"


def build_cache(per_class: int, frames: int) -> dict:
    """Cache per-frame cues for a balanced clip sample."""
    settings = get_settings()
    tuning = settings.tuning
    yolo = YoloDetector(tuning.detection)
    yolo.warmup()

    root = settings.path(settings.data_root) / "test"
    cache: dict[str, list[dict]] = {}

    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        clips = sorted(glob.glob(str(class_dir / "*.mp4")))[:per_class]
        console.print(f"  caching {class_dir.name}: {len(clips)} clips")
        for clip in clips:
            sampled = list(
                iter_frames(clip, tuning.video.sample_fps, tuning.video.infer_imgsz, frames)
            )
            observations = yolo.observe_batch(sampled)
            per_frame: list[dict] = []
            previous: list[tuple[float, float]] = []

            for frame, observation in zip(sampled, observations, strict=False):
                entry: dict[str, object] = {
                    "w": frame.width,
                    "h": frame.height,
                    "persons": [p.bbox for p in observation.persons],
                    "person_feet": [p.foot_point for p in observation.persons],
                    "person_centroids": [p.centroid for p in observation.persons],
                    "vehicles": [(v.bbox, v.area) for v in observation.vehicles],
                }

                # walkway: signed distance of each foot point to the green boundary
                mask = geo.close_mask(
                    geo.hsv_mask(frame.image, tuning.color.green_floor_line), 11, 3
                )
                contour = geo.largest_contour(
                    mask,
                    min_area=tuning.walkway.min_boundary_area_frac * frame.width * frame.height,
                )
                margins: list[float] = []
                if contour is not None:
                    for point in entry["person_feet"]:  # type: ignore[union-attr]
                        margins.append(
                            float(
                                cv2.pointPolygonTest(
                                    contour, (float(point[0]), float(point[1])), True
                                )
                            )
                        )
                entry["signed_margins"] = margins

                # vest colour per person
                greens, reds = [], []
                for box in entry["persons"]:  # type: ignore[union-attr]
                    torso = geo.crop(frame.image, geo.torso_roi(box, frame.image.shape))
                    if torso.size == 0:
                        greens.append(0.0)
                        reds.append(0.0)
                        continue
                    greens.append(geo.colour_ratio(torso, tuning.color.green_vest))
                    reds.append(
                        geo.red_ratio(torso, tuning.color.red_vest_low, tuning.color.red_vest_high)
                    )
                entry["greens"] = greens
                entry["reds"] = reds

                # motion of each person since the previous sampled frame
                motion: list[float] = []
                for cx, cy in entry["person_centroids"]:  # type: ignore[union-attr]
                    if previous:
                        motion.append(
                            min(abs(cx - px) + abs(cy - py) for px, py in previous) / frame.width
                        )
                    else:
                        motion.append(float("inf"))
                entry["motion"] = motion
                previous = list(entry["person_centroids"])  # type: ignore[arg-type]

                # forklift load geometry
                loads: list[dict] = []
                frame_area = float(frame.width * frame.height)
                for box, area in entry["vehicles"]:  # type: ignore[union-attr]
                    x1, y1, x2, y2 = box
                    bh, bw = y2 - y1, x2 - x1
                    roi = geo.clip_box(
                        (x1 - 0.12 * bw, y1 - 0.55 * bh, x2 + 0.12 * bw, y2), frame.image.shape
                    )
                    patch = geo.crop(frame.image, roi)
                    blocks = 0
                    fill = 0.0
                    if patch.size:
                        grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                        binary = cv2.adaptiveThreshold(
                            grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
                        )
                        binary = geo.close_mask(binary, 5, 1)
                        scale = (patch.shape[0] * patch.shape[1]) / (640.0 * 360.0)
                        min_area = (
                            tuning.forklift.block_min_area_px * max(0.15, min(scale, 4.0)) * 0.25
                        )
                        boxes = geo.merge_overlapping(
                            geo.rect_boxes(binary, min_area, tuning.forklift.block_aspect_ratio)
                        )
                        blocks = len(boxes)
                        fill = geo.mask_ratio(binary)
                    loads.append(
                        {
                            "area_frac": area / frame_area,
                            "aspect": bw / bh if bh else 0.0,
                            "blocks": blocks,
                            "fill": fill,
                        }
                    )
                entry["loads"] = loads
                per_frame.append(entry)

            cache[f"{class_dir.name}/{Path(clip).name}"] = per_frame

    path = get_settings().path(CACHE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(cache, handle)
    console.print(f"[green]cached {len(cache)} clips -> {path}[/green]")
    return cache


def load_cache() -> dict:
    path = get_settings().path(CACHE)
    if not path.exists():
        console.print(f"[red]{path} missing - run with --cache first[/red]")
        raise SystemExit(1)
    with path.open("rb") as handle:
        return pickle.load(handle)


def score(cache: dict, positive: str, predicate, persistence: int) -> tuple[float, float, float]:
    """Precision/recall/F1 for a per-frame predicate plus a persistence requirement."""
    tp = fp = fn = 0
    for key, frames in cache.items():
        run = 0
        fired = False
        for entry in frames:
            if predicate(entry):
                run += 1
                if run >= persistence:
                    fired = True
                    break
            else:
                run = 0
        is_positive = key.split("/")[0] == positive
        if is_positive and fired:
            tp += 1
        elif is_positive:
            fn += 1
        elif fired:
            fp += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def sweep_walkway(cache: dict) -> None:
    table = Table(title="Walkway: outside-margin threshold x persistence", title_style="bold")
    for column in ("margin_frac", "persist", "P", "R", "F1"):
        table.add_column(column, justify="right")

    best = (0.0, None)
    for margin_frac in (0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.20, 0.26):
        for persistence in (3, 5, 8):

            def predicate(entry, m=margin_frac):
                limit = m * entry["w"]
                return any(-d >= limit for d in entry["signed_margins"])

            p, r, f1 = score(cache, "0_safe_walkway_violation", predicate, persistence)
            table.add_row(
                f"{margin_frac:.2f}", str(persistence), f"{p:.2f}", f"{r:.2f}", f"{f1:.2f}"
            )
            if f1 > best[0]:
                best = (f1, (margin_frac, persistence, p, r))
    console.print(table)
    if best[1]:
        m, persistence, p, r = best[1]
        console.print(
            f"[green]best:[/green] margin_frac={m} persistence={persistence} "
            f"-> P={p:.2f} R={r:.2f} F1={best[0]:.2f}"
        )


def sweep_vest(cache: dict) -> None:
    table = Table(title="Unauthorized intervention: motion x persistence", title_style="bold")
    for column in ("green_thr", "motion", "persist", "P", "R", "F1"):
        table.add_column(column, justify="right")

    best = (0.0, None)
    for green_thr in (0.08, 0.12):
        for motion_limit in (0.010, 0.020, 0.035):
            for persistence in (4, 8, 12):

                def predicate(entry, g=green_thr, ml=motion_limit):
                    for index, box in enumerate(entry["persons"]):
                        if entry["greens"][index] >= g:
                            continue
                        if entry["motion"][index] > ml:
                            continue
                        if box[3] > 0.88 * entry["h"]:
                            continue
                        return True
                    return False

                p, r, f1 = score(cache, "1_unauthorized_intervention", predicate, persistence)
                table.add_row(
                    f"{green_thr:.2f}",
                    f"{motion_limit:.3f}",
                    str(persistence),
                    f"{p:.2f}",
                    f"{r:.2f}",
                    f"{f1:.2f}",
                )
                if f1 > best[0]:
                    best = (f1, (green_thr, motion_limit, persistence, p, r))
    console.print(table)
    if best[1]:
        g, ml, persistence, p, r = best[1]
        console.print(
            f"[green]best:[/green] green={g} motion={ml} persistence={persistence} "
            f"-> P={p:.2f} R={r:.2f} F1={best[0]:.2f}"
        )


def sweep_forklift(cache: dict) -> None:
    console.print("[bold]Forklift cue distributions (3=overload vs 7=safe)[/bold]")
    per_class: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key, frames in cache.items():
        class_name = key.split("/")[0]
        if class_name not in ("3_carrying_overload_with_forklift", "7_safe_carrying"):
            continue
        for entry in frames:
            for load in entry["loads"]:
                if not (0.01 <= load["area_frac"] <= 0.20):
                    continue
                for cue in ("blocks", "fill", "aspect", "area_frac"):
                    per_class[cue][class_name].append(float(load[cue]))

    table = Table(title="Cue percentiles", title_style="bold")
    for column in ("cue", "class", "n", "p10", "p25", "p50", "p75", "p90"):
        table.add_column(column, justify="right")
    for cue, classes in per_class.items():
        for class_name, values in sorted(classes.items()):
            if not values:
                continue
            ordered = sorted(values)

            # Bound explicitly: a closure over the loop variable would be read at call
            # time, which is correct only by accident here.
            def pct(q: float, data: list[float] = ordered) -> float:
                return data[min(len(data) - 1, round(q * (len(data) - 1)))]

            table.add_row(
                cue,
                class_name[:22],
                str(len(ordered)),
                f"{pct(0.10):.3f}",
                f"{pct(0.25):.3f}",
                f"{pct(0.50):.3f}",
                f"{pct(0.75):.3f}",
                f"{pct(0.90):.3f}",
            )
    console.print(table)

    sweep = Table(title="Forklift: block-count threshold x persistence", title_style="bold")
    for column in ("min_blocks", "persist", "P", "R", "F1"):
        sweep.add_column(column, justify="right")
    for min_blocks in (2, 3, 4, 5):
        for persistence in (2, 4, 6):

            def predicate(entry, mb=min_blocks):
                return any(
                    0.01 <= load["area_frac"] <= 0.20 and load["blocks"] >= mb
                    for load in entry["loads"]
                )

            p, r, f1 = score(cache, "3_carrying_overload_with_forklift", predicate, persistence)
            sweep.add_row(str(min_blocks), str(persistence), f"{p:.2f}", f"{r:.2f}", f"{f1:.2f}")
    console.print(sweep)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", action="store_true", help="rebuild the observation cache")
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--detector", default="all", choices=("all", "walkway", "vest", "forklift"))
    args = parser.parse_args()

    configure_logging("WARNING")
    cache = build_cache(args.per_class, args.frames) if args.cache else load_cache()
    console.print(f"[dim]{len(cache)} clips in cache[/dim]\n")

    if args.detector in ("all", "walkway"):
        sweep_walkway(cache)
    if args.detector in ("all", "vest"):
        sweep_vest(cache)
    if args.detector in ("all", "forklift"):
        sweep_forklift(cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
