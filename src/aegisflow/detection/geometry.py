"""Geometry and colour helpers shared by the four detectors.

These are the functions that read the policy's *observable indicators* off the frame:
vest colour, walkway containment, panel geometry, block shapes. Keeping them here, pure and
free of detector state, is what makes them unit-testable on synthetic inputs rather than
only on video.
"""

from __future__ import annotations

import cv2
import numpy as np

from aegisflow.core.schemas import BBox
from aegisflow.core.settings import HSVBand

# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------


def area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = min(ax2, bx2) - max(ax1, bx1)
    dy = min(ay2, by2) - max(ay1, by1)
    return dx * dy if dx > 0 and dy > 0 else 0.0


def iou(a: BBox, b: BBox) -> float:
    inter = intersection(a, b)
    if inter <= 0:
        return 0.0
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def contains_fraction(inner: BBox, outer: BBox) -> float:
    """Share of ``inner``'s area that lies inside ``outer``."""
    inner_area = area(inner)
    return intersection(inner, outer) / inner_area if inner_area > 0 else 0.0


def centre_distance(a: BBox, b: BBox) -> float:
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return float(np.hypot(ax - bx, ay - by))


def gap(a: BBox, b: BBox) -> float:
    """Shortest edge-to-edge distance; 0 when the boxes touch or overlap."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return float(np.hypot(dx, dy))


def torso_roi(person: BBox, frame_shape: tuple[int, ...]) -> BBox:
    """Upper-body region of a person box - where a vest is visible.

    The top ~12% is mostly head, and below ~55% is legs. Sampling the band between them
    keeps hair and trousers out of the colour histogram, which is what made the vest
    classifier reliable rather than noisy.
    """
    x1, y1, x2, y2 = person
    height = y2 - y1
    width = x2 - x1
    top = y1 + 0.12 * height
    bottom = y1 + 0.58 * height
    inset = 0.12 * width
    return clip_box((x1 + inset, top, x2 - inset, bottom), frame_shape)


def clip_box(box: BBox, frame_shape: tuple[int, ...]) -> BBox:
    """Clamp a box to the frame."""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box
    return (
        float(max(0.0, min(x1, width - 1))),
        float(max(0.0, min(y1, height - 1))),
        float(max(0.0, min(x2, width))),
        float(max(0.0, min(y2, height))),
    )


def crop(image: np.ndarray, box: BBox) -> np.ndarray:
    x1, y1, x2, y2 = (round(v) for v in clip_box(box, image.shape))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=image.dtype)
    return image[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


def hsv_mask(image_bgr: np.ndarray, band: HSVBand) -> np.ndarray:
    """Binary mask of pixels inside an HSV band."""
    if image_bgr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(band.lower(), np.uint8), np.array(band.upper(), np.uint8))


def mask_ratio(mask: np.ndarray) -> float:
    """Share of a mask that is set, in [0, 1]."""
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def colour_ratio(image_bgr: np.ndarray, band: HSVBand) -> float:
    return mask_ratio(hsv_mask(image_bgr, band))


def red_ratio(image_bgr: np.ndarray, low: HSVBand, high: HSVBand) -> float:
    """Red wraps around H=0, so it needs two bands combined."""
    if image_bgr.size == 0:
        return 0.0
    mask = cv2.bitwise_or(hsv_mask(image_bgr, low), hsv_mask(image_bgr, high))
    return mask_ratio(mask)


# ---------------------------------------------------------------------------
# Contours and polygons
# ---------------------------------------------------------------------------


def largest_contour(mask: np.ndarray, min_area: float = 0.0) -> np.ndarray | None:
    """Biggest contour in a mask, or None."""
    if mask.size == 0:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    return best if cv2.contourArea(best) >= min_area else None


def close_mask(mask: np.ndarray, kernel: int = 7, iterations: int = 2) -> np.ndarray:
    """Morphological close - joins the dashes of a painted floor line into one region."""
    if mask.size == 0:
        return mask
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element, iterations=iterations)


def point_in_contour(point: tuple[float, float], contour: np.ndarray) -> bool:
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    array = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return point_in_contour(point, array)


def denormalise_polygon(
    polygon: list[tuple[float, float]], width: int, height: int
) -> list[tuple[float, float]]:
    """Convert a 0..1 polygon from config into pixel coordinates."""
    return [(x * width, y * height) for x, y in polygon]


def rect_aspect(contour: np.ndarray) -> float:
    """Width/height of a contour's bounding rect (0 when degenerate)."""
    _, _, w, h = cv2.boundingRect(contour)
    return float(w) / float(h) if h > 0 else 0.0


def contour_box(contour: np.ndarray) -> BBox:
    x, y, w, h = cv2.boundingRect(contour)
    return (float(x), float(y), float(x + w), float(y + h))


def rect_boxes(
    mask: np.ndarray,
    min_area: float,
    aspect_range: tuple[float, float],
) -> list[BBox]:
    """Rectangle-ish blobs in a mask, filtered on area and aspect ratio.

    Used for block counting on the forklift forks (policy Section 6.2: the observable
    criterion is the *number of blocks visible on the forks*).
    """
    if mask.size == 0:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[BBox] = []
    lo, hi = aspect_range
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        aspect = rect_aspect(contour)
        if not (lo <= aspect <= hi):
            continue
        out.append(contour_box(contour))
    return out


def merge_overlapping(boxes: list[BBox], iou_threshold: float = 0.35) -> list[BBox]:
    """Greedy merge of overlapping boxes, largest first.

    Prevents one physical block from being counted twice when its contour fragments -
    which matters because the safe/unsafe boundary is a count of 2 vs 3.
    """
    kept: list[BBox] = []
    for box in sorted(boxes, key=area, reverse=True):
        if all(iou(box, other) < iou_threshold for other in kept):
            kept.append(box)
    return kept


def vertical_edge_strength(image_bgr: np.ndarray) -> float:
    """Mean strength of vertical edges, normalised to roughly [0, 1].

    A swung-open panel cover adds a strong vertical edge where the closed cover was flush,
    so this is the primary signal for panel state (policy Section 5.2.2).
    """
    if image_bgr.size == 0:
        return 0.0
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    return float(np.mean(np.abs(sobel)) / 255.0)


def dark_region_ratio(image_bgr: np.ndarray, value_max: int = 60) -> float:
    """Share of near-black pixels.

    An open panel exposes the unlit cavity behind the cover, which reads as a dark region
    that a closed panel does not have.
    """
    if image_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return float(np.count_nonzero(hsv[:, :, 2] < value_max)) / float(hsv[:, :, 2].size)


# ---------------------------------------------------------------------------
# Camera identity
# ---------------------------------------------------------------------------

FINGERPRINT_GRID = (9, 16)
"""Rows x columns of the scene fingerprint. Coarse on purpose: it must identify the
camera, not the frame contents."""


def scene_fingerprint(image_bgr: np.ndarray, grid: tuple[int, int] = FINGERPRINT_GRID):
    """Coarse greyscale signature identifying which camera a frame came from.

    A camera-specific calibration must not be applied to a different camera. The dataset
    mixes two fixed cameras (policy Section 7.1), and clips carry no camera metadata, so
    the detector recovers camera identity from the frame itself.

    Measured on the test split against a CAM-02 reference: clips from that camera score a
    mean absolute distance of 3-10, clips from the other camera 29-38. A tolerance of ~12
    separates them with a wide margin, and uses no labels.
    """
    if image_bgr.size == 0:
        return np.zeros(grid, dtype=np.float32)
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.resize(grey, (grid[1], grid[0]), interpolation=cv2.INTER_AREA)


def fingerprint_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two fingerprints."""
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
