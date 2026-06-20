"""
detector.py — Module 1 core: YOLOv8 + Gemini Vision hybrid detection engine.

Detection pipeline per video frame:
  1. Run YOLOv8 to detect persons, forklifts, and objects
  2. Apply compliance heuristics:
       - Class 0 (Walkway): person bbox centroid outside green zone polygons
       - Class 1 (Intervention): person near equipment without green vest colour
       - Class 2 (Panel): open-rectangle / cabinet-like region with visible interior
       - Class 3 (Forklift): forklift with 3+ block objects on forks
  3. If YOLO confidence < threshold → route to Gemini Vision for confirmation
  4. Return list of DetectionEvent objects
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class DetectionEvent:
    """A single violation event produced by the detector."""
    behavior_class_id: int       # 0-3
    behavior_name: str
    confidence: float            # 0.0 – 1.0
    frame_index: int
    timestamp_sec: float         # position in the clip
    bbox: Optional[BoundingBox]
    zone: str                    # inferred zone label
    description: str             # human-readable observation
    policy_ref: str
    detector_source: str         # "yolo" | "gemini_vlm" | "heuristic"
    raw_frame: Optional[np.ndarray] = field(default=None, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

# HSV ranges for vest colours (broad to handle factory lighting)
_GREEN_VEST_HSV = [(35, 60, 40), (90, 255, 255)]   # hue 35–90°
_RED_VEST_HSV_1 = [(0, 60, 40), (10, 255, 255)]    # hue 0–10°
_RED_VEST_HSV_2 = [(160, 60, 40), (180, 255, 255)] # hue 160–180° (red wraps)


def _dominant_colour_in_roi(
    frame: np.ndarray, bbox: BoundingBox
) -> dict[str, float]:
    """Return fraction of pixels matching green/red-black vest colours in a bbox ROI."""
    h, w = frame.shape[:2]
    x1, y1 = int(max(0, bbox.x1)), int(max(0, bbox.y1))
    x2, y2 = int(min(w, bbox.x2)), int(min(h, bbox.y2))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return {"green": 0.0, "red": 0.0, "other": 1.0}

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]

    green_mask = cv2.inRange(
        hsv,
        np.array(_GREEN_VEST_HSV[0], np.uint8),
        np.array(_GREEN_VEST_HSV[1], np.uint8),
    )
    red_mask1 = cv2.inRange(
        hsv,
        np.array(_RED_VEST_HSV_1[0], np.uint8),
        np.array(_RED_VEST_HSV_1[1], np.uint8),
    )
    red_mask2 = cv2.inRange(
        hsv,
        np.array(_RED_VEST_HSV_2[0], np.uint8),
        np.array(_RED_VEST_HSV_2[1], np.uint8),
    )
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    green_frac = float(cv2.countNonZero(green_mask)) / total
    red_frac = float(cv2.countNonZero(red_mask)) / total
    return {"green": green_frac, "red": red_frac, "other": 1 - green_frac - red_frac}


def _is_green_zone(frame: np.ndarray, cx: float, cy: float, radius: int = 20) -> bool:
    """Check if the area around (cx, cy) contains green floor markings."""
    h, w = frame.shape[:2]
    x1, y1 = int(max(0, cx - radius)), int(max(0, cy - radius))
    x2, y2 = int(min(w, cx + radius)), int(min(h, cy + radius))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([90, 255, 255]))
    frac = cv2.countNonZero(green_mask) / (roi.shape[0] * roi.shape[1] + 1e-9)
    return frac > 0.08


def _detect_open_panel(frame: np.ndarray) -> list[tuple[BoundingBox, float]]:
    """
    Heuristic: detect open electrical panels as rectangular regions with
    visible dark interior contrasting with lighter surroundings.
    Returns list of (bbox, confidence) tuples.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panels = []
    h, w = frame.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000 or area > w * h * 0.4:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) not in (4, 5, 6):
            continue
        rect = cv2.boundingRect(approx)
        rx, ry, rw, rh = rect
        aspect = rw / (rh + 1e-9)
        if not (0.3 < aspect < 3.5):
            continue
        # Check interior is darker (open panel interior)
        interior = gray[ry + 5:ry + rh - 5, rx + 5:rx + rw - 5]
        if interior.size == 0:
            continue
        mean_interior = float(np.mean(interior))
        # Surrounding brightness
        surrounding = gray[max(0, ry - 10):ry + rh + 10, max(0, rx - 10):rx + rw + 10]
        mean_surround = float(np.mean(surrounding))
        contrast = mean_surround - mean_interior
        if contrast > 20:  # Interior significantly darker → panel open
            conf = min(0.9, 0.5 + contrast / 100.0)
            bbox = BoundingBox(float(rx), float(ry), float(rx + rw), float(ry + rh), conf, "panel")
            panels.append((bbox, conf))

    return panels[:3]  # Return top 3 candidates


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Vision fallback
# ─────────────────────────────────────────────────────────────────────────────

_VLM_PROMPT = """
You are a factory safety compliance inspector reviewing a single video frame.

Your task: Identify any of these 4 unsafe behaviors present in the image:

0. Safe Walkway Violation — a person is outside the green-marked floor walkway boundaries
1. Unauthorized Intervention — a person is interacting with machinery WITHOUT a green safety vest
2. Opened Panel Cover — an electrical panel/cabinet cover is visibly open
3. Carrying Overload with Forklift — a forklift is carrying 3 or more standardized blocks

Respond ONLY as a JSON array. Each element = one detected violation:
[
  {
    "class_id": <0|1|2|3>,
    "behavior_name": "<name>",
    "confidence": <0.0-1.0>,
    "description": "<one sentence of what you observe>",
    "zone": "<describe where in the frame: top-left, center, etc.>"
  }
]

If NO violations are detected, return an empty array: []
Return ONLY the JSON, no markdown, no extra text.
"""


def _frame_to_base64(frame: np.ndarray) -> str:
    """Encode a BGR frame to base64 JPEG string for Gemini Vision."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _query_gemini_vision(frame: np.ndarray, api_key: str) -> list[dict]:
    """Send frame to Gemini Vision and return parsed violation list."""
    try:
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        img_b64 = _frame_to_base64(frame)
        img_part = {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": img_b64,
            }
        }

        response = model.generate_content(
            [_VLM_PROMPT, img_part],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            },
        )
        raw = response.text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as exc:
        logger.debug(f"Gemini VLM query failed: {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Zone labelling
# ─────────────────────────────────────────────────────────────────────────────

def _infer_zone(bbox: Optional[BoundingBox], frame_w: int, frame_h: int) -> str:
    """Map bounding box position to a named zone."""
    if bbox is None:
        return "Zone-Unknown"
    cx, cy = bbox.centroid
    col = "Left" if cx < frame_w / 3 else ("Center" if cx < 2 * frame_w / 3 else "Right")
    row = "Top" if cy < frame_h / 3 else ("Mid" if cy < 2 * frame_h / 3 else "Bottom")
    return f"Zone-{row}-{col}"


# ─────────────────────────────────────────────────────────────────────────────
# Main detector class
# ─────────────────────────────────────────────────────────────────────────────

class ComplianceDetector:
    """
    Hybrid compliance detector.
    Primary: YOLOv8 + heuristic post-processing
    Fallback: Gemini Vision (when YOLO confidence < threshold)
    """

    def __init__(
        self,
        yolo_model_name: str = "yolov8n.pt",
        yolo_conf: float = 0.45,
        vlm_fallback_conf: float = 0.50,
        api_key: str = "",
    ) -> None:
        self.yolo_conf = yolo_conf
        self.vlm_fallback_conf = vlm_fallback_conf
        self.api_key = api_key
        self._model = None
        self._model_name = yolo_model_name

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self._model_name)
                logger.info(f"YOLOv8 model '{self._model_name}' loaded")
            except Exception as exc:
                logger.error(f"Failed to load YOLO model: {exc}")
                self._model = None
        return self._model

    # ── per-frame heuristic logic ─────────────────────────────────────────────

    def _check_walkway_violation(
        self, frame: np.ndarray, persons: list[BoundingBox]
    ) -> list[DetectionEvent]:
        events = []
        h, w = frame.shape[:2]
        for p in persons:
            cx, cy = p.centroid
            # Walkway violation: person's foot position NOT in green zone
            foot_y = p.y2 - (p.y2 - p.y1) * 0.1  # near bottom of bbox
            in_walkway = _is_green_zone(frame, cx, foot_y, radius=30)
            if not in_walkway:
                events.append(DetectionEvent(
                    behavior_class_id=0,
                    behavior_name="Safe Walkway Violation",
                    confidence=p.confidence * 0.85,
                    frame_index=0,
                    timestamp_sec=0.0,
                    bbox=p,
                    zone=_infer_zone(p, w, h),
                    description=(
                        f"Person detected at position ({cx:.0f}, {cy:.0f}) outside "
                        "green-marked Designated Safe Walkway boundary."
                    ),
                    policy_ref="Section 3.3.2",
                    detector_source="yolo+heuristic",
                ))
        return events

    def _check_unauthorized_intervention(
        self, frame: np.ndarray, persons: list[BoundingBox], equipment_boxes: list[BoundingBox]
    ) -> list[DetectionEvent]:
        events = []
        h, w = frame.shape[:2]
        for p in persons:
            # Check proximity to any equipment
            pcx, pcy = p.centroid
            near_equipment = False
            for eq in equipment_boxes:
                eqcx, eqcy = eq.centroid
                dist = ((pcx - eqcx) ** 2 + (pcy - eqcy) ** 2) ** 0.5
                if dist < max(w, h) * 0.25:
                    near_equipment = True
                    break

            if near_equipment:
                colours = _dominant_colour_in_roi(frame, p)
                has_green_vest = colours["green"] > 0.05
                if not has_green_vest:
                    events.append(DetectionEvent(
                        behavior_class_id=1,
                        behavior_name="Unauthorized Intervention",
                        confidence=p.confidence * 0.80,
                        frame_index=0,
                        timestamp_sec=0.0,
                        bbox=p,
                        zone=_infer_zone(p, w, h),
                        description=(
                            "Person interacting with production equipment without "
                            "wearing the designated green authorization safety vest "
                            f"(green fraction: {colours['green']:.2f})."
                        ),
                        policy_ref="Section 4.3.2",
                        detector_source="yolo+heuristic",
                    ))
        return events

    def _check_open_panel(
        self, frame: np.ndarray
    ) -> list[DetectionEvent]:
        h, w = frame.shape[:2]
        panels = _detect_open_panel(frame)
        events = []
        for bbox, conf in panels:
            events.append(DetectionEvent(
                behavior_class_id=2,
                behavior_name="Opened Panel Cover",
                confidence=conf,
                frame_index=0,
                timestamp_sec=0.0,
                bbox=bbox,
                zone=_infer_zone(bbox, w, h),
                description=(
                    "Electrical panel cover detected in open position during "
                    "production operations — protective barrier breached."
                ),
                policy_ref="Section 5.2.2",
                detector_source="heuristic",
            ))
        return events

    def _check_forklift_overload(
        self, frame: np.ndarray, forklifts: list[BoundingBox], boxes: list[BoundingBox]
    ) -> list[DetectionEvent]:
        events = []
        h, w = frame.shape[:2]
        for fk in forklifts:
            # Count boxes/pallets in forklift fork region (lower portion of forklift bbox)
            fork_region = BoundingBox(
                fk.x1, fk.y1 + (fk.y2 - fk.y1) * 0.5,
                fk.x2, fk.y2, fk.confidence, "fork_region"
            )
            block_count = sum(
                1 for b in boxes
                if _bbox_overlap(b, fork_region) > 0.3
            )
            # If no separate boxes detected, estimate from fork region darkness patterns
            if block_count == 0:
                block_count = _estimate_block_count_from_region(frame, fork_region)

            if block_count >= 3:
                events.append(DetectionEvent(
                    behavior_class_id=3,
                    behavior_name="Carrying Overload with Forklift",
                    confidence=min(0.9, 0.6 + block_count * 0.05),
                    frame_index=0,
                    timestamp_sec=0.0,
                    bbox=fk,
                    zone=_infer_zone(fk, w, h),
                    description=(
                        f"Forklift detected carrying approximately {block_count} "
                        "standardized blocks — exceeds safe limit of 2 blocks per load "
                        "(Section 6.2 threshold: 3 or more = non-compliant)."
                    ),
                    policy_ref="Section 6.3.2",
                    detector_source="yolo+heuristic",
                ))
        return events

    # ── YOLO inference ─────────────────────────────────────────────────────────

    def _run_yolo(self, frame: np.ndarray) -> dict[str, list[BoundingBox]]:
        """Run YOLO and bucket detections by semantic class."""
        model = self._load_model()
        result_map: dict[str, list[BoundingBox]] = {
            "person": [], "forklift": [], "equipment": [], "box": []
        }
        if model is None:
            return result_map

        try:
            results = model(frame, conf=self.yolo_conf, verbose=False)
            for det in results[0].boxes:
                cls_name = model.names[int(det.cls[0])].lower()
                x1, y1, x2, y2 = det.xyxy[0].tolist()
                conf = float(det.conf[0])
                bbox = BoundingBox(x1, y1, x2, y2, conf, cls_name)

                if cls_name == "person":
                    result_map["person"].append(bbox)
                elif any(k in cls_name for k in ("forklift", "truck", "vehicle")):
                    result_map["forklift"].append(bbox)
                elif any(k in cls_name for k in ("box", "suitcase", "backpack", "sports ball")):
                    result_map["box"].append(bbox)
                elif any(k in cls_name for k in (
                    "refrigerator", "tv", "laptop", "keyboard", "microwave",
                    "oven", "toaster", "sink", "cell phone"
                )):
                    result_map["equipment"].append(bbox)
        except Exception as exc:
            logger.debug(f"YOLO inference error: {exc}")

        return result_map

    # ── public API ─────────────────────────────────────────────────────────────

    def detect_frame(
        self, frame: np.ndarray, frame_index: int = 0, fps: float = 25.0
    ) -> list[DetectionEvent]:
        """
        Run the full compliance detection pipeline on a single frame.
        Returns a (possibly empty) list of DetectionEvents.
        """
        timestamp = frame_index / max(fps, 1.0)
        detections: list[DetectionEvent] = []

        # Load dynamic rules configuration
        config_path = Path(__file__).resolve().parent.parent.parent / "outputs" / "rules_config.json"
        active_rules = {0: True, 1: True, 2: True, 3: True}
        thresholds = {0: 0.15, 1: 0.15, 2: 0.15, 3: 0.15}  # low threshold to catch and filter later
        
        if config_path.exists():
            try:
                import json
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                for cid_str, val in config.items():
                    cid = int(cid_str)
                    active_rules[cid] = val.get("active", True)
                    thresholds[cid] = val.get("confidence_threshold", 0.15)
            except Exception:
                pass

        # 1. YOLOv8 detection
        yolo_map = self._run_yolo(frame)
        persons = yolo_map["person"]
        forklifts = yolo_map["forklift"]
        equipment = yolo_map["equipment"]
        boxes = yolo_map["box"]

        # 2. Apply heuristic compliance checks
        detections += self._check_walkway_violation(frame, persons)
        detections += self._check_unauthorized_intervention(frame, persons, equipment)
        detections += self._check_open_panel(frame)
        detections += self._check_forklift_overload(frame, forklifts, boxes)

        # 3. Gemini VLM fallback: if all YOLO confidences were low OR no detections
        yolo_max_conf = max((d.confidence for d in detections), default=0.0)
        if (yolo_max_conf < self.vlm_fallback_conf or not detections) and self.api_key:
            logger.debug(f"Frame {frame_index}: YOLO conf={yolo_max_conf:.2f} → VLM fallback")
            vlm_results = _query_gemini_vision(frame, self.api_key)
            for v in vlm_results:
                # Only add if not already detected by YOLO with higher confidence
                existing_ids = {d.behavior_class_id for d in detections}
                if v.get("class_id") not in existing_ids and v.get("confidence", 0) > 0.4:
                    h, w = frame.shape[:2]
                    detections.append(DetectionEvent(
                        behavior_class_id=v["class_id"],
                        behavior_name=v.get("behavior_name", "Unknown"),
                        confidence=float(v.get("confidence", 0.5)),
                        frame_index=frame_index,
                        timestamp_sec=timestamp,
                        bbox=None,
                        zone=v.get("zone", "Zone-Unknown"),
                        description=v.get("description", "Detected by Gemini Vision"),
                        policy_ref=_CLASS_TO_POLICY_REF.get(v.get("class_id", -1), "Unknown"),
                        detector_source="gemini_vlm",
                        raw_frame=frame,
                    ))

        # 4. Stamp frame metadata and filter by active/confidence rules
        filtered = []
        for ev in detections:
            ev.frame_index = frame_index
            ev.timestamp_sec = timestamp
            ev.raw_frame = frame
            
            cid = ev.behavior_class_id
            if active_rules.get(cid, True) and ev.confidence >= thresholds.get(cid, 0.45):
                filtered.append(ev)

        return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

_CLASS_TO_POLICY_REF = {
    0: "Section 3.3.2",
    1: "Section 4.3.2",
    2: "Section 5.2.2",
    3: "Section 6.3.2",
}


def _bbox_overlap(a: BoundingBox, b: BoundingBox) -> float:
    """Return IoU (intersection-over-union) between two bboxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / (union + 1e-9)


def _estimate_block_count_from_region(frame: np.ndarray, region: BoundingBox) -> int:
    """
    Estimate number of rectangular block objects in a region using contour analysis.
    Used when YOLO doesn't detect separate 'box' objects on forklift forks.
    """
    h, w = frame.shape[:2]
    x1, y1 = int(max(0, region.x1)), int(max(0, region.y1))
    x2, y2 = int(min(w, region.x2)), int(min(h, region.y2))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = roi.shape[0] * roi.shape[1]
    block_contours = [
        c for c in contours
        if roi_area * 0.03 < cv2.contourArea(c) < roi_area * 0.6
    ]
    return len(block_contours)
