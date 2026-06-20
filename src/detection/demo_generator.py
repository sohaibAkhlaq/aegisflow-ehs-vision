"""
demo_generator.py — Generates synthetic demo violation events when no real
video clips are available.

Creates a set of realistic ClipViolation records (one per behavior class)
with rendered thumbnail images so the dashboard is fully populated on first run.
"""

from __future__ import annotations

import base64
import io
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .video_processor import ClipViolation

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Frame painters — generate synthetic annotated frames
# ─────────────────────────────────────────────────────────────────────────────

def _make_base_frame(label: str, bg_colour=(30, 30, 40)) -> np.ndarray:
    """Create a 640×360 dark frame with a watermark label."""
    frame = np.full((360, 640, 3), bg_colour, dtype=np.uint8)
    # Simulate factory floor texture
    for i in range(0, 360, 40):
        cv2.line(frame, (0, i), (640, i), (40, 40, 50), 1)
    for j in range(0, 640, 40):
        cv2.line(frame, (j, 0), (j, 360), (40, 40, 50), 1)
    cv2.putText(frame, "DEMO FRAME — NO REAL CLIP LOADED", (30, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    cv2.putText(frame, label, (30, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    return frame


def _draw_walkway_violation(frame: np.ndarray) -> np.ndarray:
    """Paint a scene showing a person outside green walkway markings."""
    f = frame.copy()
    # Green walkway markings
    cv2.rectangle(f, (260, 50), (380, 310), (0, 180, 0), 3)
    cv2.putText(f, "SAFE WALKWAY", (265, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
    # Person OUTSIDE walkway (left side)
    cv2.rectangle(f, (80, 100), (140, 280), (0, 0, 220), 2)   # red box
    cv2.circle(f, (110, 90), 20, (200, 170, 140), -1)          # head
    cv2.putText(f, "PERSON", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(f, "WALKWAY VIOLATION", (50, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    # Alert badge
    cv2.rectangle(f, (460, 10), (630, 50), (0, 0, 200), -1)
    cv2.putText(f, "!CRITICAL", (470, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return f


def _draw_unauthorized_intervention(frame: np.ndarray) -> np.ndarray:
    """Paint a person in red vest touching equipment."""
    f = frame.copy()
    # Machine/equipment (grey box)
    cv2.rectangle(f, (300, 80), (560, 290), (80, 80, 80), -1)
    cv2.rectangle(f, (300, 80), (560, 290), (120, 120, 120), 2)
    cv2.putText(f, "PRODUCTION MACHINE", (310, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    # Person in RED vest near machine
    cv2.rectangle(f, (220, 100), (310, 280), (0, 80, 220), 2)   # orange
    cv2.circle(f, (265, 90), 22, (200, 170, 140), -1)
    # Red vest torso
    cv2.rectangle(f, (232, 130), (298, 240), (0, 30, 180), -1)
    cv2.putText(f, "RED VEST", (225, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 80, 255), 1)
    cv2.putText(f, "UNAUTHORIZED", (60, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)
    cv2.rectangle(f, (460, 10), (630, 50), (0, 100, 200), -1)
    cv2.putText(f, "! HIGH", (490, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return f


def _draw_open_panel(frame: np.ndarray) -> np.ndarray:
    """Paint an open electrical panel."""
    f = frame.copy()
    # Panel body
    cv2.rectangle(f, (200, 80), (420, 290), (60, 60, 60), -1)
    cv2.rectangle(f, (200, 80), (420, 290), (100, 100, 100), 2)
    # Open door (rotated)
    pts = np.array([[200, 80], [200, 290], [130, 260], [130, 110]], np.int32)
    cv2.fillPoly(f, [pts], (70, 70, 70))
    cv2.polylines(f, [pts], True, (110, 110, 110), 2)
    # Wiring inside
    for y in range(100, 280, 25):
        colour = (
            (0, 0, 200) if y % 75 == 100 else
            (0, 200, 0) if y % 75 == 25 else
            (0, 200, 200)
        )
        cv2.line(f, (210, y), (410, y), colour, 2)
    cv2.putText(f, "OPEN PANEL", (210, 310),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100), 2)
    cv2.putText(f, "ELECTRICAL HAZARD", (180, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.rectangle(f, (460, 10), (630, 50), (0, 150, 0), -1)
    cv2.putText(f, "LOW", (500, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return f


def _draw_forklift_overload(frame: np.ndarray) -> np.ndarray:
    """Paint a forklift carrying 3 blocks."""
    f = frame.copy()
    # Forklift body
    cv2.rectangle(f, (100, 150), (380, 320), (50, 90, 130), -1)
    cv2.rectangle(f, (100, 150), (380, 320), (80, 120, 160), 2)
    cv2.putText(f, "FORKLIFT", (170, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 240), 2)
    # Forks
    cv2.rectangle(f, (380, 260), (570, 275), (80, 80, 80), -1)
    cv2.rectangle(f, (380, 285), (570, 300), (80, 80, 80), -1)
    # 3 blocks on forks
    block_colours = [(0, 120, 200), (0, 160, 220), (0, 180, 240)]
    for i, col in enumerate(block_colours):
        bx = 390 + i * 60
        cv2.rectangle(f, (bx, 220), (bx + 50, 265), col, -1)
        cv2.rectangle(f, (bx, 220), (bx + 50, 265), (200, 200, 200), 1)
        cv2.putText(f, str(i + 1), (bx + 18, 248),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(f, "3 BLOCKS — OVERLOAD", (100, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
    cv2.rectangle(f, (440, 10), (630, 50), (0, 100, 200), -1)
    cv2.putText(f, "! HIGH", (460, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Demo violations
# ─────────────────────────────────────────────────────────────────────────────

_PAINTERS = [
    _draw_walkway_violation,
    _draw_unauthorized_intervention,
    _draw_open_panel,
    _draw_forklift_overload,
]

_DEMO_META = [
    {
        "class_id": 0, "name": "Safe Walkway Violation",
        "zone": "Zone-Mid-Left", "policy_ref": "Section 3.3.2",
        "clip_id": "demo_clip_walkway",
        "description": (
            "DEMO: Person detected outside green-marked Designated Safe Walkway boundary, "
            "moving into an area designated for forklift and machinery operation."
        ),
        "source": "demo_generator",
    },
    {
        "class_id": 1, "name": "Unauthorized Intervention",
        "zone": "Zone-Mid-Center", "policy_ref": "Section 4.3.2",
        "clip_id": "demo_clip_intervention",
        "description": (
            "DEMO: Person in red-black safety vest (general personnel) detected "
            "directly interacting with production equipment without the green authorization vest."
        ),
        "source": "demo_generator",
    },
    {
        "class_id": 2, "name": "Opened Panel Cover",
        "zone": "Zone-Mid-Right", "policy_ref": "Section 5.2.2",
        "clip_id": "demo_clip_panel",
        "description": (
            "DEMO: Electrical panel cover detected in open position. "
            "Exposed wiring creates electrocution risk during active production operations."
        ),
        "source": "demo_generator",
    },
    {
        "class_id": 3, "name": "Carrying Overload with Forklift",
        "zone": "Zone-Bottom-Center", "policy_ref": "Section 6.3.2",
        "clip_id": "demo_clip_forklift",
        "description": (
            "DEMO: Forklift observed carrying 3 standardized blocks — exceeds the "
            "safe limit of 2 blocks per load (Section 6.2). Vehicle instability risk."
        ),
        "source": "demo_generator",
    },
]


def generate_demo_violations(count_per_class: int = 2) -> list[ClipViolation]:
    """
    Generate synthetic demo violations for all 4 behavior classes.
    Used when data/ directory contains no real video clips.
    """
    violations: list[ClipViolation] = []
    base_time = time.time()

    for rep in range(count_per_class):
        for i, meta in enumerate(_DEMO_META):
            base_frame = _make_base_frame(f"KMP-OHS-POL-001 | {meta['name']}")
            painter = _PAINTERS[meta["class_id"]]
            annotated = painter(base_frame)
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            thumb = base64.b64encode(buf.tobytes()).decode("utf-8")

            violations.append(ClipViolation(
                event_id=str(uuid.uuid4()),
                clip_id=f"{meta['clip_id']}_{rep}",
                clip_path=f"data/{meta['clip_id']}_{rep}.mp4",
                frame_index=rep * 120 + i * 30,
                timestamp_sec=float(rep * 8 + i * 2),
                behavior_class_id=meta["class_id"],
                behavior_name=meta["name"],
                confidence=0.87 - (i * 0.03) - (rep * 0.01),
                zone=meta["zone"],
                description=meta["description"],
                policy_ref=meta["policy_ref"],
                detector_source=meta["source"],
                thumbnail_b64=thumb,
            ))

    logger.info(f"Generated {len(violations)} demo violations")
    return violations
