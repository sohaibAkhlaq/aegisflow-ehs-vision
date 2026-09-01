"""Camera identification and per-camera commissioned regions.

The policy describes two fixed cameras covering different parts of the floor (Section 7.1),
and several detectors need camera-specific commissioning data: where the walkway is, where
the electrical panel is, which zone a record belongs to.

Clips carry no camera metadata. Resolving the camera from the *class folder* would leak the
ground-truth label into inference, so identity is recovered from the frame itself using the
coarse scene fingerprints built by ``scripts/calibrate_cameras.py``. Measured separation on
this dataset: ~6 within a camera, ~30 between cameras.

Everything here degrades gracefully. With no registry, :meth:`CameraRegistry.identify`
returns ``None`` and callers fall back to their uncommissioned behaviour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from aegisflow.core.logging import get_logger
from aegisflow.core.settings import Settings, get_settings
from aegisflow.detection import geometry as geo

log = get_logger(__name__)

CAMERA_REGISTRY = "artifacts/models/camera_registry.json"
REGION_BASELINE = "artifacts/models/region_baseline.json"


@dataclass
class CameraRegistry:
    """Maps a frame to the camera that produced it."""

    fingerprints: dict[str, np.ndarray] = field(default_factory=dict)
    tolerance: float = 12.0

    @property
    def available(self) -> bool:
        return bool(self.fingerprints)

    def identify(self, image: np.ndarray) -> str | None:
        """Nearest camera by fingerprint, or None if none is close enough."""
        if not self.fingerprints:
            return None
        fingerprint = geo.scene_fingerprint(image)
        best_name, best_distance = None, float("inf")
        for name, reference in self.fingerprints.items():
            distance = geo.fingerprint_distance(fingerprint, reference)
            if distance < best_distance:
                best_name, best_distance = name, distance
        if best_distance > self.tolerance:
            # An unknown view: better to say so than to apply another camera's calibration.
            log.debug("no camera within tolerance (nearest %s at %.1f)", best_name, best_distance)
            return None
        return best_name


@dataclass
class RegionBaseline:
    """Per-camera commissioned regions."""

    walkway_polygons: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    area_fractions: dict[str, float] = field(default_factory=dict)

    def walkway(self, camera: str | None) -> list[tuple[float, float]] | None:
        if camera is None:
            return None
        return self.walkway_polygons.get(camera)

    def walkway_is_usable(self, camera: str | None, max_area: float = 0.45) -> bool:
        """Reject a polygon so large it would exclude nobody.

        A hull covering most of the frame is not a boundary; treating it as one produces
        confident nonsense. Better to abstain and say the camera needs more commissioning
        data.
        """
        if camera is None or camera not in self.walkway_polygons:
            return False
        return self.area_fractions.get(camera, 1.0) <= max_area


@lru_cache(maxsize=1)
def load_camera_registry(root: str | None = None) -> CameraRegistry:
    settings = get_settings()
    path = Path(root) / CAMERA_REGISTRY if root else settings.path(CAMERA_REGISTRY)
    if not path.exists():
        log.debug("no camera registry at %s", path)
        return CameraRegistry()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("camera registry unreadable (%s)", exc)
        return CameraRegistry()

    fingerprints = {
        entry["camera"]: np.asarray(entry["fingerprint"], dtype=np.float32)
        for entry in data.get("cameras", [])
        if entry.get("fingerprint")
    }
    registry = CameraRegistry(
        fingerprints=fingerprints,
        tolerance=float(data.get("match_tolerance", 12.0)),
    )
    log.debug("camera registry: %s (tolerance %.1f)", list(fingerprints), registry.tolerance)
    return registry


@lru_cache(maxsize=1)
def load_region_baseline(root: str | None = None) -> RegionBaseline:
    settings = get_settings()
    path = Path(root) / REGION_BASELINE if root else settings.path(REGION_BASELINE)
    if not path.exists():
        log.debug("no region baseline at %s", path)
        return RegionBaseline()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("region baseline unreadable (%s)", exc)
        return RegionBaseline()

    polygons: dict[str, list[tuple[float, float]]] = {}
    areas: dict[str, float] = {}
    for camera, entry in data.get("cameras", {}).items():
        polygon = entry.get("walkway_polygon") or []
        if len(polygon) >= 3:
            polygons[camera] = [(float(x), float(y)) for x, y in polygon]
            areas[camera] = float(entry.get("area_fraction", 1.0))
    return RegionBaseline(walkway_polygons=polygons, area_fractions=areas)


def reset_caches() -> None:
    """Drop cached artefacts. For tests and after re-commissioning."""
    load_camera_registry.cache_clear()
    load_region_baseline.cache_clear()


def zone_for_camera(camera: str | None, settings: Settings | None = None) -> str | None:
    """Map a camera onto the zone that declares it in ``config/zones.yaml``."""
    if camera is None:
        return None
    settings = settings or get_settings()
    for zone, definition in settings.zones.zones.items():
        if definition.camera == camera:
            return zone
    return None
