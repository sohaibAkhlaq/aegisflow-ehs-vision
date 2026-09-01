"""Clip -> facility zone resolution.

``zone`` is a mandatory field on every compliance record (assignment Module 4), and the
policy describes two cameras covering the production floor (Section 7.1). The dataset clips
carry no zone metadata, so we resolve one deterministically.

Resolution order:
    1. an explicit ``clip_overrides`` entry in ``config/zones.yaml``
    2. the ``class_default`` for the clip's behaviour class
    3. ``Zone-Unassigned``
"""

from __future__ import annotations

from pathlib import Path

from aegisflow.core.enums import BehaviorClass
from aegisflow.core.settings import Settings, get_settings

UNASSIGNED = "Zone-Unassigned"


def clip_id_for(path: str | Path) -> str:
    """Stable clip identifier: the filename, as the assignment's field table allows."""
    return Path(path).name


def behavior_from_clip_path(path: str | Path) -> BehaviorClass | None:
    """Infer the ground-truth class from the dataset folder name.

    Used *only* for evaluation and zone defaults - never by the detectors, which would be
    circular. The dataset lays clips out as ``data/raw/<split>/<class_folder>/clip.mp4``.
    """
    parent = Path(path).parent.name
    try:
        return BehaviorClass.from_policy_name(parent)
    except KeyError:
        return None


def resolve_zone(
    clip_path: str | Path,
    behavior: BehaviorClass | None = None,
    settings: Settings | None = None,
) -> str:
    """Resolve the zone label for a clip."""
    settings = settings or get_settings()
    zones = settings.zones
    clip_id = clip_id_for(clip_path)

    override = zones.clip_overrides.get(clip_id)
    if override:
        return override

    behavior = behavior or behavior_from_clip_path(clip_path)
    if behavior is not None:
        default = zones.class_default.get(behavior.value)
        if default:
            return default

    return UNASSIGNED


def zone_label(zone: str, settings: Settings | None = None) -> str:
    """Human-readable name for a zone, for the dashboard and PDF reports."""
    settings = settings or get_settings()
    definition = settings.zones.zones.get(zone)
    return definition.label if definition else zone


def camera_for(zone: str, settings: Settings | None = None) -> str | None:
    settings = settings or get_settings()
    definition = settings.zones.zones.get(zone)
    return definition.camera if definition else None


def walkway_polygon(
    zone: str, settings: Settings | None = None
) -> list[tuple[float, float]] | None:
    """Static normalised walkway polygon for a zone, if one has been annotated.

    The walkway detector prefers live green-line segmentation and falls back to this when
    the painted lines are occluded (policy Section 3.2).
    """
    settings = settings or get_settings()
    camera = camera_for(zone, settings) or zone
    polygon = settings.zones.walkway_polygons.get(camera) or settings.zones.walkway_polygons.get(
        zone
    )
    if not polygon:
        return None
    return [(float(x), float(y)) for x, y in polygon]
