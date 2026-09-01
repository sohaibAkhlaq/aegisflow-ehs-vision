"""Typed configuration: ``.env`` for secrets and switches, ``config/*.yaml`` for tuning.

Every module reads configuration through :func:`get_settings`. Nothing else should touch
``os.environ`` or open a YAML file - that keeps the whole system configurable from one place
and testable by overriding one object.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from aegisflow.core.enums import LLMProviderName
from aegisflow.core.errors import ConfigError


def repo_root() -> Path:
    """Repository root, resolved from this file's location.

    ``src/aegisflow/core/settings.py`` -> up four levels. Resolved rather than assumed from
    the CWD so the CLI, the API and pytest all agree on where ``config/`` lives.
    """
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# YAML-backed tuning blocks (config/settings.yaml)
# ---------------------------------------------------------------------------


class HSVBand(BaseModel):
    """An HSV range in OpenCV convention: H 0-179, S 0-255, V 0-255."""

    h: tuple[int, int]
    s: tuple[int, int]
    v: tuple[int, int]

    def lower(self) -> tuple[int, int, int]:
        return (self.h[0], self.s[0], self.v[0])

    def upper(self) -> tuple[int, int, int]:
        return (self.h[1], self.s[1], self.v[1])


class VideoConfig(BaseModel):
    sample_fps: float = 4.0
    infer_imgsz: int = 640
    max_frames_per_clip: int = 64


class DetectionConfig(BaseModel):
    weights: str = "artifacts/models/yolov8n.pt"
    device: str = "cpu"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.50
    coco_classes: dict[str, int] = Field(
        default_factory=lambda: {"person": 0, "truck": 7, "car": 2}
    )
    persistence_frames: dict[str, int] = Field(default_factory=dict)
    batch_size: int = 8
    torch_threads: int | None = Field(
        default=None, description="None = use every core. Set only to cap CPU usage."
    )

    def persistence_for(self, behavior_key: str, default: int = 3) -> int:
        return int(self.persistence_frames.get(behavior_key, default))


class ColorConfig(BaseModel):
    green_vest: HSVBand
    green_floor_line: HSVBand
    red_vest_low: HSVBand
    red_vest_high: HSVBand
    min_vest_pixel_ratio: float = 0.12


class WalkwayConfig(BaseModel):
    min_boundary_margin_frac: float = 0.055
    min_boundary_area_frac: float = 0.02


class ForkliftConfig(BaseModel):
    safe_block_count: int = 2
    block_min_area_px: int = 900
    block_aspect_ratio: tuple[float, float] = (0.6, 2.2)
    max_vehicle_area_frac: float = 0.20
    min_vehicle_area_frac: float = 0.01
    offline_detection_enabled: bool = False


class PanelConfig(BaseModel):
    baseline: str = "artifacts/models/panel_baseline.json"
    min_contour_area_px: int = 1500
    confidence_span: float = 18.0


class VLMTiebreakConfig(BaseModel):
    enabled: bool = True
    band: float = 0.15
    max_calls_per_clip: int = 2
    cache: bool = True


class LoggingConfig(BaseModel):
    # populate_by_name lets the model_dump() -> model_validate() round-trip in
    # Settings.tuning work; without it the dumped `json_output` key would not
    # match the `json` alias on re-validation.
    model_config = ConfigDict(populate_by_name=True)

    level: str = "INFO"
    # Aliased because a field literally named `json` shadows BaseModel.json.
    json_output: bool = Field(default=False, alias="json")


class TuningConfig(BaseModel):
    """The whole of ``config/settings.yaml``, typed.

    Engineering knobs only. No behaviour class, no severity tier, no compliance rule -
    those are parsed from the policy PDF at runtime. See CLAUDE.md.
    """

    video: VideoConfig = VideoConfig()
    detection: DetectionConfig = DetectionConfig()
    color: ColorConfig
    walkway: WalkwayConfig = WalkwayConfig()
    forklift: ForkliftConfig = ForkliftConfig()
    panel: PanelConfig = PanelConfig()
    vlm_tiebreak: VLMTiebreakConfig = VLMTiebreakConfig()
    logging: LoggingConfig = LoggingConfig()


# ---------------------------------------------------------------------------
# YAML-backed zone map (config/zones.yaml)
# ---------------------------------------------------------------------------


class ZoneDef(BaseModel):
    label: str
    camera: str
    domains: list[str] = Field(default_factory=list)


class ZoneConfig(BaseModel):
    zones: dict[str, ZoneDef] = Field(default_factory=dict)
    class_default: dict[str, str] = Field(default_factory=dict)
    clip_overrides: dict[str, str] = Field(default_factory=dict)
    walkway_polygons: dict[str, list[tuple[float, float]]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment-backed settings (.env)
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Environment configuration, prefixed ``AEGISFLOW_`` except for provider API keys."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AEGISFLOW_",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM provider ---
    llm_provider: LLMProviderName = LLMProviderName.OFFLINE
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_text_model: str = Field(default="openai/gpt-oss-120b", validation_alias="GROQ_TEXT_MODEL")
    groq_vision_model: str = Field(default="qwen/qwen3.8-27b", validation_alias="GROQ_VISION_MODEL")
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", validation_alias="GEMINI_MODEL")

    # --- detection overrides (YAML supplies the defaults) ---
    yolo_weights: str | None = None
    device: str | None = None
    sample_fps: float | None = None
    infer_imgsz: int | None = None
    conf_threshold: float | None = None
    vlm_tiebreak_band: float | None = None

    # --- persistence & API ---
    db_url: str = "sqlite+aiosqlite:///./aegisflow.db"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"

    # --- paths ---
    policy_pdf: str = "compliance_policy.pdf"
    rules_json: str = "artifacts/policy/rules.json"
    data_root: str = "data/raw"
    outputs_root: str = "outputs"
    cache_root: str = "data/processed"

    # ---------------- derived helpers ----------------

    @property
    def root(self) -> Path:
        return repo_root()

    def path(self, relative: str | Path) -> Path:
        """Resolve a repo-relative path. Absolute inputs pass through untouched."""
        p = Path(relative)
        return p if p.is_absolute() else (self.root / p)

    @functools.cached_property
    def tuning(self) -> TuningConfig:
        """``config/settings.yaml`` with any environment overrides applied on top."""
        raw = _load_yaml(self.path("config/settings.yaml"))
        cfg = TuningConfig.model_validate(raw)

        overrides: dict[str, dict[str, Any]] = {}
        if self.sample_fps is not None:
            overrides.setdefault("video", {})["sample_fps"] = self.sample_fps
        if self.infer_imgsz is not None:
            overrides.setdefault("video", {})["infer_imgsz"] = self.infer_imgsz
        if self.yolo_weights is not None:
            overrides.setdefault("detection", {})["weights"] = self.yolo_weights
        if self.device is not None:
            overrides.setdefault("detection", {})["device"] = self.device
        if self.conf_threshold is not None:
            overrides.setdefault("detection", {})["conf_threshold"] = self.conf_threshold
        if self.vlm_tiebreak_band is not None:
            overrides.setdefault("vlm_tiebreak", {})["band"] = self.vlm_tiebreak_band

        if not overrides:
            return cfg
        merged = cfg.model_dump()
        for block, values in overrides.items():
            merged[block].update(values)
        return TuningConfig.model_validate(merged)

    @functools.cached_property
    def zones(self) -> ZoneConfig:
        return ZoneConfig.model_validate(_load_yaml(self.path("config/zones.yaml")))

    @property
    def llm_key(self) -> str:
        """API key for the selected provider ('' when none is configured)."""
        if self.llm_provider is LLMProviderName.GROQ:
            return self.groq_api_key
        if self.llm_provider is LLMProviderName.GEMINI:
            return self.gemini_api_key
        return ""

    @property
    def llm_available(self) -> bool:
        """True only when a non-offline provider is selected *and* has a key."""
        return self.llm_provider is not LLMProviderName.OFFLINE and bool(self.llm_key)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"missing configuration file: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"expected a mapping at the top level of {path}")
    return data


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Tests that need a different configuration call ``get_settings.cache_clear()`` after
    patching the environment.
    """
    return Settings()
