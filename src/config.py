"""
config.py — Central configuration for the AegisFlow EHS Vision Platform.

All paths, thresholds, and environment-driven settings are defined here.
Import this module everywhere instead of repeating os.getenv calls.
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# ─────────────────────────────────────────────────────────────────────────────
# Project root (one level above src/)
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Reads from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── API Keys ───────────────────────────────────────────────────────────────
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # ── Server ─────────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    debug: bool = Field(default=True, alias="DEBUG")

    # ── Paths ──────────────────────────────────────────────────────────────────
    data_dir: Path = Field(default=ROOT_DIR / "data", alias="DATA_DIR")
    outputs_dir: Path = Field(default=ROOT_DIR / "outputs", alias="OUTPUTS_DIR")
    db_path: Path = Field(default=ROOT_DIR / "outputs" / "compliance.db", alias="DB_PATH")
    rules_config_path: Path = Field(default=ROOT_DIR / "outputs" / "rules_config.json", alias="RULES_CONFIG_PATH")
    policy_pdf: Path = Field(default=ROOT_DIR / "Compliance_Policy_Manual.pdf")

    # ── YOLO / Detection ───────────────────────────────────────────────────────
    yolo_model: str = Field(default="yolov8n.pt")
    yolo_conf: float = Field(default=0.45, alias="YOLO_CONFIDENCE_THRESHOLD")
    vlm_fallback_conf: float = Field(default=0.50, alias="VLM_FALLBACK_CONFIDENCE")

    # ── Escalation ─────────────────────────────────────────────────────────────
    high_crit_alert: bool = Field(default=True, alias="HIGH_CRIT_WEBSOCKET_ALERT")

    def ensure_dirs(self) -> None:
        """Create all required output directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Compliance class definitions — derived from KMP-OHS-POL-001 policy document
# These are the ground-truth labels used throughout the pipeline.
# ─────────────────────────────────────────────────────────────────────────────

BEHAVIOR_CLASSES: dict[int, dict] = {
    0: {
        "id": 0,
        "name": "Safe Walkway Violation",
        "domain": "Pedestrian Movement",
        "policy_ref": "Section 3.3.2",
        "indicator": "Person outside green floor markings",
        "safe_pair": "Safe Walkway",
        "severity": "CRITICAL",
    },
    1: {
        "id": 1,
        "name": "Unauthorized Intervention",
        "domain": "Equipment Interaction",
        "policy_ref": "Section 4.3.2",
        "indicator": "Person interacting with equipment without green authorization vest",
        "safe_pair": "Authorized Intervention",
        "severity": "HIGH",
    },
    2: {
        "id": 2,
        "name": "Opened Panel Cover",
        "domain": "Electrical Safety",
        "policy_ref": "Section 5.2.2",
        "indicator": "Electrical panel cover in open position during production operations",
        "safe_pair": "Closed Panel Cover",
        "severity": "LOW",
    },
    3: {
        "id": 3,
        "name": "Carrying Overload with Forklift",
        "domain": "Forklift Load Management",
        "policy_ref": "Section 6.3.2",
        "indicator": "Forklift carrying 3 or more standardized blocks in a single load",
        "safe_pair": "Safe Carrying",
        "severity": "HIGH",
    },
}

# Severity color palette (used by dashboard + reports)
SEVERITY_COLORS: dict[str, str] = {
    "LOW": "#22c55e",       # green
    "MEDIUM": "#f59e0b",    # amber
    "HIGH": "#f97316",      # orange
    "CRITICAL": "#ef4444",  # red
}

# Singleton settings instance
settings = Settings()
settings.ensure_dirs()
