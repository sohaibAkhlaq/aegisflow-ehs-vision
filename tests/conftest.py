"""Shared fixtures.

The suite runs fully offline: no network, no model weights, no dataset. Tests that need
any of those are marked ``slow`` or ``llm`` and skip themselves when the resource is absent,
so ``pytest`` is green on a clean checkout.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from aegisflow.core.enums import (
    BehaviorClass,
    DetectionMethod,
    PolicyCallout,
    Severity,
)
from aegisflow.core.schemas import (
    DetectionRecord,
    FrameContext,
    PolicyRule,
    PolicyRuleSet,
    ViolationEvent,
)
from aegisflow.core.settings import Settings, get_settings
from aegisflow.db.models import Base

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="session")
def dataset_root(settings: Settings) -> Path:
    return settings.path(settings.data_root)


@pytest.fixture
def sample_clip(dataset_root: Path) -> Path:
    """A real clip, or skip. Keeps the suite runnable without the 9.4 GB dataset."""
    clips = sorted(dataset_root.glob("test/*/*.mp4"))
    if not clips:
        pytest.skip("dataset not present under data/raw/")
    return clips[0]


@pytest.fixture(scope="session")
def policy_pdf(settings: Settings) -> Path:
    path = settings.path(settings.policy_pdf)
    if not path.exists():
        pytest.skip("compliance_policy.pdf not present")
    return path


# ---------------------------------------------------------------------------
# Policy fixtures - handmade, so unit tests never depend on PDF parsing
# ---------------------------------------------------------------------------


def _rule(
    behavior: BehaviorClass,
    section: str,
    callout: PolicyCallout,
    **kwargs: object,
) -> PolicyRule:
    return PolicyRule(
        behavior_class=behavior,
        domain=behavior.display_name,
        section_ref=section,
        observable_indicator=f"indicator for {behavior.value}",
        unsafe_description=f"{behavior.display_name} is defined as a test condition.",
        callout=callout,
        callout_text=f"{callout.value} text for {section}",
        source_quote=f"{behavior.display_name} is defined as a test condition.",
        validated=True,
        **kwargs,
    )


@pytest.fixture
def rule_set() -> PolicyRuleSet:
    """A rule set mirroring what KMP-OHS-POL-001 actually yields."""
    return PolicyRuleSet(
        source_path="compliance_policy.pdf",
        source_sha256="0" * 64,
        rules=(
            _rule(
                BehaviorClass.SAFE_WALKWAY_VIOLATION,
                "Section 3.3.2",
                PolicyCallout.WARNING,
                high_frequency=True,
            ),
            _rule(
                BehaviorClass.UNAUTHORIZED_INTERVENTION,
                "Section 4.3.2",
                PolicyCallout.CRITICAL_SAFETY_NOTICE,
                unambiguous=True,
            ),
            _rule(
                BehaviorClass.OPENED_PANEL_COVER,
                "Section 5.2.2",
                PolicyCallout.WARNING,
                standalone_condition=True,
            ),
            _rule(
                BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT,
                "Section 6.3.2",
                PolicyCallout.CRITICAL_SAFETY_NOTICE,
                unambiguous=True,
                numeric_threshold=2,
            ),
        ),
        sections={"3.3.2": "Safe Walkway Violation", "6.3.2": "Carrying Overload"},
    )


@pytest.fixture
def make_detection():
    """Factory for detection records with controllable context."""

    def _make(
        behavior: BehaviorClass,
        confidence: float = 0.9,
        clip_id: str = "test_clip.mp4",
        **context: object,
    ) -> DetectionRecord:
        return DetectionRecord(
            clip_id=clip_id,
            behavior_class=behavior,
            confidence=confidence,
            detection_method=DetectionMethod.YOLO,
            first_frame_index=0,
            first_timestamp_s=0.0,
            frame_count=4,
            description=f"{behavior.display_name} observed in a test.",
            zone="Zone-1",
            context=FrameContext(**context),  # type: ignore[arg-type]
        )

    return _make


@pytest.fixture
def make_event():
    """Factory for violation events."""

    def _make(
        behavior: BehaviorClass = BehaviorClass.SAFE_WALKWAY_VIOLATION,
        severity: Severity = Severity.MEDIUM,
        clip_id: str = "test_clip.mp4",
        **kwargs: object,
    ) -> ViolationEvent:
        fields = {
            "clip_id": clip_id,
            "zone": "Zone-1",
            "behavior_class": behavior,
            "policy_rule_ref": "Section 3.3.2",
            "event_description": f"{behavior.display_name} observed.",
            "severity": severity,
            "confidence": 0.8,
            "detection_method": DetectionMethod.HSV,
            "severity_rationale": "test rationale",
        }
        fields.update(kwargs)
        return ViolationEvent(**fields)  # type: ignore[arg-type]

    return _make


# ---------------------------------------------------------------------------
# Database - in-memory, per test
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Isolated in-memory SQLite session."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session
        await session.rollback()
    await engine.dispose()


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((360, 640, 3), dtype=np.uint8)


@pytest.fixture
def green_patch() -> np.ndarray:
    """A solid green BGR image - a stand-in for a green safety vest."""
    image = np.zeros((80, 60, 3), dtype=np.uint8)
    image[:, :] = (40, 180, 40)
    return image


@pytest.fixture
def red_patch() -> np.ndarray:
    image = np.zeros((80, 60, 3), dtype=np.uint8)
    image[:, :] = (40, 40, 190)
    return image
