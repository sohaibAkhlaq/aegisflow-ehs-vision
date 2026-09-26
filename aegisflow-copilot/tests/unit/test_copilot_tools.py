"""Unit tests for the copilot's DB-reading tools.

Uses an in-memory SQLite database (same pattern as the core project's own test
fixtures) so these run fast and never touch a real aegisflow.db file.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from aegisflow.copilot import tools
from aegisflow.db.models import Base, ViolationEventRow


@pytest_asyncio.fixture
async def seeded_session(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    from sqlalchemy.ext.asyncio import async_sessionmaker

    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        session.add(
            ViolationEventRow(
                event_id="evt-1",
                timestamp=datetime.now(UTC),
                clip_id="clip-1.mp4",
                zone="Zone-1",
                behavior_class="carrying_overload_with_forklift",
                policy_rule_ref="Section 6.3.2",
                event_description="3 blocks detected on forks",
                severity="CRITICAL",
                escalation_action="alerted",
                detection_method="vlm",
            )
        )
        await session.commit()

    import contextlib

    @contextlib.asynccontextmanager
    async def _fake_session_scope():
        async with factory() as session:
            yield session
            await session.commit()

    monkeypatch.setattr(tools, "session_scope", _fake_session_scope)
    yield
    await engine.dispose()


@pytest.mark.asyncio
async def test_query_events_returns_seeded_row(seeded_session):
    results = await tools.query_events(limit=10)
    assert len(results) == 1
    assert results[0]["behavior_class"] == "carrying_overload_with_forklift"


@pytest.mark.asyncio
async def test_query_events_filters_by_severity(seeded_session):
    results = await tools.query_events(severity="low", limit=10)
    assert results == []


@pytest.mark.asyncio
async def test_get_stats_counts_seeded_row(seeded_session):
    stats = await tools.get_stats()
    assert stats["total_events"] == 1
    assert stats["by_severity"]["CRITICAL"] == 1
