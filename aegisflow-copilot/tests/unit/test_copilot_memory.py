"""Unit tests for ConversationMemory - the copilot's episodic session store."""

from __future__ import annotations

import contextlib

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aegisflow.copilot import memory as memory_module
from aegisflow.copilot.memory import ConversationMemory
from aegisflow.db.models import Base


@pytest_asyncio.fixture
async def memory(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    @contextlib.asynccontextmanager
    async def _fake_session_scope():
        async with factory() as session:
            yield session
            await session.commit()

    monkeypatch.setattr(memory_module, "session_scope", _fake_session_scope)
    yield ConversationMemory()
    await engine.dispose()


@pytest.mark.asyncio
async def test_append_and_get_history_preserves_order(memory):
    await memory.append("session-1", "user", "first message")
    await memory.append("session-1", "assistant", "first reply")
    await memory.append("session-1", "user", "second message")

    history = await memory.get_history("session-1")
    assert [m.content for m in history] == ["first message", "first reply", "second message"]


@pytest.mark.asyncio
async def test_history_is_scoped_per_session(memory):
    await memory.append("session-a", "user", "a message")
    await memory.append("session-b", "user", "b message")

    history_a = await memory.get_history("session-a")
    assert len(history_a) == 1
    assert history_a[0].content == "a message"


@pytest.mark.asyncio
async def test_clear_session_removes_only_that_session(memory):
    await memory.append("session-1", "user", "keep me? no")
    await memory.append("session-2", "user", "keep me")

    deleted = await memory.clear_session("session-1")
    assert deleted == 1

    assert await memory.get_history("session-1") == []
    assert len(await memory.get_history("session-2")) == 1
