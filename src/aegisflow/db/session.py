"""Async engine and session management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from aegisflow.core.logging import get_logger
from aegisflow.core.settings import Settings, get_settings
from aegisflow.db.models import Base

log = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _resolve_url(settings: Settings) -> str:
    """Make a relative SQLite path absolute against the repo root.

    Without this, ``sqlite+aiosqlite:///./aegisflow.db`` resolves against the process CWD,
    so the CLI and the API would happily read two different database files depending on
    where they were launched from.
    """
    url = settings.db_url
    prefix = "sqlite+aiosqlite:///"
    if not url.startswith(prefix):
        return url
    raw = url[len(prefix) :]
    if raw.startswith(":memory:") or raw == "":
        return url
    path = Path(raw)
    if not path.is_absolute():
        path = (settings.root / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return prefix + path.as_posix()


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    """Process-wide engine."""
    global _engine, _session_factory
    if _engine is None:
        settings = settings or get_settings()
        url = _resolve_url(settings)
        _engine = create_async_engine(url, echo=False, future=True)
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)
        log.debug("database engine created: %s", url)
    return _engine


def get_session_factory(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    get_engine(settings)
    assert _session_factory is not None
    return _session_factory


async def init_db(settings: Settings | None = None) -> None:
    """Create tables if they do not exist.

    No migration framework: the schema is append-only and small, and the product story is
    a single-file SQLite database on a factory PC. If this grows, add Alembic.
    """
    engine = get_engine(settings)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    log.debug("database schema ensured")


@asynccontextmanager
async def session_scope(settings: Settings | None = None) -> AsyncIterator[AsyncSession]:
    """Transactional session: commits on success, rolls back on error."""
    factory = get_session_factory(settings)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Close the engine. Called on API shutdown and between tests."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
