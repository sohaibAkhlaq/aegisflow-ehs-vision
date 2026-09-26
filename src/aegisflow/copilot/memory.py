"""Episodic session memory for the copilot chat.

A minimal analogue of the ``episodic_store.py`` pattern: one SQLite table, keyed by
``session_id``, storing each turn so a follow-up question ("and how does that compare
to last week?") has the prior turns to refer back to.

Deliberately its own table (``copilot_messages``), not touching ``violation_events`` -
this is conversational scratch space, not part of the compliance audit trail, and must
never be confused with it. Unlike the audit trail, this table is NOT append-only-forever:
:func:`ConversationMemory.clear_session` exists because chat history has no compliance
value once a session ends, whereas violation events must never be deletable.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Index, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from aegisflow.copilot.schemas import ChatMessage
from aegisflow.db.models import Base
from aegisflow.db.session import session_scope


class CopilotMessageRow(Base):
    """One turn of copilot conversation. Session-scoped, not part of the audit trail."""

    __tablename__ = "copilot_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    __table_args__ = (Index("ix_copilot_messages_session", "session_id"),)


class ConversationMemory:
    """Thin async wrapper around :class:`CopilotMessageRow` for session history."""

    async def append(self, session_id: str, role: str, content: str) -> None:
        async with session_scope() as session:
            session.add(
                CopilotMessageRow(session_id=session_id, role=role, content=content)
            )

    async def get_history(self, session_id: str, limit: int = 20) -> list[ChatMessage]:
        """Most recent ``limit`` turns for this session, oldest first."""
        async with session_scope() as session:
            stmt = (
                select(CopilotMessageRow)
                .where(CopilotMessageRow.session_id == session_id)
                .order_by(CopilotMessageRow.created_at.desc())
                .limit(limit)
            )
            rows = (await session.execute(stmt)).scalars().all()

        rows = list(reversed(rows))  # chronological order for the caller
        return [
            ChatMessage(role=row.role, content=row.content, created_at=row.created_at)
            for row in rows
        ]

    async def clear_session(self, session_id: str) -> int:
        """Delete a session's history. Returns the number of rows removed.

        Only touches ``copilot_messages`` - the compliance audit trail
        (``violation_events``) has no delete path anywhere in this codebase, and this
        function does not add one.
        """
        async with session_scope() as session:
            stmt = select(CopilotMessageRow).where(CopilotMessageRow.session_id == session_id)
            rows = (await session.execute(stmt)).scalars().all()
            for row in rows:
                await session.delete(row)
            return len(rows)
