"""Function-calling tools for the copilot agent.

Each tool is a plain async function over the *existing* database and policy artifacts -
no new storage, no new schema. Kept separate from ``aegisflow.db.crud`` (the graded,
append-only CRUD layer) because these are read-only query helpers for the copilot, not
part of the audit-trail write path; nothing here ever inserts, updates or deletes a row.

Exposed three ways from this single source of truth:

1. Directly, as plain async functions (used by :mod:`aegisflow.copilot.agent`'s
   fallback keyword router).
2. Wrapped as LangChain ``StructuredTool`` objects (used by the LangChain tool-calling
   agent, when installed) - see :func:`as_langchain_tools`.
3. Wrapped as MCP tools (used by :mod:`aegisflow.copilot.mcp_server`) - see that module.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from aegisflow.copilot import CopilotDependencyError
from aegisflow.db.models import PolicyRuleRow, ViolationEventRow
from aegisflow.db.session import session_scope


async def query_events(
    severity: str | None = None,
    behavior_class: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Query logged violation events, optionally filtered.

    Args:
        severity: one of LOW, MEDIUM, HIGH, CRITICAL (case-insensitive), or None for all.
        behavior_class: e.g. "carrying_overload_with_forklift", or None for all.
        start_date: ISO date string (inclusive), or None.
        end_date: ISO date string (inclusive), or None.
        limit: maximum rows to return (capped at 200 to keep the copilot's answers short).
    """
    limit = min(limit, 200)
    stmt = select(ViolationEventRow).order_by(ViolationEventRow.timestamp.desc()).limit(limit)

    if severity:
        stmt = stmt.where(ViolationEventRow.severity == severity.upper())
    if behavior_class:
        stmt = stmt.where(ViolationEventRow.behavior_class == behavior_class)
    if start_date:
        stmt = stmt.where(ViolationEventRow.timestamp >= datetime.fromisoformat(start_date))
    if end_date:
        stmt = stmt.where(ViolationEventRow.timestamp <= datetime.fromisoformat(end_date))

    async with session_scope() as session:
        result = await session.execute(stmt)
        rows = result.scalars().all()

    return [
        {
            "event_id": row.event_id,
            "timestamp": row.timestamp.isoformat(),
            "clip_id": row.clip_id,
            "zone": row.zone,
            "behavior_class": row.behavior_class,
            "severity": row.severity,
            "policy_rule_ref": row.policy_rule_ref,
            "event_description": row.event_description,
            "escalation_action": row.escalation_action,
        }
        for row in rows
    ]


async def get_stats(behavior_class: str | None = None) -> dict[str, Any]:
    """Aggregate event counts, by severity and by behaviour class.

    Args:
        behavior_class: restrict the count to one behaviour class, or None for all.
    """
    stmt = select(ViolationEventRow.severity, func.count()).group_by(ViolationEventRow.severity)
    if behavior_class:
        stmt = stmt.where(ViolationEventRow.behavior_class == behavior_class)

    async with session_scope() as session:
        by_severity_rows = (await session.execute(stmt)).all()

        total_stmt = select(func.count()).select_from(ViolationEventRow)
        if behavior_class:
            total_stmt = total_stmt.where(ViolationEventRow.behavior_class == behavior_class)
        total = (await session.execute(total_stmt)).scalar_one()

        by_behavior_stmt = select(ViolationEventRow.behavior_class, func.count()).group_by(
            ViolationEventRow.behavior_class
        )
        by_behavior_rows = (await session.execute(by_behavior_stmt)).all()

    return {
        "total_events": total,
        "by_severity": {sev: count for sev, count in by_severity_rows},
        "by_behavior_class": {cls: count for cls, count in by_behavior_rows},
        "filtered_to_behavior_class": behavior_class,
    }


async def get_policy_rule(section_ref: str) -> dict[str, Any] | None:
    """Look up the parsed policy rule for a given section reference.

    Args:
        section_ref: e.g. "Section 6.3.2". Matching is exact against what the policy
            parser stored; use :func:`list_policy_rules` first if unsure of the exact
            string.
    """
    async with session_scope() as session:
        stmt = (
            select(PolicyRuleRow)
            .where(PolicyRuleRow.section_ref == section_ref)
            .order_by(PolicyRuleRow.parsed_at.desc())
            .limit(1)
        )
        row = (await session.execute(stmt)).scalar_one_or_none()

    if row is None:
        return None
    return {
        "behavior_class": row.behavior_class,
        "domain": row.domain,
        "section_ref": row.section_ref,
        "observable_indicator": row.observable_indicator,
        "callout": row.callout,
        "source_quote": row.source_quote,
        "numeric_threshold": row.numeric_threshold,
    }


_rag_singleton: Any = None


def _get_rag(settings: Any = None) -> Any:
    """Lazily build (once per process) and cache the :class:`PolicyRAG` index."""
    global _rag_singleton
    if _rag_singleton is None:
        from aegisflow.copilot.rag import PolicyRAG
        from aegisflow.core.settings import get_settings

        settings = settings or get_settings()
        root = settings.root
        rag = PolicyRAG(
            pdf_path=root / "compliance_policy.pdf",
            rules_path=root / "artifacts" / "policy" / "rules.json",
            persist_dir=root / "artifacts" / "chroma_db",
        )
        rag.build_index()
        _rag_singleton = rag
    return _rag_singleton


async def retrieve_policy_context(question: str, k: int = 3) -> list[dict[str, Any]]:
    """Retrieve the policy excerpts most relevant to a natural-language question.

    Use this before answering any question about WHAT a rule says, what counts as a
    violation, or what severity language the policy uses - as opposed to questions
    about logged events, which should use :func:`query_events` or :func:`get_stats`
    instead.

    Args:
        question: the natural-language question to retrieve context for.
        k: how many excerpts to return (default 3).
    """
    rag = _get_rag()
    chunks = rag.query(question, k=k)
    return [
        {"text": c.text, "section_ref": c.section_ref, "score": round(c.score, 3)}
        for c in chunks
    ]


async def list_policy_rules() -> list[dict[str, Any]]:
    """List every currently-parsed policy rule (all four behaviour classes)."""
    async with session_scope() as session:
        stmt = select(PolicyRuleRow).order_by(PolicyRuleRow.parsed_at.desc())
        rows = (await session.execute(stmt)).scalars().all()

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.behavior_class in seen:
            continue  # keep only the most recent parse per behaviour class
        seen.add(row.behavior_class)
        out.append(
            {
                "behavior_class": row.behavior_class,
                "section_ref": row.section_ref,
                "callout": row.callout,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Tool registry - the single source of truth for name/description/schema,
# consumed by the LangChain agent, the MCP server, and the fallback router.
# --------------------------------------------------------------------------- #

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "query_events": {
        "fn": query_events,
        "description": "Query logged compliance violation events, optionally filtered "
        "by severity, behaviour class, or date range.",
    },
    "get_stats": {
        "fn": get_stats,
        "description": "Get aggregate counts of violation events, broken down by "
        "severity and behaviour class.",
    },
    "get_policy_rule": {
        "fn": get_policy_rule,
        "description": "Look up the parsed policy rule for a specific section "
        "reference (e.g. 'Section 6.3.2').",
    },
    "list_policy_rules": {
        "fn": list_policy_rules,
        "description": "List all four currently-parsed policy rules with their "
        "section references.",
    },
    "retrieve_policy_context": {
        "fn": retrieve_policy_context,
        "description": "Retrieve the policy excerpts most relevant to a question "
        "about what a rule says, what counts as a violation, or severity language. "
        "Use this for policy/definition questions, not for questions about logged "
        "events (use query_events or get_stats for those).",
    },
}


def as_langchain_tools() -> list[Any]:
    """Wrap the tool registry as LangChain ``StructuredTool`` objects.

    Raises :class:`CopilotDependencyError` if ``langchain-core`` is not installed;
    callers (the LangChain agent path in :mod:`aegisflow.copilot.agent`) should catch
    this and fall back to the plain keyword router.
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise CopilotDependencyError("The LangChain tool-calling agent", "langchain-core") from exc

    tools = []
    for name, spec in TOOL_REGISTRY.items():
        tools.append(
            StructuredTool.from_function(
                coroutine=spec["fn"],
                name=name,
                description=spec["description"],
            )
        )
    return tools
