"""The copilot's tool-calling agent loop.

Two implementations, chosen automatically at call time:

* **LangChain tool-calling agent** (preferred) - ``ChatGroq`` bound to the tool
  registry via ``create_tool_calling_agent`` + ``AgentExecutor``. The model itself
  decides which tool(s) to call, in the classic agent-loop pattern (see the ReAct
  paper notes this demonstrates).
* **Keyword-routed fallback** (zero extra dependencies) - a small heuristic router
  that picks one tool based on simple keyword matching, calls it directly, and
  formats the result as plain text. No model call is needed for this path at all,
  which means the copilot's data-query features keep working even fully offline.

Either path is wrapped by :func:`handle_message`, which also manages session memory
(:mod:`aegisflow.copilot.memory`) so a follow-up question has context.
"""

from __future__ import annotations

import json
import re

from aegisflow.copilot import CopilotDependencyError
from aegisflow.copilot.memory import ConversationMemory
from aegisflow.copilot.schemas import ChatResponse, ToolCallRecord
from aegisflow.copilot.tools import (
    TOOL_REGISTRY,
    get_policy_rule,
    get_stats,
    list_policy_rules,
    query_events,
    retrieve_policy_context,
)

_memory = ConversationMemory()


async def _run_langchain_agent(question: str, history: list[dict[str, str]]) -> ChatResponse:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_groq import ChatGroq

    from aegisflow.copilot.tools import as_langchain_tools

    tools = as_langchain_tools()
    model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the AegisFlow EHS copilot. Use the available tools to answer "
                "questions about logged compliance events and the compliance policy. "
                "Always call retrieve_policy_context before answering a question about "
                "what the policy says. Cite section references you relied on.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(model, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True)

    chat_history = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in history
    ]
    result = await executor.ainvoke({"input": question, "chat_history": chat_history})

    tool_calls = [
        ToolCallRecord(
            tool_name=step[0].tool,
            arguments=step[0].tool_input,
            result_summary=str(step[1])[:200],
        )
        for step in result.get("intermediate_steps", [])
    ]
    citations = sorted(
        {
            call.arguments.get("section_ref", "")
            for call in tool_calls
            if call.tool_name == "get_policy_rule"
        }
        | _extract_citations_from_tool_results(tool_calls)
    )
    return ChatResponse(
        session_id="",  # filled in by handle_message
        answer=result["output"],
        tool_calls=tool_calls,
        citations=[c for c in citations if c],
    )


def _extract_citations_from_tool_results(tool_calls: list[ToolCallRecord]) -> set[str]:
    citations: set[str] = set()
    for call in tool_calls:
        if call.tool_name == "retrieve_policy_context":
            for match in re.findall(r"Section \d+(?:\.\d+)*", call.result_summary):
                citations.add(match)
    return citations


# --------------------------------------------------------------------------- #
# Fallback: keyword-routed, no LangChain required
# --------------------------------------------------------------------------- #

_STATS_KEYWORDS = ("how many", "count", "stats", "statistics", "how much")
_EVENTS_KEYWORDS = ("show me", "list", "events", "violations", "recent")
_SEVERITY_WORDS = ("low", "medium", "high", "critical")


async def _run_fallback_router(question: str) -> ChatResponse:
    lowered = question.lower()
    tool_calls: list[ToolCallRecord] = []

    severity = next((w.upper() for w in _SEVERITY_WORDS if w in lowered), None)
    section_match = re.search(r"section\s+(\d+(?:\.\d+)*)", lowered)

    if section_match:
        section_ref = f"Section {section_match.group(1)}"
        rule = await get_policy_rule(section_ref)
        tool_calls.append(
            ToolCallRecord(
                tool_name="get_policy_rule",
                arguments={"section_ref": section_ref},
                result_summary=json.dumps(rule)[:200] if rule else "not found",
            )
        )
        if rule:
            answer = (
                f"{section_ref} ({rule['callout']}) covers {rule['behavior_class']}: "
                f'"{rule["source_quote"]}"'
            )
            return ChatResponse(session_id="", answer=answer, tool_calls=tool_calls, citations=[section_ref])

    if any(kw in lowered for kw in _STATS_KEYWORDS):
        stats = await get_stats()
        tool_calls.append(
            ToolCallRecord(tool_name="get_stats", arguments={}, result_summary=json.dumps(stats)[:200])
        )
        answer = (
            f"There are {stats['total_events']} logged events in total. "
            f"By severity: {stats['by_severity']}. "
            f"By behaviour: {stats['by_behavior_class']}."
        )
        return ChatResponse(session_id="", answer=answer, tool_calls=tool_calls, citations=[])

    if any(kw in lowered for kw in _EVENTS_KEYWORDS) or severity:
        events = await query_events(severity=severity, limit=5)
        tool_calls.append(
            ToolCallRecord(
                tool_name="query_events",
                arguments={"severity": severity, "limit": 5},
                result_summary=json.dumps(events)[:200],
            )
        )
        if not events:
            answer = "No matching events found."
        else:
            lines = [
                f"- [{e['severity']}] {e['behavior_class']} in {e['zone']} "
                f"({e['policy_rule_ref']}) at {e['timestamp']}"
                for e in events
            ]
            answer = "Most recent matching events:\n" + "\n".join(lines)
        return ChatResponse(session_id="", answer=answer, tool_calls=tool_calls, citations=[])

    # Default: treat as a policy question, use RAG.
    chunks = await retrieve_policy_context(question, k=3)
    tool_calls.append(
        ToolCallRecord(
            tool_name="retrieve_policy_context",
            arguments={"question": question},
            result_summary=json.dumps(chunks)[:200],
        )
    )
    from aegisflow.copilot.rag import RetrievedChunk
    from aegisflow.copilot.chains import answer_with_citations

    retrieved = [RetrievedChunk(text=c["text"], section_ref=c["section_ref"], score=c["score"], source="tool") for c in chunks]
    answer, citations = await answer_with_citations(question, retrieved)
    return ChatResponse(session_id="", answer=answer, tool_calls=tool_calls, citations=citations)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


async def handle_message(session_id: str, message: str) -> ChatResponse:
    """Answer one chat message, using session memory for context.

    Tries the LangChain tool-calling agent first; falls back to the keyword router
    on any missing-dependency error, so this function always returns an answer.
    """
    await _memory.append(session_id, "user", message)
    history = [m.model_dump() for m in await _memory.get_history(session_id, limit=10)]

    try:
        response = await _run_langchain_agent(message, history[:-1])  # exclude message just added
    except (CopilotDependencyError, ImportError):
        response = await _run_fallback_router(message)

    response.session_id = session_id
    await _memory.append(session_id, "assistant", response.answer)
    return response
