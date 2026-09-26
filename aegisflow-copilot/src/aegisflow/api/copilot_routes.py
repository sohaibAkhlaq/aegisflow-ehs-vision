"""REST routes for the copilot chat panel.

Kept in its own router (``copilot_router``), separate from ``aegisflow.api.routes``
(the graded Module 5 routes) - included from ``app.py`` as one extra line, so the
core API surface described in ``docs/api-contract.md`` is never touched by this
additive feature.

=========================================  ======================================
Route                                       Purpose
=========================================  ======================================
``POST /api/copilot/chat``                  Ask the copilot a question
``DELETE /api/copilot/sessions/{id}``       Clear a chat session's memory
=========================================  ======================================
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from aegisflow.copilot.agent import handle_message
from aegisflow.copilot.memory import ConversationMemory
from aegisflow.copilot.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/copilot", tags=["copilot"])
_memory = ConversationMemory()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Ask the copilot a question. Session memory is keyed by ``session_id``, which
    the frontend generates once per browser tab and reuses for the conversation."""
    try:
        return await handle_message(request.session_id, request.message)
    except Exception as exc:  # noqa: BLE001 - copilot failures must not crash the API
        raise HTTPException(
            status_code=502, detail=f"copilot could not answer: {exc}"
        ) from exc


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str) -> dict[str, int]:
    """Clear a chat session's memory. Only affects ``copilot_messages`` - never the
    compliance audit trail, which has no delete path anywhere in this codebase."""
    deleted = await _memory.clear_session(session_id)
    return {"deleted": deleted}
