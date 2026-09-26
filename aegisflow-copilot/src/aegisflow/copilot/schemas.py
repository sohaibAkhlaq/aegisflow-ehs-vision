"""Typed contracts for the copilot subsystem.

Kept separate from ``aegisflow.core.schemas`` (the graded pipeline's contracts) on
purpose: the copilot is an additive feature, and its data shapes should never be
confused with - or accidentally imported into - the audit-trail models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """One turn in a copilot conversation, as stored in session memory."""

    role: Literal["user", "assistant", "tool"]
    content: str
    created_at: datetime


class ToolCallRecord(BaseModel):
    """A single tool invocation the agent made while answering a question."""

    tool_name: str
    arguments: dict
    result_summary: str


class ChatRequest(BaseModel):
    """Incoming request to ``POST /api/copilot/chat``."""

    session_id: str = Field(..., description="Client-generated id, groups a conversation")
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    """Response returned to the dashboard's chat panel."""

    session_id: str
    answer: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    citations: list[str] = Field(
        default_factory=list, description="Policy section refs backing the answer, if any"
    )


class IncidentDraft(BaseModel):
    """Output of the drafting agent in the LangGraph incident workflow."""

    event_id: str
    summary: str
    cited_section: str


class IncidentReviewVerdict(BaseModel):
    """Output of the reviewing agent: approve the draft, or send it back with feedback."""

    approved: bool
    feedback: str = ""


class IncidentReport(BaseModel):
    """Final artifact of the incident workflow, attached to the triggering event."""

    event_id: str
    summary: str
    cited_section: str
    review_rounds: int
    graph_trace: list[str] = Field(
        default_factory=list, description="Node names visited, in order, for traceability"
    )
