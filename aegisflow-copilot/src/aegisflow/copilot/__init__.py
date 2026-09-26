"""AegisFlow Copilot - an additive, optional chat assistant layered on top of the
core compliance pipeline (Modules 1-5).

Nothing in this package is required for the graded pipeline to run. It exists to
demonstrate a second, independent skill set on the same real data: retrieval-augmented
generation over the policy PDF, a tool-calling agent, a LangGraph incident-drafting
workflow with a second reviewing agent, session memory, and an MCP server exposing the
same tools to external clients (e.g. Claude Desktop).

If none of the optional dependencies (langchain, langgraph, chromadb,
sentence-transformers, mcp) are installed, importing this package still succeeds -
individual modules raise a clear ``CopilotDependencyError`` only when a function that
needs the missing library is actually called. This mirrors the project's own
"offline must always work" rule: the base system must never break because an optional
extra isn't installed.
"""

from __future__ import annotations


class CopilotDependencyError(RuntimeError):
    """Raised when a copilot feature is used without its optional dependency installed."""

    def __init__(self, feature: str, package: str) -> None:
        super().__init__(
            f"{feature} requires the '{package}' package. "
            f"Install it with: pip install -r requirements-copilot.txt"
        )


__all__ = ["CopilotDependencyError"]
