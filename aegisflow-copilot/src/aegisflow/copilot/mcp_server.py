"""MCP server exposing the copilot's tools to external MCP clients (Claude Desktop,
the MCP Inspector, or any other MCP-speaking application).

This deliberately reuses :mod:`aegisflow.copilot.tools`'s registry as the single
source of truth - the same functions the in-dashboard chat agent calls are exposed
here, so "ask the dashboard" and "ask from Claude Desktop" are two clients of one
tool implementation, not two separate copies of the query logic.

Run standalone for local testing with the MCP Inspector:

    python -m aegisflow.copilot.mcp_server

Or configure it as a Claude Desktop MCP server by adding to
``claude_desktop_config.json``::

    {
      "mcpServers": {
        "aegisflow-ehs": {
          "command": "python",
          "args": ["-m", "aegisflow.copilot.mcp_server"],
          "cwd": "/absolute/path/to/aegisflow-ehs-vision"
        }
      }
    }

Requires the ``mcp`` package (``pip install mcp``, included in
``requirements-copilot.txt``).
"""

from __future__ import annotations

from aegisflow.copilot import CopilotDependencyError
from aegisflow.copilot.tools import (
    get_policy_rule,
    get_stats,
    list_policy_rules,
    query_events,
    retrieve_policy_context,
)


def build_server() -> "object":
    """Build the FastMCP server instance. Raises :class:`CopilotDependencyError` if
    the ``mcp`` package is not installed."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise CopilotDependencyError("The MCP server", "mcp") from exc

    server = FastMCP("aegisflow-ehs")

    # Each tool is registered directly from aegisflow.copilot.tools - no re-implementation.
    server.add_tool(query_events)
    server.add_tool(get_stats)
    server.add_tool(get_policy_rule)
    server.add_tool(list_policy_rules)
    server.add_tool(retrieve_policy_context)

    return server


def main() -> None:
    """Entry point for ``python -m aegisflow.copilot.mcp_server``."""
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
