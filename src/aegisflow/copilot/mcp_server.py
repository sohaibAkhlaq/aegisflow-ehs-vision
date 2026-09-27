"""MCP server exposing AegisFlow Copilot tools to external MCP clients."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aegisflow.copilot import CopilotDependencyError
from aegisflow.copilot.tools import (
    get_policy_rule,
    get_stats,
    list_policy_rules,
    query_events,
    retrieve_policy_context,
)


def build_server() -> object:
    """Build the MCP server using the MCP 1.1.2 Server API."""
    try:
        from mcp.server import Server
        from mcp.server.models import InitializationOptions
        from mcp.server.stdio import stdio_server
        from mcp.types import CallToolResult, TextContent, Tool
    except ImportError as exc:
        raise CopilotDependencyError("The MCP server", "mcp") from exc

    server = Server("aegisflow-ehs")

    @server.list_tools()
    async def handle_list_tools() -> list[Tool]:
        return [
            Tool(
                name="query_events",
                description="Query EHS violation events using optional filters.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string"},
                        "behavior_class": {"type": "string"},
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                        "limit": {"type": "integer", "default": 20},
                    },
                },
            ),
            Tool(
                name="get_stats",
                description="Get statistics about EHS events.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "behavior_class": {"type": "string"},
                    },
                },
            ),
            Tool(
                name="get_policy_rule",
                description="Retrieve a policy rule using its section reference.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "section_ref": {"type": "string"},
                    },
                    "required": ["section_ref"],
                },
            ),
            Tool(
                name="list_policy_rules",
                description="List all available policy rules.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="retrieve_policy_context",
                description="Retrieve policy context relevant to a question.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "k": {"type": "integer", "default": 3},
                    },
                    "required": ["question"],
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult:
        if name == "query_events":
            result = await query_events(**arguments)
        elif name == "get_stats":
            result = await get_stats(**arguments)
        elif name == "get_policy_rule":
            result = await get_policy_rule(**arguments)
        elif name == "list_policy_rules":
            result = await list_policy_rules()
        elif name == "retrieve_policy_context":
            result = await retrieve_policy_context(**arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(result, default=str),
                )
            ]
        )

    return server, stdio_server, InitializationOptions


async def run_server() -> None:
    server, stdio_server, initialization_options = build_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            initialization_options(
                server_name="aegisflow-ehs",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=server.notification_options,
                    experimental_capabilities={},
                ),
            ),
        )


def main() -> None:
    asyncio.run(run_server())


if __name__ == "__main__":
    main()