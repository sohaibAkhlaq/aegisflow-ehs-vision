# AegisFlow Copilot — Architecture

An additive chat assistant layered on top of AegisFlow's core compliance pipeline
(Modules 1–5, unmodified). Nothing here is required for the graded pipeline to run;
this exists to demonstrate a second, independent skill set — agentic AI — on the same
real data, using the same policy PDF and the same logged events.

## Top layer: what a person actually sees

A new "💬 Ask AI" tab on the existing dashboard. A safety officer types a question in
plain English and gets an answer that either:

- Quotes the real policy, with a section citation (RAG), or
- Reports real numbers from the logged events (tool-calling), or
- Automatically explains a serious event that just fired (the incident workflow),
  drafted by one agent and checked by a second before it's finalized.

## Component map

```
User question (dashboard chat)
        │
        ▼
POST /api/copilot/chat  ──────────────────────────────────────────┐
        │                                                          │
        ▼                                                          │
  agent.handle_message()                                           │
        │                                                          │
        ├─ loads session history ──── memory.py (episodic store)   │
        │                                                          │
        ├─ LangChain tool-calling agent (preferred) ────────────────┤
        │     bound to: query_events, get_stats, get_policy_rule,  │
        │     list_policy_rules, retrieve_policy_context           │
        │                                                          │
        └─ keyword-routed fallback (if LangChain not installed) ───┘
              same tools, called directly


retrieve_policy_context tool
        │
        ▼
  rag.py (PolicyRAG)
        │
        ├─ ChromaDB + sentence-transformers (preferred)
        └─ pure-Python TF-IDF (automatic fallback, zero extra deps)


HIGH/CRITICAL event logged (Module 3 escalation, unchanged)
        │
        ▼
  graph.run_incident_workflow()
        │
        ▼
  gather_context → draft (Drafter agent) → review (Reviewer agent) ─┐
                        ▲                         │                 │
                        └── retry, up to 3 rounds ┘                 │
                                                                     ▼
                                                                finalize
                                                          (IncidentReport)


Same tools, also exposed via:
        │
        ▼
  mcp_server.py  ──►  Claude Desktop / MCP Inspector / any MCP client
```

## Curriculum mapping

| Piece | File(s) | Demonstrates |
|---|---|---|
| RAG over the policy PDF | `copilot/rag.py` | Embeddings, vector search (Week 3) |
| LangChain / LCEL | `copilot/chains.py` | `prompt \| model \| parser` composition (Week 1) |
| Tool-calling agent | `copilot/tools.py`, `copilot/agent.py` | Function calling, agent loop (Week 2) |
| LangGraph workflow | `copilot/graph.py` | State machine, conditional routing, retry loop (Week 5) |
| Multi-agent drafter/reviewer | `copilot/agents/drafter.py`, `copilot/agents/reviewer.py` | Debate/consensus pattern (Week 6) |
| Session memory | `copilot/memory.py` | Episodic memory store (Week 7) |
| MCP server | `copilot/mcp_server.py` | Tool exposure via MCP (Week 8) |

## Design principles carried over from the core project

- **Everything degrades gracefully.** Every optional dependency (LangChain, LangGraph,
  ChromaDB, sentence-transformers, MCP) has a fallback path that still answers the
  question, just via a simpler method. Nothing crashes because an extra package isn't
  installed — same "offline must always work" rule as `CLAUDE.md`.
- **Read-only, additive, and isolated.** `copilot/` never imports from — or is
  imported by — the graded pipeline modules (`detection/`, `severity/`, `escalation/`,
  `reports/`). The one exception is reading (never writing) `violation_events` and
  `policy_rules` for its tools, and adding one new table (`copilot_messages`) for its
  own session memory, which — unlike the audit trail — is deletable, because chat
  history has no compliance value.
- **Two different agent-building styles, on purpose.** The chat agent is built with
  LangChain's `create_tool_calling_agent`; the drafter/reviewer pair calls the core
  project's own `aegisflow.llm.build_provider()` directly. This is deliberate — it
  shows both an "I used the framework" and an "I built this from the primitives up"
  approach, rather than only ever demonstrating one library.

## Setup

```bash
pip install -r requirements-copilot.txt
```

Then merge the three snippets in `frontend/copilot_snippet.html` into your existing
`frontend/index.html`, and add these two lines to `src/aegisflow/api/app.py`:

```python
from aegisflow.api import copilot_routes
from aegisflow.copilot import memory as copilot_memory  # noqa: F401 - registers the table

# inside create_app(), alongside the existing app.include_router(routes.router):
app.include_router(copilot_routes.router)
```

No other file needs to change. Run `aegisflow serve` as usual — the new tab and
endpoint are live immediately.
