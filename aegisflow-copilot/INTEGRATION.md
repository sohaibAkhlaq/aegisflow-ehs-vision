# AegisFlow Copilot — Integration Guide

This zip contains the **AI Safety Copilot** feature described in
`docs/copilot-architecture.md` — an additive chat assistant that sits on top of your
existing AegisFlow EHS project. It does **not** modify or replace anything in your
graded pipeline (Modules 1–5).

## What's in this zip

```
src/aegisflow/copilot/         ← the whole feature, new package
src/aegisflow/api/copilot_routes.py   ← new API endpoint
config/copilot.yaml            ← copilot-specific settings
frontend/assets/copilot.js     ← chat panel logic
frontend/assets/copilot.css    ← chat panel styling
frontend/copilot_snippet.html  ← HTML to paste into your index.html
docs/copilot-architecture.md   ← full architecture writeup
tests/unit/test_copilot_*.py   ← unit tests
requirements-copilot.txt       ← optional new dependencies
```

## How to merge this into your existing `aegisflow-ehs-vision` project

### 1. Copy the files over

Copy everything in this zip's `src/`, `config/`, `frontend/`, `docs/`, and `tests/`
folders into the matching folders in your project — nothing here overwrites an
existing file (it's all new files, except the two small edits in Step 2 and Step 3).

### 2. Install the optional dependencies

```bash
pip install -r requirements-copilot.txt
```

Everything degrades gracefully if you skip this — see "What happens if you don't
install these" below — but install them if you want the full LangChain/LangGraph/
RAG/MCP behavior working, not just the fallback paths.

### 3. Wire in the new API router (one small edit to `src/aegisflow/api/app.py`)

Open `src/aegisflow/api/app.py`. Add these two lines near the top, with the other
imports:

```python
from aegisflow.api import copilot_routes
from aegisflow.copilot import memory as copilot_memory  # noqa: F401 - registers the table
```

Then inside `create_app()`, right after the existing:
```python
app.include_router(routes.router)
app.include_router(ws.router)
```
add:
```python
app.include_router(copilot_routes.router)
```

That's the only change to an existing file in the whole integration.

### 4. Add the chat panel to your dashboard

Open `frontend/copilot_snippet.html` in this zip — it has 3 small pieces with clear
instructions on exactly where to paste each one into your existing
`frontend/index.html` (a nav button, a section, and two `<link>`/`<script>` tags).
No other part of `index.html` or `app.js` needs to change.

### 5. Run it

```bash
python -m aegisflow serve
```

Open the dashboard — there's now a "💬 Ask AI" tab. The new `copilot_messages`
database table is created automatically on next startup (via the existing
`init_db()` call, since the model registers itself on `Base.metadata` when
`aegisflow.copilot.memory` is imported).

## What happens if you don't install `requirements-copilot.txt`

Every feature still works, just via a simpler fallback:

| Feature | With deps installed | Without |
|---|---|---|
| Policy Q&A | Real vector search (ChromaDB) | Pure-Python TF-IDF search |
| Chat agent | LangChain tool-calling agent | Keyword-routed direct tool calls |
| Answer generation | LCEL chain via `langchain-groq` | Direct call via the core project's own `aegisflow.llm` provider |
| Incident drafting | LangGraph state machine | Plain Python retry loop, same logic |
| MCP server | Works | Raises a clear error if you try to run it |

Nothing crashes the core AegisFlow pipeline either way — see
`src/aegisflow/copilot/__init__.py`'s `CopilotDependencyError` for how this is enforced
throughout.

## Running the new tests

```bash
pip install -r requirements-copilot.txt   # needed for pytest-asyncio
pytest tests/unit/test_copilot_rag.py tests/unit/test_copilot_tools.py tests/unit/test_copilot_memory.py
```

## Trying the MCP server standalone

```bash
pip install mcp
python -m aegisflow.copilot.mcp_server
```

Or point the MCP Inspector / Claude Desktop at it — see the docstring at the top of
`src/aegisflow/copilot/mcp_server.py` for the exact Claude Desktop config JSON.

## Full architecture writeup, and how each piece maps to the internship curriculum

See `docs/copilot-architecture.md`.
