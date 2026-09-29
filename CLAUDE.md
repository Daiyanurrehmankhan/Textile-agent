# Textile Mill Reconciliation Agent

## Overview

An autonomous LangGraph agent that reconciles textile mill order and inventory data — it
investigates order delays and inventory mismatches by reasoning over multi-step tool calls
against a Postgres DB, not a fixed pipeline. This is a portfolio piece demonstrating agentic
AI (LLM-driven tool selection + looped reasoning) for job interviews, so code clarity matters
as much as correctness. Two investigation paths: **delay** (why is this order stuck?) and
**mismatch** (why doesn't this material's recorded stock add up?). See `agent/graph.py` for
how they're wired.

## Critical commands

```bash
pytest                                   # full test suite (tool tests + agent graph tests)
pytest tests/test_tools                  # tool-layer tests only
pytest tests/test_agent                  # graph/agent tests only (2 make live Gemini calls,
                                          #   skipped automatically if GOOGLE_API_KEY is unset)
python scripts/run_agent_demo.py delay [order_id]        # live end-to-end demo, delay path
python scripts/run_agent_demo.py mismatch [material_id]  # live end-to-end demo, mismatch path
                                          #   (both args optional -- auto-picks the most anomalous case)
python scripts/verify_data.py            # sanity-check seeded dev data + the 5 known anomalies
python db/generate_sample_data.py        # regenerate dev sample data (idempotent, truncates first)
python setup_db.py                       # (re)apply schema.sql to the dev DB
```

## Architecture

- `config/settings.py` — the ONLY place env vars are read. Everything else imports `settings`.
- `db/` — connection helper (`db/__init__.py`) + dev sample-data generator (`db/generate_sample_data.py`).
- `tools/` — **tool execution logic**: 5 functions, one per file, each a parameterized-SQL query
  against the DB. `tools/_common.py` has the shared `ToolQueryError` + validators + `db_cursor`.
- `agent/` — **reasoning/decision logic**: `state.py` (the shared state schema), `nodes.py`
  (reasoning/tool/explanation node functions — this is where the LLM gets called and tool
  results get interpreted), `graph.py` (wires nodes into the LangGraph `StateGraph`).
  **Do not conflate the two** — `tools/` never decides *when* to call itself; `agent/nodes.py`
  never runs SQL directly, it only dispatches to `tools/`.
- `tests/` — `test_tools/` (one file per tool) and `test_agent/` (graph wiring, router, cap,
  live end-to-end). `tests/conftest.py` seeds an isolated `test_textile` Postgres schema per
  test session — tests never touch dev data.
- `docs/` — currently just a placeholder; see Project status below.

## Tech stack

- Python 3.13, Postgres 18 (via Neon, serverless).
- LangGraph (`agent/graph.py`) for the agent loop/routing.
- **`langchain-google-genai`, model `gemini-3.5-flash`** (with `gemini-3.6-flash` /
  `gemini-3.5-flash-lite` as automatic fallbacks, `max_retries=1` per model so failover
  is fast rather than burning ~90s of internal retries per model — see
  `agent/nodes.py::_get_llm_with_fallbacks`) is the reasoning model for the agent itself.
- pytest for testing.

**This project does NOT use the Anthropic/Claude API for the agent's own reasoning.** Claude
Code is only the tool used to *build* this project. Do not reintroduce `anthropic`/`ChatAnthropic`
as a runtime dependency of `agent/` — the provider was deliberately migrated from Claude to
Gemini; see git history / prior session context if the reasoning is unclear.

## Coding standards (non-negotiable)

- Secrets/config: **only** `config/settings.py` calls `os.getenv`/`os.environ`. No other file
  reads an env var directly.
- SQL: always parameterized (`%s` placeholders via psycopg2). Never string-format a value into
  a query.
- Tools: validate inputs, catch DB errors, and raise the shared `ToolQueryError` (from
  `tools/_common.py`) rather than leaking raw driver error text. Every tool function caps its
  result size.
- Tests: every new tool or agent node ships with tests under `tests/`, run against the isolated
  `test_textile` schema (via the `db_conn`/`anomalies` fixtures in `tests/conftest.py`) —
  never against dev data.

## Workflow preferences

I'm new to LangGraph. **Any LangGraph-related change must be explained before or alongside the
code** — what mechanism is being used (node, conditional edge, reducer, entry point, etc.) and
why it's the right one for the situation — not just what the code does. Code touching
`agent/graph.py` or `agent/nodes.py` should carry comments explaining the LangGraph-specific
mechanics inline, not just the business logic. See `.claude/skills/langgraph-explainer/SKILL.md`
for the checklist this follows.

## Project status

**Built:** schema (`schema.sql`), seeded dev data with 5 known anomalies (`db/generate_sample_data.py`),
5 tools with full test coverage, both LangGraph investigation paths (delay + inventory-mismatch,
including router + max-iteration-cap tests), Gemini 3.5 Flash integration with automatic model
fallback, `README.md`, and an architecture diagram (`docs/architecture.md`). **Pending:** nothing
tracked right now -- next likely additions are a CI workflow and/or a lightweight web UI over
`scripts/run_agent_demo.py`.
