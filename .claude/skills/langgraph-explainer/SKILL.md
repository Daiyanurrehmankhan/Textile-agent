---
name: langgraph-explainer
description: >
  Use whenever a change touches this project's LangGraph agent code
  (agent/graph.py, agent/nodes.py, agent/state.py, or adding/removing a node,
  edge, conditional edge, entry point, or state field). The user is new to
  LangGraph and wants every such change explained, not just implemented:
  which LangGraph mechanism is being used (node, edge, conditional edge,
  reducer, entry point, config vs. state, ...) and why it's the right one
  here -- given before or alongside the code -- plus inline code comments
  explaining the LangGraph-specific mechanics, not just the business logic.
  Trigger on requests like "add a node", "change the routing", "extend the
  graph", "add a new investigation path", or any edit to the files above.
---

# LangGraph explainer

This project's agent (`agent/graph.py`, `agent/nodes.py`, `agent/state.py`) is built with
LangGraph. The user is learning LangGraph through this project, so any change to it needs to
teach, not just work.

## Before or alongside writing the code

State, in plain terms:
- **What LangGraph mechanism is involved** — a node, a plain edge, a conditional edge, a
  (conditional) entry point, a state reducer (`Annotated[list, operator.add]` vs. a plain
  field), or the `config`/`configurable` channel for runtime dependencies.
- **Why that mechanism is the right one** for what's being asked — e.g. "this needs a
  conditional edge, not a plain one, because the destination depends on state" or "this field
  needs a reducer because it has to accumulate across loop iterations, not get overwritten."
- If it changes something that already existed (an existing node's contract, an existing edge's
  target), say what changed and why the rest of the graph didn't need to.

## In the code itself

Comments on new/changed LangGraph code should explain the mechanics, not restate the line:
- A new node: why it's a separate node rather than folded into an existing one.
- A new edge/conditional edge: what decides the routing and where that decision lives (in the
  routing function, not the node) — see `agent/graph.py`'s `route_after_reasoning` for the
  established pattern (cap enforcement lives in the router, not the reasoning node).
- A new/changed state field: whether it needs a reducer, and why.

## Keep it proportional

A one-line change (e.g. renaming a node) doesn't need a lecture — a sentence is enough. Save
the fuller explanation for genuinely new mechanics (a new kind of edge, a new reducer, a new
routing pattern) the user hasn't seen in this project yet.

## Reference

`agent/graph.py` and `agent/nodes.py` already carry this style of comment throughout — match
it rather than inventing a new format.
