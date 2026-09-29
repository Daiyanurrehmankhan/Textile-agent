"""Tests for the polish-pass fixes to failure handling:
  - tool_node must not crash on an exception type it didn't specifically
    expect (e.g. a malformed LLM-generated tool call raising TypeError).
  - a reasoning node must not crash the graph if the LLM call fails outright
    (every model in the fallback chain unavailable); it should report
    honestly instead.
"""
from agent.nodes import explanation_node, reasoning_node_delay, tool_node


def test_tool_node_survives_an_unexpected_exception_type(db_conn):
    """query_order_status doesn't take a `bogus` kwarg -- this reproduces the
    TypeError an LLM-malformed tool call could trigger, which isn't a
    ToolQueryError or ValueError."""
    state = {
        "current_step": 1,
        "tool_call_count": 0,
        "max_tool_calls": 6,
        "pending_tool_calls": [{"id": "call-1", "name": "query_order_status", "input": {"bogus": 1}}],
    }
    result = tool_node(state, config={"configurable": {"conn": db_conn}})

    assert result["tool_call_count"] == 1
    entry = result["tool_call_history"][0]
    assert entry["is_error"] is True
    assert "unexpected failure" in entry["result"]
    # The raw TypeError text (which could reference internal argument names)
    # must not reach the model-facing content.
    assert "bogus" not in entry["result"]
    assert "TypeError" not in entry["result"]


class _AlwaysFailsLLM:
    """Stub LLM: every model in the (simulated) fallback chain is down."""
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        raise RuntimeError("simulated total outage")


def test_reasoning_node_reports_honestly_when_the_llm_is_unavailable(db_conn):
    state = {
        "investigation_type": "delay",
        "investigation_target": {"order_id": 6},
        "messages": [],
        "current_step": 0,
        "max_tool_calls": 6,
        "findings": [],
    }
    update = reasoning_node_delay(state, config={"configurable": {"conn": db_conn, "llm": _AlwaysFailsLLM()}})

    # Doesn't crash, doesn't pretend to have concluded.
    assert update["concluded_naturally"] is False
    assert update["pending_tool_calls"] == []
    assert "reasoning_error" in update and update["reasoning_error"]

    # explanation_node must not phrase this as a confident conclusion.
    state.update(update)
    state["tool_call_history"] = []
    state["tool_call_count"] = 0
    final = explanation_node(state, config={})["final_explanation"]
    assert "could not be completed" in final
    assert "order_id=6" in final
