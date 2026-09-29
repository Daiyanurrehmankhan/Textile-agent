"""Prove the max-tool-calls safety cap actually halts the reason<->tool loop,
using a stubbed reasoning node that always asks for another tool call --
no real Gemini API call needed for this test.
"""
from langchain_core.messages import AIMessage

from agent.graph import build_graph
from agent.state import initial_state
from agent.nodes import tool_node, explanation_node


def _always_requests_a_tool_call(state, config):
    """Stub reasoning node: never concludes on its own, always asks for the
    same (cheap, valid) tool call. Same signature as the real reasoning
    nodes so build_graph() can swap it in transparently.
    """
    return {
        "messages": [AIMessage(content="(stub) requesting another tool call")],
        "current_step": state["current_step"] + 1,
        "pending_tool_calls": [
            {"id": f"stub-{state['current_step']}", "name": "get_average_stage_duration",
             "input": {"stage": "dyeing"}}
        ],
        "concluded_naturally": False,
    }


class _AlwaysFailsLLM:
    """Stub LLM standing in for explanation_node's cutoff-synthesis call (see
    agent/nodes.py::_synthesize_cutoff_explanation). Raising here forces that
    call down its own except-clause fallback, so this test stays offline/
    deterministic (no live Gemini call, no wording assumptions about a real
    model's synthesized report) while still exercising the raw-dump fallback
    text this test asserts on. Same shape as test_failure_handling.py's stub
    of the same name, used for the analogous reasoning-node failure path.
    """
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        raise RuntimeError("simulated total outage")


def test_cap_halts_the_loop_even_when_reasoning_never_concludes(db_conn):
    # This test targets the delay path -- initial_state() below builds a
    # "delay" investigation, so the stub is registered as reasoning_node_delay
    # (the graph's conditional entry point routes "delay" runs here).
    max_tool_calls = 3
    app = build_graph(reasoning_node_delay=_always_requests_a_tool_call, tool_node=tool_node,
                       explanation_node=explanation_node)
    state = initial_state(order_id=1, max_tool_calls=max_tool_calls)

    result = app.invoke(state, config={"configurable": {"conn": db_conn, "llm": _AlwaysFailsLLM()}})

    # The loop must stop at exactly the cap, not run away.
    assert result["tool_call_count"] == max_tool_calls
    # It never got a natural conclusion -- the cap forced the explain branch.
    assert result["concluded_naturally"] is False
    # explanation_node still ran and produced a graceful (not crashed) report --
    # the cutoff-synthesis call fails too (stub raises), so this is the
    # raw-dump fallback text, not a synthesized report.
    assert result["final_explanation"] is not None
    assert "tool call" in result["final_explanation"].lower()
    # Every one of the forced tool calls actually executed and was recorded.
    assert len(result["tool_call_history"]) == max_tool_calls
