"""Prove the graph's conditional entry point sends a delay investigation
into reason_delay and a mismatch investigation into reason_mismatch --
using two stub reasoning nodes that each emit a distinctive marker, so this
needs no real LLM call and can't pass by accident (a mixed-up route would
produce the wrong marker).
"""
from langchain_core.messages import AIMessage

from agent.graph import build_graph
from agent.nodes import explanation_node, tool_node
from agent.state import initial_mismatch_state, initial_state


def _stub_concludes_with(marker):
    def node(state, config):
        return {
            "messages": [AIMessage(content=marker)],
            "current_step": state["current_step"] + 1,
            "pending_tool_calls": [],
            "concluded_naturally": True,  # conclude immediately -- no tool_node involvement needed here
        }
    return node


def test_delay_state_is_routed_to_the_delay_reasoning_node(db_conn):
    app = build_graph(
        reasoning_node_delay=_stub_concludes_with("DELAY_PATH_TAKEN"),
        reasoning_node_mismatch=_stub_concludes_with("MISMATCH_PATH_TAKEN"),
        tool_node=tool_node,
        explanation_node=explanation_node,
    )
    result = app.invoke(initial_state(order_id=1, max_tool_calls=3), config={"configurable": {"conn": db_conn}})
    assert result["final_explanation"] == "DELAY_PATH_TAKEN"


def test_mismatch_state_is_routed_to_the_mismatch_reasoning_node(db_conn):
    app = build_graph(
        reasoning_node_delay=_stub_concludes_with("DELAY_PATH_TAKEN"),
        reasoning_node_mismatch=_stub_concludes_with("MISMATCH_PATH_TAKEN"),
        tool_node=tool_node,
        explanation_node=explanation_node,
    )
    result = app.invoke(
        initial_mismatch_state(material_id="MAT-001", max_tool_calls=3),
        config={"configurable": {"conn": db_conn}},
    )
    assert result["final_explanation"] == "MISMATCH_PATH_TAKEN"
