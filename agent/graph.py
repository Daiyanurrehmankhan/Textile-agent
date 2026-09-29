"""Wires the delay- and inventory-mismatch-investigation nodes into a
LangGraph StateGraph.

Graph shape:

                                 ,-- investigation_type=="delay" --> reason_delay --,
    (conditional entry point) --+                                                   +--> (loop / explain, below)
                                 `-- investigation_type=="mismatch" --> reason_mismatch --'

    reason_delay / reason_mismatch --(pending tool call, cap not reached)--> tool
    reason_delay / reason_mismatch --(concluded naturally, OR cap reached)--> explain --> END

    tool --(investigation_type=="delay")--> reason_delay
    tool --(investigation_type=="mismatch")--> reason_mismatch

Two reasoning nodes exist (see agent/nodes.py) purely because they're bound
to different system prompts; `tool` and `explain` are the exact same two
node functions for both paths -- no duplication, no per-type branching
inside them. What changed from the single-path (delay-only) version: the
entry point became conditional instead of fixed, and the "tool -> reason"
edge became conditional (routing back to whichever reasoning node the
investigation started at) instead of a single unconditional edge.

`build_graph()` takes all four node functions as parameters (instead of
importing agent.nodes.* directly) purely so tests can swap in stubs -- e.g.
a reasoning node that always requests another tool call (max-iteration
cap test), or two stubs that each emit a distinctive marker (router test)
-- without needing a real LLM API call.
"""
from langgraph.graph import END, StateGraph

from agent.nodes import explanation_node, reasoning_node_delay, reasoning_node_mismatch, tool_node
from agent.state import AgentState, initial_mismatch_state, initial_state
from config.settings import settings


def route_entry(state: AgentState) -> str:
    """Conditional entry point: the very first routing decision in the
    graph, before any node has run. Reads state["investigation_type"] --
    set once when the initial state was built (initial_state /
    initial_mismatch_state), not decided by an LLM call -- and sends the
    run into the matching reasoning node's name.
    """
    return state["investigation_type"]


def route_after_reasoning(state: AgentState) -> str:
    """The conditional edge's routing function: decide what happens after
    a reasoning node runs. Registered identically for both reason_delay and
    reason_mismatch below -- it only reads state, not which node produced
    it, so one function serves both paths.

    This is where the max-tool-calls safety cap is actually enforced --
    deliberately here, not inside the reasoning nodes. Their only job is
    "ask the model what to do next"; they shouldn't also decide whether the
    loop is allowed to continue. By checking the cap in the routing
    function instead, the graph structurally cannot loop back into another
    tool call past the cap, regardless of what the model asked for or
    whether the reasoning node was ever told to stop -- the cap is the
    graph's rule, not a request to the LLM it could ignore.
    """
    if state["tool_call_count"] >= state["max_tool_calls"]:
        return "explain"
    if state["pending_tool_calls"]:
        return "tool"
    return "explain"


def route_after_tool(state: AgentState) -> str:
    """After tool_node runs, loop back to whichever reasoning node this
    investigation started at. In the single-path version this was a fixed
    add_edge("tool", "reason"); with two reasoning nodes it has to be a
    conditional edge instead, because "go back to reasoning" now means two
    different possible destinations. investigation_type already carries
    the answer, so this is the same kind of state-only lookup as
    route_entry -- no new state needed, just one more place reading it.
    """
    return state["investigation_type"]


def build_graph(
    reasoning_node_delay=reasoning_node_delay,
    reasoning_node_mismatch=reasoning_node_mismatch,
    tool_node=tool_node,
    explanation_node=explanation_node,
):
    """Build and compile the StateGraph. See module docstring for why the
    nodes are parameters rather than hardcoded.
    """
    # StateGraph(AgentState) tells LangGraph the shape of the state dict and,
    # via the Annotated[...] fields in AgentState, how to merge each node's
    # partial return value into it (see agent/state.py).
    graph = StateGraph(AgentState)

    # Register each node function under a name -- edges below refer to nodes
    # by these string names, not by the function objects themselves. "tool"
    # and "explain" are registered once each and reached from both reasoning
    # paths -- that's the "reuse without duplicating" part.
    graph.add_node("reason_delay", reasoning_node_delay)
    graph.add_node("reason_mismatch", reasoning_node_mismatch)
    graph.add_node("tool", tool_node)
    graph.add_node("explain", explanation_node)

    # Conditional entry point: instead of one fixed start node, LangGraph
    # calls route_entry(state) on the very first state (before any node has
    # run) and starts at whichever node its label maps to. This is the
    # "router" -- it's a routing function wired at the graph's entry, not a
    # node that itself executes work.
    graph.set_conditional_entry_point(route_entry, {"delay": "reason_delay", "mismatch": "reason_mismatch"})

    # Each reasoning node gets its own conditional edge to the shared
    # tool/explain nodes, using the SAME routing function (route_after_reasoning)
    # both times -- registering it twice doesn't duplicate its logic, it just
    # wires two different source nodes to the same two destinations.
    graph.add_conditional_edges("reason_delay", route_after_reasoning, {"tool": "tool", "explain": "explain"})
    graph.add_conditional_edges("reason_mismatch", route_after_reasoning, {"tool": "tool", "explain": "explain"})

    # The loop-back from "tool": which reasoning node to return to depends on
    # investigation_type, so this is a conditional edge (route_after_tool)
    # rather than the single add_edge("tool", "reason") the delay-only graph
    # used.
    graph.add_conditional_edges("tool", route_after_tool, {"delay": "reason_delay", "mismatch": "reason_mismatch"})

    # "explain" always ends the run, regardless of which path got there.
    graph.add_edge("explain", END)

    # .compile() validates the graph (entry point set, every edge target
    # exists) and returns a CompiledStateGraph -- the actual runnable. This
    # step does not execute any node; it just turns the definition into
    # something you can call .invoke(...) on, possibly many times.
    return graph.compile()


def run_investigation(order_id: int, conn=None, max_tool_calls: int | None = None, llm=None):
    """Convenience entry point: run one full DELAY investigation and return
    the final AgentState.

    conn: DB connection to run tools against (e.g. a test-schema connection
    in tests; None lets each tool open/close its own connection against the
    configured DATABASE_URL).
    llm: override for tests/mocking; None uses the real ChatGoogleGenerativeAI
    client, credentialed via config/settings.py.
    """
    app = build_graph()
    state = initial_state(order_id, max_tool_calls or settings.default_max_tool_calls_delay)
    # `config["configurable"]` is how runtime dependencies (DB connection,
    # LLM client) reach node functions without living in `state` -- see the
    # docstring at the top of agent/nodes.py for why.
    config = {"configurable": {"conn": conn, "llm": llm}}
    return app.invoke(state, config=config)


def run_mismatch_investigation(material_id: str, conn=None, max_tool_calls: int | None = None, llm=None):
    """Convenience entry point: run one full INVENTORY-MISMATCH investigation
    and return the final AgentState. Same shape as run_investigation --
    only the starting state differs (initial_mismatch_state vs initial_state)."""
    app = build_graph()
    state = initial_mismatch_state(material_id, max_tool_calls or settings.default_max_tool_calls_mismatch)
    config = {"configurable": {"conn": conn, "llm": llm}}
    return app.invoke(state, config=config)
