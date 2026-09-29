"""Shared state schema for the delay- and inventory-mismatch-investigation
LangGraph agent.

LangGraph passes one dict-like state object between nodes. Every node reads
whatever keys it needs from `state` and returns a *partial* dict of the keys
it wants to change; LangGraph merges that partial dict into the running
state before calling the next node. The TypedDict below documents the shape
-- LangGraph doesn't enforce the types at runtime, but a StateGraph needs a
schema class to know which keys exist and how to merge them.

Most fields use plain types, which means "the last node to write this key
wins" (a normal dict update). Fields that accumulate across the
reasoning<->tool loop instead use `Annotated[list, operator.add]` -- a
"reducer". It tells LangGraph: when a node returns a value for this key,
don't overwrite state's list with it, concatenate the two lists
(`operator.add` is just `list.__add__`). Without that annotation, a tool
node returning one new history entry would replace the *entire* history
built up by every earlier pass through the loop, instead of appending to it.

No separate "expected_quantity" / "actual_quantity" / "discrepancy_source"
fields for the mismatch path: explanation_node never runs a second LLM call
to extract structured numbers back out of the model's prose (see its
docstring for why), so nothing would ever populate them -- they'd be
speculative fields nobody writes to. The tool results and findings already
carry the actual numbers (tool_call_history has the full JSON); add
structured fields like these only if something downstream needs to consume
the discrepancy programmatically rather than read the report text.
"""
from typing import Annotated, Optional, TypedDict
import operator

from langchain_core.messages import HumanMessage


class AgentState(TypedDict):
    investigation_type: str
    # "delay" or "mismatch" -- decided once, when the initial state is built
    # (initial_state / initial_mismatch_state below), never by an LLM call.
    # The graph's conditional entry point reads this to pick which reasoning
    # node starts the run; the loop-back edge after tool_node reads it again
    # to return to that same reasoning node. It's plain (no reducer) because
    # it's set once and never changes for the life of a run.

    investigation_target: dict
    # e.g. {"order_id": 6} for a delay investigation, or {"material_id":
    # "MAT-012"} for a mismatch investigation -- what this run is
    # investigating. Deliberately a free-form dict (not separate order_id/
    # material_id fields) so explanation_node's fallback text and logging
    # can describe either kind of target the same way, without a per-type
    # branch: `", ".join(f"{k}={v}" for k, v in investigation_target.items())`.

    messages: Annotated[list, operator.add]
    # Full conversation as LangChain message objects (HumanMessage/AIMessage/
    # ToolMessage). This is what actually gets sent back to the LLM on every
    # reasoning call -- the API is stateless, so the whole history has to
    # travel each request. Using LangChain's message types here (rather than
    # a provider's raw wire format) is what let the provider swap from
    # Claude to Gemini without touching this schema or the graph wiring.

    tool_call_history: Annotated[list, operator.add]
    # One structured record per tool call across the whole investigation:
    # {step, tool, args, result, is_error}. This is the "reasoning trace" --
    # printed at the end so a human can see exactly what the agent did and why.

    findings: Annotated[list, operator.add]
    # Short one-line human-readable notes, one per successful tool call --
    # a condensed version of tool_call_history for the final report/fallback.

    pending_tool_calls: list
    # The tool call(s) the model just asked for. NOT accumulated (no
    # reducer) -- each reasoning pass replaces this with its own request(s);
    # it's a mailbox from the reasoning node to the tool node, not a log.

    current_step: int          # reasoning-loop iteration counter, for display/debugging
    tool_call_count: int       # tool calls made so far -- checked against max_tool_calls
    max_tool_calls: int        # safety cap, configurable per investigation (see config/settings.py)
    concluded_naturally: bool  # True once the model stops asking for tools on its own (vs. hitting the cap)
    reasoning_error: Optional[str]
    # Set by a reasoning node only when the LLM call itself fails on every
    # model in the fallback chain (not a tool failure -- those are handled
    # entirely within tool_node and never reach this far). None normally;
    # explanation_node checks it to give an honest "could not complete"
    # report instead of either crashing or being mistaken for a natural
    # conclusion.
    final_explanation: Optional[str]  # set by the explanation node; None until the graph finishes


def _base_state(investigation_type: str, investigation_target: dict, task: str, max_tool_calls: int) -> AgentState:
    """Shared scaffolding for both investigation types -- everything except
    the type, the target, and the opening task description is identical."""
    return AgentState(
        investigation_type=investigation_type,
        investigation_target=investigation_target,
        messages=[HumanMessage(content=task)],
        tool_call_history=[],
        findings=[],
        pending_tool_calls=[],
        current_step=0,
        tool_call_count=0,
        max_tool_calls=max_tool_calls,
        concluded_naturally=False,
        reasoning_error=None,
        final_explanation=None,
    )


def initial_state(order_id: int, max_tool_calls: int) -> AgentState:
    """Build the starting state for a DELAY investigation into one order."""
    task = (
        f"Investigate order_id={order_id} in our textile mill's production system. "
        "Figure out whether it's delayed or has a data inconsistency, and if so, why. "
        "Use the available tools to gather evidence before concluding -- don't guess."
    )
    return _base_state("delay", {"order_id": order_id}, task, max_tool_calls)


def initial_mismatch_state(material_id: str, max_tool_calls: int) -> AgentState:
    """Build the starting state for an INVENTORY-MISMATCH investigation into
    one material."""
    task = (
        f"Investigate material_id={material_id} in our textile mill's inventory system. "
        "Figure out whether its recorded stock is consistent with recent order demand and "
        "material movement, and if there's a mismatch, what's likely causing it. "
        "Use the available tools to gather evidence before concluding -- don't guess."
    )
    return _base_state("mismatch", {"material_id": material_id}, task, max_tool_calls)
