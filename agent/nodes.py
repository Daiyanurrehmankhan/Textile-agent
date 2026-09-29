"""The node functions for the delay- and inventory-mismatch-investigation
graph: two reasoning nodes (one per investigation type, sharing a common
LLM-calling helper), plus tool_node and explanation_node, which are used
by both investigation types unchanged.

LangGraph node signature: `def node(state, config) -> dict`.
  - `state` is the accumulated AgentState so far (see agent/state.py).
  - `config` carries per-invocation *runtime* dependencies that don't belong
    in state -- an open DB connection, an LLM client. State gets merged and
    (optionally) checkpointed by LangGraph; a live socket connection can't
    be serialized into a checkpoint, so it travels via config instead,
    under config["configurable"].
  - The return value is a partial state update (only the keys this node
    changed) -- see agent/state.py's docstring for how merging works.

Tests inject a fake DB connection (isolated test schema) and even a fake
reasoning node this same way, which is why the DB connection and LLM client
are read from `config` rather than imported/constructed globally.

LLM provider: ChatGoogleGenerativeAI (Gemini), via langchain-google-genai --
not the raw Google SDK -- so tool binding and tool-call parsing go through
LangChain's standard chat-model interface. That's what keeps this file (and
only this file, plus a couple of lines in tool_node/explanation_node for
message-format reasons) as the sole place that knows which LLM provider is
in use; the graph, state schema, and tool dispatch logic don't.
"""
import json
import logging

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings
from tools._common import STAGES, ToolQueryError
from tools.check_recent_inventory_movements import check_recent_inventory_movements
from tools.find_similar_past_delays import find_similar_past_delays
from tools.get_average_stage_duration import get_average_stage_duration
from tools.query_inventory import query_inventory
from tools.query_order_status import query_order_status

# Library code only gets a logger and logs to it -- it never configures
# handlers/formatters itself (that's the application entry point's job, e.g.
# scripts/run_agent_demo.py). That's what lets the same logging calls show
# up on the console during a demo run and in a log file for later analysis,
# without this module caring which -- or both -- are wired up.
logger = logging.getLogger(__name__)

TOOL_REGISTRY = {
    "query_order_status": query_order_status,
    "get_average_stage_duration": get_average_stage_duration,
    "query_inventory": query_inventory,
    "find_similar_past_delays": find_similar_past_delays,
    "check_recent_inventory_movements": check_recent_inventory_movements,
}

# Tool definitions, in LangChain's schema shape: {"name", "description",
# "parameters"} where "parameters" is a JSON Schema object. This is the same
# information Anthropic's format carried (that used "input_schema" for this
# key) -- ChatGoogleGenerativeAI.bind_tools() (like every LangChain chat
# model) expects "parameters", so that's the one thing that had to change
# about these definitions when the provider swapped.
TOOL_SCHEMAS = [
    {
        "name": "query_order_status",
        "description": "Look up one order's current stage, dates, and full stage history by order_id.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "integer", "description": "The order to look up."}},
            "required": ["order_id"],
        },
    },
    {
        "name": "get_average_stage_duration",
        "description": "Average number of days orders normally spend in a given production stage, "
                        "based on completed (not in-progress) stage history. Use this as the baseline "
                        "to judge whether an order is stuck.",
        "parameters": {
            "type": "object",
            "properties": {"stage": {"type": "string", "enum": list(STAGES)}},
            "required": ["stage"],
        },
    },
    {
        "name": "query_inventory",
        "description": "Look up inventory batches, optionally filtered by exact material_name or material_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "material_name": {"type": "string"},
                "material_id": {"type": "string"},
            },
        },
    },
    {
        "name": "find_similar_past_delays",
        "description": "Find past (completed) orders that spent at least min_days in a given stage -- "
                        "i.e. has a delay like this happened before?",
        "parameters": {
            "type": "object",
            "properties": {
                "stage": {"type": "string", "enum": list(STAGES)},
                "min_days": {"type": "number", "description": "Minimum days in stage to qualify."},
            },
            "required": ["stage", "min_days"],
        },
    },
    {
        "name": "check_recent_inventory_movements",
        "description": "Recent production activity (a proxy for stock movement) for orders using a given "
                        "material_id, within a lookback window in days.",
        "parameters": {
            "type": "object",
            "properties": {
                "material_id": {"type": "string"},
                "days": {"type": "integer", "description": "Lookback window in days (default 30)."},
            },
            "required": ["material_id"],
        },
    },
]

# Two system prompts, one per investigation type -- this is the only thing
# that actually differs between the delay and mismatch paths. Both share the
# same closing paragraph (tool-call budget + "write a plain-language
# conclusion") so the two reasoning nodes below can stay tiny wrappers
# around one shared LLM-calling helper (_reason) rather than duplicating it.
_CLOSING = """You may call up to {max_tool_calls} tools in this investigation. Once you have \
enough evidence to explain what's going on (or to confidently say nothing is wrong), STOP \
calling tools and write your conclusion as plain text: a short, plain-language explanation \
for a plant manager (not a developer) covering what you found, why it happened if you can \
tell, and what evidence supports it."""

DELAY_SYSTEM_PROMPT = """You are a reconciliation agent for a textile mill's production system. \
You investigate one order at a time, looking for delays or data inconsistencies between \
orders, inventory, and production logs.

You have tools to query order status, stage-duration baselines, inventory, historical \
delays, and recent material movement. Use them to gather concrete evidence -- don't \
speculate about numbers you haven't looked up.

""" + _CLOSING

MISMATCH_SYSTEM_PROMPT = """You are a reconciliation agent for a textile mill's inventory system. \
You investigate one material at a time, looking for inventory mismatches -- recorded stock \
that doesn't line up with recent order demand or production activity for that material.

You have tools to look up inventory batches (query_inventory) and recent order/production \
activity for a material (check_recent_inventory_movements) -- those two are the most directly \
relevant here, but order-status and stage-duration tools may help explain *why* a mismatch \
happened (e.g. an order consuming more than expected). Use them to gather concrete evidence \
-- don't speculate about numbers you haven't looked up.

""" + _CLOSING

# Used only by _synthesize_cutoff_explanation, below -- a distinct prompt (not a variant of
# DELAY_SYSTEM_PROMPT/MISMATCH_SYSTEM_PROMPT) because the framing is different in a way that
# matters: those two are told "you have tools, decide whether to use them"; this one is told
# "there are no tools, work with what's already in front of you" and is asked to flag that
# explicitly rather than write a normal-looking report.
_CUTOFF_SYNTHESIS_PROMPT = """You are the same reconciliation agent, continuing an investigation \
that has just run out of its tool-call budget. You have NO tools available in this turn and \
must not attempt to call any -- base your answer only on the evidence already gathered in the \
conversation above.

Write the best-effort diagnosis the evidence so far supports, in the same report style you'd \
use if you had concluded naturally: a short, plain-language explanation for a plant manager \
(not a developer) covering what you found, why it happened if you can tell, and what evidence \
supports it.

Because the investigation was cut short, open with one sentence making that explicit -- that \
this is a partial, best-effort read based on the evidence collected before the tool-call limit \
was reached, not a fully confident conclusion."""


def _get_conn(config):
    return (config.get("configurable") or {}).get("conn")


def _get_llm_with_fallbacks(config, tools=TOOL_SCHEMAS):
    """Build the tool-bound model to call, with automatic backup models on failure.

    `tools` defaults to the real TOOL_SCHEMAS (what _reason uses); pass `tools=[]` to get a
    model with NO tools attached at all -- that's what _synthesize_cutoff_explanation below
    uses, so that call structurally cannot request a tool call (there's nothing to call). Every
    model, including a test override, still goes through `.bind_tools(...)` -- binding an empty
    list, rather than skipping the call -- so this stays the one code path both cases share.

    `.bind_tools(...)` has to happen per-model (it's what attaches TOOL_SCHEMAS
    to that specific client's requests), so each candidate model is built and
    bound individually *before* chaining them -- `.with_fallbacks()` is a
    generic Runnable combinator, not something bind_tools understands, so it
    has to wrap the already-tool-bound runnables, not the raw ChatGoogleGenerativeAI
    instances.

    `primary.with_fallbacks([backup1, backup2, ...])` returns a new Runnable:
    calling `.invoke()` on it tries `primary` first, and on ANY exception
    (rate limit, quota exhausted, transient outage -- the default
    `exceptions_to_handle=(Exception,)` is deliberately broad) retries the
    same input against each backup in order, returning the first success.
    _reason() below doesn't need to know any of this happened -- it just
    calls `.invoke()` on whatever this function returns.

    `max_retries=1` on each model matters here: ChatGoogleGenerativeAI has
    its OWN retry layer underneath ours (default max_retries=6, with
    exponential backoff -- 1s, 2s, 4s, 8s, 16s...), which retries the SAME
    model several times before ever letting an exception surface to
    `.with_fallbacks()` above. Left at the default, a single failed call can
    burn ~90 seconds hammering an already-quota-exhausted model before our
    fallback chain even gets a turn. Since a 429/quota error won't resolve
    itself in a few seconds anyway, there's nothing to gain from the extra
    internal retries -- fail fast and let `.with_fallbacks()` move to the
    next model instead.

    Tests can still inject a single fake/mock model via
    config["configurable"]["llm"] to bypass the real models and fallback
    chain entirely.
    """
    override = (config.get("configurable") or {}).get("llm")
    if override is not None:
        return override.bind_tools(tools)

    model_names = [settings.google_model, *settings.google_fallback_models]
    bound_models = [
        ChatGoogleGenerativeAI(model=name, google_api_key=settings.google_api_key, max_retries=1)
        .bind_tools(tools)
        for name in model_names
    ]
    primary, *backups = bound_models
    return primary.with_fallbacks(backups) if backups else primary


def _reason(state, config, system_prompt: str):
    """Ask Gemini what to do next: call a tool, or conclude the investigation.

    This is the one place that actually talks to the LLM -- shared by both
    reasoning_node_delay and reasoning_node_mismatch below, which differ
    only in which system_prompt they pass in. Splitting it out this way
    means the delay/mismatch branching is purely "which prompt frames the
    task" -- the call, the tool binding, and the response parsing are
    identical for both investigation types, so there's exactly one place
    that logic could be wrong instead of two.

    Sends the full message history + tool schemas to the model. If the
    response includes tool calls, they're stashed in pending_tool_calls for
    the tool node to execute. Otherwise the response text IS the conclusion,
    and concluded_naturally is set so the explanation node knows to use it
    as-is.

    `.bind_tools(TOOL_SCHEMAS)` (done inside _get_llm_with_fallbacks, per
    candidate model) returns a new Runnable wrapping a model with those
    tools attached to every request it makes -- this is LangChain's standard
    tool-binding interface, the same call shape regardless of provider (an
    Anthropic model would bind the exact same way via ChatAnthropic). What
    differs per provider is hidden behind it: Gemini's function-calling wire
    format vs. Anthropic's tool_use blocks. LangChain normalizes both into
    `response.tool_calls` -- a flat list of {"name", "args", "id"} dicts --
    so the code below doesn't need to know which provider (or, now, which of
    the backup models) produced them.
    """
    step = state["current_step"] + 1
    logger.info("[step %d] reasoning (%s): calling %s (+ %d backup model(s) if it fails) "
                "(%d messages in history)",
                step, state["investigation_type"], settings.google_model,
                len(settings.google_fallback_models), len(state["messages"]) + 1)

    llm = _get_llm_with_fallbacks(config)
    system_message = SystemMessage(content=system_prompt.format(max_tool_calls=state["max_tool_calls"]))
    try:
        response = llm.invoke([system_message, *state["messages"]])
    except Exception:
        # Every model in the fallback chain failed (total outage, invalid key,
        # network down, ...). This is the one call in the whole graph with no
        # further fallback below it, so it has to be caught here rather than
        # left to propagate -- otherwise a single bad API day would crash the
        # entire investigation instead of ending it gracefully. Full detail
        # logged server-side only; explanation_node turns this into an honest
        # "could not complete" report rather than crashing or (worse) silently
        # treating it as if the investigation concluded on its own.
        logger.exception("[step %d] reasoning: LLM call failed on every model in the fallback chain", step)
        return {
            "messages": [],
            "current_step": step,
            "pending_tool_calls": [],
            "concluded_naturally": False,
            "reasoning_error": "the reasoning step failed (LLM call unavailable after trying "
                                f"{1 + len(settings.google_fallback_models)} model(s))",
        }

    # response_metadata carries which model actually answered -- useful in
    # the logs to see whether a backup had to step in.
    model_used = (getattr(response, "response_metadata", None) or {}).get("model_name", settings.google_model)
    if model_used != settings.google_model:
        logger.warning("[step %d] reasoning: primary model failed, backup %s answered instead",
                        step, model_used)

    update = {
        "messages": [response],
        "current_step": step,
    }
    if response.tool_calls:
        # response.tool_calls is LangChain's normalized shape: each entry is
        # {"name": str, "args": dict, "id": str, "type": "tool_call"}.
        # We only keep the three fields tool_node actually needs, under our
        # own key names ("input" instead of "args") -- that's what lets
        # tool_node stay provider-agnostic: it was written against this
        # dict shape, not against whatever Gemini or Claude calls it.
        update["pending_tool_calls"] = [
            {"id": tc["id"], "name": tc["name"], "input": tc["args"]} for tc in response.tool_calls
        ]
        update["concluded_naturally"] = False
        logger.info("[step %d] reasoning: requested %d tool call(s): %s",
                    step, len(response.tool_calls), [tc["name"] for tc in response.tool_calls])
    else:
        update["pending_tool_calls"] = []
        update["concluded_naturally"] = True
        logger.info("[step %d] reasoning: concluded naturally (no tool call requested)", step)
    return update


def reasoning_node_delay(state, config):
    """Reasoning node for the delay-investigation path. See _reason above --
    this is just _reason bound to the delay system prompt. A distinct
    function (rather than passing the prompt around at graph-build time)
    exists because LangGraph registers nodes by name/callable, and the
    graph needs two distinct names to route between: the conditional entry
    point sends "delay" runs here, and the loop-back edge after tool_node
    needs a name to return to for a delay investigation specifically."""
    return _reason(state, config, DELAY_SYSTEM_PROMPT)


def reasoning_node_mismatch(state, config):
    """Reasoning node for the inventory-mismatch-investigation path --
    _reason bound to the mismatch system prompt. See reasoning_node_delay
    for why this needs to be its own named function."""
    return _reason(state, config, MISMATCH_SYSTEM_PROMPT)


def tool_node(state, config):
    """Execute every tool Gemini just requested and report results back.

    A tool raising ToolQueryError (or a validation ValueError) is not a
    crash -- it's turned into an error-status ToolMessage so the model sees
    the failure as an observation and can adapt (retry differently, try
    another tool, or note the limitation in its conclusion), rather than the
    graph crashing. Any OTHER exception is also caught here (not just those
    two expected types): `call["input"]` is LLM-generated, and an LLM can
    call a tool with a subtly wrong argument (wrong name, wrong type) that
    raises a plain TypeError before the tool's own validation even runs --
    that's still "a tool call went wrong," not a reason to crash the whole
    graph, so it gets the same is_error=True treatment (with the real
    exception logged in full for debugging, kept out of what the model sees).

    Unlike Anthropic's wire protocol (which requires every tool_result from
    one turn to be batched into a single follow-up message), LangChain's
    convention is one ToolMessage per tool call, each carrying the
    tool_call_id it answers -- so each pending call becomes its own message
    here, appended to state["messages"] via the list reducer.
    """
    conn = _get_conn(config)
    tool_messages = []
    trace_entries = []
    findings = []
    count = state["tool_call_count"]

    for call in state["pending_tool_calls"]:
        fn = TOOL_REGISTRY.get(call["name"])
        try:
            if fn is None:
                raise ValueError(f"Unknown tool {call['name']!r}")
            result = fn(**call["input"], conn=conn)
            content = json.dumps(result, default=str)  # default=str handles date/datetime/Decimal
            is_error = False
            findings.append(f"{call['name']}({call['input']}) -> {content[:200]}")
        except (ToolQueryError, ValueError) as exc:
            content = f"Tool error: {exc}"
            is_error = True
            logger.warning("[step %d] tool %s(%s) failed: %s", state["current_step"], call["name"],
                            call["input"], content)
        except Exception:
            # Unexpected exception type (see docstring) -- keep the model-facing
            # message generic (don't echo str(exc), which could be anything,
            # e.g. a raw TypeError referencing internal argument names) and log
            # the real traceback server-side only.
            content = f"Tool error: unexpected failure calling {call['name']}."
            is_error = True
            logger.exception("[step %d] tool %s(%s) failed unexpectedly", state["current_step"],
                              call["name"], call["input"])
        else:
            logger.info("[step %d] tool %s(%s) -> %s", state["current_step"], call["name"],
                        call["input"], content[:200])
        count += 1
        tool_messages.append(
            ToolMessage(content=content, tool_call_id=call["id"], status="error" if is_error else "success")
        )
        trace_entries.append(
            {"step": state["current_step"], "tool": call["name"], "args": call["input"],
             "result": content, "is_error": is_error}
        )

    logger.info("[step %d] tool_call_count now %d/%d", state["current_step"], count, state["max_tool_calls"])
    return {
        "messages": tool_messages,
        "tool_call_history": trace_entries,
        "findings": findings,
        "tool_call_count": count,
        "pending_tool_calls": [],
    }


def _extract_text(content) -> str:
    """AIMessage.content is a plain string for Gemini's text responses
    (Anthropic's equivalent was a list of typed content blocks) -- handle
    both shapes so this doesn't break if a provider returns parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


def _synthesize_cutoff_explanation(state, config) -> str | None:
    """One extra LLM call, made only when the tool-call cap cut the loop off
    before the model concluded on its own -- turns the raw findings into an
    actual readable report instead of a semicolon-joined dump of tool
    outputs. Returns the synthesized text, or None if this call itself fails
    (so the caller can fall back to the raw dump rather than crash).

    Why this doesn't reopen the reasoning<->tool loop: route_after_reasoning
    (agent/graph.py) is what enforces the cap, and it only runs after a
    *reasoning* node -- reason_delay/reason_mismatch. explanation_node has no
    outgoing conditional edge at all (`graph.add_edge("explain", END)` is
    unconditional), so nothing this function does can route back into
    "tool". The stronger guarantee is structural, not just "the graph
    doesn't happen to route there": `_get_llm_with_fallbacks(config,
    tools=[])` binds zero tools to the model, so `response.tool_calls` is
    guaranteed empty -- there's nothing for the model to ask for, so its
    reply can only ever be plain text. That's why this is a plain Python
    call inside explanation_node rather than a new node: a new node would
    imply a new edge decision to make (LangGraph would need to know what
    comes after it), but this call has exactly one possible outcome (text,
    or an exception), so there's no routing decision left to wire.
    """
    llm = _get_llm_with_fallbacks(config, tools=[])
    system_message = SystemMessage(content=_CUTOFF_SYNTHESIS_PROMPT)
    try:
        response = llm.invoke([system_message, *state["messages"]])
    except Exception:
        # Same reasoning as _reason's except clause: every model in the
        # fallback chain failing here is a real (if rare) possibility, and
        # explanation_node must still return *something* -- the raw-dump
        # fallback the caller uses on None is the pre-existing behavior,
        # so a failure here degrades to that instead of crashing the graph
        # on its very last node.
        logger.exception("cutoff-synthesis LLM call failed; falling back to raw findings dump")
        return None
    return _extract_text(response.content).strip() or None


def explanation_node(state, config):
    """Produce the final plain-language report.

    This node deliberately does NOT make another LLM call when the model
    concluded naturally (concluded_naturally=True): its own last response
    already IS the plain-language explanation the system prompt asked for --
    re-asking would just pay for a second API call to restate it. This
    node's job is just to lift that text into final_explanation.

    If the investigation was cut off by the tool-call cap, there's no such
    text to lift, but there IS a full evidence trail worth synthesizing into
    a real report -- see _synthesize_cutoff_explanation above. If the
    reasoning step itself failed outright (see _reason's except clause),
    there's no point making another LLM call (the model is presumably still
    unavailable), so that case keeps the plain raw-findings fallback. All
    three cases get distinct, honest wording -- none is phrased as if the
    investigation reached a confident conclusion when it didn't.
    """
    # investigation_target is a free-form dict ({"order_id": 6} or
    # {"material_id": "MAT-012"}) precisely so this node doesn't need an
    # if/else on investigation_type to describe what was investigated.
    target_desc = ", ".join(f"{k}={v}" for k, v in state["investigation_target"].items())
    findings_summary = "; ".join(state["findings"]) or "no findings recorded"

    if state["concluded_naturally"]:
        last_message = state["messages"][-1]
        final = _extract_text(last_message.content).strip() or "(Model concluded with no text content.)"
    elif state.get("reasoning_error"):
        final = (
            f"Investigation of {target_desc} could not be completed: {state['reasoning_error']}. "
            f"Findings gathered before the failure: {findings_summary}."
        )
    else:
        synthesized = _synthesize_cutoff_explanation(state, config)
        final = synthesized or (
            f"Investigation of {target_desc} stopped after reaching the limit of "
            f"{state['max_tool_calls']} tool calls before reaching a conclusion. "
            f"Findings gathered so far: {findings_summary}."
        )
    logger.info("%s investigation of %s finished (concluded_naturally=%s, tool_calls_used=%d): %s",
                state["investigation_type"], target_desc, state["concluded_naturally"],
                state["tool_call_count"], final[:300])
    return {"final_explanation": final}
