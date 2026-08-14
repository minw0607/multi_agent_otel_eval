"""
Customer-support multi-agent system, fully instrumented for token attribution.

    Supervisor (deterministic, 0 tokens) → Planner → Navigator → Validator

Differences from the Mind2Web MAS in `agents.py`:

  * Tools perform **real I/O** (`support_tools.py`), including real local writes.
  * Every LLM call is recorded individually (`LLMCallRecorder`), so re-planning,
    per-turn context growth, and hidden reasoning are observable.
  * Reasoning artifacts are **kept**, not discarded: the plan text and the
    validator's rationale are stored and attached to spans (spec R1).
  * Tool spans carry output size and argument/result fingerprints (spec T2),
    which is what makes duplicate retrieval detectable.

The Supervisor deliberately makes no LLM call. Its routing is a dictionary
lookup, so "orchestration" costs exactly zero tokens — an honest structural
finding rather than a gap (spec §8).
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from .config import Config
from .otel import (LLMCallRecorder, Usage, estimated_usage,
                   measure_tool_schema_tokens, tool_schema_tokens)
from .support_dataset import SupportTicket
from .support_tools import get_support_tools
from .tracer import (GEN_AI_ATTRIBUTES as A, SPAN_NAMES, HierarchicalTracer,
                     SpanKind, TracingManager)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """You are the PLANNING agent on a customer-support desk.

Decompose the ticket into 2-5 concrete steps for the Navigator to execute.
Each step MUST name the exact tool to use, from this list:
  search_kb, read_policy, read_order, read_ticket, web_search,
  draft_response, escalate

Format exactly:
PLAN:
1. TOOL: <tool_name> - <what it should find or do>
2. TOOL: <tool_name> - <what it should find or do>

Do not execute anything. Produce only the plan."""

NAVIGATOR_PROMPT = """You are the NAVIGATION agent on a customer-support desk.

Execute the provided plan using your tools. Ground every factual claim about an
order in an actual read_order lookup - never invent order details. Consult
search_kb for customer-facing guidance and read_policy for authoritative rules.

Finish by calling draft_response with the reply text and the KB article ids you
relied on. If a mandatory escalation trigger applies (third contact on the same
issue, legal action or chargeback threatened, remedy over $500, an exception to
published policy, a request for a manager, or a business/bulk order), call
escalate instead of promising a remedy."""

VALIDATOR_PROMPT = """You are the VALIDATION agent. Check the drafted reply against policy.

Assess whether the reply cites the correct guidance, states the correct policy,
makes no unsupported factual claims about the order, and escalates when a
mandatory trigger applies.

Respond in EXACTLY this format:
COMPLETION: [YES/PARTIAL/NO]
POLICY_CORRECT: [YES/PARTIAL/NO]
ESCALATION_CORRECT: [YES/NO/NA]
CONFIDENCE: [0.0-1.0]
REASONING: [2-3 sentences explaining the judgement]"""

_ROUTING = {"start": "planner", "planner": "navigator",
            "navigator": "validator", "validator": "FINISH"}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SupportRunResult:
    """Everything one ticket produced — telemetry, artifacts, and outcomes."""
    ticket_id: str
    trace_id: str = ""
    plan: str = ""
    navigator_output: str = ""
    validation: str = ""
    verdict: Dict[str, str] = field(default_factory=dict)
    reasoning_artifacts: Dict[str, str] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    recorders: Dict[str, LLMCallRecorder] = field(default_factory=dict)
    cited_articles: List[str] = field(default_factory=list)
    escalated: bool = False
    latency_ms: float = 0.0
    total_cost: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def all_calls(self) -> List:
        out = []
        for r in self.recorders.values():
            out.extend(r.calls)
        return out

    def total_usage(self) -> Usage:
        total = Usage(source="api")
        for r in self.recorders.values():
            total = total + r.total_usage()
        return total


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def create_support_mas(config: Config = None, allowlist: List[str] = None,
                       measure_schema: bool = True) -> Dict:
    """
    Build the support MAS. Each specialist gets its own model (spec §8).

    `measure_schema=True` spends two tiny calls to measure the real tool-schema
    cost instead of estimating it from JSON. Worth it: the JSON estimate runs
    ~1.4x high, and since schemas are the single largest input bucket that error
    otherwise dominates the residual.
    """
    cfg = config or Config
    tools = get_support_tools(allowlist)
    nav_llm = cfg.create_llm(role="agent", model=cfg.SUPPORT_NAVIGATOR_MODEL)
    schema_tokens = (measure_tool_schema_tokens(nav_llm, tools) if measure_schema
                     else tool_schema_tokens(tools))
    return {
        "planner":   cfg.create_llm(role="agent", model=cfg.SUPPORT_PLANNER_MODEL),
        "navigator": nav_llm,
        "validator": cfg.create_llm(role="judge", model=cfg.SUPPORT_VALIDATOR_MODEL),
        "navigator_agent": create_react_agent(nav_llm, tools, checkpointer=MemorySaver(),
                                              prompt=NAVIGATOR_PROMPT),
        "tools": tools,
        "tool_schema_tokens": schema_tokens,
        "models": {
            "supervisor": None,
            "planner":   cfg.SUPPORT_PLANNER_MODEL,
            "navigator": cfg.SUPPORT_NAVIGATOR_MODEL,
            "validator": cfg.SUPPORT_VALIDATOR_MODEL,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fingerprint(value) -> str:
    import hashlib
    norm = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha1(norm.strip().encode("utf-8")).hexdigest()[:12]


def _est_tokens(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.encoding_for_model("gpt-4").encode(text or ""))
    except Exception:
        return int(len((text or "").split()) * 1.3)


def _parse_verdict(text: str) -> Dict[str, str]:
    """Extract the validator's structured fields INCLUDING the rationale.

    The Mind2Web validator drops REASONING on the floor — it is generated,
    billed, and discarded. Here it is captured (spec R1).
    """
    out: Dict[str, str] = {}
    keys = ("COMPLETION", "POLICY_CORRECT", "ESCALATION_CORRECT", "CONFIDENCE", "REASONING")
    current = None
    for line in (text or "").splitlines():
        stripped = line.strip()
        matched = next((k for k in keys if stripped.upper().startswith(k)), None)
        if matched:
            current = matched
            out[matched] = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
        elif current == "REASONING" and stripped:
            out["REASONING"] = (out.get("REASONING", "") + " " + stripped).strip()
    return out


def _extract_tool_events(messages) -> List[Dict[str, Any]]:
    """
    Pair each tool call with its result so spans can carry both fingerprints
    and the output size that becomes input context on the next turn (spec T2).
    """
    results: Dict[str, str] = {}
    for m in messages:
        if type(m).__name__ == "ToolMessage":
            results[getattr(m, "tool_call_id", "")] = str(getattr(m, "content", ""))
    events = []
    for m in messages:
        for tc in (getattr(m, "tool_calls", None) or []):
            out = results.get(tc.get("id", ""), "")
            events.append({
                "tool": tc.get("name", "?"),
                "args": tc.get("args", {}),
                "output_chars": len(out),
                "output_tokens_est": _est_tokens(out),
                "args_fingerprint": _fingerprint(tc.get("args", {})),
                "result_fingerprint": _fingerprint(out) if out else "",
            })
    return events


def _cited_from_tool_calls(events: List[Dict]) -> List[str]:
    for e in events:
        if e["tool"] == "draft_response":
            raw = str(e["args"].get("cited_articles", ""))
            return [c.strip() for c in raw.split(",") if c.strip()]
    return []


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_support_mas(ticket: SupportTicket, mas: Dict, tracer: HierarchicalTracer,
                    tracing_manager: TracingManager = None,
                    config: Config = None) -> SupportRunResult:
    """
    Run one ticket through the support MAS with per-call instrumentation.

    Returns a SupportRunResult carrying telemetry (per-call records), reasoning
    artifacts (plan, rationale), and outcomes (cited articles, escalation).
    """
    cfg = config or Config
    models = mas["models"]
    started = time.time()
    res = SupportRunResult(ticket_id=ticket.id)
    trace = tracing_manager.start_trace(int(re.sub(r"\D", "", ticket.id) or 0)) if tracing_manager else None

    root = tracer.start_trace(SPAN_NAMES["TASK_ROOT"], attributes={
        "ticket.id": ticket.id, "ticket.difficulty": ticket.difficulty,
        "ticket.trap": ticket.trap or "none", "system.type": "support_mas",
    })
    res.trace_id = root.trace_id

    def _cost(model: str, u: Usage) -> float:
        if not u:
            return 0.0
        return ((u.input_tokens / 1e6) * cfg.get_cost_rate(model, "input") +
                (u.output_tokens / 1e6) * cfg.get_cost_rate(model, "output"))

    try:
        # ---- Supervisor: deterministic routing, zero tokens ----
        sup = tracer.start_span(SPAN_NAMES["AGENT_SUPERVISOR"], attributes={
            A["AGENT_NAME"]: "supervisor", A["AGENT_ROLE"]: "orchestrator",
            A["INPUT_TOKENS"]: 0, A["OUTPUT_TOKENS"]: 0,
            A["USAGE_SOURCE"]: "api", A["USAGE_TIER"]: "verified",
            "routing.deterministic": True,
        })
        tracer.end_span(sup, attributes={"supervisor.routed_to": _ROUTING["start"]})

        # ---- Planner (reasoning model) ----
        sp = tracer.start_span(SPAN_NAMES["AGENT_PLANNER"], attributes={
            A["AGENT_NAME"]: "planner", A["AGENT_ROLE"]: "task_decomposer",
            A["REQUEST_MODEL"]: models["planner"]})
        prec = LLMCallRecorder("planner", tracer, sp, models["planner"])
        res.recorders["planner"] = prec
        plan = mas["planner"].invoke(
            [SystemMessage(content=PLANNER_PROMPT),
             HumanMessage(content=ticket.as_prompt())],
            config={"callbacks": [prec]},
        ).content.strip()
        res.plan = plan
        p_usage = prec.total_usage()
        p_cost = _cost(models["planner"], p_usage)
        tracer.end_span(sp, attributes={
            **p_usage.to_span_attributes(),
            A["COST_USD"]: round(p_cost, 6),
            "plan.num_steps": len(re.findall(r"^\s*\d+\.", plan, re.M)),
            # Externalized reasoning kept as evidence, not discarded (R1).
            "reasoning.plan_text": plan[:2000],
            "reasoning.provenance": "externalized",
            "reasoning.hidden_tokens": p_usage.reasoning_tokens,
        })

        # ---- Navigator (tools, multi-turn) ----
        sn = tracer.start_span(SPAN_NAMES["AGENT_NAVIGATOR"], attributes={
            A["AGENT_NAME"]: "navigator", A["AGENT_ROLE"]: "tool_executor",
            A["REQUEST_MODEL"]: models["navigator"]})
        nrec = LLMCallRecorder("navigator", tracer, sn, models["navigator"],
                               tool_schema_tokens=mas["tool_schema_tokens"])
        res.recorders["navigator"] = nrec
        nav_input = (f"{ticket.as_prompt()}\n\nPlan from the Planner:\n{plan}\n\n"
                     f"Execute this plan.")
        nav_result = mas["navigator_agent"].invoke(
            {"messages": [HumanMessage(content=nav_input)]},
            config={"configurable": {"thread_id": f"support_{ticket.id}"},
                    "recursion_limit": 30, "callbacks": [nrec]},
        )
        msgs = nav_result["messages"]
        res.navigator_output = "\n".join(
            str(m.content) for m in msgs if getattr(m, "content", None))
        events = _extract_tool_events(msgs)
        res.tool_calls = events
        res.cited_articles = _cited_from_tool_calls(events)
        res.escalated = any(e["tool"] == "escalate" for e in events)

        # Tool spans carry cost-relevant size and fingerprints (T2).
        for i, e in enumerate(events):
            ts = tracer.start_span(SPAN_NAMES["TOOL_CALL"], kind=SpanKind.CLIENT.value,
                                   attributes={
                                       A["TOOL_NAME"]: e["tool"],
                                       "tool.index": i,
                                       "tool.output_chars": e["output_chars"],
                                       "tool.output_tokens_est": e["output_tokens_est"],
                                       "tool.args_fingerprint": e["args_fingerprint"],
                                       "tool.result_fingerprint": e["result_fingerprint"],
                                   }, parent=sn)
            tracer.end_span(ts)

        n_usage = nrec.total_usage()
        n_cost = _cost(models["navigator"], n_usage)
        tracer.end_span(sn, attributes={
            **n_usage.to_span_attributes(),
            A["COST_USD"]: round(n_cost, 6),
            "tools.count": len(events),
            "llm.calls": len(nrec.calls),
        })

        # ---- Validator ----
        sv = tracer.start_span(SPAN_NAMES["AGENT_VALIDATOR"], attributes={
            A["AGENT_NAME"]: "validator", A["AGENT_ROLE"]: "quality_checker",
            A["REQUEST_MODEL"]: models["validator"]})
        vrec = LLMCallRecorder("validator", tracer, sv, models["validator"])
        res.recorders["validator"] = vrec
        v_input = (f"{ticket.as_prompt()}\n\nPlan:\n{plan[:800]}\n\n"
                   f"Navigator output:\n{res.navigator_output[:2000]}\n\n"
                   f"Cited articles: {', '.join(res.cited_articles) or 'none'}\n"
                   f"Escalated: {res.escalated}")
        validation = mas["validator"].invoke(
            [SystemMessage(content=VALIDATOR_PROMPT), HumanMessage(content=v_input)],
            config={"callbacks": [vrec]},
        ).content.strip()
        res.validation = validation
        res.verdict = _parse_verdict(validation)
        v_usage = vrec.total_usage()
        v_cost = _cost(models["validator"], v_usage)
        tracer.end_span(sv, attributes={
            **v_usage.to_span_attributes(),
            A["COST_USD"]: round(v_cost, 6),
            "validation.completion": res.verdict.get("COMPLETION", "?"),
            "validation.policy_correct": res.verdict.get("POLICY_CORRECT", "?"),
            "reasoning.rationale": res.verdict.get("REASONING", "")[:1000],
            "reasoning.provenance": "externalized",
        })

        # ---- Roll up ----
        res.total_cost = p_cost + n_cost + v_cost
        res.latency_ms = (time.time() - started) * 1000
        res.reasoning_artifacts = {
            "plan": plan,
            "validator_rationale": res.verdict.get("REASONING", ""),
        }
        total = res.total_usage()
        root.set_attribute("task.total_tokens", total.total_tokens)
        root.set_attribute("task.total_cost_usd", round(res.total_cost, 6))
        root.set_attribute("task.hidden_reasoning_tokens", total.reasoning_tokens)
        root.set_attribute("task.llm_calls", len(res.all_calls))
        root.set_attribute("task.tool_calls", len(events))

        if trace is not None:
            trace.tool_calls = [{"tool": e["tool"], "args": e["args"]} for e in events]
            trace.agent_input_tokens = total.input_tokens
            trace.agent_output_tokens = total.output_tokens
            trace.reasoning_tokens = total.reasoning_tokens
            trace.cached_tokens = total.cached_tokens
            trace.total_cost = res.total_cost
            trace.tokens_source = total.source
            # reasoning_steps now holds actual reasoning (spec R1)
            trace.reasoning_steps = [plan, res.verdict.get("REASONING", "")]
            trace.agent_summaries = [
                f"planner: {p_usage.total_tokens} tok (hidden {p_usage.reasoning_tokens}) ${p_cost:.5f}",
                f"navigator: {len(nrec.calls)} calls, {len(events)} tools, {n_usage.total_tokens} tok ${n_cost:.5f}",
                f"validator: {v_usage.total_tokens} tok ${v_cost:.5f}",
            ]
            trace.finish()

    except Exception as e:
        res.errors.append(f"{type(e).__name__}: {e}")
        res.latency_ms = (time.time() - started) * 1000
        if trace is not None:
            trace.errors.append(res.errors[-1])
            trace.finish()
    finally:
        tracer.end_trace()

    return res
