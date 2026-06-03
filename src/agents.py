"""
Agent factory for single-agent and multi-agent (supervisor) patterns.

create_baseline_agent()  → single ReAct agent + judge LLM
create_multi_agent()     → supervisor + planner + navigator + validator
run_agent()              → execute one Mind2Web task with full tracing
run_multi_agent()        → execute via multi-agent pipeline
"""

import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from .config import Config
from .tools import ALL_TOOLS
from .tracer import TracingManager, HierarchicalTracer, SPAN_NAMES, SpanStatus


AGENT_SYSTEM_PROMPT = """You are an AI agent completing real-world web navigation tasks.

TOOL SELECTION:
  site_navigation    → clicking buttons, links, typing, selecting dropdowns
  site_search        → searching within a specific website
  web_search         → broad web research
  filter_content     → sort or filter results (price, rating, category)
  get_page_info      → check current page content or state
  check_availability → verify if an item, room, or slot is available
  get_price_info     → look up prices
  book_reservation   → book restaurants, hotels, flights, or cars
  make_purchase      → complete a purchase or checkout
  submit_form        → submit forms or applications
  budget_calculator  → calculate costs and budgets

STRATEGY:
1. Start with site_search or web_search to locate the right page
2. Use site_navigation for sequential click/type/select actions
3. Use filter_content for any sort or filter operations
4. Complete the task fully — do not stop at partial results

OUTPUT FORMAT:
**Final Plan:**
1) TOOL: [tool_name]([args]) - [what it accomplishes]
...

**Final Answer:**
[2–3 sentence summary of what was accomplished]
"""


@dataclass
class Mind2WebTask:
    """Represents one Mind2Web evaluation task."""
    idx: int
    annotation_id: str
    website: str
    domain: str
    confirmed_task: str
    action_reprs: List[str]

    @classmethod
    def from_dict(cls, example: Dict, idx: int) -> "Mind2WebTask":
        return cls(
            idx=idx,
            annotation_id=example.get("annotation_id", ""),
            website=example.get("website", ""),
            domain=example.get("domain", ""),
            confirmed_task=example.get("confirmed_task", ""),
            action_reprs=example.get("action_reprs", []),
        )


def create_baseline_agent(config: Config = None):
    """Return (agent, judge_llm) using settings from Config."""
    cfg = config or Config
    agent_llm = cfg.create_llm(role="agent")
    judge_llm = cfg.create_llm(role="judge")
    memory    = MemorySaver()
    agent     = create_react_agent(agent_llm, ALL_TOOLS, checkpointer=memory,
                                   prompt=AGENT_SYSTEM_PROMPT)
    return agent, judge_llm


def run_agent(task: Mind2WebTask, agent, tracing_manager: TracingManager,
              config: Config = None, hier_tracer: HierarchicalTracer = None) -> Tuple[str, object]:
    """
    Run the baseline agent on one task and return (agent_output, trace).
    All token counting and cost tracking is done inside this function.

    If `hier_tracer` is provided, an OpenTelemetry span tree is also emitted
    (task.execute → agent.react.execute → tool.execute per tool call), so the
    baseline path produces the same observability artifacts as the multi-agent path.
    """
    cfg   = config or Config
    trace = tracing_manager.start_trace(task.idx)

    task_input = (f"Task: {task.confirmed_task}\n"
                  f"Website: {task.website}\nDomain: {task.domain}\n\n"
                  "Complete this task using available tools.")

    root = None
    if hier_tracer is not None:
        root = hier_tracer.start_trace(
            SPAN_NAMES["TASK_ROOT"],
            attributes={"task_id": task.idx, "website": task.website,
                        "gen_ai.agent.name": "baseline_react"},
        )

    try:
        run_config = {
            "configurable": {"thread_id": f"task_{task.idx}"},
            "recursion_limit": 50,
        }
        agent_span = hier_tracer.start_span("agent.react.execute") if hier_tracer else None
        result      = agent.invoke({"messages": [HumanMessage(content=task_input)]}, config=run_config)
        agent_output = ""
        tool_calls   = []

        for msg in result["messages"]:
            if hasattr(msg, "content") and msg.content:
                agent_output += str(msg.content) + "\n"
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "?")
                    tool_calls.append({"tool": name, "args": tc.get("args", {})})
                    tracing_manager.log_tool_call(name, tc.get("args", {}), "called")
                    if hier_tracer is not None:
                        ts = hier_tracer.start_span(SPAN_NAMES["TOOL_CALL"],
                                                    attributes={"gen_ai.tool.name": name})
                        hier_tracer.end_span(ts)

        if hier_tracer is not None:
            hier_tracer.end_span(agent_span, attributes={"tool_calls": len(tool_calls)})

        # Token counting
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-4")
            in_tok  = len(enc.encode(AGENT_SYSTEM_PROMPT + task_input))
            out_tok = len(enc.encode(agent_output))
        except Exception:
            in_tok  = int(len((AGENT_SYSTEM_PROMPT + task_input).split()) * 1.3)
            out_tok = int(len(agent_output.split()) * 1.3)

        agent_model = cfg.AGENT_MODEL
        trace.agent_input_tokens  = in_tok
        trace.agent_output_tokens = out_tok
        trace.judge_input_tokens  = out_tok + 500   # estimate for judge call
        trace.judge_output_tokens = 150
        trace.agent_cost = (
            (in_tok  / 1_000_000) * cfg.get_cost_rate(agent_model, "input") +
            (out_tok / 1_000_000) * cfg.get_cost_rate(agent_model, "output")
        )
        judge_model = cfg.JUDGE_MODEL
        trace.judge_cost = (
            (trace.judge_input_tokens  / 1_000_000) * cfg.get_cost_rate(judge_model, "input") +
            (trace.judge_output_tokens / 1_000_000) * cfg.get_cost_rate(judge_model, "output")
        )
        trace.total_cost = trace.agent_cost + trace.judge_cost
        trace.tool_calls = tool_calls
        trace.finish()
        if hier_tracer is not None:
            root.set_attribute("gen_ai.usage.input_tokens", in_tok)
            root.set_attribute("gen_ai.usage.output_tokens", out_tok)
            root.set_attribute("gen_ai.usage.cost_usd", round(trace.total_cost, 6))
            hier_tracer.end_trace()
        return agent_output, trace

    except Exception as e:
        trace.errors.append(f"{type(e).__name__}: {e}")
        trace.finish()
        if hier_tracer is not None:
            hier_tracer.end_trace()
        return f"Error: {e}", trace


# ===========================================================================
# Multi-Agent System (Supervisor pattern)
#
#   Supervisor → Planner → Navigator → Validator
#
# Each specialist has a distinct role, its own model, and its own OpenTelemetry
# span carrying GenAI Semantic Convention attributes. Per-agent token usage and
# cost are tracked and rolled up into the task trace.
# ===========================================================================

_PLANNER_PROMPT = """You are a PLANNING agent. Decompose a web-navigation task into
concrete, executable steps (3–5 max). Each step starts with an action verb
(NAVIGATE, SEARCH, CLICK, FILTER, EXTRACT, BOOK, PURCHASE, VERIFY), is specific,
and builds on the previous one.

Format:
PLAN:
1. [ACTION]: [specific step]
2. [ACTION]: [specific step]
...

Do NOT execute — only produce the plan. The Navigator will execute it."""

_NAVIGATOR_PROMPT = """You are a NAVIGATION agent. Execute the provided plan using the
available tools (site_navigation, site_search, web_search, filter_content,
get_page_info, check_availability, get_price_info, book_reservation, make_purchase,
submit_form, budget_calculator). Follow the plan, call tools as needed, and describe
what you accomplished. End with a short Final Answer."""

_VALIDATOR_PROMPT = """You are a VALIDATION agent assessing an AI agent's performance in a
SANDBOX (not production). READ ops may return real or mock data; WRITE ops return
simulated confirmations — mock responses are expected and acceptable. Judge PROCESS
and REASONING quality, not whether real-world actions succeeded.

Respond in EXACTLY this format:
COMPLETION: [YES/PARTIAL/NO]
TOOL_USAGE: [YES/PARTIAL/NO]
QUALITY: [HIGH/MEDIUM/LOW]
CONFIDENCE: [0.0-1.0]
REASONING: [1-2 sentences]"""

# Sequential routing the Supervisor follows.
_ROUTING = {"start": "planner", "planner": "navigator",
            "navigator": "validator", "validator": "FINISH"}


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.encoding_for_model("gpt-4").encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


def create_multi_agent(config: Config = None) -> Dict:
    """
    Build the multi-agent system: a dict of specialist LLMs (each with its own
    model from Config) plus a tool-enabled Navigator ReAct agent.
    """
    cfg = config or Config
    navigator_llm = cfg.create_llm(role="agent", model=cfg.NAVIGATOR_MODEL)
    return {
        "supervisor":      cfg.create_llm(role="agent", model=cfg.SUPERVISOR_MODEL),
        "planner":         cfg.create_llm(role="agent", model=cfg.PLANNER_MODEL),
        "navigator":       navigator_llm,
        "validator":       cfg.create_llm(role="judge", model=cfg.VALIDATOR_MODEL),
        "navigator_agent": create_react_agent(navigator_llm, ALL_TOOLS,
                                              checkpointer=MemorySaver(),
                                              prompt=_NAVIGATOR_PROMPT),
        "models": {
            "supervisor": cfg.SUPERVISOR_MODEL, "planner": cfg.PLANNER_MODEL,
            "navigator":  cfg.NAVIGATOR_MODEL,  "validator": cfg.VALIDATOR_MODEL,
        },
    }


def run_multi_agent(task: Mind2WebTask, agents: Dict,
                    tracer: HierarchicalTracer,
                    tracing_manager: TracingManager,
                    config: Config = None) -> Tuple[str, object]:
    """
    Run a task through Supervisor → Planner → Navigator → Validator with full
    hierarchical OTel tracing and per-agent cost tracking.

    Returns (final_output, trace). `trace.total_cost` is the summed cost of all
    specialists; `trace.decision_points` holds the per-agent cost/token breakdown.
    """
    cfg    = config or Config
    models = agents["models"]
    trace  = tracing_manager.start_trace(task.idx)
    root   = tracer.start_trace(
        SPAN_NAMES["TASK_ROOT"],
        attributes={"task_id": task.idx, "website": task.website,
                    "domain": task.domain, "system.type": "multi_agent"},
    )

    def _cost(model, in_tok, out_tok):
        return ((in_tok / 1e6) * cfg.get_cost_rate(model, "input") +
                (out_tok / 1e6) * cfg.get_cost_rate(model, "output"))

    per_agent: Dict[str, Dict] = {}

    try:
        # ---- Supervisor: route start → planner ----
        sup = tracer.start_span(SPAN_NAMES["AGENT_SUPERVISOR"],
                                attributes={"gen_ai.agent.name": "supervisor",
                                            "gen_ai.agent.role": "orchestrator"})
        step = _ROUTING["start"]
        tracer.end_span(sup, attributes={"supervisor.routed_to": step})

        # ---- Planner: pure reasoning, no tools ----
        sp = tracer.start_span(SPAN_NAMES["AGENT_PLANNER"])
        p_user = f"Task: {task.confirmed_task}\nWebsite: {task.website}\nDomain: {task.domain}"
        plan = agents["planner"].invoke(
            [SystemMessage(content=_PLANNER_PROMPT), HumanMessage(content=p_user)]
        ).content.strip()
        p_in, p_out = _count_tokens(_PLANNER_PROMPT + p_user), _count_tokens(plan)
        p_cost = _cost(models["planner"], p_in, p_out)
        per_agent["planner"] = {"cost": p_cost, "tokens": p_in + p_out}
        tracer.end_span(sp, attributes={
            "gen_ai.agent.name": "planner", "gen_ai.agent.role": "task_decomposer",
            "gen_ai.request.model": models["planner"],
            "gen_ai.usage.input_tokens": p_in, "gen_ai.usage.output_tokens": p_out,
            "gen_ai.usage.cost_usd": round(p_cost, 6), "plan.num_steps": plan.count("\n") + 1,
        })

        # ---- Navigator: executes plan with tools ----
        sn = tracer.start_span(SPAN_NAMES["AGENT_NAVIGATOR"])
        n_user = f"Task: {task.confirmed_task}\nWebsite: {task.website}\n\nPlan:\n{plan}\n\nExecute it."
        nav_result = agents["navigator_agent"].invoke(
            {"messages": [HumanMessage(content=n_user)]},
            config={"configurable": {"thread_id": f"nav_{task.idx}"}, "recursion_limit": 50},
        )
        nav_output, nav_tools = "", []
        for msg in nav_result["messages"]:
            if hasattr(msg, "content") and msg.content:
                nav_output += str(msg.content) + "\n"
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "?")
                    nav_tools.append({"tool": name, "args": tc.get("args", {})})
                    ts = tracer.start_span(SPAN_NAMES["TOOL_CALL"],
                                           attributes={"gen_ai.tool.name": name})
                    tracer.end_span(ts)
        n_in, n_out = _count_tokens(_NAVIGATOR_PROMPT + n_user), _count_tokens(nav_output)
        n_cost = _cost(models["navigator"], n_in, n_out)
        per_agent["navigator"] = {"cost": n_cost, "tokens": n_in + n_out}
        tracer.end_span(sn, attributes={
            "gen_ai.agent.name": "navigator", "gen_ai.agent.role": "tool_executor",
            "gen_ai.request.model": models["navigator"],
            "gen_ai.usage.input_tokens": n_in, "gen_ai.usage.output_tokens": n_out,
            "gen_ai.usage.cost_usd": round(n_cost, 6), "tools.count": len(nav_tools),
        })

        # ---- Validator: structured quality check ----
        sv = tracer.start_span(SPAN_NAMES["AGENT_VALIDATOR"])
        v_user = (f"Task: {task.confirmed_task}\n\nPlan:\n{plan[:500]}\n\n"
                  f"Navigation Output:\n{nav_output[:1500]}\n\nProvide your assessment.")
        validation = agents["validator"].invoke(
            [SystemMessage(content=_VALIDATOR_PROMPT), HumanMessage(content=v_user)]
        ).content.strip()
        verdict = {}
        for line in validation.split("\n"):
            for key in ("COMPLETION", "TOOL_USAGE", "QUALITY", "CONFIDENCE"):
                if line.strip().upper().startswith(key):
                    verdict[key] = line.split(":", 1)[1].strip()
        v_in, v_out = _count_tokens(_VALIDATOR_PROMPT + v_user), _count_tokens(validation)
        v_cost = _cost(models["validator"], v_in, v_out)
        per_agent["validator"] = {"cost": v_cost, "tokens": v_in + v_out}
        tracer.end_span(sv, attributes={
            "gen_ai.agent.name": "validator", "gen_ai.agent.role": "quality_checker",
            "gen_ai.request.model": models["validator"],
            "gen_ai.usage.input_tokens": v_in, "gen_ai.usage.output_tokens": v_out,
            "gen_ai.usage.cost_usd": round(v_cost, 6),
            "validation.completion": verdict.get("COMPLETION", "?"),
            "validation.quality": verdict.get("QUALITY", "?"),
        })

        # ---- Roll up ----
        total_cost   = sum(a["cost"] for a in per_agent.values())
        total_tokens = sum(a["tokens"] for a in per_agent.values())
        trace.tool_calls          = nav_tools
        trace.agent_input_tokens  = p_in + n_in
        trace.agent_output_tokens = p_out + n_out
        trace.judge_input_tokens  = v_in
        trace.judge_output_tokens = v_out
        trace.agent_cost = per_agent["planner"]["cost"] + per_agent["navigator"]["cost"]
        trace.judge_cost = per_agent["validator"]["cost"]
        trace.total_cost = total_cost
        trace.decision_points = [{"agent": k, **v} for k, v in per_agent.items()]
        trace.reasoning_steps = [
            f"Planner: {p_in + p_out} tok / ${p_cost:.5f}",
            f"Navigator: {len(nav_tools)} tools, {n_in + n_out} tok / ${n_cost:.5f}",
            f"Validator: {verdict.get('COMPLETION','?')}/{verdict.get('QUALITY','?')}, ${v_cost:.5f}",
        ]
        trace.finish()

        root.set_attribute("task.total_cost_usd", round(total_cost, 6))
        root.set_attribute("task.total_tokens", total_tokens)
        root.set_attribute("task.agents_invoked", 3)
        tracer.end_trace()

        final_output = (f"[PLAN]\n{plan}\n\n[EXECUTION]\n{nav_output}\n\n"
                        f"[VALIDATION]\n{validation}")
        return final_output, trace

    except Exception as e:
        trace.errors.append(f"{type(e).__name__}: {e}")
        trace.finish()
        tracer.end_trace()
        return f"Error: {e}", trace
