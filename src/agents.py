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
              config: Config = None) -> Tuple[str, object]:
    """
    Run the baseline agent on one task and return (agent_output, trace).
    All token counting and cost tracking is done inside this function.
    """
    cfg   = config or Config
    trace = tracing_manager.start_trace(task.idx)

    task_input = (f"Task: {task.confirmed_task}\n"
                  f"Website: {task.website}\nDomain: {task.domain}\n\n"
                  "Complete this task using available tools.")

    try:
        run_config = {
            "configurable": {"thread_id": f"task_{task.idx}"},
            "recursion_limit": 50,
        }
        result      = agent.invoke({"messages": [HumanMessage(content=task_input)]}, config=run_config)
        agent_output = ""
        tool_calls   = []

        for msg in result["messages"]:
            if hasattr(msg, "content") and msg.content:
                agent_output += str(msg.content) + "\n"
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({"tool": tc.get("name", "?"), "args": tc.get("args", {})})
                    tracing_manager.log_tool_call(tc.get("name", "?"), tc.get("args", {}), "called")

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
        return agent_output, trace

    except Exception as e:
        trace.errors.append(f"{type(e).__name__}: {e}")
        trace.finish()
        return f"Error: {e}", trace


# ---------------------------------------------------------------------------
# Multi-agent (supervisor pattern)
# ---------------------------------------------------------------------------

_SUPERVISOR_PROMPT = """You are the supervisor of a 3-agent pipeline.
Given a web task and a plan from the Planner, route work to the Navigator,
then pass its output to the Validator.
Return a JSON decision: {"route": "planner"|"navigator"|"validator"|"done", "reason": "..."}
"""

_PLANNER_PROMPT = """You are the Planner. Given a web task, produce a numbered step-by-step
action plan (CLICK, TYPE, SELECT, SCROLL, SEARCH, FILTER, BOOK, PURCHASE, SUBMIT).
Be concise and specific. Return the plan as a numbered list."""

_NAVIGATOR_PROMPT = """You are the Navigator. Execute each step in the provided plan using tools.
Call the appropriate tool for each step and report what was accomplished."""

_VALIDATOR_PROMPT = """You are the Validator. Given the original task and Navigator output,
check whether the task was completed correctly. Return:
{"passed": true/false, "score": 0.0-1.0, "feedback": "..."}"""


def create_multi_agent(config: Config = None):
    """Return a dict of specialist LLMs for the supervisor pattern."""
    cfg = config or Config
    return {
        "supervisor": cfg.create_llm(role="agent"),
        "planner":    cfg.create_llm(role="agent"),
        "navigator":  cfg.create_llm(role="agent"),
        "validator":  cfg.create_llm(role="judge"),
    }


def run_multi_agent(task: Mind2WebTask, agents: Dict,
                    tracer: HierarchicalTracer,
                    tracing_manager: TracingManager,
                    config: Config = None) -> Tuple[str, object]:
    """
    Execute a task through the supervisor → planner → navigator → validator pipeline.
    Returns (final_output, trace).
    """
    cfg   = config or Config
    trace = tracing_manager.start_trace(task.idx)
    root  = tracer.start_trace(SPAN_NAMES["TASK_ROOT"],
                                attributes={"task_id": task.idx, "website": task.website})

    def _call(llm, system_msg: str, user_msg: str) -> str:
        resp = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
        return resp.content.strip()

    try:
        # 1. Planner
        sp = tracer.start_span(SPAN_NAMES["AGENT_PLANNER"])
        plan = _call(agents["planner"], _PLANNER_PROMPT,
                     f"Task: {task.confirmed_task}\nWebsite: {task.website}")
        tracer.end_span(sp, attributes={"plan_length": len(plan.split())})

        # 2. Navigator (with tools via baseline agent)
        nav_agent = create_react_agent(agents["navigator"], ALL_TOOLS,
                                       checkpointer=MemorySaver(),
                                       prompt=_NAVIGATOR_PROMPT)
        sn = tracer.start_span(SPAN_NAMES["AGENT_NAVIGATOR"])
        nav_input = f"Plan:\n{plan}\n\nTask: {task.confirmed_task}\nWebsite: {task.website}"
        nav_result = nav_agent.invoke(
            {"messages": [HumanMessage(content=nav_input)]},
            config={"configurable": {"thread_id": f"nav_{task.idx}"}},
        )
        nav_output = ""
        nav_tools  = []
        for msg in nav_result["messages"]:
            if hasattr(msg, "content") and msg.content:
                nav_output += str(msg.content) + "\n"
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    nav_tools.append({"tool": tc.get("name", "?"), "args": tc.get("args", {})})
        tracer.end_span(sn, attributes={"tool_calls": len(nav_tools)})

        # 3. Validator
        sv = tracer.start_span(SPAN_NAMES["AGENT_VALIDATOR"])
        validation = _call(agents["validator"], _VALIDATOR_PROMPT,
                           f"Task: {task.confirmed_task}\n\nNavigator output:\n{nav_output[:1000]}")
        tracer.end_span(sv)

        final_output = f"PLAN:\n{plan}\n\nNAVIGATOR:\n{nav_output}\n\nVALIDATION:\n{validation}"
        trace.tool_calls = nav_tools
        trace.finish()
        tracer.end_trace()
        return final_output, trace

    except Exception as e:
        trace.errors.append(f"{type(e).__name__}: {e}")
        trace.finish()
        tracer.end_trace()
        return f"Error: {e}", trace
