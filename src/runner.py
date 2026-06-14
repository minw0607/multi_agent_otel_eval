"""
Batch evaluation runner.

`evaluate_batch` runs N Mind2Web tasks through either the single-agent baseline
or the multi-agent system, applies the full evaluation stack (task completion,
tool correctness, safety), logs cost/health, and returns a tidy DataFrame.

Keeping this loop in src/ lets the notebook run a full evaluation in one line
per architecture — so single-vs-multi comparison stays coding-light.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from .agents import Mind2WebTask, run_agent, run_multi_agent
from .config import Config
from .evaluator import HybridEvaluator, ToolCorrectnessEval
from .safety import SafetyValidator


def evaluate_batch(
    tasks: List[Dict],
    n: int,
    mode: str,                       # "single" or "multi"
    evaluator: HybridEvaluator,
    hier_tracer,
    tracing_manager,
    agent=None,                      # required for mode="single"
    multi_agents: Dict = None,       # required for mode="multi"
    cost_tracker=None,
    health_monitor=None,
    config: Config = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Run `n` tasks and return (results_df, raw_rows).

    mode="single" → baseline ReAct agent (pass `agent`)
    mode="multi"  → supervisor/planner/navigator/validator (pass `multi_agents`)
    """
    cfg = config or Config
    if mode == "single" and agent is None:
        raise ValueError("mode='single' requires agent=...")
    if mode == "multi" and multi_agents is None:
        raise ValueError("mode='multi' requires multi_agents=...")

    hier_tracer.reset()
    tracing_manager.reset()
    if health_monitor is not None:
        health_monitor.reset()

    label = "Single-Agent" if mode == "single" else "Multi-Agent"
    if verbose:
        print(f"{'='*64}\n{label}: running {n} tasks\n{'='*64}")

    rows: List[Dict] = []
    for i in range(n):
        task = Mind2WebTask.from_dict(tasks[i], idx=i)
        if verbose:
            print(f"[{i+1}/{n}] {task.website}: {task.confirmed_task[:55]}…")

        if mode == "single":
            output, trace = run_agent(task, agent, tracing_manager, cfg, hier_tracer)
        else:
            output, trace = run_multi_agent(task, multi_agents, hier_tracer, tracing_manager, cfg)

        safety = SafetyValidator.validate_all(output, task.confirmed_task)
        ev     = evaluator.evaluate(output, task, safety)
        tool   = ToolCorrectnessEval.evaluate(output, task.action_reprs, trace.tool_calls)

        if health_monitor is not None:
            health_monitor.log(i, ev.passed, ev.total_score, trace.latency_ms, len(trace.errors))
        if cost_tracker is not None:
            cost_tracker.log(i, cfg.AGENT_MODEL,
                             trace.agent_input_tokens, trace.agent_output_tokens, trace.total_cost)

        if verbose:
            status = "✅" if ev.passed else "❌"
            print(f"  {status} score={ev.total_score:.2f} tool_f1={tool.f1:.2f} "
                  f"tools={len(trace.tool_calls)} cost=${trace.total_cost:.4f} "
                  f"latency={trace.latency_ms:.0f}ms")

        rows.append({
            "task_id":        i,
            "website":        task.website,
            "system":         label,
            "task_score":     ev.total_score,
            "task_passed":    ev.passed,
            "rule_score":     ev.rule_score,
            "llm_score":      ev.llm_score,
            "tool_f1":        tool.f1,
            "tool_precision": tool.precision,
            "tool_recall":    tool.recall,
            "n_tool_calls":   len(trace.tool_calls),
            "latency_ms":     trace.latency_ms,
            "agent_tokens":   trace.agent_input_tokens + trace.agent_output_tokens,
            "judge_tokens":   trace.judge_input_tokens + trace.judge_output_tokens,
            "total_cost":     trace.total_cost,
            "tokens_source":  trace.tokens_source,
            "safety_passed":  SafetyValidator.is_safe(safety),
            "errors":         len(trace.errors),
        })

    if verbose:
        print(f"{label} done.\n")
    return pd.DataFrame(rows), rows
