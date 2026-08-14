"""
Token attribution — where the tokens actually went.

This is the deliverable the transparency thesis rests on. It turns per-call
spans into an auditable breakdown, and it obeys the spec invariants:

  I1  Total = SUM over LLM calls of (input + output). Every row is a
      DECOMPOSITION of that total, never an addition to it. Tool outputs are a
      slice of input context, not an extra row — they are already inside the
      next call's input_tokens.
  I2  Every number carries how it was obtained: api | estimated | residual.
  I3  Every number carries how far it can be trusted:
        verified  - we counted it ourselves from content we assembled
        trusted   - the vendor reported it; we cannot audit it
        asserted  - reported but uninspectable (reasoning tokens)
  I4  The residual is reported, never hidden. A large residual means the
      decomposition is wrong, and saying so is the point.
  I5  Cost is an estimate even when tokens are exact (cache state varies).

Functions accept either live `OTelSpan` objects or spans parsed from exported
OTLP JSON, so the same analysis runs on the support desk and on Mind2Web — the
cross-check that shows the instrument is not tuned to data we authored.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

# Context buckets, in the order they should be presented.
CONTEXT_BUCKETS = [
    ("system_tokens",      "System prompts"),
    ("tool_schema_tokens", "Tool definitions"),
    ("task_tokens",        "Task / ticket"),
    ("tool_output_tokens", "Retrieved knowledge (tool output)"),
    ("history_tokens",     "Conversation history"),
]

STAGE_BY_AGENT = {
    "supervisor": "Routing / orchestration",
    "planner":    "Planning",
    "navigator":  "Navigation & response",
    "validator":  "Verification",
}


# ---------------------------------------------------------------------------
# Span normalisation — accept live spans or exported OTLP
# ---------------------------------------------------------------------------

def _attrs(span) -> Dict[str, Any]:
    """Return a span's attributes as a plain dict, from either representation."""
    a = getattr(span, "attributes", None)
    if isinstance(a, dict):
        return a
    if isinstance(span, dict):
        raw = span.get("attributes", {})
        if isinstance(raw, dict):
            return raw
        out = {}
        for item in raw or []:              # OTLP list-of-KV form
            v = item.get("value", {})
            out[item.get("key")] = next(iter(v.values())) if v else None
        return out
    return {}


def _name(span) -> str:
    return getattr(span, "name", None) or (span.get("name", "") if isinstance(span, dict) else "")


def _num(attrs: Dict, key: str, default=0):
    v = attrs.get(key, default)
    try:
        return type(default)(v)
    except (TypeError, ValueError):
        return default


def normalize_spans(source) -> List[Any]:
    """
    Accept a HierarchicalTracer, a trace-id-keyed dict, a list of spans, or an
    exported OTLP payload, and return a flat list of spans.
    """
    if source is None:
        return []
    if hasattr(source, "traces"):                       # HierarchicalTracer
        return [s for spans in source.traces.values() for s in spans]
    if isinstance(source, dict) and "resourceSpans" in source:
        out = []
        for rs in source["resourceSpans"]:
            for ss in rs.get("scopeSpans", []):
                out.extend(ss.get("spans", []))
        return out
    if isinstance(source, dict):                        # {trace_id: [spans]}
        return [s for spans in source.values() for s in spans]
    return list(source)


def llm_call_spans(spans: Iterable) -> List:
    return [s for s in spans if _name(s) == "llm.chat.completion"]


def tool_spans(spans: Iterable) -> List:
    return [s for s in spans if _name(s) == "tool.execute"]


# ---------------------------------------------------------------------------
# Headline: token attribution table
# ---------------------------------------------------------------------------

@dataclass
class AttributionRow:
    dimension: str      # INPUT | OUTPUT
    category: str
    tokens: int
    share: float
    method: str         # api | estimated | residual
    tier: str           # verified | trusted | asserted
    calls: Optional[int] = None


def token_attribution(source) -> Dict[str, Any]:
    """
    The headline artifact: a full decomposition of every token consumed.

    Input tokens decompose into context categories plus a residual; output
    tokens decompose into visible text and hidden reasoning. Together they sum
    to exactly the measured total (I1) — no row is added on top.
    """
    spans = normalize_spans(source)
    calls = llm_call_spans(spans)

    total_in = total_out = total_reasoning = total_cached = 0
    buckets = defaultdict(int)
    est_calls = 0
    for s in calls:
        a = _attrs(s)
        total_in        += _num(a, "gen_ai.usage.input_tokens")
        total_out       += _num(a, "gen_ai.usage.output_tokens")
        total_reasoning += _num(a, "gen_ai.usage.reasoning_tokens")
        total_cached    += _num(a, "gen_ai.usage.cached_tokens")
        if a.get("gen_ai.usage.source") == "estimated":
            est_calls += 1
        for key, _ in CONTEXT_BUCKETS:
            buckets[key] += _num(a, f"context.{key}")

    total = total_in + total_out
    accounted = sum(buckets.values())
    residual = total_in - accounted

    def _row(dim, cat, tok, method, tier, n=None):
        return AttributionRow(dim, cat, tok, (tok / total if total else 0.0), method, tier, n)

    rows: List[AttributionRow] = []
    for key, label in CONTEXT_BUCKETS:
        n_calls = None
        if key == "tool_output_tokens":
            n_calls = len(tool_spans(spans))
        rows.append(_row("INPUT", label, buckets[key], "estimated", "verified", n_calls))
    # I4: the residual is a row, not a rounding adjustment.
    rows.append(_row("INPUT", "Unaccounted (residual)", residual, "residual", "—"))

    visible = max(total_out - total_reasoning, 0)
    rows.append(_row("OUTPUT", "Visible response", visible, "api", "trusted", len(calls)))
    rows.append(_row("OUTPUT", "Hidden reasoning", total_reasoning, "api", "asserted"))

    return {
        "rows": rows,
        "total_tokens": total,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "reasoning_tokens": total_reasoning,
        "cached_tokens": total_cached,
        "llm_calls": len(calls),
        "accounted_tokens": accounted,
        "residual_tokens": residual,
        "residual_share": abs(residual) / total_in if total_in else 0.0,
        "estimated_calls": est_calls,
        "hidden_share": total_reasoning / total_out if total_out else 0.0,
    }


def stage_breakdown(source) -> List[Dict[str, Any]]:
    """
    The same total, cut by workflow stage instead of token role.

    An orthogonal view of `token_attribution` — both sum to 100% of the measured
    total, and neither is added to the other.
    """
    spans = normalize_spans(source)
    calls = llm_call_spans(spans)
    by_agent: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"tokens": 0, "reasoning": 0, "calls": 0})
    for s in calls:
        a = _attrs(s)
        agent = a.get("gen_ai.agent.name", "unknown")
        d = by_agent[agent]
        d["tokens"] += _num(a, "gen_ai.usage.input_tokens") + _num(a, "gen_ai.usage.output_tokens")
        d["reasoning"] += _num(a, "gen_ai.usage.reasoning_tokens")
        d["calls"] += 1

    # Supervisor makes no LLM call by construction — report the zero explicitly
    # rather than omitting the row (spec §8).
    if "supervisor" not in by_agent and any(
            _attrs(s).get("gen_ai.agent.name") == "supervisor" for s in spans):
        by_agent["supervisor"] = {"tokens": 0, "reasoning": 0, "calls": 0}

    total = sum(d["tokens"] for d in by_agent.values())
    order = ["supervisor", "planner", "navigator", "validator"]
    out = []
    for agent in sorted(by_agent, key=lambda x: order.index(x) if x in order else 99):
        d = by_agent[agent]
        out.append({
            "stage": STAGE_BY_AGENT.get(agent, agent),
            "agent": agent,
            "tokens": d["tokens"],
            "share": d["tokens"] / total if total else 0.0,
            "calls": d["calls"],
            "hidden_reasoning": d["reasoning"],
            "method": "api" if d["calls"] else "verified",
            "tier": "trusted" if d["calls"] else "verified",
        })
    return out


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def context_growth(source) -> Dict[str, Any]:
    """
    How input context grows across turns.

    Every ReAct turn resends all prior turns, so history accumulates roughly
    quadratically. This is usually the largest single driver in long runs and
    the line item a reader will not predict.
    """
    spans = normalize_spans(source)
    calls = sorted(llm_call_spans(spans),
                   key=lambda s: (_attrs(s).get("gen_ai.agent.name", ""),
                                  _num(_attrs(s), "gen_ai.call.index")))
    turns = []
    for s in calls:
        a = _attrs(s)
        turns.append({
            "agent": a.get("gen_ai.agent.name", "?"),
            "index": _num(a, "gen_ai.call.index"),
            "input_tokens": _num(a, "gen_ai.usage.input_tokens"),
            "history_tokens": _num(a, "context.history_tokens"),
            "tool_output_tokens": _num(a, "context.tool_output_tokens"),
            "tool_schema_tokens": _num(a, "context.tool_schema_tokens"),
        })
    nav = [t for t in turns if t["agent"] == "navigator"] or turns
    first = nav[0]["input_tokens"] if nav else 0
    last = nav[-1]["input_tokens"] if nav else 0
    resent = sum(t["history_tokens"] + t["tool_output_tokens"] for t in nav)
    total_in = sum(t["input_tokens"] for t in nav) or 1
    return {
        "turns": turns,
        "first_input_tokens": first,
        "last_input_tokens": last,
        "growth_factor": (last / first) if first else 0.0,
        "resent_tokens": resent,
        "resent_share": resent / total_in,
    }


def tool_schema_overhead(source) -> Dict[str, Any]:
    """
    Cost of re-declaring tools the agent may never call.

    Tool definitions are resent on every turn. Measured at 1,104 tokens for the
    11 Mind2Web tools against a 77-token system prompt — a 14x ratio invisible
    to standard accounting, and directly actionable (trim the tool set).
    """
    spans = normalize_spans(source)
    calls = llm_call_spans(spans)
    per_turn = max((_num(_attrs(s), "context.tool_schema_tokens") for s in calls), default=0)
    turns_with_tools = sum(1 for s in calls if _num(_attrs(s), "context.tool_schema_tokens") > 0)
    total_schema = sum(_num(_attrs(s), "context.tool_schema_tokens") for s in calls)
    total_in = sum(_num(_attrs(s), "gen_ai.usage.input_tokens") for s in calls) or 1
    used = {_attrs(s).get("gen_ai.tool.name") for s in tool_spans(spans)}
    return {
        "tokens_per_turn": per_turn,
        "turns_charged": turns_with_tools,
        "total_tokens": total_schema,
        "share_of_input": total_schema / total_in,
        "distinct_tools_used": len([u for u in used if u]),
    }


def duplicate_retrievals(source) -> Dict[str, Any]:
    """
    Repeated identical tool calls, found by argument fingerprint.

    Costs nothing to detect and answers a question standard accounting cannot:
    did the agent pay twice for the same information?
    """
    spans = tool_spans(normalize_spans(source))
    seen = Counter()
    wasted = 0
    detail = []
    for s in spans:
        a = _attrs(s)
        key = (a.get("gen_ai.tool.name"), a.get("tool.args_fingerprint"))
        seen[key] += 1
        if seen[key] > 1:
            wasted += _num(a, "tool.output_tokens_est")
            detail.append({"tool": key[0], "fingerprint": key[1], "occurrence": seen[key]})
    dupes = {k: v for k, v in seen.items() if v > 1}
    return {
        "total_calls": len(spans),
        "unique_calls": len(seen),
        "duplicate_calls": sum(v - 1 for v in dupes.values()),
        "duplicate_groups": len(dupes),
        "wasted_tokens_est": wasted,
        "detail": detail,
    }


def replanning_count(source) -> Dict[str, Any]:
    """
    How many times each agent invoked its model.

    A planner with >1 call re-planned. A navigator's call count is its turn
    count — distinct from its tool-call count, because models can emit several
    tool calls in a single turn.
    """
    spans = normalize_spans(source)
    per_agent = Counter(_attrs(s).get("gen_ai.agent.name", "?") for s in llm_call_spans(spans))
    n_tools = len(tool_spans(spans))
    nav_turns = per_agent.get("navigator", 0)
    return {
        "calls_per_agent": dict(per_agent),
        "planner_calls": per_agent.get("planner", 0),
        "replans": max(per_agent.get("planner", 0) - 1, 0),
        "navigator_turns": nav_turns,
        "tool_calls": n_tools,
        "tools_per_turn": (n_tools / nav_turns) if nav_turns else 0.0,
    }


def verification_share(source) -> Dict[str, Any]:
    """What fraction of spend went to checking the work rather than doing it."""
    stages = {s["agent"]: s for s in stage_breakdown(source)}
    total = sum(s["tokens"] for s in stages.values()) or 1
    v = stages.get("validator", {}).get("tokens", 0)
    p = stages.get("planner", {}).get("tokens", 0)
    return {
        "verification_tokens": v,
        "verification_share": v / total,
        "planning_tokens": p,
        "planning_share": p / total,
        "overhead_share": (v + p) / total,   # not executing the task
    }


def retrieval_share(source) -> Dict[str, Any]:
    """Is retrieved knowledge dominating input context?"""
    spans = normalize_spans(source)
    calls = llm_call_spans(spans)
    tool_out = sum(_num(_attrs(s), "context.tool_output_tokens") for s in calls)
    total_in = sum(_num(_attrs(s), "gen_ai.usage.input_tokens") for s in calls) or 1
    return {
        "tool_output_tokens": tool_out,
        "share_of_input": tool_out / total_in,
        "retrieval_calls": sum(1 for s in tool_spans(spans)
                               if _attrs(s).get("gen_ai.tool.name") in
                               ("search_kb", "read_policy", "read_order", "web_search")),
    }


def hidden_reasoning(source) -> Dict[str, Any]:
    """
    Output tokens billed but never returned to the caller.

    The sharpest transparency exhibit: this is not a technical limit, it is a
    disclosure choice. The provider generated the text and charged for it.
    """
    spans = normalize_spans(source)
    calls = llm_call_spans(spans)
    by_agent = defaultdict(lambda: {"output": 0, "reasoning": 0})
    for s in calls:
        a = _attrs(s)
        d = by_agent[a.get("gen_ai.agent.name", "?")]
        d["output"] += _num(a, "gen_ai.usage.output_tokens")
        d["reasoning"] += _num(a, "gen_ai.usage.reasoning_tokens")
    total_out = sum(d["output"] for d in by_agent.values())
    total_hidden = sum(d["reasoning"] for d in by_agent.values())
    return {
        "hidden_tokens": total_hidden,
        "output_tokens": total_out,
        "hidden_share": total_hidden / total_out if total_out else 0.0,
        "by_agent": {k: {**v, "hidden_share": (v["reasoning"] / v["output"]) if v["output"] else 0.0}
                     for k, v in by_agent.items()},
    }


def reasoning_vs_complexity(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Was reasoning proportional to task difficulty?

    Requires a difficulty-graded dataset, which the support corpus provides by
    construction. A flat or negative relationship means the reasoning is
    decorative rather than load-bearing.

    Args:
        records: dicts with keys `difficulty`, `reasoning_tokens`, `total_tokens`,
                 and optionally `passed`.
    """
    by_diff = defaultdict(list)
    for r in records:
        by_diff[r.get("difficulty", "unknown")].append(r)

    order = {"easy": 0, "medium": 1, "hard": 2}
    summary = {}
    for diff in sorted(by_diff, key=lambda d: order.get(d, 99)):
        rs = by_diff[diff]
        n = len(rs)
        summary[diff] = {
            "n": n,
            "avg_reasoning_tokens": sum(r.get("reasoning_tokens", 0) for r in rs) / n,
            "avg_total_tokens": sum(r.get("total_tokens", 0) for r in rs) / n,
            "pass_rate": (sum(1 for r in rs if r.get("passed")) / n) if any("passed" in r for r in rs) else None,
        }

    # Rank correlation between difficulty and reasoning spend.
    pairs = [(order.get(r.get("difficulty"), 1), r.get("reasoning_tokens", 0)) for r in records
             if r.get("difficulty") in order]
    corr = _spearman(pairs) if len(pairs) > 2 else None
    return {
        "by_difficulty": summary,
        "difficulty_reasoning_correlation": corr,
        "interpretation": _interpret_corr(corr),
    }


def _spearman(pairs) -> float:
    def rank(vals):
        order_idx = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        for pos, i in enumerate(order_idx):
            r[i] = pos + 1
        return r
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    rx, ry = rank(xs), rank(ys)
    n = len(pairs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def _interpret_corr(c: Optional[float]) -> str:
    if c is None:
        return "insufficient data"
    if c > 0.4:
        return "reasoning scales with task difficulty — plausibly load-bearing"
    if c < -0.1:
        return "reasoning is inversely related to difficulty — investigate"
    return "reasoning is roughly flat across difficulty — possibly decorative"


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

def attribution_dataframe(source):
    """The headline table as a pandas DataFrame, ready to display."""
    import pandas as pd
    att = token_attribution(source)
    df = pd.DataFrame([{
        "Dimension": r.dimension, "Category": r.category, "Tokens": r.tokens,
        "Share": f"{r.share:.1%}", "Calls": r.calls if r.calls is not None else "—",
        "Method": r.method, "Tier": r.tier,
    } for r in att["rows"]])
    total = att["total_tokens"]
    df.loc[len(df)] = ["TOTAL", "All tokens", total, "100.0%", att["llm_calls"], "api", "trusted"]
    return df


def summarize(source) -> Dict[str, Any]:
    """Everything at once — convenient for the notebook and the report."""
    return {
        "attribution":  token_attribution(source),
        "stages":       stage_breakdown(source),
        "context":      context_growth(source),
        "tool_schema":  tool_schema_overhead(source),
        "duplicates":   duplicate_retrievals(source),
        "replanning":   replanning_count(source),
        "verification": verification_share(source),
        "retrieval":    retrieval_share(source),
        "hidden":       hidden_reasoning(source),
    }
