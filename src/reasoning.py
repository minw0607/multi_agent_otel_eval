"""
Reasoning transparency — capturing reasoning, and checking it against reality.

Three layers get called "reasoning", and only one is genuinely inaccessible:

  latent computation      activations, forward passes    impossible for anyone
  hidden reasoning tokens real text, BILLED, withheld    size measurable only
  externalized reasoning  plans, ReAct steps, rationales fully capturable

**Architecture is a transparency choice.** A single reasoning-model call hides
its planning in the opaque layer; decomposing into Planner -> Navigator ->
Validator moves that same reasoning into inspectable inter-agent messages. The
multi-agent system costs more AND makes reasoning auditable — a dimension of the
single-vs-multi trade-off that quality and cost comparisons miss.

**The central caution (spec R4):** a reasoning narrative is a *claim about*
process, not process. Chain-of-thought is frequently unfaithful — models reach
answers via factors they do not mention, then rationalize. Fluent reasoning is
strong evidence of a well-trained writer, not of a sound process.

So this module evaluates reasoning by its **consequences and its consistency
with the trace**, not by reading it. `plan_execution_divergence` is the
strongest tool here precisely because it is deterministic: it compares what the
Planner said it would do against what the Navigator actually did.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .attribution import _attrs, _name, _num, llm_call_spans, normalize_spans, tool_spans

KNOWN_TOOLS = [
    "search_kb", "read_policy", "read_order", "read_ticket", "web_search",
    "draft_response", "escalate",
    # Mind2Web tool set, so the same check runs on both corpora
    "site_navigation", "site_search", "filter_content", "get_page_info",
    "check_availability", "get_price_info", "book_reservation",
    "make_purchase", "submit_form", "budget_calculator",
]

# Verbs a planner may use instead of naming a tool outright.
VERB_TO_TOOL = {
    "search": "search_kb", "look up": "search_kb", "find": "search_kb",
    "policy": "read_policy", "verify": "read_policy",
    "order": "read_order", "lookup order": "read_order",
    "draft": "draft_response", "reply": "draft_response", "respond": "draft_response",
    "escalate": "escalate",
}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

REASONING_PROVENANCE = {
    "externalized": "Verified — the full text was produced as output and captured.",
    "summarized":   "Asserted — a model-generated paraphrase of reasoning, not the reasoning.",
    "hidden":       "Unavailable — generated and billed, but never returned to the caller.",
}


@dataclass
class ReasoningInventory:
    """What reasoning exists for a run, and how much of it we can actually see."""
    externalized_chars: int = 0
    externalized_tokens_est: int = 0
    hidden_tokens: int = 0
    visible_output_tokens: int = 0
    artifacts: Dict[str, str] = field(default_factory=dict)

    @property
    def hidden_share(self) -> float:
        total = self.hidden_tokens + self.visible_output_tokens
        return self.hidden_tokens / total if total else 0.0

    def as_rows(self) -> List[Dict[str, Any]]:
        return [
            {"layer": "Externalized (plans, rationales)", "tokens": self.externalized_tokens_est,
             "provenance": "externalized", "auditable": "yes",
             "note": REASONING_PROVENANCE["externalized"]},
            {"layer": "Hidden reasoning tokens", "tokens": self.hidden_tokens,
             "provenance": "hidden", "auditable": "no (billed)",
             "note": REASONING_PROVENANCE["hidden"]},
            {"layer": "Latent computation", "tokens": None,
             "provenance": "n/a", "auditable": "impossible",
             "note": "Not observable by any API consumer, including the vendor's."},
        ]


def reasoning_inventory(source, artifacts: Dict[str, str] = None) -> ReasoningInventory:
    """Inventory reasoning by layer, with provenance labels (spec R3)."""
    spans = normalize_spans(source)
    calls = llm_call_spans(spans)
    hidden = sum(_num(_attrs(s), "gen_ai.usage.reasoning_tokens") for s in calls)
    out = sum(_num(_attrs(s), "gen_ai.usage.output_tokens") for s in calls)

    artifacts = dict(artifacts or {})
    if not artifacts:
        for s in spans:
            a = _attrs(s)
            if a.get("reasoning.plan_text"):
                artifacts["plan"] = a["reasoning.plan_text"]
            if a.get("reasoning.rationale"):
                artifacts["validator_rationale"] = a["reasoning.rationale"]

    text = " ".join(str(v) for v in artifacts.values())
    try:
        import tiktoken
        est = len(tiktoken.encoding_for_model("gpt-4").encode(text))
    except Exception:
        est = int(len(text.split()) * 1.3)

    return ReasoningInventory(
        externalized_chars=len(text), externalized_tokens_est=est,
        hidden_tokens=hidden, visible_output_tokens=max(out - hidden, 0),
        artifacts=artifacts,
    )


# ---------------------------------------------------------------------------
# Plan-execution divergence (spec R2) — the deterministic faithfulness check
# ---------------------------------------------------------------------------

@dataclass
class DivergenceResult:
    planned: List[str]
    executed: List[str]
    followed: List[str]        # planned and actually used
    skipped: List[str]         # planned but never used
    unplanned: List[str]       # used but never planned
    fidelity: float            # |followed| / |planned|
    order_match: float         # LCS of planned vs executed order
    verdict: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "planned": self.planned, "executed": self.executed,
            "followed": self.followed, "skipped": self.skipped,
            "unplanned": self.unplanned, "fidelity": round(self.fidelity, 3),
            "order_match": round(self.order_match, 3), "verdict": self.verdict,
        }

    def summary(self) -> str:
        return (f"plan named {len(self.planned)} tool(s); navigator used "
                f"{len(self.followed)} of them"
                + (f" + {len(self.unplanned)} unplanned" if self.unplanned else "")
                + (f"; skipped {', '.join(self.skipped)}" if self.skipped else ""))


def extract_planned_tools(plan_text: str) -> List[str]:
    """
    Pull the tool sequence out of a plan.

    Prefers explicit `TOOL: name` markers (which the planner prompt requests),
    falls back to bare tool names, then to action verbs. Consecutive duplicates
    are collapsed, since a plan naming the same tool twice in a row is one step.
    """
    text = plan_text or ""
    found: List[str] = []

    explicit = re.findall(r"TOOL:\s*([a-z_]+)", text, re.I)
    if explicit:
        found = [t.lower() for t in explicit if t.lower() in KNOWN_TOOLS]

    if not found:
        for line in text.splitlines():
            low = line.lower()
            hit = next((t for t in KNOWN_TOOLS if t in low), None)
            if hit is None:
                hit = next((tool for verb, tool in VERB_TO_TOOL.items() if verb in low), None)
            if hit:
                found.append(hit)

    collapsed: List[str] = []
    for t in found:
        if not collapsed or collapsed[-1] != t:
            collapsed.append(t)
    return collapsed


def extract_executed_tools(source) -> List[str]:
    """Actual tool sequence from the trace (spans), in call order."""
    spans = tool_spans(normalize_spans(source))
    indexed = []
    for s in spans:
        a = _attrs(s)
        indexed.append((_num(a, "tool.index"), a.get("gen_ai.tool.name", "?")))
    indexed.sort(key=lambda x: x[0])
    return [name for _, name in indexed]


def _lcs(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def plan_execution_divergence(plan_text: str, source=None,
                              executed: List[str] = None) -> DivergenceResult:
    """
    Compare the Planner's stated steps against the Navigator's actual tool calls.

    This is a **faithfulness check on stated-vs-actual**, and it needs no judge:
    the plan is a testable claim, and the trace is the ground truth about what
    happened. It is uniquely enabled by the multi-agent architecture — a single
    agent that plans internally leaves nothing to compare against.

    A low fidelity score means the plan did not drive execution, which makes the
    planning tokens suspect regardless of how good the plan reads.
    """
    planned = extract_planned_tools(plan_text)
    if executed is None:
        executed = extract_executed_tools(source)

    exec_collapsed: List[str] = []
    for t in executed:
        if not exec_collapsed or exec_collapsed[-1] != t:
            exec_collapsed.append(t)

    pset, eset = set(planned), set(executed)
    followed = [t for t in planned if t in eset]
    skipped = [t for t in planned if t not in eset]
    unplanned = [t for t in dict.fromkeys(executed) if t not in pset]

    fidelity = len(followed) / len(planned) if planned else 0.0
    order_match = (_lcs(planned, exec_collapsed) / len(planned)) if planned else 0.0

    if not planned:
        verdict = "no plan detected — cannot assess faithfulness"
    elif fidelity == 1.0 and not unplanned:
        verdict = "faithful — executed exactly the planned tools"
    elif fidelity >= 0.7:
        verdict = "mostly faithful — minor deviation"
    elif fidelity >= 0.4:
        verdict = "partial divergence — plan only loosely followed"
    else:
        verdict = "diverged — the plan did not drive execution"

    return DivergenceResult(planned, executed, followed, skipped, unplanned,
                            fidelity, order_match, verdict)


# ---------------------------------------------------------------------------
# Consistency + correlation checks
# ---------------------------------------------------------------------------

def verdict_consistency(verdict: Dict[str, str]) -> Dict[str, Any]:
    """
    Does the validator's rationale agree with its own structured verdict?

    Cheap, deterministic, and catches a real failure mode: a confident score
    attached to a rationale that says the opposite.
    """
    completion = (verdict.get("COMPLETION") or "").upper()
    reasoning = (verdict.get("REASONING") or "").lower()
    try:
        conf = float(re.findall(r"[\d.]+", verdict.get("CONFIDENCE", "") or "0")[0])
    except (IndexError, ValueError):
        conf = 0.0

    negative = any(w in reasoning for w in
                   ("did not", "failed", "incorrect", "missing", "wrong", "should have"))
    positive = any(w in reasoning for w in
                   ("correct", "appropriate", "accurate", "properly", "cited the right"))

    issues = []
    if completion == "YES" and negative and not positive:
        issues.append("verdict says YES but the rationale describes failures")
    if completion == "NO" and positive and not negative:
        issues.append("verdict says NO but the rationale is positive")
    if conf >= 0.9 and negative:
        issues.append("high confidence stated alongside a rationale describing problems")
    if not reasoning:
        issues.append("no rationale captured — the verdict is unexplained")

    return {"consistent": not issues, "issues": issues,
            "completion": completion, "confidence": conf}


def reasoning_outcome_correlation(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Do runs that spend more on reasoning actually do better?

    Correlational, not causal — but a flat or negative relationship is strong
    evidence that the reasoning spend is decorative. Establishing causality
    requires ablation (remove the plan, rerun), which is deliberately out of
    v1 scope because it costs a full re-run per condition.

    Args:
        records: dicts with `reasoning_tokens` (or `planning_tokens`) and a
                 numeric `score` or boolean `passed`.
    """
    from .attribution import _spearman
    pairs = []
    for r in records:
        spend = r.get("reasoning_tokens", r.get("planning_tokens", 0))
        outcome = r.get("score", 1.0 if r.get("passed") else 0.0)
        pairs.append((spend, outcome))
    if len(pairs) < 3:
        return {"correlation": None, "n": len(pairs),
                "interpretation": "insufficient data (need at least 3 runs)"}
    c = _spearman(pairs)
    if c > 0.4:
        interp = "more reasoning spend tracks better outcomes — plausibly load-bearing"
    elif c < -0.1:
        interp = "more reasoning tracks WORSE outcomes — likely struggling, not thinking"
    else:
        interp = "reasoning spend is uncorrelated with outcome — likely decorative"
    return {"correlation": round(c, 3), "n": len(pairs), "interpretation": interp}


# ---------------------------------------------------------------------------
# Roll-up
# ---------------------------------------------------------------------------

def reasoning_report(source, plan_text: str = "", verdict: Dict[str, str] = None,
                     artifacts: Dict[str, str] = None) -> Dict[str, Any]:
    """Everything about reasoning for one run, with provenance throughout."""
    inv = reasoning_inventory(source, artifacts)
    div = plan_execution_divergence(plan_text or inv.artifacts.get("plan", ""), source)
    cons = verdict_consistency(verdict or {})
    return {
        "inventory": inv,
        "layers": inv.as_rows(),
        "divergence": div,
        "consistency": cons,
        "headline": (
            f"{inv.hidden_share:.0%} of output tokens were reasoning you cannot read; "
            f"{div.summary()}"
        ),
    }
