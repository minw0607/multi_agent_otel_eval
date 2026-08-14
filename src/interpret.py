"""
Reading the charts — deterministic rubrics first, LLM narrative second.

A chart that needs a human to explain it is only half a result. This module
turns each attribution visual into stated findings.

**Rubrics, not vibes.** Every finding below is produced by an explicit threshold
on a measured quantity, so the same numbers always yield the same reading and a
reviewer can check the rule rather than trust the prose. Thresholds are declared
as module constants precisely so they can be argued with.

An LLM narrative is available on top, but it is:
  * optional and off unless a judge is supplied,
  * given the **numbers**, never the image — it does not "look at" the chart,
  * always labelled as an AI Assessment, per the repo's existing discipline.

The rubric is the finding; the narrative is presentation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --- Rubric thresholds (declared so they can be challenged) ----------------
T_GROWTH_STEEP        = 3.0    # x, input growth first→last turn
T_GROWTH_MODERATE     = 1.8
T_RESENT_DOMINANT     = 0.50   # share of input that is re-sent context
T_RESENT_NOTABLE      = 0.30
T_SCHEMA_DOMINANT     = 0.25   # share of input spent re-declaring tools
T_SCHEMA_NOTABLE      = 0.15
T_HIDDEN_MAJORITY     = 0.50   # share of output that is unreadable
T_HIDDEN_NOTABLE      = 0.20
T_RESIDUAL_CONCERN    = 0.10   # |residual| / input
T_RESIDUAL_WATCH      = 0.05
T_DUP_CONCERN         = 0.20   # duplicate share of tool calls
T_STAGE_IMBALANCE     = 0.85   # one stage dominating total tokens
T_OVERHEAD_HIGH       = 0.30   # planning + verification share

SEVERITY_ORDER = {"info": 0, "watch": 1, "concern": 2}


@dataclass
class Finding:
    severity: str          # info | watch | concern
    text: str
    evidence: str = ""

    def __str__(self) -> str:
        icon = {"info": "•", "watch": "▲", "concern": "!"}[self.severity]
        return f"{icon} {self.text}" + (f"  [{self.evidence}]" if self.evidence else "")


@dataclass
class Interpretation:
    title: str
    headline: str
    findings: List[Finding] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    narrative: Optional[str] = None     # LLM-generated, clearly labelled

    @property
    def severity(self) -> str:
        return max((f.severity for f in self.findings),
                   key=lambda s: SEVERITY_ORDER[s], default="info")

    def show(self):
        print(f"{self.title}\n{'-' * len(self.title)}")
        print(self.headline + "\n")
        for f in sorted(self.findings, key=lambda x: -SEVERITY_ORDER[x.severity]):
            print(f"  {f}")
        if self.narrative:
            print("\n  🤖 AI Assessment (advisory; the findings above are rule-based)")
            for line in _wrap(self.narrative):
                print(f"     {line}")

    def _repr_markdown_(self):
        rows = "\n".join(f"- {f}" for f in
                         sorted(self.findings, key=lambda x: -SEVERITY_ORDER[x.severity]))
        md = f"**{self.title}** — {self.headline}\n\n{rows}"
        if self.narrative:
            md += (f"\n\n> 🤖 **AI Assessment** (advisory; findings above are rule-based)\n> "
                   + self.narrative.replace("\n", "\n> "))
        return md


def _wrap(text: str, width: int = 88) -> List[str]:
    import textwrap
    return textwrap.wrap(text, width=width)


# ---------------------------------------------------------------------------
# Optional LLM narrative
# ---------------------------------------------------------------------------

_NARRATIVE_SYSTEM = """You are a model-risk reviewer summarising telemetry from an AI agent run.

Write ONE paragraph, at most 90 words, in a neutral analytical register.

Rules:
- Use ONLY the numbers supplied. Never invent or extrapolate a figure.
- Explain what the pattern MEANS operationally and what could be done about it.
- Do not restate every number; interpret them.
- No markdown, no headings, no bullet points, no preamble.
- If the numbers are unremarkable, say so plainly rather than inflating them."""


def add_narrative(interp: Interpretation, judge_llm, extra_context: str = "") -> Interpretation:
    """
    Attach an LLM narrative to an interpretation.

    The judge receives the *measured numbers and rule-based findings* — not the
    image. It is summarising a table, not performing vision. Failures degrade to
    no narrative rather than a fabricated one.
    """
    if judge_llm is None:
        return interp
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        facts = "\n".join(f"- {k}: {v}" for k, v in interp.metrics.items())
        rules = "\n".join(f"- {f.text}" for f in interp.findings)
        prompt = (f"Chart: {interp.title}\n\nMeasured values:\n{facts}\n\n"
                  f"Rule-based findings:\n{rules}\n{extra_context}")
        resp = judge_llm.invoke([SystemMessage(content=_NARRATIVE_SYSTEM),
                                 HumanMessage(content=prompt)])
        interp.narrative = resp.content.strip()
    except Exception as e:
        interp.narrative = None
        interp.findings.append(Finding("info", f"AI narrative unavailable ({type(e).__name__})"))
    return interp


# ---------------------------------------------------------------------------
# Interpreters
# ---------------------------------------------------------------------------

def interpret_context_growth(source, judge_llm=None) -> Interpretation:
    """Read the per-turn context chart: what accumulates, and what it costs."""
    from .attribution import context_growth, tool_schema_overhead
    g = context_growth(source)
    ts = tool_schema_overhead(source)
    turns = [t for t in g["turns"] if t["agent"] == "navigator"] or g["turns"]
    n = len(turns)

    m = {
        "navigator turns": n,
        "input first turn": f"{g['first_input_tokens']:,} tokens",
        "input last turn": f"{g['last_input_tokens']:,} tokens",
        "growth factor": f"{g['growth_factor']:.1f}x",
        "re-sent share of input": f"{g['resent_share']:.0%}",
        "tool definitions per turn": f"{ts['tokens_per_turn']:,} tokens",
        "tool definitions share of input": f"{ts['share_of_input']:.0%}",
    }

    f: List[Finding] = []
    gf, rs, ss = g["growth_factor"], g["resent_share"], ts["share_of_input"]

    if gf >= T_GROWTH_STEEP:
        f.append(Finding("concern",
            f"Input context grew {gf:.1f}x across {n} turns — each turn is markedly "
            "more expensive than the last.",
            f"{g['first_input_tokens']:,} → {g['last_input_tokens']:,} tokens"))
    elif gf >= T_GROWTH_MODERATE:
        f.append(Finding("watch",
            f"Input context grew {gf:.1f}x across {n} turns.",
            f"{g['first_input_tokens']:,} → {g['last_input_tokens']:,} tokens"))
    else:
        f.append(Finding("info",
            f"Input context was broadly flat across {n} turns ({gf:.1f}x)."))

    if rs >= T_RESENT_DOMINANT:
        f.append(Finding("concern",
            f"{rs:.0%} of navigator input was re-sent context rather than new "
            "information — the majority of spend buys nothing new.",
            "history + prior tool output"))
    elif rs >= T_RESENT_NOTABLE:
        f.append(Finding("watch", f"{rs:.0%} of navigator input was re-sent context."))

    if ss >= T_SCHEMA_DOMINANT:
        f.append(Finding("concern",
            f"Tool definitions consumed {ss:.0%} of all input — "
            f"{ts['tokens_per_turn']:,} tokens re-declared every turn, of which only "
            f"{ts['distinct_tools_used']} tool(s) were actually used. Trimming the tool "
            "set for this task class is the single clearest saving available.",
            f"{ts['total_tokens']:,} tokens total"))
    elif ss >= T_SCHEMA_NOTABLE:
        f.append(Finding("watch",
            f"Tool definitions were {ss:.0%} of input ({ts['tokens_per_turn']:,}/turn)."))

    if n >= 4 and gf >= T_GROWTH_MODERATE:
        f.append(Finding("info",
            "Growth is superlinear in turn count: every turn resends all prior turns, "
            "so cost rises faster than the work performed. Capping turns or summarising "
            "history bounds it."))

    interp = Interpretation(
        title="Context growth per navigator turn",
        headline=(f"{gf:.1f}x input growth over {n} turns; {rs:.0%} of input was re-sent, "
                  f"{ss:.0%} was tool definitions."),
        findings=f, metrics=m)
    return add_narrative(interp, judge_llm) if judge_llm else interp


def interpret_hidden_reasoning(source, judge_llm=None) -> Interpretation:
    """Read the visible-vs-hidden output chart."""
    from .attribution import hidden_reasoning
    h = hidden_reasoning(source)
    share = h["hidden_share"]
    reasoning_agents = {a: d for a, d in h["by_agent"].items() if d["reasoning"] > 0}

    m = {"output tokens": f"{h['output_tokens']:,}",
         "hidden reasoning tokens": f"{h['hidden_tokens']:,}",
         "hidden share of output": f"{share:.0%}",
         "agents with hidden reasoning": ", ".join(reasoning_agents) or "none"}

    f: List[Finding] = []
    if share >= T_HIDDEN_MAJORITY:
        f.append(Finding("concern",
            f"{share:.0%} of all output tokens were reasoning that was billed but never "
            "returned — most of what you paid for on the output side is unreadable.",
            f"{h['hidden_tokens']:,} of {h['output_tokens']:,} tokens"))
    elif share >= T_HIDDEN_NOTABLE:
        f.append(Finding("watch",
            f"{share:.0%} of output tokens were hidden reasoning.",
            f"{h['hidden_tokens']:,} tokens"))
    elif share > 0:
        f.append(Finding("info", f"{share:.0%} of output was hidden reasoning."))
    else:
        f.append(Finding("info",
            "No hidden reasoning reported — no reasoning model was used, so all output "
            "tokens are readable."))

    for a, d in reasoning_agents.items():
        f.append(Finding("info",
            f"{a}: {d['hidden_share']:.0%} of its output was unreadable "
            f"({d['reasoning']:,} of {d['output']:,} tokens)."))

    if reasoning_agents:
        f.append(Finding("info",
            "This is a disclosure choice, not a technical limit: the provider generated "
            "this text and charged for it. Auditing whether that reasoning was sound is "
            "impossible from outside — only its size is knowable."))

    interp = Interpretation(
        title="Hidden reasoning — billed but unreadable",
        headline=(f"{h['hidden_tokens']:,} of {h['output_tokens']:,} output tokens "
                  f"({share:.0%}) were never returned to the caller."),
        findings=f, metrics=m)
    return add_narrative(interp, judge_llm) if judge_llm else interp


def interpret_tool_usage(source, judge_llm=None) -> Interpretation:
    """Read the tool-usage chart: retrieval volume and wasted duplicates."""
    from .attribution import duplicate_retrievals, retrieval_share, tool_schema_overhead
    d = duplicate_retrievals(source)
    r = retrieval_share(source)
    ts = tool_schema_overhead(source)
    dup_share = d["duplicate_calls"] / d["total_calls"] if d["total_calls"] else 0.0

    m = {"tool calls": d["total_calls"], "unique calls": d["unique_calls"],
         "duplicate calls": d["duplicate_calls"],
         "wasted tokens (est)": f"{d['wasted_tokens_est']:,}",
         "tool output share of input": f"{r['share_of_input']:.0%}",
         "distinct tools used": ts["distinct_tools_used"]}

    f: List[Finding] = []
    if dup_share >= T_DUP_CONCERN:
        f.append(Finding("concern",
            f"{d['duplicate_calls']} of {d['total_calls']} tool calls repeated an identical "
            f"request, costing about {d['wasted_tokens_est']:,} tokens to re-obtain "
            "information the agent already had. Caching results within a run removes this.",
            f"{dup_share:.0%} of calls"))
    elif d["duplicate_calls"]:
        f.append(Finding("watch",
            f"{d['duplicate_calls']} duplicate tool call(s) detected "
            f"(~{d['wasted_tokens_est']:,} tokens)."))
    else:
        f.append(Finding("info",
            "No duplicate retrievals — every tool call fetched something new."))

    if r["share_of_input"] >= 0.35:
        f.append(Finding("watch",
            f"Retrieved content was {r['share_of_input']:.0%} of all input. Retrieval "
            "dominates context, so tightening what each tool returns has more leverage "
            "than shortening prompts."))
    else:
        f.append(Finding("info",
            f"Retrieved content was {r['share_of_input']:.0%} of input."))

    interp = Interpretation(
        title="Tool usage and duplicate retrievals",
        headline=(f"{d['total_calls']} tool calls, {d['unique_calls']} unique; retrieved "
                  f"content was {r['share_of_input']:.0%} of all input."),
        findings=f, metrics=m)
    return add_narrative(interp, judge_llm) if judge_llm else interp


def interpret_stage_bars(source, per_task: List[Dict] = None,
                         judge_llm=None) -> Interpretation:
    """Read the per-ticket stage chart: where effort concentrates, and how evenly."""
    from .attribution import stage_breakdown, verification_share
    stages = {s["stage"]: s for s in stage_breakdown(source)}
    v = verification_share(source)
    total = sum(s["tokens"] for s in stages.values()) or 1
    nav = stages.get("Navigation & response", {}).get("tokens", 0) / total

    m = {stage: f"{s['tokens']:,} ({s['share']:.0%})" for stage, s in stages.items()}
    m["planning + verification overhead"] = f"{v['overhead_share']:.0%}"

    f: List[Finding] = []
    if nav >= T_STAGE_IMBALANCE:
        f.append(Finding("watch",
            f"Navigation accounts for {nav:.0%} of all tokens. Execution dominates, so "
            "optimisation effort belongs there rather than in planning or verification."))
    if v["overhead_share"] >= T_OVERHEAD_HIGH:
        f.append(Finding("concern",
            f"Planning and verification together consumed {v['overhead_share']:.0%} of "
            "tokens without executing the task. That overhead is justified only if it "
            "measurably improves outcomes."))
    else:
        f.append(Finding("info",
            f"Planning and verification overhead was {v['overhead_share']:.0%} of spend "
            f"(verification alone {v['verification_share']:.0%})."))

    if "Routing / orchestration" in stages and stages["Routing / orchestration"]["tokens"] == 0:
        f.append(Finding("info",
            "Orchestration cost zero tokens: routing is deterministic code, not an LLM "
            "call. An LLM-routed supervisor would add cost on every task."))

    if per_task:
        vals = [sum(t["stages"].values()) for t in per_task if t.get("stages")]
        if len(vals) > 1:
            lo, hi = min(vals), max(vals)
            spread = hi / lo if lo else 0
            labels = {sum(t["stages"].values()): t["label"] for t in per_task if t.get("stages")}
            m["cheapest / dearest ticket"] = f"{lo:,} / {hi:,} tokens"
            sev = "watch" if spread >= 2.5 else "info"
            f.append(Finding(sev,
                f"Per-ticket cost varied {spread:.1f}x ({labels.get(lo,'?')} {lo:,} → "
                f"{labels.get(hi,'?')} {hi:,} tokens). "
                + ("Wide variance means per-ticket budgeting, not an average, should drive "
                   "capacity planning." if spread >= 2.5 else "Cost is fairly predictable.")))

    interp = Interpretation(
        title="Tokens by workflow stage",
        headline=(f"Navigation {nav:.0%} of spend; planning + verification overhead "
                  f"{v['overhead_share']:.0%}."),
        findings=f, metrics=m)
    return add_narrative(interp, judge_llm) if judge_llm else interp


def interpret_attribution(source, judge_llm=None) -> Interpretation:
    """Read the headline attribution table, including the residual."""
    from .attribution import token_attribution, tool_schema_overhead
    a = token_attribution(source)
    ts = tool_schema_overhead(source)
    by_cat = {r.category: r for r in a["rows"]}
    top = max((r for r in a["rows"] if r.dimension == "INPUT" and r.method != "residual"),
              key=lambda r: r.tokens, default=None)

    m = {"total tokens": f"{a['total_tokens']:,}", "LLM calls": a["llm_calls"],
         "input / output": f"{a['input_tokens']:,} / {a['output_tokens']:,}",
         "residual": f"{a['residual_tokens']:+,} ({a['residual_share']:.1%} of input)",
         "hidden reasoning share of output": f"{a['hidden_share']:.0%}",
         "cached input tokens": f"{a['cached_tokens']:,}"}

    f: List[Finding] = []
    if top:
        f.append(Finding("info",
            f"Largest single category is {top.category.lower()} at {top.share:.0%} of all "
            f"tokens ({top.tokens:,}).") )

    out_share = a["output_tokens"] / a["total_tokens"] if a["total_tokens"] else 0
    f.append(Finding("info",
        f"Spend is input-dominated: {1-out_share:.0%} input vs {out_share:.0%} output. "
        "Since output is typically priced several times higher per token, the cost split "
        "is less lopsided than the token split."))

    rs = a["residual_share"]
    if rs >= T_RESIDUAL_CONCERN:
        f.append(Finding("concern",
            f"Residual is {rs:.1%} of input — the decomposition disagrees materially with "
            "the API's reported total, so category shares should be treated as approximate.",
            f"{a['residual_tokens']:+,} tokens"))
    elif rs >= T_RESIDUAL_WATCH:
        f.append(Finding("watch",
            f"Residual is {rs:.1%} of input — small but non-trivial.",
            f"{a['residual_tokens']:+,} tokens"))
    else:
        f.append(Finding("info",
            f"Residual is {rs:.1%} of input: the decomposition closely matches the reported "
            "total.", f"{a['residual_tokens']:+,} tokens"))

    if a["residual_tokens"] > 0:
        f.append(Finding("info",
            "The residual is positive, i.e. we under-account. Per-message framing overhead "
            "is charged by the provider but not represented in any message body, so a small "
            "positive residual is the expected resting state."))
    elif a["residual_tokens"] < 0:
        f.append(Finding("watch",
            "The residual is negative, i.e. we over-account. Some category is being "
            "over-estimated — tool definitions are the usual cause."))

    if a["cached_tokens"]:
        f.append(Finding("info",
            f"{a['cached_tokens']:,} input tokens were served from cache and billed at a "
            "different rate. Token counts are reproducible; the resulting cost is not."))

    interp = Interpretation(
        title="Token attribution",
        headline=(f"{a['total_tokens']:,} tokens across {a['llm_calls']} LLM calls; "
                  f"residual {a['residual_share']:.1%}."),
        findings=f, metrics=m)
    return add_narrative(interp, judge_llm) if judge_llm else interp
