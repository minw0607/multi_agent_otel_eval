# Token & Reasoning Transparency — Implementation Spec (v2)

**Supersedes** `token_attribution_spec.md` (v1). Scope widened from token attribution to
**token *and* reasoning transparency**, with a concrete use case, real tools, and a
model assignment chosen to demonstrate the thesis rather than merely describe it.

**Goal:** move this repo from token *accounting* (how many tokens were consumed) to
token *attribution* (where they went, why, and whether that was proportionate) — and
extend the same discipline to reasoning.

Background: [The Token Transparency Gap](https://minwu-ai.github.io/the-token-transparency-gap-why-agentic-ai-still-hides-where-computation-goes/).
That post argues APIs should report a per-stage breakdown. This repo already emits
per-stage spans, so it can **demonstrate the proposal rather than argue for it.** An
existence proof is a stronger claim than a recommendation.

---

## 0. What changed from v1

| # | Amendment | Why |
|---|---|---|
| 1 | **Single-total invariant** — every table row is a *decomposition*, never an addition | v1's headline table double-counted tool outputs (they are already inside the next call's `input_tokens`) |
| 2 | **Tracer decision stated**: build on the local `HierarchicalTracer` | v1 was silent; OpenInference already emits per-call spans, but only in the optional Phoenix path, which doesn't feed the report |
| 3 | **Context decomposition is tractable** via typed messages | v1 (and my first review) overstated the difficulty for the navigator |
| 4 | **Supervisor is structurally zero tokens** | Deterministic `_ROUTING` lookup — an honest finding, not a gap |
| 5 | **Span-parenting concurrency hazard** flagged | Callback-driven spans make the shared `active_spans` stack unsafe |
| 6 | **Verified / trusted / asserted** epistemic tiers | "We own the stack" ≠ omniscience; we still take the vendor's word on usage |
| 7 | **Reasoning transparency** added (§10) | Reasoning deserves the same provenance discipline as tokens |
| 8 | Probe results replace assumptions (§2) | `reasoning_tokens` and `cached_tokens` confirmed available |
| 9 | Use case, real tools, data strategy, model assignment | v1 had no concrete scenario |

---

## 1. Thesis

Three claims, in order of strength:

1. **Here is the ceiling.** What is knowable when you control every layer except the model
   itself — demonstrated with a working artifact.
2. **Most of the gap is withheld, not impossible.** Per-call counts and reasoning/cache
   details are already returned by the API. A vendor not showing them is making a
   *disclosure choice*, not hitting a technical limit.
3. **The delta between (1) and what commercial agents disclose is the transparency gap** —
   quantified rather than asserted.

### The disclosure ladder

| Level | Why *we* can get it | Could a vendor disclose it? | Typically do they? |
|---|---|---|---|
| Per-call token counts | API already returns it | **Yes, trivially** | Rarely |
| Reasoning / cached tokens | Provider reports it | **Yes** — pure passthrough | Almost never |
| Context composition | We assembled the prompt | Yes *in aggregate* (shares without content) | No |
| Inside the model | — | **No — genuinely impossible** | n/a |

Only the last row is a technical limit. Conceding the legitimate objection to row 3
(exposing prompt engineering) strengthens the argument for rows 1–2.

---

## 2. What the capability probe established (measured, 2026-08)

Run against the project's own Azure deployment. **These are facts, not assumptions.**

**Hidden reasoning is real and measurable:**

| Model | Output tokens | Reasoning (hidden) | Visible | Hidden share |
|---|---|---|---|---|
| `o3-mini-20250131-gs` | 494 | 384 | 110 | **78%** |
| `o4-mini-20250416-gs` | 291 | 128 | 163 | **44%** |
| `gpt-5-4-20260305-gs` | — | 0 | all | 0% |

→ You are billed for 44–78% of output tokens you cannot read. Non-reasoning models
correctly report `0`, so the field is safe to read unconditionally.

**Cache economics — and cost non-reproducibility:**

```
identical 2,098-token prompt, sent twice
call 1:  input=2098   cached=0
call 2:  input=2098   cached=1920      (91% cache read)
```

→ Identical tokens, materially different cost, determined by state we do not control.
**Attribution by token is reproducible; attribution by dollar is an estimate.**

**Implementation facts confirmed:**

- o-series requires `max_completion_tokens` and rejects `temperature` → `Config.create_llm`
  needs an o-series branch.
- Two independent read paths work: `usage_metadata.output_token_details.reasoning` and
  `response_metadata.token_usage.completion_tokens_details.reasoning_tokens`. Read
  normalized, fall back to raw.
- Bonus: the Azure gateway returns `latency_checkpoint` (`engine_ttft_ms`,
  `service_ttft_ms`, `total_duration_ms`) — separates model time from gateway overhead.
  No standard OTel attribute captures this; record it.

---

## 3. Core invariants

Non-negotiable rules that keep the output auditable. Violating any of these produces a
table that looks authoritative and isn't.

### I1 — Single-total invariant

> **Total = Σ over LLM calls of (input_tokens + output_tokens).**
> Every reported row is a *decomposition of that total*, never an addition to it.

Tool outputs are a **slice of input context**, not an independent row. A projected
"this retrieval will cost N downstream" figure is a *separate diagnostic*, never summed
into the stage table.

### I2 — Provenance on every number

Every figure carries **how it was obtained**: `api` | `estimated` | `derived` | `residual`.

### I3 — Epistemic tier on every number

| Tier | Example | Meaning |
|---|---|---|
| **Verified** | Context composition, tool schemas, history | We can independently count it — we built the prompt |
| **Trusted** | `input_tokens` / `output_tokens` | We accept the vendor's number; we cannot audit it |
| **Asserted** | `reasoning_tokens` | Reported but uninspectable — only its claimed size is known |

We can verify the context we constructed; we take the vendor's word on what it cost.
Saying so is what makes this more credible than vendor reporting, not less.

### I4 — Residual is reported, never hidden

```
context.accounted_tokens = Σ(system, tool_schema, task, plan, tool_output, history)
context.residual_tokens  = reported_input − accounted
```

A large residual means the decomposition is wrong. Surfacing it is the honesty mechanism.
Expect a few percent from tokenizer mismatch and per-message framing overhead.

### I5 — Cost is presented with lower confidence than tokens

Per §2. Never render `cost_usd` with the same authority as a token count.

---

## 4. Scope: two notebooks, one `src/`

| | `agentic_otel_demo_notebook.ipynb` (existing) | `token_transparency_notebook.ipynb` (new) |
|---|---|---|
| Domain | Mind2Web web navigation | Customer-support desk |
| Tools | Hybrid real/mock, WRITE mocked | **Real reads, real local writes** |
| Evaluation | Trajectory (proxy) | **Trajectory + outcome** |
| Role | Framework demo | **Transparency highlight** |
| Changes | 13 steps intact; + capability probe cell | New |

Both consume the same `src/`. All existing modules are **extended, never replaced.**

---

## 5. Use case: customer-support desk triage

Chosen because it does three jobs at once: makes tools genuinely real, creates ground
truth (enabling outcome evaluation), and generates retrieval-heavy traffic that makes
attribution metrics non-trivial.

**Workflow:** ticket arrives → Planner decomposes → Navigator retrieves KB/policy/order
data and drafts a response → Validator checks it against policy → draft written to disk
or escalated.

---

## 6. Real tools & the sandbox contract

| Tool | Real action | Effect class |
|---|---|---|
| `search_kb(query)` | Real BM25/keyword retrieval over local KB corpus | read, local |
| `read_ticket(id)` / `read_order(id)` | Real read from local system-of-record | read, local |
| `read_policy(topic)` | Real read of policy docs | read, local |
| `web_search(query)` | Real Tavily (if key present) | read, **external** |
| `draft_response(ticket_id, text)` | **Real file write** → `workspace/drafts/` | **write, local** |
| `escalate(ticket_id, reason)` | **Real append** → `workspace/escalations.jsonl` | **write, local** |

**Sandbox contract — enforced in code, not by convention:**

1. All writes confined to a `workspace/` root; **path-traversal guard** (`Path.resolve()`
   then assert prefix) on every write.
2. **No network writes.** External access is read-only.
3. Explicit per-run tool allowlist.
4. `SafetyValidator` runs on **inputs to write-tools**, not only on outputs.
5. `workspace/` is gitignored and reset per run.

This is a meaningful upgrade over the Mind2Web notebook: **real reads, real local
writes, zero external mutations.**

---

## 7. Data strategy: synthetic instrument + public anchor

**Synthetic corpus = the measuring instrument.** Like a resolution chart for a camera
lens — used because its properties are *known*, which is what makes the metrics
falsifiable rather than merely produced.

Planted properties, each tied to a metric it validates:

| Planted property | Validates |
|---|---|
| Tickets answerable only via 2+ KB articles | `context_growth`, history accumulation |
| Bait for re-fetching the same article | `duplicate_retrievals` (known correct answer) |
| One ticket where the obvious article is **wrong** | verification value, `plan_execution_divergence` |
| **Difficulty grading (easy/medium/hard)** | **`reasoning_vs_complexity`** — optional in v1, now core |

Size: ~18 KB articles, ~15 labeled tickets. Ground truth per ticket: correct KB
article(s), escalate y/n, policy cited. Committed to the repo.

**Public anchor = external validity.** Mind2Web (already in notebook 1). Because
attribution is post-hoc on traces, **the same `attribution.py` runs on both** — if the
metrics behave sensibly on a public benchmark *and* on our corpus, the instrument is
demonstrably not tuned to data we authored.

Future scope, not now: **τ-bench** (Sierra) for leaderboard comparability.

---

## 8. Model assignment

| Agent | Model | Rationale |
|---|---|---|
| Supervisor | *none* — deterministic routing | **0 tokens.** Report as a structural finding |
| **Planner** | **`o4-mini-20250416-gs`** | Reasoning model → emits `reasoning_tokens` we count but cannot read. The exhibit |
| Navigator | `gpt-5-4-20260305-gs` | Strong tool use; continuity with existing runs |
| Validator | `gpt-4-1-20250414-gs` | Different family → self-evaluation bias reduction |

**Controlled experiment (own notebook section):** run the Planner as `o4-mini` vs
`gpt-5-4-mini` on identical tickets; compare hidden-reasoning share, total cost, plan
quality, and downstream navigator turns. Tests *"is the reasoning you're billed for
buying anything?"* — the thesis reduced to a measurement.

---

## 9. Token attribution tasks

### T1 — Per-LLM-call spans *(prerequisite for everything)*

Use the already-defined-but-unused `SPAN_NAMES["LLM_CALL"]`. Every model invocation gets
a child span under its agent's span, carrying `gen_ai.usage.*`, model id, and a sequence
index.

For the Navigator, attach a **custom callback handler** firing per call
(`on_llm_end`), not `UsageMetadataCallbackHandler`, which aggregates.

**⚠️ Concurrency hazard:** `HierarchicalTracer.active_spans` is a shared mutable stack
using "last opened" as parent. Callback-driven spans may fire off-thread. Fix with
`contextvars` or by passing explicit `parent_span_id` to `start_span`. **Do this in T1,
not later** — retrofitting corrupted parenting is worse than preventing it.

**Acceptance:** a task produces ≥3 `llm.chat.completion` spans whose token counts sum to
the parent agent spans' totals.

### T2 — Tool spans carry their cost

```
tool.output_chars          int
tool.output_tokens_est     int    # PROJECTED downstream cost — diagnostic only (I1)
tool.args_fingerprint      str    # sha1 of normalised args, first 12 chars
tool.result_fingerprint    str
```

**Acceptance:** two identical tool calls produce identical `args_fingerprint`.

### T3 — Capture the whole usage payload

```python
@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0      # billed differently — not a subset to ignore
    reasoning_tokens: int = 0   # hidden output: reported, unreadable
    source: str = "api"         # "api" | "estimated"
    tier: str = "trusted"       # I3
```

Both extra fields confirmed available (§2).

### T4 — Context composition per call

Classify the message list from `on_chat_model_start` by type — this is what makes the
navigator tractable:

| Message type | Bucket |
|---|---|
| `SystemMessage` | `context.system_tokens` |
| bound tool schemas (from request, not messages) | `context.tool_schema_tokens` |
| first `HumanMessage` | `context.task_tokens` / `context.plan_tokens` |
| `ToolMessage` | `context.tool_output_tokens` |
| prior `AIMessage`/`ToolMessage` pairs | `context.history_tokens` |

`history_tokens` deserves special attention: every turn resends all prior turns, so it
grows roughly quadratically and usually dominates in long runs. It is the line item a
reader will not predict.

Labelled `estimated`/`verified` per I2–I3, reconciled per **I4**.

**Known baseline:** the 11 bound tools serialize to **1,104 tokens resent every navigator
turn**, versus a 77-token navigator system prompt — a 14× ratio. Confirming its share of
total input is a headline result.

### T5 — Derived metrics module (`src/attribution.py`)

| Function | Question |
|---|---|
| `stage_breakdown(spans)` | Where did the tokens go, by stage? **Headline table** |
| `context_growth(spans)` | How does input context grow across turns? |
| `duplicate_retrievals(spans)` | Repeated identical retrievals? |
| `replanning_count(spans)` | Re-planning / retries? |
| `verification_share(spans)` | Was verification most of the cost? |
| `retrieval_share(spans)` | Is retrieval dominating? |
| `reasoning_vs_complexity(traces, difficulty)` | Proportional to task difficulty? *(now core — §7)* |

`stage_breakdown` returns per stage: tokens, share, call count, **method** (I2), and
**tier** (I3).

### T6 — Per-span token provenance

Move `tokens_source` from `ExecutionTrace` to a span attribute `gen_ai.usage.source`.
Keep the trace-level roll-up for compatibility but **derive it from spans**.

### T7 — Report & visual

`src/report.py` gains an attribution section; `src/visualizer.py` gains a stacked bar
(tokens by stage, one bar per task). The trace tree shows *structure*; this shows *mass*.

---

## 10. Reasoning transparency tasks

### Three layers — only one is genuinely inaccessible

| Layer | Accessible? |
|---|---|
| Latent computation (activations) | ❌ Impossible for anyone |
| **Hidden reasoning tokens** | ⚠️ Real text. **Billed. Withheld by policy.** Size measurable (§2) |
| Externalized reasoning (plans, ReAct steps, rationales) | ✅ Fully capturable |

**Architecture is a transparency choice.** A single reasoning-model call hides planning
in the opaque layer; our MAS decomposition moves that same reasoning into inspectable
inter-agent messages. This is a new axis for the single-vs-multi comparison: the MAS
costs ~1.2× *and* converts hidden reasoning into auditable artifacts.

### R1 — Capture what is currently discarded

Three concrete gaps in today's code:

- Validator prompt requests `REASONING: [1-2 sentences]`; the parser extracts only
  `COMPLETION/TOOL_USAGE/QUALITY/CONFIDENCE` — **the rationale is generated, billed, and
  thrown away.**
- Plan **text** is not a span attribute (only `plan.num_steps`).
- `trace.reasoning_steps` contains **no reasoning** — it holds cost strings. Rename.

### R2 — `plan_execution_divergence` *(the v1 deterministic metric)*

We have the Planner's stated steps **and** the Navigator's actual tool calls. Compare
them mechanically — no judge required.

> *"The plan said FILTER by price; the navigator never called `filter_content`."*

This is a **faithfulness check on stated-vs-actual**, uniquely enabled by the MAS
architecture, and it is the single highest-value reasoning metric available.

**Acceptance:** for a task where the plan names a tool never invoked, the metric flags it.

### R3 — Reasoning provenance labels

Mirror the token tiers: **externalized** (verified) / **summarized** (asserted) /
**hidden** (unavailable — *and note it is billed*).

### R4 — Evaluation policy for reasoning

A reasoning narrative is a **claim about process, not process.** Chain-of-thought is
frequently unfaithful — models reach answers via factors they don't mention and
rationalize afterwards. Fluent reasoning is strong evidence of a well-trained writer, not
of a sound process.

Therefore, **evaluate reasoning by its consequences and its consistency with the trace,
not by reading it.** Ranked:

| Method | Strength | Note |
|---|---|---|
| **Plan–execution divergence** (R2) | Strongest — deterministic | Stated vs actual |
| **Ablation** (remove the plan, rerun) | Causal | Only method proving the reasoning did work |
| **Reasoning-cost vs outcome correlation** | Indirect but causal-ish | Flat correlation ⇒ decorative reasoning |
| Internal consistency (rationale vs verdict) | Cheap, catches real failures | |
| LLM-judged plausibility | **Weakest** | Grades fluency; shares blind spots with generator |

**Who evaluates:** layered. Deterministic where possible; LLM judge only for the
remainder, and **only when calibrated against a human-labeled sample with agreement
reported.** An uncalibrated judge score is unfalsifiable. (Microsoft's ASSERT reports
80–90% human agreement for exactly this reason; this repo's existing `tokens_source` and
selective-judging discipline should extend to reasoning.)

---

## 11. Module map

| Module | Status | Purpose |
|---|---|---|
| `src/support_tools.py` | **new** | Real tool set + sandbox guards (§6) |
| `src/support_dataset.py` | **new** | KB/ticket loader + ground truth (§7) |
| `src/attribution.py` | **new** | T5 metrics |
| `src/reasoning.py` | **new** | R2 divergence, R3 provenance |
| `src/otel.py` | extend | `Usage` dataclass, per-call handler (T1, T3) |
| `src/agents.py` | extend | Per-call spans, context split, capture rationale/plan |
| `src/tracer.py` | extend | Per-span provenance (T6), rename `reasoning_steps` |
| `src/config.py` | extend | o-series branch (§2) |
| `src/evaluator.py` | extend | Outcome checks (correct KB cited, escalation correct) |
| `src/visualizer.py` | extend | Stacked bar by stage (T7) |
| `src/report.py` | extend | Attribution + reasoning sections (T7) |
| `docs/token-transparency.md` | **new** | Public-facing guide; disclosure ladder |
| `examples/sample_traces.jsonl` | **new** | ~88 KB — analysis runs without credentials |

---

## 12. Acceptance criteria

The build is done when a single ticket produces:

| Stage | Tokens | Share | Calls | Method | Tier |
|---|---|---|---|---|---|
| System prompts | … | … | — | estimated | verified |
| Tool definitions | … | … | — | estimated | verified |
| Retrieved knowledge (tool output) | … | … | 4 | estimated | verified |
| Conversation history | … | … | — | estimated | verified |
| Planning (visible) | … | … | 2 | api | trusted |
| **Internal reasoning (hidden)** | … | … | — | **api** | **asserted** |
| Verification | … | … | 1 | api | trusted |
| Final response | … | … | 1 | api | trusted |
| **Unaccounted** | … | … | — | **residual** | — |

**plus** — *"2 of 4 retrievals were duplicates"*, *"1 re-plan"*, *"plan named 3 tools;
navigator used 2 of them + 1 unplanned"*, *"44% of planner output was reasoning you
cannot read"*, *"identical prompt cost differed 91% between calls due to cache."*

Every row sums into the single total (I1). Every row carries method and tier (I2, I3).
The residual is visible (I4).

---

## 13. Non-goals & honest limits

**This works because we own the agent.** Toggling components, decomposing context, and
instrumenting every call are possible only from inside. None of it recovers hidden
computation in someone else's product.

State that as the result, not a caveat: this establishes **the achievable ceiling when
you control the stack.** The distance between that ceiling and what a commercial agent
discloses *is* the transparency gap — and quantifying it is sharper than either half.

Even at the ceiling we are not omniscient: we **verify** the context we built, **trust**
the vendor's usage counts, and can only **assert** reasoning size (I3).

Explicitly out of scope:

- Auditing third-party agents (impossible per above; governance work)
- Budget-conformance *testing* (belongs in `genai_alignment`, consumes this output)
- Presenting dollar figures with token-level confidence (I5)
- τ-bench integration (future)

---

## 14. Sequencing

| # | Step | Depends on |
|---|---|---|
| 1 | ✅ Capability probe | — |
| 2 | ✅ This spec | 1 |
| 3 | Support corpus + real tools + sandbox guards | 2 |
| 4 | Instrumentation: T1 (+ concurrency fix), T3, T6 | 2 |
| 5 | T4 context decomposition, R1 capture | 4 |
| 6 | `attribution.py` (T5), `reasoning.py` (R2, R3) | 5 |
| 7 | New notebook | 3, 6 |
| 8 | T7 report/visual, README highlight, `docs/token-transparency.md` | 7 |

Steps 3 and 4 are independent and can proceed in parallel.
