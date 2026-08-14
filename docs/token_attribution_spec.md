# Token Attribution for Multi-Agent Systems — Implementation Spec

**Goal:** move this repo from token *accounting* (how many tokens were consumed) to token *attribution* (where they went, why, and whether that was proportionate).

Background: [The Token Transparency Gap](https://minwu-ai.github.io/the-token-transparency-gap-why-agentic-ai-still-hides-where-computation-goes/). That post argues APIs should report a per-stage token breakdown. **This repo already emits per-stage spans — so it can demonstrate the proposal rather than merely argue for it.** An existence proof is a stronger claim than a recommendation, and that is what this work is for.

---

## Why this repo, and not one of the others

The proposal in the post is a breakdown by workflow stage: planning, internal reasoning, verification, tool outputs, final response. This repo's `SPAN_NAMES` (`src/tracer.py`) is already almost exactly that list:

| Post's proposed category | Existing span |
|---|---|
| Planning | `agent.planner.plan` |
| Verification | `agent.validator.validate` |
| Tool outputs | `tool.execute` |
| Routing / orchestration | `agent.supervisor.route` |
| Final response | `agent.navigator.execute` |

The instrumentation exists. **Nothing rolls it up and reports it as an attribution table.** That gap is the whole task.

Two things belong elsewhere and should not be built here:

- **Budget-conformance testing** (*was the system told to stay under N tool calls, and did it?*) belongs in `genai_alignment` as a scenario, and consumes what this repo produces.
- **The vendor-transparency audit** (*which of these categories can a customer verify at all?*) is governance work. Its answers are mostly "no", and that is the point of it.

---

## What already works — do not rebuild

Credit where due; this is a real foundation:

- Per-agent spans carry `gen_ai.usage.input_tokens`, `output_tokens`, `cost_usd` (`src/agents.py`, `run_multi_agent`).
- **`tokens_source` distinguishes `"api"` from `"estimated"`** (`ExecutionTrace`, `src/tracer.py`). This is the single most important thing already correct — a measured number and a tiktoken guess must never be silently mixed, and this repo already refuses to.
- Real usage comes from LangChain's `UsageMetadataCallbackHandler` (`make_usage_callback`), falling back to an estimate only when the callback yields nothing.
- Per-agent cost/token breakdown lands in `trace.decision_points`.
- Spans export to OTLP JSON (`OTelSpan.to_otlp_dict`), so any backend can consume this.

---

## The gaps, each tied to a question it blocks

| # | Gap | Blocks |
|---|---|---|
| 1 | **`tool.execute` spans carry no tokens.** In `run_multi_agent` a tool span is opened and closed with only `gen_ai.tool.name` — no output size, no argument record | *How many tokens went to retrieval? Is retrieval dominating cost?* |
| 2 | **The navigator is one lump.** It is a LangGraph agent with `recursion_limit: 50` making many internal LLM calls; all of it collapses into a single `n_in` / `n_out` pair | *Did planning need retries? Is the agent re-planning? Was verification most of the cost?* |
| 3 | **`SPAN_NAMES["LLM_CALL"]` is defined but never used.** Finest granularity available today is the agent, not the call | Everything per-step |
| 4 | **No context composition.** Input tokens per agent are one scalar — no split into system prompt / task / plan / tool output / history | *How much context came from system prompts? From retrieved knowledge?* |
| 5 | **`usage_from_callback` drops `cached_tokens` and `reasoning_tokens`**, summing only input/output | *Was reasoning proportional to task complexity?* — and cache economics entirely |
| 6 | **Tool arguments are not fingerprinted**, so repeated identical calls are invisible | *Did agents repeatedly retrieve identical documents?* |
| 7 | **`tokens_source` is per-trace, not per-span.** If the planner had API counts and the navigator estimated, the whole trace reads `"mixed"` and no individual number can be trusted | Interpreting any of the above honestly |

---

## Tasks, in dependency order

### Task 1 — Per-LLM-call spans

Use the already-defined `SPAN_NAMES["LLM_CALL"]`. Every model invocation gets its own child span under its agent's span, carrying `gen_ai.usage.*`, the model id, and a sequence index.

For the navigator, this means attaching a callback that fires per LLM call rather than summing over the whole graph run. LangChain's callback system exposes `on_llm_end` with per-call usage — prefer a small custom handler over `UsageMetadataCallbackHandler` here, since the latter aggregates.

**This is the prerequisite for everything else.** Without per-call spans, re-planning and retries are unobservable by construction.

**Acceptance:** a multi-agent task produces ≥ 3 `llm.chat.completion` spans, and their token counts sum to the parent agent spans' totals.

### Task 2 — Tool spans carry their cost

A tool call's real token cost is not the call — it is **the output, which becomes input to the next LLM call.** Attach to each `tool.execute` span:

```
tool.output_chars          int
tool.output_tokens_est     int      # tokens this output will add downstream
tool.args_fingerprint      str      # sha1 of normalised args, first 12 chars
tool.result_fingerprint    str      # same, over the result
```

Fingerprints are what make gap 6 answerable, and they cost nothing.

**Acceptance:** for a task where the navigator calls the same tool twice with identical arguments, both spans carry the same `tool.args_fingerprint`.

### Task 3 — Capture the whole usage payload

Replace `usage_from_callback`'s `(int, int)` return with a dataclass:

```python
@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0       # billed differently; not a subset to ignore
    reasoning_tokens: int = 0    # hidden output, reported but not visible
    source: str = "api"          # "api" | "estimated"
```

Both extra fields are already present in OpenAI-compatible responses (`prompt_tokens_details.cached_tokens`, `completion_tokens_details.reasoning_tokens`) and are simply discarded today.

**Note on cost, and state it in the output:** token counts are deterministic — the same messages always tokenize identically. **Cost is not.** Cache hits depend on TTL and eviction you do not control, so the same request billed twice can differ. Attribution *by token* is reproducible; attribution *by dollar* is an estimate. Do not present `cost_usd` with the same confidence as token counts.

### Task 4 — Context composition per call

The largest single driver in agent workflows is input context, and it is currently one number. Where the caller assembles the prompt, record the split:

```
context.system_tokens
context.task_tokens
context.plan_tokens
context.tool_output_tokens
context.history_tokens
```

**`history_tokens` deserves its own attention.** Every turn resends all prior turns, so it grows roughly quadratically in a long run. In extended agent tasks it usually dominates everything else in the table, and it is the one line item a reader will not predict.

Compute these with the same tokenizer used for the estimate fallback, and label them `estimated` — they are a decomposition of a measured total, not measurements themselves. **The decomposition must be reconciled against the API's reported `input_tokens`**, with the residual reported rather than hidden:

```
context.accounted_tokens   = sum of the above
context.residual_tokens    = reported_input - accounted
```

A large residual means the decomposition is wrong. Surfacing it is what keeps this honest.

### Task 5 — A derived-metrics module (`src/attribution.py`)

The deliverable. One function per question from the post, each operating on a trace's spans:

| Function | Question |
|---|---|
| `stage_breakdown(spans)` | Where did the tokens go, by stage? **The headline table.** |
| `retrieval_share(spans)` | Is retrieval dominating cost? |
| `duplicate_retrievals(spans)` | Did agents repeatedly retrieve identical documents? |
| `replanning_count(spans)` | Is the agent re-planning? How often? |
| `verification_share(spans)` | Was verification responsible for most of the cost? |
| `reasoning_vs_complexity(traces, difficulty)` | Was reasoning proportional to task difficulty? |
| `context_growth(spans)` | How does input context grow across steps? |

`stage_breakdown` should return, per stage: tokens, share of total, call count, and **the method by which each number was obtained** (`api` / `estimated` / `derived`). A number without its provenance is not auditable, and this repo's existing `tokens_source` discipline should extend to every reported figure.

`reasoning_vs_complexity` needs a difficulty-graded dataset and is the only item here that cannot be answered from traces alone — treat it as optional scope.

### Task 6 — Per-span token provenance

Move `tokens_source` from `ExecutionTrace` to a span attribute (`gen_ai.usage.source`). Keep the trace-level roll-up for backwards compatibility, but derive it from the spans rather than from a parallel list.

### Task 7 — Report and visual

Extend `src/report.py` with an attribution section, and add a stacked bar (tokens by stage, one bar per task) to `src/visualizer.py`. The existing trace-tree visual already shows structure; this shows *mass*.

---

## What "done" looks like

A single task produces a table like:

| Stage | Tokens | Share | Calls | Method |
|---|---|---|---|---|
| System & tool definitions | 1,847 | 12% | — | estimated |
| Retrieved knowledge / tool outputs | 6,203 | 41% | 4 | estimated |
| Planning | 1,412 | 9% | 2 | api |
| Internal reasoning | 890 | 6% | — | api (reported) |
| Verification | 2,104 | 14% | 1 | api |
| Final response | 1,655 | 11% | 1 | api |
| Conversation history | 890 | 6% | — | estimated |
| **Unaccounted** | **204** | **1%** | — | residual |

Plus: *"2 of 4 retrievals were duplicates"*, *"1 re-plan"*, *"verification was 14% of spend"*.

That table is the artifact the post says does not exist. Producing it — with an explicit residual and per-row provenance — is the whole point.

---

## Non-goals, and the honest limit

**This works because we own the agent.** Toggling components, decomposing context, and instrumenting every call are all possible only from inside. None of it recovers hidden computation in someone else's product.

That boundary is worth stating in the output rather than treating as a caveat: this repo establishes **the achievable ceiling when you control the stack.** The distance between that ceiling and what a commercial agent actually discloses is the transparency gap itself — and quantifying that distance is a sharper result than either half alone.

Explicitly out of scope:

- Auditing third-party agents (impossible by the above; belongs in governance work)
- Budget-conformance *testing* (belongs in `genai_alignment`)
- Presenting dollar figures with the same confidence as token counts (see Task 3)
