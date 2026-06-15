> 📄 **Sample report** — example output of Step 13 (`src/report.py`) from a real run
> (token/cost from **real API usage**). Findings carry **High / Medium / Low ratings**
> plus an **overall testing rating**. Uses the default selective LLM-judge policy
> (🤖 AI Assessment only where deterministic reading is ambiguous). Latency is the
> median. See also the [HTML executive summary](sample_executive_summary.html).

---

# Agent Evaluation Report

**Multi-Agent Observability & Evaluation Framework** · Mind2Web benchmark

- **Generated:** 2026-06-15 14:44
- **Agent model:** GPT 5-4 (Azure) · **Judge model:** GPT 4-1 (Azure)
- **Tasks evaluated:** 10 (per architecture) · **Pass threshold:** 0.70
- **Overall rating:** 🟢 **High**
- **Report type:** Pre-deployment sandbox evaluation

---

## 1. Executive Summary

This report evaluates an AI web-navigation agent on **10 Mind2Web tasks**, comparing a single-agent baseline against a multi-agent system (MAS) using the supervisor pattern (Planner → Navigator → Validator). Every run is instrumented with OpenTelemetry-compliant tracing and scored across task completion, tool selection, safety, cost, and latency.

### Key Findings

| Finding | Rating | Detail |
|---|---|---|
| **F1** | 🟢 High | **Task completion:** MAS pass rate **90%** vs. single-agent **70%** (+20 pp). *(see §4.1)* |
| **F2** | 🟢 High | **Quality:** MAS average score **0.816** vs. **0.724** for the single agent. *(see §4.1)* |
| **F3** | 🟡 Medium | **Tool correctness:** MAS tool-F1 **0.432** vs. **0.383**. *(see §4.2)* |
| **F4** | 🟢 High | **Cost:** MAS **$0.0386/task** vs. **$0.0330/task** (1.2× the single-agent cost). *(see §4.4)* |
| **F5** | 🟡 Medium | **Latency (median):** MAS **19710 ms/task** vs. **11279 ms/task**. *(see §4.4)* |
| **F6** | 🟢 High | **Safety:** **100%** of MAS outputs passed all safety checks (PII, injection, harmful content). *(see §4.3)* |
| **F7** | 🟢 High | **Overall:** the **Multi-Agent System** delivered higher task quality on this sample. *(see §4.5)* |

**Overall testing rating: 🟢 High** — derived from task completion, quality, and safety.


> *Rule-based assessment is unambiguous here; the LLM judge was not invoked for this section.*


---

## 2. Testing Scope

### 2.1 What We Are Testing

We test an LLM **web-navigation agent** in two architectures, built on the **LangChain** + **LangGraph** stack, and compare them on identical tasks.

**System A — Single Agent (baseline).** A single **ReAct** agent created with LangGraph's `create_react_agent`, given all 11 hybrid tools and a focused system prompt. It plans, selects tools, executes, and produces a final answer in one reasoning loop. Model: GPT 5-4 (Azure).

**System B — Multi-Agent System (supervisor pattern).** Work is decomposed across four specialists, each a LangChain chat model with its own role, prompt, and (optionally) its own model; per-agent token use and cost are tracked individually:

| Specialist | Role | Framework component | Tools | Model |
|---|---|---|---|---|
| **Supervisor** | Routes the pipeline (sequential) | LangChain chat model | no | GPT 5-4 (Azure) |
| **Planner** | Decomposes the task into 3–5 steps | LangChain chat model | no | GPT 5-4 (Azure) |
| **Navigator** | Executes the plan with tools | LangGraph `create_react_agent` | **yes** | GPT 5-4 (Azure) |
| **Validator** | Independently judges completion & quality | LangChain chat model | no | GPT 4-1 (Azure) |

- **Tooling:** 11 hybrid tools (`src/tools.py`) — READ tools fetch live data when API keys are present (Tavily) else realistic mocks; WRITE tools (book / purchase / submit) are **always mocked**.
- **Capabilities assessed:** task planning, tool selection & sequencing, instruction following, output quality, safety, cost, and latency.
- **Observability:** every step is wrapped in an OpenTelemetry span (`HierarchicalTracer`, `src/tracer.py`) following GenAI Semantic Conventions.
- **Out of scope:** live browser execution against production websites (plans are scored in a sandbox; WRITE actions are mocked).

### 2.2 Testing Data

- **Benchmark:** Mind2Web (NeurIPS 2023, OSU NLP) — natural-language web tasks across 137 real websites and 31 domains.
- **Sample:** 10 tasks drawn from the cached corpus of 300 streamed tasks.
- **Reference labels:** gold `action_reprs` sequences used to score tool correctness.
- **Domains represented in this run:** budget, discogs, ign, resy, rottentomatoes, united.

### 2.3 Applicable Regulations & Compliance

| Framework | Requirement | How this evaluation addresses it |
|---|---|---|
| **SR 11-7** (Model Risk Mgmt) | Effective challenge & outcome analysis | Independent validator + rule-based scoring + failure visibility |
| **NIST AI RMF** — MEASURE 2.5 | Ongoing monitoring of AI outputs | Per-task tracing, health metrics, drift detection |
| **NIST AI RMF** — GOVERN 1.7 | Transparency & explainability | OTel trace tree + AI-assessment disclosure on every LLM-generated block |
| **EU AI Act** — Art. 12 | Automatic logging / record-keeping | OTLP-compliant span export per task (`outputs/traces/`) |
| **OpenTelemetry GenAI SemConv** | Standardized AI observability | Spans carry `gen_ai.*` attributes, portable to any OTel backend |

---

## 3. Testing Approach

**Hybrid sandbox evaluation.** Agents run with real reasoning and real READ tools (live web search/scraping when keys are present, realistic mocks otherwise) while WRITE tools (book / purchase / submit) are always mocked — capturing authentic agent behavior with zero real-world side effects.

**Scoring stack (per task):**

1. **Task completion** — hybrid score = 0.4 × rule-based + 0.6 × LLM-as-judge (pass ≥ 0.70). Rules cover length, specificity, goal alignment, action verbs, and overlap with the reference sequence.
2. **Tool correctness** — precision / recall / F1 against the gold action sequence, with flexible search-tool equivalence and LCS order accuracy.
3. **Safety** — deterministic scans for PII, prompt injection, and harmful content.
4. **Cost / latency / health** — per-call token & cost tracking (agent vs. judge separated), rolling-window success rate, and latency percentiles.

**Judge model.** The LLM-as-judge and the AI-assessment blocks in this report use GPT 4-1 (Azure), separate from the agent model to reduce self-evaluation bias.

---

## 4. Testing Results

### 4.1 Task Completion

**What this measures.** Whether each agent actually accomplished the task. We score the agent's plan with a hybrid metric: 40% deterministic rules (length, specificity, goal-keyword alignment, action verbs, overlap with the gold action sequence) + 60% LLM-as-judge (holistic 0–1 quality). A task **passes** when the total score ≥ 0.70. This is the headline quality signal cross-referenced by **Executive Summary F1 & F2**.

| Metric | Single Agent | Multi-Agent |
|---|---|---|
| Pass rate | 70% | 90% |
| Avg total score | 0.724 | 0.816 |
| Avg rule score | 0.759 | 0.809 |
| Avg LLM score | 0.700 | 0.820 |

**Assessment:** (ref. **F1, F2**) 🟢 Strong — multi-agent completion is higher than the single-agent baseline (90% vs. 70% pass rate).


> *Rule-based assessment is unambiguous here; the LLM judge was not invoked for this section.*

![Task Completion](mas_eval_dashboard.png)

*Multi-agent evaluation dashboard (score, pass/fail, tool-F1, cost-vs-latency).*

### 4.2 Tool Correctness

**What this measures.** Whether the agent invoked the *right tools in the right order*. We compare the tools actually called (captured from the execution trace) against the tools implied by Mind2Web's gold `action_reprs`, reporting precision, recall, and F1. Search tools are treated as interchangeable (flexible equivalence), and order accuracy uses longest-common-subsequence. Cross-referenced by **F3**.

| Metric | Single Agent | Multi-Agent |
|---|---|---|
| Avg F1 | 0.383 | 0.432 |
| Avg precision | 0.286 | 0.345 |
| Avg recall | 0.692 | 0.733 |
| Avg tool calls | 7.5 | 10.5 |

**Assessment:** (ref. **F3**) 🟡 Adequate — tool-selection alignment with the reference sequence (MAS F1 0.432 vs. single 0.383).


> 🤖 **AI Assessment** — the text in this block is generated by an LLM judge from the run's metrics. It is advisory only and **requires independent human review** before use in any regulatory, audit, or production decision.
>
> Tool selection improved under the multi-agent system (F1 0.43 vs 0.38), consistent with the Planner providing explicit step structure. Scores remain in the moderate band due to Mind2Web's compressed action vocabulary; precision is the main limiter while recall stays high. No systematic misuse was observed.

### 4.3 Safety & Robustness

**What this measures.** Whether agent outputs are safe to surface. Each output is scanned deterministically for **PII** (SSN, credit card, email, phone), **prompt injection** (XSS, SQL, code execution), and **harmful-content** keywords; we also count execution errors. WRITE actions are mocked, so no real-world side effects are possible. Cross-referenced by **F6**.

| Check | MAS pass rate |
|---|---|
| Overall safety | 100% |
| Tasks with errors | 0 / 10 |

**Assessment:** (ref. **F6**) 🟢 Strong — no PII leakage, injection, or harmful content detected in passing outputs.


> *Rule-based assessment is unambiguous here; the LLM judge was not invoked for this section.*

### 4.4 Cost & Performance

**What this measures.** The operational price of each architecture. Cost is computed per call from token usage (agent and judge/validator tracked separately) using the rate table in `Config`; latency is wall-clock per task. The MAS runs 3–4 LLM roles per task vs. 1 for the baseline, so this section quantifies the overhead behind **F4 (cost)** and **F5 (latency)**. Token counts are sourced from **real API usage** for 100% of multi-agent tasks (remainder estimated via tiktoken).

| Metric | Single Agent | Multi-Agent |
|---|---|---|
| Avg cost / task | $0.0330 | $0.0386 |
| Total cost (10 tasks) | $0.3297 | $0.3858 |
| Median latency | 11279 ms | 19710 ms |
| P95 latency | 21276 ms | 30093 ms |

**Assessment:** (ref. **F4, F5**) 🟢 Strong — multi-agent overhead is 1.2× cost and 1.7× latency.


> *Rule-based assessment is unambiguous here; the LLM judge was not invoked for this section.*

![Cost & Performance](mas_telemetry.png)

*Multi-agent telemetry: tokens, cost, latency percentiles, rolling pass rate.*

### 4.5 Single-Agent vs. Multi-Agent System

**What this measures.** The head-to-head trade-off, consolidating §4.1–4.4 onto the same tasks. The question is not which architecture is universally better, but **when the multi-agent system's extra cost and latency are justified by higher quality**. This is the basis for **Executive Summary F7** and the recommendation in §5.

| Metric | Single Agent | Multi-Agent |
|---|---|---|
| System | Single Agent | Multi-Agent |
| Pass rate | 70% | 90% |
| Avg score | 0.724 | 0.816 |
| Tool F1 | 0.383 | 0.432 |
| Avg cost | $0.0330 | $0.0386 |
| Avg latency | 13438ms | 20046ms |
| Avg tools | 7.5 | 10.5 |

**Assessment:** (ref. **F7**) On this sample the **Multi-Agent System** wins on quality. Multi-agent decomposition tends to help most on complex, multi-step tasks; simple lookups favor the lower-cost single agent.


> *Rule-based assessment is unambiguous here; the LLM judge was not invoked for this section.*

![Single-Agent vs. Multi-Agent System](baseline_vs_multi.png)

*Pass rate, average score, and cost per task — single vs. multi-agent.*

### 4.6 Observability (OpenTelemetry)

Each task produces a hierarchical span tree (`task.execute → agent.* → tool.execute`) with GenAI Semantic Convention attributes and per-agent cost attribution, exported to OTLP JSON.

![Trace tree](mas_trace_tree.png)

*Multi-agent OTel trace tree for a representative task.*

**Assessment:** 🟢 Strong — full traceability with portable, audit-ready spans.

---

## 5. Conclusion

The multi-agent system achieved the higher task quality on this 10-task sample (score 0.816). Based on the cost/quality trade-off, we recommend: **adopt the multi-agent system for complex tasks while keeping the single agent as a low-cost default for simple lookups.** All outputs passed safety screening at 100%, and every decision is traceable via OTLP spans.

**Recommended next steps:** (1) expand to a larger, complexity-stratified sample; (2) use a distinct judge model to further reduce evaluation bias; (3) wire the exported OTLP traces into a production observability backend.


> *Rule-based assessment is unambiguous here; the LLM judge was not invoked for this section.*


---

## 6. Appendices

### 6.1 Artifacts

| Artifact | File |
|---|---|
| Single-agent results | `single_agent_*.csv` |
| Multi-agent results | `multi_agent_*.csv` |
| Comparison table | `comparison_*.csv` |
| OTLP traces | `traces/all_otel_traces.jsonl` |
| Dashboards | `*.png` |

### 6.2 Audit Trail — Per-Task Record (Multi-Agent)

| Task | Website | Score | Pass | Tool F1 | Tools | Cost $ | Latency ms | Safe |
|---|---|---|---|---|---|---|---|---|
| 0 | united | 0.80 | ✅ | 0.29 | 28 | 0.0631 | 22660 | ✅ |
| 1 | ign | 0.86 | ✅ | 0.67 | 6 | 0.0249 | 15148 | ✅ |
| 2 | discogs | 0.67 | ❌ | 0.00 | 3 | 0.0164 | 11299 | ✅ |
| 3 | discogs | 0.80 | ✅ | 0.57 | 3 | 0.0186 | 16678 | ✅ |
| 4 | discogs | 0.83 | ✅ | 0.25 | 9 | 0.0398 | 24220 | ✅ |
| 5 | budget | 0.86 | ✅ | 0.33 | 16 | 0.0547 | 26492 | ✅ |
| 6 | budget | 0.81 | ✅ | 0.57 | 9 | 0.0379 | 21192 | ✅ |
| 7 | budget | 0.84 | ✅ | 0.44 | 18 | 0.0759 | 33038 | ✅ |
| 8 | resy | 0.84 | ✅ | 0.80 | 5 | 0.0210 | 11500 | ✅ |
| 9 | rottentomatoes | 0.85 | ✅ | 0.40 | 8 | 0.0336 | 18229 | ✅ |

### 6.3 AI Disclosure

**LLM judge: enabled (selective).** The LLM was invoked only where the deterministic reading was ambiguous — sections: *tool*. All other sections were interpreted by rule alone.

Blocks labeled **🤖 AI Assessment** are LLM-generated and advisory only. All tables, metrics, pass/fail decisions, and the audit trail are computed deterministically from the run and are audit-safe. Independent human review is required before relying on any AI-generated content for regulatory or production use.
