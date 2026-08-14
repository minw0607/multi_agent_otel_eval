# Multi-Agent OTel Evaluation Framework

<div align="center">

**See what your AI agents did, prove they did it right, and account for every token.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-GenAI%20SemConv-425CC7?logo=opentelemetry)](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
[![LangChain](https://img.shields.io/badge/LangChain-LangGraph-1C3C3C)](https://langchain.com)
[![Provider agnostic](https://img.shields.io/badge/LLM-provider--agnostic-412991)](docs/provider-setup.md)

*A provider-agnostic framework for evaluating and instrumenting multi-agent LLM systems —*
*OpenTelemetry tracing · token attribution · reasoning transparency · audit-grade reporting*

</div>

---

## Two notebooks, two questions

| Notebook | Question it answers | Benchmark |
|---|---|---|
| **[`agentic_otel_demo_notebook.ipynb`](agentic_otel_demo_notebook.ipynb)** | *How well does the agent perform, and can I see what it did?* | [Mind2Web](https://osu-nlp-group.github.io/Mind2Web/) web navigation |
| **[`token_transparency_notebook.ipynb`](token_transparency_notebook.ipynb)** ⭐ | *Where did every token go, and how much of it can anyone see?* | Customer-support desk (real tools) |

Both run on the same `src/` package. The notebooks stay deliberately coding-light — every
component lives in `src/`, so each notebook reads as a narrative rather than a script.

---

## What makes this different

**1. Token attribution, not just token counting.** Every agent API reports a total. This
one reports **where the tokens went** — system prompts, tool definitions you never used,
retrieved documents, conversation history re-sent for the tenth time, and reasoning the
model performed and billed you for but never showed you.

**2. Reasoning transparency.** A multi-agent architecture externalizes reasoning that a
single reasoning-model call hides. We capture the plan and the rationale, then check them
against the trace — deterministically, without asking an LLM whether the reasoning "looks
good."

**3. Honest provenance on every number.** Each figure carries how it was obtained
(`api` / `estimated` / `residual`) and how far it can be trusted (`verified` / `trusted` /
`asserted`). When our decomposition disagrees with the provider's total, the **residual is
printed, not hidden**.

**4. Real OpenTelemetry.** Not OTLP-shaped JSON — the actual SDK, auto-instrumenting
LangChain, exportable to Phoenix, Datadog, Jaeger, or Langfuse.

**5. Provider-agnostic.** Azure OpenAI, OpenAI, Ollama, Groq, Together, LM Studio. No
vendor assumptions in the code or in the output.

---

## Results

### Where the tokens actually went

A 15-ticket support-desk run: **114,462 tokens, 87 LLM calls, $0.34.**

![Token attribution](docs/token_attribution.png)

| Finding | Number |
|---|---|
| **Tool definitions** — resent every turn, whether used or not | **23.7%** of all tokens |
| Retrieved knowledge (tool output) | 27.1% |
| Conversation history re-sent | 6.8% |
| **Hidden reasoning** — billed, never returned | **33% of all output** |
| Residual (decomposition vs. reported total) | **+2.1%**, disclosed |

Nearly a quarter of every token went to *re-declaring tools*, most of which the agent
never called. That is invisible in any per-request total, and it is directly actionable.

### Hidden reasoning is a model choice, not a workload property

![Hidden reasoning](docs/hidden_reasoning.png)

The planner runs a reasoning model; the other agents don't. You pay for the red.

### Single-agent vs. multi-agent

On Mind2Web, orchestration bought a real quality gain at a modest cost premium:

| Metric | Single agent | Multi-agent | |
|---|---|---|---|
| Pass rate | 60% | **100%** | 🟢 |
| Avg task score | 0.697 | **0.809** | 🟢 |
| Cost / task | $0.0346 | $0.0414 (1.2×) | 🟡 |
| Median latency | 8.3 s | 19.5 s | 🟡 |

📄 **[Sample evaluation report](docs/sample_evaluation_report.md)** · **[HTML executive summary](docs/sample_executive_summary.html)**

---

## Quickstart

```bash
git clone https://github.com/minw0607/multi_agent_otel_eval.git
cd multi_agent_otel_eval
pip install -r requirements.txt
cp .env.example .env          # add your provider credentials
jupyter notebook token_transparency_notebook.ipynb
```

The provider is auto-detected: set `OPENAI_API_VERSION` for Azure OpenAI, leave it blank
for OpenAI / Ollama / Groq / any compatible endpoint. Full setup for each provider is in
**[docs/provider-setup.md](docs/provider-setup.md)**.

> **Running locally is recommended.** The reference Azure setup uses IP-allowlist access
> (no interactive login), which keeps evaluation runs uninterrupted but means Colab
> cannot reach it. Colab works fine with any IP-independent provider.

Optional — stream live traces to a real backend:

```python
from src import setup_phoenix
setup_phoenix()      # Phoenix UI at http://localhost:6006
```

---

## How it works

```
┌───────────────────────────────────────────────────────────────────────────┐
│  DATA              AGENTS                EXECUTION           EVALUATION    │
│                                                                            │
│  Mind2Web       Single ReAct         Real + mock tools    Hybrid scoring   │
│  benchmark      ── or ──             OTel span tracing    Tool correctness │
│                 Multi-agent MAS      per-call token       Safety checks    │
│  Support        supervisor →         attribution          Outcome vs       │
│  corpus         planner →            reasoning capture    ground truth     │
│  (real tools)   navigator →                               Audit report     │
│                 validator                                                  │
└───────────────────────────────────────────────────────────────────────────┘
```

**The multi-agent system** is built with LangChain + LangGraph. Only the Navigator is a
true ReAct graph; the Supervisor is deterministic routing that costs **zero tokens**, and
the Planner and Validator are single model calls with role prompts. Each specialist can
run a different model.

**Instrumentation** attaches through LangChain callbacks, so every LLM call is recorded
individually — which is what makes re-planning, per-turn context growth, and hidden
reasoning observable at all.

### Documentation

| Guide | Covers |
|---|---|
| **[Token & reasoning transparency](docs/transparency_spec.md)** | The methodology: invariants, provenance tiers, what's measurable and what isn't |
| **[Observability](docs/observability.md)** | OpenTelemetry from first principles + setup for Phoenix, Jaeger, Tempo, Datadog, Splunk, Langfuse |
| **[Evaluation](docs/evaluation.md)** | Metrics reference, trajectory vs. outcome scoring, tool environments, judge policy |
| **[Provider setup](docs/provider-setup.md)** | Step-by-step for Azure, OpenAI, Ollama, Groq, Together, LM Studio |

---

## Repo structure

```
multi_agent_otel_eval/
├── agentic_otel_demo_notebook.ipynb    ← evaluation + observability demo
├── token_transparency_notebook.ipynb   ← ⭐ token & reasoning attribution
│
├── src/
│   ├── config.py            Provider-agnostic LLM factory, cost table
│   ├── tracer.py            OTel spans, per-span provenance
│   ├── otel.py              Real OTel export · Usage · per-call recorder
│   ├── agents.py            Mind2Web single + multi-agent systems
│   ├── support_agents.py    Support-desk MAS (real tools, instrumented)
│   ├── support_tools.py     Real tools under an enforced sandbox contract
│   ├── support_dataset.py   Labeled corpus + ground truth
│   ├── attribution.py       Token attribution and derived metrics
│   ├── reasoning.py         Plan–execution divergence, reasoning provenance
│   ├── interpret.py         Rule-based chart interpretation
│   ├── evaluator.py         Hybrid scoring, tool correctness, outcome eval
│   ├── visualizer.py        Dashboards, trace trees, attribution charts
│   ├── report.py            Audit-grade Markdown + HTML reports
│   └── runner.py            Batch evaluation
│
├── data/support/            KB articles, policies, orders, labeled tickets
├── docs/                    Guides (see table above)
└── outputs/                 Results, traces, charts (gitignored)
```

---

## Limitations

- **Trajectory ≠ outcome.** The Mind2Web notebook scores the agent's *plan* against
  reference actions; WRITE actions are mocked, so real consequences are unobservable by
  design. The support desk closes part of this gap with ground truth. See
  [docs/evaluation.md](docs/evaluation.md).
- **We trust the vendor's token counts.** We verify context we assembled ourselves, but
  cannot audit the provider's reported usage — hence the `trusted` tier.
- **Hidden reasoning is only ever a number.** Size is reported; content is withheld. No
  amount of instrumentation recovers it.
- **Cost is an estimate even when tokens are exact.** Cache hits depend on state outside
  your control, so identical requests can bill differently.
- **This is the ceiling for a stack you own.** None of it audits a third-party agent — and
  the distance between this ceiling and what commercial agents disclose *is* the
  transparency gap.
- **LLM-as-judge bias.** Use a different model for the judge than the agent, and calibrate
  against human labels before trusting a judge score.

---

## Citation

```bibtex
@inproceedings{deng2023mind2web,
  title={Mind2Web: Towards a Generalist Agent for the Web},
  author={Deng, Xiang and Gu, Yu and Zheng, Boyuan and Chen, Shijie and
          Stevens, Samuel and Wang, Boshi and Sun, Huan and Su, Yu},
  booktitle={NeurIPS},
  year={2023}
}
```

Aligned with the [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

---

## Disclaimer

Research and evaluation framework. The support-desk corpus is synthetic; company names,
policies, and orders in it are fictional. Numbers shown are from specific runs and will
vary by model, provider, and sample. Nothing here constitutes a vendor audit.

---

<div align="center">

[Open an issue](https://github.com/minw0607/multi_agent_otel_eval/issues) ·
[Transparency spec](docs/transparency_spec.md) ·
[Observability](docs/observability.md) ·
[Evaluation](docs/evaluation.md) ·
[Provider setup](docs/provider-setup.md)

</div>
