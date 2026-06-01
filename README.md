# Multi-Agent OTel Evaluation Framework

<div align="center">

**A provider-agnostic GenAI observability & evaluation framework benchmarked on Mind2Web**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minw0607/multi_agent_otel_eval/blob/main/demo_notebook.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-GenAI%20SemConv-425CC7?logo=opentelemetry)](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
[![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-0078D4?logo=microsoftazure)](docs/provider-setup.md)
[![OpenAI](https://img.shields.io/badge/OpenAI-Compatible-412991?logo=openai)](docs/provider-setup.md)
[![Ollama](https://img.shields.io/badge/Ollama-Local-black)](docs/provider-setup.md)

*Plug in any OpenAI-compatible LLM and run a rigorous, observable evaluation of single- and multi-agent web-navigation systems —*  
*OpenTelemetry-compliant tracing · hybrid rule + LLM-judge scoring · tool-correctness metrics · cost & health monitoring · safety validation*

</div>

---

## Why This Framework

Most agent evaluation toolkits answer one question: *"Did the agent complete the task?"*

This framework answers **five** — and captures an OpenTelemetry-compliant trace that tells you *how*:

| Question | How |
|---|---|
| Did the agent **complete** the task? | Hybrid rule-based + LLM-as-judge score |
| Did it pick the **right tools**? | Precision / recall / F1 with flexible tool-equivalence mapping |
| Is it **safe**? | PII, injection, harmful-content, and budget-violation checks |
| What does it **cost** to run? | Per-call token + cost tracking, agent vs. judge separation |
| Is it **healthy** over time? | Rolling-window success rate, latency percentiles, drift detection |

> **Novel contributions:** The full OpenTelemetry GenAI Semantic Convention span tree (portable to Datadog, Splunk, Phoenix, Langfuse) and the flexible tool-equivalence mapping are purpose-built for production agent observability — not found in standard evaluation toolkits.

---

## At a Glance

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       Agent Evaluation Pipeline                              │
├──────────┬─────────────────┬──────────────────────────┬──────────────────────┤
│  DATA    │     AGENTS      │       EXECUTION          │      EVALUATION       │
│          │                 │                          │                       │
│ Mind2Web │ Single ReAct    │ Hybrid real/mock tools   │ Hybrid score          │
│ web-nav  │ + multi-agent   │ OTel span tracing        │ (rule + LLM judge)    │
│ benchmark│ supervisor      │ token + cost tracking    │ Tool correctness F1   │
│ (NeurIPS │ (planner /      │ health monitoring        │ Safety validation     │
│  2023)   │  navigator /    │                          │ Cost & latency        │
│          │  validator)     │                          │ Audit-ready traces    │
└──────────┴─────────────────┴──────────────────────────┴──────────────────────┘
```

---

## OpenTelemetry-Compliant Tracing

Every agent, LLM call, and tool invocation creates a **span** that follows the
[OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).
Spans form a parent-child tree and export to **OTLP JSON** — portable to any
observability backend (Datadog, Splunk, Phoenix, Langfuse, Jaeger).

```
task.execute                                  ← root span
├── agent.planner.plan          (gen_ai.agent.role = task_decomposer)
├── agent.navigator.execute     (gen_ai.agent.role = executor)
│   ├── tool.execute            (gen_ai.tool.name = site_search)
│   ├── tool.execute            (gen_ai.tool.name = filter_content)
│   └── tool.execute            (gen_ai.tool.name = site_navigation)
└── agent.validator.validate    (gen_ai.evaluation.score = 0.82)
```

Standard attributes captured per span: `gen_ai.system`, `gen_ai.request.model`,
`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.agent.name`,
`gen_ai.tool.name`, `gen_ai.usage.cost_usd`, plus custom evaluation extensions.

---

## Evaluation Metrics

### 🎯 Task Completion (HybridEvaluator)

| Component | Method | Weight |
|---|---|---|
| **Length adequacy** | ≥ 40 words in the plan | 0.2 (rule) |
| **Specificity** | Contains numeric detail | 0.1 (rule) |
| **Goal alignment** | Task ↔ plan keyword overlap | 0.3 (rule) |
| **Structured format** | Uses action verbs (CLICK, TYPE, …) | 0.2 (rule) |
| **Action overlap** | Plan verbs ↔ reference verbs | 0.2 (rule) |
| **LLM judge** | Holistic 0.0–1.0 quality rating | 0.6 (total) |

`total_score = 0.4 × rule_score + 0.6 × llm_score` (weights configurable)

### 🛠️ Tool Correctness (ToolCorrectnessEval)

| Metric | Method |
|---|---|
| **Precision / Recall / F1** | Predicted vs. reference tools, flexible equivalence |
| **Exact Match** | No missing or extra tools |
| **Order Accuracy** | Longest common subsequence vs. reference order |

> **Flexible tool equivalence** recognizes that Mind2Web's basic action set
> (CLICK, TYPE) maps to the agent's richer toolset — `site_navigation ≈ site_search ≈
> filter_content` for navigation, `site_search ≈ web_search` for search.

### 🛡️ Safety (SafetyValidator)

| Check | Detects |
|---|---|
| **PII** | SSN, credit card, email, phone |
| **Injection** | XSS, SQL injection, code injection |
| **Harmful content** | Jailbreak / bypass / exploit keywords |
| **Financial** | Prices exceeding a stated budget |

---

## Single-Agent vs. Multi-Agent

The framework runs the **same evaluation** against two architectures so you can
quantify the cost/quality trade-off of orchestration:

| Architecture | Pattern | When it helps |
|---|---|---|
| **Baseline** | One ReAct agent with all tools | Simple, fast, lower cost |
| **Multi-agent** | Supervisor → Planner → Navigator → Validator | Complex multi-step tasks, better decomposition, full span tree |

Both produce identical metrics (task score, tool F1, cost, latency), enabling a
direct head-to-head comparison.

---

## Hybrid Real + Mock Tool Environment

The industry-standard approach for **safe** pre-deployment agent evaluation:

```
READ tools     (real when API keys present, mock fallback)
  web_search · site_search · get_price_info · check_availability
  site_navigation · filter_content · get_page_info

WRITE tools    (ALWAYS mocked — no real-world side effects)
  book_reservation · make_phone_call · submit_form · make_purchase

COMPUTE        (always real)
  budget_calculator
```

This balances **rigor** (real agent behavior, real web data via Tavily) with
**safety** (no real bookings, purchases, or form submissions). Set `TAVILY_API_KEY`
to enable live web search; without it, READ tools return realistic mock data.

---

## Quickstart

### Option A — Google Colab (no local install)

> **Azure OpenAI with IP allowlisting:** Colab runs on Google Cloud — its IPs are
> typically not on corporate allowlists. Use Option B (local) for Azure deployments
> behind a firewall.

Click the **Open in Colab** badge at the top of this README, then:

1. Add your credentials as Colab Secrets (🔑 in the left sidebar):

   | Secret name | Example value |
   |---|---|
   | `OPENAI_API_KEY` | `sk-...` |
   | `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
   | `OPENAI_API_VERSION` | `2025-04-01-preview` *(Azure only — leave blank for OpenAI)* |
   | `AGENT_MODEL` | `gpt-4o` |
   | `JUDGE_MODEL` | `gpt-4o` |
   | `TAVILY_API_KEY` | `tvly-...` *(optional — enables real web search)* |

2. Run all cells in order.

---

### Option B — Local Setup

#### 1. Clone and install

```bash
git clone https://github.com/minw0607/multi_agent_otel_eval.git
cd multi_agent_otel_eval
pip install -r requirements.txt
```

#### 2. Configure your LLM provider

```bash
cp .env.example .env
# Edit .env — uncomment the section for your provider and fill in credentials
```

The provider is **auto-detected** from `OPENAI_API_VERSION`:
- **Set** (e.g. `2025-04-01-preview`) → Azure OpenAI (`AzureChatOpenAI`)
- **Blank** → OpenAI direct, Ollama, Groq, or any compatible endpoint (`ChatOpenAI`)

See [docs/provider-setup.md](docs/provider-setup.md) for step-by-step instructions per provider.

#### 3. Run the demo notebook

```bash
jupyter notebook demo_notebook.ipynb
```

Run cells in order. The Mind2Web dataset streams from HuggingFace on first run and
caches locally — subsequent runs skip the download.

---

## Repo Structure

```
multi_agent_otel_eval/
├── README.md
├── requirements.txt
├── .env.example                    ← Copy to .env and fill in credentials (never committed)
├── .gitignore
├── LICENSE
│
├── demo_notebook.ipynb             ← ★ Start here — coding-light, all heavy lifting in src/
├── Enhanced_Agentic_Framework_Multi_Agent.ipynb   ← Original monolithic research notebook
│
├── src/                            ← Importable Python modules
│   ├── config.py                   ← Provider-agnostic LLM factory, reads from .env
│   ├── tracer.py                   ← OTel spans (HierarchicalTracer) + ExecutionTrace
│   ├── monitors.py                 ← CostTracker + HealthMonitor (rolling window)
│   ├── safety.py                   ← PII, injection, harmful-content, budget checks
│   ├── tools.py                    ← Hybrid real/mock web-navigation tools
│   ├── dataset.py                  ← Mind2Web streaming loader with local cache
│   ├── agents.py                   ← Baseline ReAct + multi-agent supervisor pipeline
│   ├── evaluator.py                ← HybridEvaluator + ToolCorrectnessEval
│   └── visualizer.py               ← Eval dashboard, trace tree, telemetry charts
│
├── docs/
│   └── provider-setup.md           ← Step-by-step setup for Azure, OpenAI, Ollama, Groq
│
└── outputs/                        ← Results, traces, charts (gitignored)
    ├── traces/                     ← OTLP JSON span exports
    ├── data/                       ← Cached Mind2Web dataset
    ├── eval_dashboard.png
    └── quick_test_results_*.csv
```

---

## Provider Compatibility

The framework auto-detects your provider from `.env` — no code changes required.

| `OPENAI_API_VERSION` | Provider | LangChain class |
|---|---|---|
| Set (e.g. `2025-04-01-preview`) | Azure OpenAI | `AzureChatOpenAI` |
| Blank | OpenAI / Ollama / Groq / etc. | `ChatOpenAI` |

**Supported providers:**

| Provider | `OPENAI_BASE_URL` | Notes |
|---|---|---|
| **Azure OpenAI** | `https://<resource>.openai.azure.com` | Set `OPENAI_API_VERSION` |
| **OpenAI (direct)** | `https://api.openai.com/v1` | Default |
| **Ollama** (local) | `http://localhost:11434/v1` | Free, no API key |
| **Groq** | `https://api.groq.com/openai/v1` | — |
| **Together AI** | `https://api.together.xyz/v1` | — |
| **LM Studio** | `http://localhost:1234/v1` | — |

See [docs/provider-setup.md](docs/provider-setup.md) for step-by-step setup and troubleshooting.

---

## About the Mind2Web Dataset

[**Mind2Web**](https://osu-nlp-group.github.io/Mind2Web/) is the first large-scale
benchmark for evaluating AI agents that perform web-navigation tasks from natural
language instructions. Published at **NeurIPS 2023** (Spotlight) by The Ohio State
University, it spans 2,000+ tasks across 137 real websites and 31 domains.

This framework streams the lightweight text metadata (task descriptions, websites,
reference action sequences) from the `osunlp/Multimodal-Mind2Web` HuggingFace
dataset, skipping the heavy HTML and screenshot fields.

---

## Limitations

- **LLM-as-judge bias**: When the agent and judge use the same model, self-evaluation biases toward higher scores. A separate judge model is preferable when budget allows — set `JUDGE_MODEL` to a different model.
- **Hypothetical-plan scoring**: The evaluator scores the agent's *plan* (intended actions), not live browser execution. Mind2Web tasks run against mock/scraped data, not the live target site.
- **Tool-equivalence mapping**: The flexible mapping is tuned for Mind2Web's basic action set. Custom toolsets may need their own equivalence rules in `evaluator.py`.
- **Synchronous execution**: The evaluation loop is single-threaded. For large runs (>100 tasks), parallelise at the process level.

---

## Citation

If you use this framework, please cite the Mind2Web benchmark:

```bibtex
@inproceedings{deng2023mind2web,
  title={Mind2Web: Towards a Generalist Agent for the Web},
  author={Deng, Xiang and Gu, Yu and Zheng, Boyuan and Chen, Shijie and
          Stevens, Samuel and Wang, Boshi and Sun, Huan and Su, Yu},
  booktitle={NeurIPS},
  year={2023}
}
```

And the OpenTelemetry GenAI Semantic Conventions:

```
https://opentelemetry.io/docs/specs/semconv/gen-ai/
```

---

<div align="center">

Made with ❤️ for rigorous, observable agent evaluation  
[Open an issue](https://github.com/minw0607/multi_agent_otel_eval/issues) · [Provider setup guide](docs/provider-setup.md)

</div>
