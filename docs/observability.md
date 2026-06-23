# Observability Guide — OpenTelemetry & Backends

A systematic guide to **how this framework emits telemetry** and **how to view it in any
backend** (Phoenix, Jaeger, Grafana Tempo, Datadog, Splunk, Langfuse, or a generic
OpenTelemetry Collector).

If you've never used distributed tracing for LLM apps, start at §1. If you just want to
wire up a specific backend, jump to §6.

---

## 1. Why observability for agents

A single agent run is a black box: a prompt goes in, an answer comes out. A *multi-agent*
run is worse — a supervisor routes to a planner, which hands a plan to a navigator, which
calls tools, which feeds a validator. When something is slow, expensive, or wrong, you
need to see **what happened, in what order, for how long, and at what cost**.

That record is a **trace**. Producing and shipping traces in a standard format is what
OpenTelemetry does.

---

## 2. Core concepts

| Term | Meaning |
|---|---|
| **Span** | One unit of work — an LLM call, a tool invocation, an agent step. Has a name, start/end time, status, and **attributes**. |
| **Trace** | A tree of spans for one end-to-end operation (e.g. one Mind2Web task). Spans link via parent/child IDs. |
| **Attributes** | Key/value metadata on a span (model name, token counts, cost, tool name…). |
| **OTel (OpenTelemetry)** | The vendor-neutral **standard** for generating, collecting, and exporting telemetry. CNCF project. |
| **OTLP** | OpenTelemetry Protocol — the wire format spans travel in (gRPC on port **4317**, HTTP on **4318**). Every major backend ingests OTLP. |
| **GenAI Semantic Conventions** | OTel's standard attribute names for LLM apps: `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.agent.name`, `gen_ai.tool.name`, etc. Using them makes traces portable. |
| **Instrumentation** | Code that *produces* spans. Can be manual or automatic (auto-patches a library). |
| **OpenInference** | Arize's open instrumentation standard for AI frameworks. `openinference-instrumentation-langchain` auto-emits OTel spans for LangChain/LangGraph. |

---

## 3. The pipeline: producer → protocol → backend

The single most important mental model. These are **layers, not alternatives**:

```
  Your LangChain / LangGraph agents
        │
        │  (A) INSTRUMENTATION  ── the "producer"
        │      OpenInference auto-patches LangChain so each call emits an OTel span
        ▼
  OpenTelemetry SDK
        │
        │  (B) OTLP  ── the open "wire" protocol (gRPC :4317 / HTTP :4318)
        ▼
  A BACKEND  ── the "consumer": stores, indexes, and visualizes spans
        Phoenix · Jaeger · Tempo · Datadog · Splunk · Langfuse · …
```

- **(A) is the same regardless of backend.** Swapping Phoenix for Datadog changes only
  *where* spans are sent, not how they're produced.
- **(B) is a standard.** Any OTLP-compatible backend works without code changes — usually
  just an endpoint + auth header.

> **"Phoenix vs LangChain OTel" is a category error.** The LangChain instrumentation is the
> *producer* (A); Phoenix is one possible *backend* (B). They sit in the same pipeline.

---

## 4. How this framework emits telemetry

This repo has **two independent tracing systems** that run in parallel:

| Tracer | Module | Output | Purpose |
|---|---|---|---|
| **Local `HierarchicalTracer`** | `src/tracer.py` | OTLP-shaped JSONL (`outputs/traces/`) + matplotlib trace-tree chart | Offline artifacts for the audit report; zero dependencies; always on |
| **Real OpenTelemetry** | `src/otel.py` → `setup_phoenix()` | Live OTel spans over OTLP to a backend | Production-grade, interactive observability |

They are complementary: the local tracer feeds the **report** (Step 13), real OTel feeds the
**live UI**. Enabling one does not disable the other.

To turn on real OTel:

```python
from src import setup_phoenix
setup_phoenix()                                  # local Phoenix at http://localhost:6006
setup_phoenix(endpoint="http://collector:4317")  # or any OTLP endpoint
```

Once called, `openinference-instrumentation-langchain` auto-traces every agent/LLM/tool
call — **no other code changes needed**.

---

## 5. Reading a trace

In any backend you'll see the same structure (the names below are what this framework
produces):

```
LangGraph                         ← root span for one multi-agent task (kind: chain)
├── AzureChatOpenAI               ← Planner LLM call      (kind: llm)
│   └── ChatCompletion            ←   underlying OpenAI client call
├── AzureChatOpenAI               ← Navigator LLM call    (kind: llm)
│   ├── ChatCompletion
│   └── (tool calls)
└── AzureChatOpenAI               ← Validator LLM call    (kind: llm)
```

How to read it:

- **`kind`**: `chain` = a LangGraph/LangChain step, `llm` = a model call, `tool` = a tool.
- **Root vs nested**: a "root span" is the top of one task (the `LangGraph` run); nested
  spans are the steps inside it. Most UIs let you filter to root spans only.
- **Identify the specialist** by the system prompt in the span's `input` —
  `"You are a PLANNING…"`, `"…NAVIGATION…"`, `"…VALIDATION…"`, or `"You are a model-risk…"`
  (the report's AI-Assessment judge).
- **Per-span vs per-task latency**: span-level P50 (e.g. ~1 s) is one LLM call; a whole task
  is many spans (e.g. ~20 s). Don't confuse the two.
- **Cost**: backends compute cost from the `gen_ai.usage.*` token attributes; the cumulative
  figure spans *all* runs in the selected time window, not one run.

**To debug:** open a single trace → inspect each span's input/output. This is how you find
why an agent chose the wrong tool or produced a low-scoring answer.

---

## 6. Choosing a backend

| Backend | Best for | Hosting | Cost | LLM-aware UI |
|---|---|---|---|---|
| **Phoenix** (Arize) | LLM/agent dev, local-first | Local or cloud | Free (OSS) | ✅ purpose-built |
| **Langfuse** | LLM product analytics, prompt mgmt | Cloud or self-host | Free tier / OSS | ✅ purpose-built |
| **Jaeger** | General distributed tracing, quick local | Local (Docker) | Free (OSS) | ⚪ generic |
| **Grafana Tempo** | Traces alongside metrics/logs in Grafana | Self-host / Grafana Cloud | Free (OSS) / paid | ⚪ generic |
| **Datadog** | Enterprise APM + LLM Observability | SaaS | Paid | ✅ LLM Obs module |
| **Splunk** | Enterprise observability/SIEM shops | SaaS / self-host | Paid | ⚪ generic APM |

**Rule of thumb:** Phoenix or Langfuse for AI-native development; Datadog/Splunk if your org
already runs them; Jaeger/Tempo for a free, self-hosted general-purpose option.

---

## 7. Backend setup tutorials

All backends speak OTLP, so the pattern is always: **point the OTLP exporter at the
backend's endpoint, add any required auth header.** The cleanest vendor-neutral way is the
standard OpenTelemetry environment variables (read automatically by the SDK):

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://<host>:4317"      # gRPC (or :4318 for HTTP)
export OTEL_EXPORTER_OTLP_PROTOCOL="grpc"                    # or "http/protobuf"
export OTEL_EXPORTER_OTLP_HEADERS="authorization=Bearer ..." # backend-specific, optional
export OTEL_SERVICE_NAME="multi-agent-otel-eval"
```

Then initialize tracing once (see the generic snippet in §8). For Phoenix specifically, the
repo's `setup_phoenix()` is the shortcut.

### 7.1 Phoenix (recommended starting point)

```bash
pip install arize-phoenix openinference-instrumentation-langchain \
            opentelemetry-sdk opentelemetry-exporter-otlp
```

```python
from src import setup_phoenix
setup_phoenix()                  # launches local UI at http://localhost:6006, ingests on :4317
```

- **Phoenix Cloud:** sign up, then
  `setup_phoenix(endpoint="https://app.phoenix.arize.com/v1/traces", launch_local=False)`
  and set `OTEL_EXPORTER_OTLP_HEADERS="api_key=<your-key>"`.

### 7.2 Jaeger

Jaeger ingests OTLP natively (v1.35+).

```bash
docker run --rm -p 16686:16686 -p 4317:4317 -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
```

Run the generic init (§8). View traces at **http://localhost:16686** (pick service
`multi-agent-otel-eval`).

### 7.3 Grafana Tempo

Run Tempo (Docker) with OTLP enabled, then add it as a data source in Grafana.

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"   # Tempo OTLP ingest
```

Explore traces in Grafana → Explore → Tempo. Best when you already use Grafana for metrics
and want traces in the same pane.

### 7.4 Datadog

Two common paths:

**A) Via the Datadog Agent's OTLP receiver** (recommended):
1. Install the Datadog Agent with `DD_API_KEY` and `DD_SITE` set.
2. Enable its OTLP receiver (in `datadog.yaml` or via
   `DD_OTLP_CONFIG_RECEIVER_PROTOCOLS_GRPC_ENDPOINT=0.0.0.0:4317`).
3. Point the app at the agent:
   ```bash
   export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
   ```
View under **APM → Traces** (and **LLM Observability** if enabled).

**B) Via the OpenTelemetry Collector** with the `datadog` exporter — use this when you want
one collector fanning out to several backends (see §7.7).

### 7.5 Splunk Observability Cloud

Send OTLP/HTTP directly to the Splunk ingest endpoint (replace `<realm>`, e.g. `us1`):

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://ingest.<realm>.signalfx.com"
export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
export OTEL_EXPORTER_OTLP_HEADERS="X-SF-Token=<access-token>"
```

Or run the **Splunk Distribution of the OpenTelemetry Collector** and point the app at it
(localhost:4317). View under **APM**.

### 7.6 Langfuse

Langfuse exposes an OTLP endpoint and works with OpenInference.

```bash
# Basic-auth header = base64("<public-key>:<secret-key>")
AUTH=$(printf "pk-...:sk-..." | base64)
export OTEL_EXPORTER_OTLP_ENDPOINT="https://cloud.langfuse.com/api/public/otel"
export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic ${AUTH}"
```

(Self-hosted Langfuse: swap the host.) View traces in the Langfuse UI.

### 7.7 Generic OpenTelemetry Collector (the universal hub)

The Collector is a standalone process that receives OTLP and **fans out to one or more
backends** — the most flexible setup. Minimal `config.yaml`:

```yaml
receivers:
  otlp:
    protocols:
      grpc: { endpoint: 0.0.0.0:4317 }
      http: { endpoint: 0.0.0.0:4318 }
exporters:
  debug: {}                 # prints spans to stdout
  # datadog: { api: { key: ${DD_API_KEY}, site: datadoghq.com } }
  # otlphttp/langfuse: { endpoint: https://cloud.langfuse.com/api/public/otel, headers: { Authorization: "Basic ..." } }
service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [debug]    # add your backend exporters here
```

Run it, then point every app at `http://localhost:4317`. Switch backends by editing the
collector config — no app changes.

---

## 8. The vendor-neutral init snippet

For any non-Phoenix backend, this ~12-line setup uses the **same OpenInference producer**
with a standard OTLP exporter, honoring the `OTEL_EXPORTER_OTLP_*` env vars from §7:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.langchain import LangChainInstrumentor

provider = TracerProvider(resource=Resource.create({"service.name": "multi-agent-otel-eval"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))  # reads OTEL_EXPORTER_OTLP_* env
trace.set_tracer_provider(provider)
LangChainInstrumentor().instrument()   # auto-trace LangChain/LangGraph
```

> Want this as a built-in? It can be added to the repo as `setup_otel(endpoint, headers)`
> alongside `setup_phoenix()` — open an issue or ask.

---

## 9. Cost & token attributes

Backends compute cost from the GenAI usage attributes this framework sets:

| Attribute | Meaning |
|---|---|
| `gen_ai.usage.input_tokens` | Prompt tokens (real, from API usage metadata) |
| `gen_ai.usage.output_tokens` | Completion tokens |
| `gen_ai.usage.cost_usd` | Per-call cost (computed from the rate table in `src/config.py`) |
| `gen_ai.request.model` | Model/deployment used |
| `gen_ai.agent.name` / `gen_ai.agent.role` | Which specialist (supervisor/planner/navigator/validator) |
| `gen_ai.tool.name` | Tool invoked by the navigator |

Some backends (Phoenix, Datadog LLM Obs) auto-price known models; for custom Azure
deployments, keep `COST_PER_1M_TOKENS` in `src/config.py` accurate so the framework's own
cost is authoritative.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Phoenix not installed` warning | optional deps missing | `pip install arize-phoenix openinference-instrumentation-langchain opentelemetry-sdk opentelemetry-exporter-otlp` |
| No spans appear in the backend | wrong endpoint/port, or init ran *after* the agent | Verify `OTEL_EXPORTER_OTLP_ENDPOINT`; call `setup_*` **before** running tasks |
| Spans rejected (401/403) | missing/invalid auth header | Check `OTEL_EXPORTER_OTLP_HEADERS` (token, base64 basic-auth) |
| gRPC connection refused | backend listening on HTTP only | Use `:4318` and `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf` |
| Cost shows $0 | backend can't price the model | Rely on the framework's own cost (`config.py` rates) |
| Too many traces from old runs | accumulated history | Use the backend's time-range filter or per-run session |

---

## 11. Further reading

- OpenTelemetry GenAI Semantic Conventions — https://opentelemetry.io/docs/specs/semconv/gen-ai/
- OpenInference — https://github.com/Arize-ai/openinference
- Arize Phoenix — https://arize.com/docs/phoenix
- OpenTelemetry Collector — https://opentelemetry.io/docs/collector/
- Langfuse OpenTelemetry — https://langfuse.com/docs/opentelemetry/get-started

---

← Back to the [README](../README.md) · See also [provider-setup.md](provider-setup.md)
