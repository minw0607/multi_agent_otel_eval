# Provider Setup Guide

The framework auto-detects your LLM provider from a single environment variable,
`OPENAI_API_VERSION`:

| `OPENAI_API_VERSION` | Provider | LangChain class used |
|---|---|---|
| **Set** (e.g. `2025-04-01-preview`) | Azure OpenAI | `AzureChatOpenAI` |
| **Blank / unset** | OpenAI, Ollama, Groq, Together, LM Studio | `ChatOpenAI` |

All configuration lives in `.env` (copy from `.env.example`). No code changes needed
to switch providers.

---

## Azure OpenAI

> **Reference setup for this demo:** an **IP-allowlisted endpoint with API-key auth**
> (no interactive Microsoft login). This keeps evaluation runs uninterrupted — no
> login popup on every kernel restart, and long eval loops never break on a token
> expiry. The trade-off is that **Google Colab cannot connect** (its IPs aren't on
> the allowlist), so run the notebook **locally**. If you instead use an
> IP-independent endpoint that requires interactive login, you'd need to add
> `azure-identity` and a bearer-token provider to `src/config.py` — not covered here,
> because it adds friction that hurts a live demo.

### Standard endpoint

```bash
OPENAI_API_KEY=your-azure-key
OPENAI_BASE_URL=https://<your-resource>.openai.azure.com
OPENAI_API_VERSION=2025-04-01-preview
AGENT_MODEL=gpt-4o          # your DEPLOYMENT name (not the base model name)
JUDGE_MODEL=gpt-4o          # your DEPLOYMENT name
```

- `OPENAI_BASE_URL` is your **resource endpoint** from Azure Portal → *Keys and Endpoint*.
- `AGENT_MODEL` / `JUDGE_MODEL` must be your **deployment names**, which may differ
  from the underlying model (e.g. a deployment named `gpt-4o-prod`).
- Find your `OPENAI_API_VERSION` in Azure Portal → *Model deployments*.

### Behind an API Management (APIM) gateway

Some corporate Azure setups route through an APIM gateway that requires a custom
subscription-key header:

```bash
OPENAI_API_KEY=your-gateway-key
OPENAI_BASE_URL=https://your-gateway-host/your-path
OPENAI_API_VERSION=2025-04-01-preview
AGENT_MODEL=gpt-4o
JUDGE_MODEL=gpt-4o

# Custom header (only if your gateway requires one)
OPENAI_APIM_HEADER_NAME=Your-Subscription-Key-Header
OPENAI_APIM_SUBSCRIPTION_KEY=your-subscription-key
```

The custom header is added automatically when `OPENAI_APIM_HEADER_NAME` is set.

> **IP allowlisting:** If your Azure deployment restricts inbound IPs, Google Colab
> will be blocked (it runs on Google Cloud IPs). Run locally instead.

---

## OpenAI (direct)

```bash
OPENAI_API_KEY=sk-...
AGENT_MODEL=gpt-4o
JUDGE_MODEL=gpt-4o
# OPENAI_BASE_URL defaults to https://api.openai.com/v1
# OPENAI_API_VERSION must be BLANK
```

Use `gpt-4o-mini` for a cheaper agent and `gpt-4o` for the judge to cut cost:

```bash
AGENT_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o
```

---

## Ollama (local, free)

[Install Ollama](https://ollama.com), then pull a model:

```bash
ollama pull llama3
```

```bash
OPENAI_API_KEY=ollama          # any non-empty string
OPENAI_BASE_URL=http://localhost:11434/v1
AGENT_MODEL=llama3
JUDGE_MODEL=llama3
# OPENAI_API_VERSION must be BLANK
```

No API key cost. Tool-calling quality is lower than GPT-4o — expect lower tool-F1 scores.

---

## Groq

```bash
OPENAI_API_KEY=gsk_...
OPENAI_BASE_URL=https://api.groq.com/openai/v1
AGENT_MODEL=llama-3.1-70b-versatile
JUDGE_MODEL=llama-3.1-70b-versatile
# OPENAI_API_VERSION must be BLANK
```

Very fast inference. Check Groq's model list for current tool-calling-capable models.

---

## Together AI

```bash
OPENAI_API_KEY=your-together-key
OPENAI_BASE_URL=https://api.together.xyz/v1
AGENT_MODEL=meta-llama/Llama-3-70b-chat-hf
JUDGE_MODEL=meta-llama/Llama-3-70b-chat-hf
# OPENAI_API_VERSION must be BLANK
```

---

## LM Studio (local)

Start the LM Studio local server, then:

```bash
OPENAI_API_KEY=lm-studio       # any non-empty string
OPENAI_BASE_URL=http://localhost:1234/v1
AGENT_MODEL=your-loaded-model-name
JUDGE_MODEL=your-loaded-model-name
# OPENAI_API_VERSION must be BLANK
```

---

## Optional: Real Web Search (Tavily)

Without a Tavily key, the READ tools (`web_search`, `site_search`, `get_price_info`,
`check_availability`) return realistic **mock** data. To enable **live** web search:

```bash
TAVILY_API_KEY=tvly-...
```

Get a free key at [tavily.com](https://tavily.com). WRITE tools (bookings, purchases,
forms) remain mocked regardless — the framework never executes real transactions.

---

## Cost Tuning

Update per-model rates in `src/config.py` (`COST_PER_1M_TOKENS`) to match your
provider's pricing. The framework tracks **agent** and **judge** costs separately so
you can see exactly what evaluation overhead adds on top of agent execution.

Evaluation settings (override in `.env`):

```bash
EVAL_PASS_THRESHOLD=0.7        # min total score to "pass"
RULE_WEIGHT=0.4                # weight on rule-based score
LLM_WEIGHT=0.6                 # weight on LLM-judge score
QUICK_TEST_N=10                # tasks in the quick test
MIND2WEB_TARGET_TASKS=300      # tasks to stream from HuggingFace
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `AuthenticationError` | Wrong key or endpoint | Verify `OPENAI_API_KEY` and `OPENAI_BASE_URL` |
| Azure: `DeploymentNotFound` | `AGENT_MODEL` is a base model name, not a deployment | Use your Azure **deployment** name |
| `Connection refused` (Ollama/LM Studio) | Local server not running | Start the local server first |
| Tool calls always empty | Model lacks tool-calling support | Use a tool-calling-capable model (GPT-4o, Llama-3.1-70b) |
| Judge returns un-parseable scores | Weak judge model | Set `JUDGE_MODEL` to a stronger model |
| Colab + Azure timeout | IP allowlisting blocks Colab | Run locally |
