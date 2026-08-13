"""
Real OpenTelemetry integration via Arize Phoenix (optional).

This upgrades the framework from hand-rolled OTLP-shaped JSON to **real
OpenTelemetry spans** that stream to a live backend. Phoenix is free, runs
locally (`pip install arize-phoenix`), and auto-instruments LangChain/LangGraph
through OpenInference — so once `setup_phoenix()` is called, every agent run,
LLM call, and tool invocation is captured as a genuine OTel span with no further
code changes.

Everything here degrades gracefully: if the optional packages aren't installed,
`setup_phoenix()` prints an install hint and returns None, and the rest of the
framework (local `HierarchicalTracer`, dashboards, reports) keeps working.

Install:
    pip install arize-phoenix openinference-instrumentation-langchain \
                opentelemetry-sdk opentelemetry-exporter-otlp

Backends other than Phoenix (Datadog, Jaeger, Grafana Tempo, Langfuse, …) work
too — point `endpoint` at any OTLP collector.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def setup_phoenix(
    project_name: str = "multi-agent-otel-eval",
    endpoint: Optional[str] = None,
    launch_local: bool = True,
):
    """
    Register a real OpenTelemetry tracer provider and auto-instrument LangChain.

    Args:
        project_name : project/grouping name shown in the Phoenix UI.
        endpoint     : OTLP collector endpoint. None → Phoenix default
                       (http://localhost:6006). Set this to ship to Datadog,
                       Jaeger, Grafana Tempo, Langfuse, etc.
        launch_local : if True and no endpoint is given, start a local Phoenix
                       app (UI at http://localhost:6006).

    Returns the tracer provider on success, or None if Phoenix/OpenInference
    are not installed (with an install hint printed).
    """
    try:
        from phoenix.otel import register
    except ImportError:
        print("⚠️  Phoenix not installed — real OTel export is disabled.\n"
              "    Install with:\n"
              "      pip install arize-phoenix openinference-instrumentation-langchain \\\n"
              "                  opentelemetry-sdk opentelemetry-exporter-otlp\n"
              "    The framework still works with the local HierarchicalTracer.")
        return None

    # Optionally spin up a local Phoenix app (no-op if one is already running).
    if launch_local and endpoint is None:
        try:
            import phoenix as px
            if getattr(px, "active_session", lambda: None)() is None:
                px.launch_app()
                print("🌐 Phoenix UI: http://localhost:6006")
        except Exception as e:
            print(f"   (Could not auto-launch Phoenix app: {e} — continuing.)")

    try:
        kwargs = {"project_name": project_name, "auto_instrument": True}
        if endpoint:
            kwargs["endpoint"] = endpoint
        tracer_provider = register(**kwargs)
        print(f"✅ Real OpenTelemetry tracing active → project '{project_name}'"
              + (f" → {endpoint}" if endpoint else " → Phoenix (localhost:6006)"))
        print("   LangChain / LangGraph runs are now auto-traced as OTel spans.")
        return tracer_provider
    except Exception as e:
        print(f"⚠️  Could not register OTel tracer provider: {e}")
        return None


# ---------------------------------------------------------------------------
# Real token/cost accounting via LangChain usage callbacks
# ---------------------------------------------------------------------------

def make_usage_callback():
    """
    Return a LangChain callback that aggregates real per-model token usage from
    API responses, or None if unavailable. Attach via config={"callbacks": [cb]}.
    """
    try:
        from langchain_core.callbacks import UsageMetadataCallbackHandler
        return UsageMetadataCallbackHandler()
    except Exception:
        return None


def usage_from_callback(cb):
    """
    Extract (input_tokens, output_tokens) summed across models from a usage
    callback, or None if no real usage was captured (caller should fall back to
    a tiktoken estimate).

    Kept for backwards compatibility. New code should prefer `LLMCallRecorder`,
    which captures the full `Usage` payload per call rather than an aggregate.
    """
    if cb is None:
        return None
    meta = getattr(cb, "usage_metadata", None)
    if not meta:
        return None
    tin = tout = 0
    for usage in meta.values():
        tin  += usage.get("input_tokens", 0)
        tout += usage.get("output_tokens", 0)
    if tin == 0 and tout == 0:
        return None
    return tin, tout


# ---------------------------------------------------------------------------
# Full usage payload — per-call, with provenance
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    """
    One LLM call's token usage, with provenance.

    `cached_tokens` and `reasoning_tokens` are returned by OpenAI-compatible
    APIs and are routinely discarded by tooling. They are not optional detail:

      cached_tokens     — billed at a different rate, and dependent on cache
                          state the caller does not control. This is why cost
                          is not reproducible even when tokens are.
      reasoning_tokens  — output tokens generated, billed, and never returned
                          to the caller. Measured at 44-78% of output on
                          o-series models (docs/transparency_spec.md §2).

    `source` records whether counts came from the API or a tokenizer estimate.
    `tier` records how much the number can be trusted (spec invariant I3):
      verified  — we counted it ourselves from content we assembled
      trusted   — the vendor reported it; we cannot audit it
      asserted  — reported but uninspectable (reasoning tokens)
    """
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    source: str = "api"          # "api" | "estimated"
    tier: str = "trusted"        # "verified" | "trusted" | "asserted"
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def visible_output_tokens(self) -> int:
        """Output tokens the caller can actually read."""
        return max(self.output_tokens - self.reasoning_tokens, 0)

    @property
    def hidden_share(self) -> float:
        """Fraction of output tokens billed but not returned."""
        return self.reasoning_tokens / self.output_tokens if self.output_tokens else 0.0

    def to_span_attributes(self) -> dict:
        from .tracer import GEN_AI_ATTRIBUTES as A
        attrs = {
            A["INPUT_TOKENS"]:  self.input_tokens,
            A["OUTPUT_TOKENS"]: self.output_tokens,
            A["USAGE_SOURCE"]:  self.source,
            A["USAGE_TIER"]:    self.tier,
        }
        if self.model:
            attrs[A["REQUEST_MODEL"]] = self.model
        if self.reasoning_tokens:
            attrs[A["REASONING_TOKENS"]] = self.reasoning_tokens
        if self.cached_tokens:
            attrs[A["CACHED_TOKENS"]] = self.cached_tokens
        return attrs

    def __add__(self, other: "Usage") -> "Usage":
        if other is None:
            return self
        # Provenance degrades to the weaker of the two.
        source = "api" if self.source == other.source == "api" else "mixed"
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            source=source,
            tier=self.tier if self.tier == other.tier else "trusted",
            model=self.model if self.model == other.model else "",
        )


def extract_usage(response, model: str = "") -> Optional[Usage]:
    """
    Build a `Usage` from a LangChain response or LLMResult generation.

    Reads LangChain's normalized `usage_metadata` first, then falls back to the
    raw provider payload. Both paths were confirmed working against Azure
    (docs/transparency_spec.md §2); reading both makes this resilient to
    langchain-core version drift.
    """
    um = getattr(response, "usage_metadata", None)
    if um:
        return Usage(
            input_tokens=um.get("input_tokens", 0),
            output_tokens=um.get("output_tokens", 0),
            cached_tokens=(um.get("input_token_details") or {}).get("cache_read", 0) or 0,
            reasoning_tokens=(um.get("output_token_details") or {}).get("reasoning", 0) or 0,
            source="api", tier="trusted", model=model,
        )

    rm = getattr(response, "response_metadata", None) or {}
    tu = rm.get("token_usage") or rm.get("usage") or {}
    if tu:
        return Usage(
            input_tokens=tu.get("prompt_tokens", 0),
            output_tokens=tu.get("completion_tokens", 0),
            cached_tokens=(tu.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
            reasoning_tokens=(tu.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
            source="api", tier="trusted", model=rm.get("model_name", model),
        )
    return None


def estimated_usage(input_text: str, output_text: str, model: str = "") -> Usage:
    """Fallback when the provider reports nothing. Explicitly marked `estimated`."""
    def _count(t: str) -> int:
        try:
            import tiktoken
            return len(tiktoken.encoding_for_model("gpt-4").encode(t or ""))
        except Exception:
            return int(len((t or "").split()) * 1.3)
    return Usage(input_tokens=_count(input_text), output_tokens=_count(output_text),
                 source="estimated", tier="trusted", model=model)


# ---------------------------------------------------------------------------
# Per-LLM-call recorder (spec T1)
# ---------------------------------------------------------------------------

@dataclass
class LLMCallRecord:
    """One model invocation, with context composition (spec T4)."""
    index: int
    agent: str
    model: str = ""
    usage: Optional[Usage] = None
    # Context composition — a decomposition of `usage.input_tokens`, never an
    # addition to it (spec invariant I1). Estimated by tokenizer, hence
    # reconciled against the reported total via `residual_tokens`.
    context: Dict[str, int] = field(default_factory=dict)
    n_messages: int = 0
    n_tool_messages: int = 0
    duration_ms: float = 0.0

    @property
    def accounted_tokens(self) -> int:
        return sum(self.context.values())

    @property
    def residual_tokens(self) -> int:
        """
        Reported input minus what the decomposition accounts for.
        A large residual means the decomposition is wrong — surfacing it is the
        honesty mechanism (spec invariant I4).
        """
        reported = self.usage.input_tokens if self.usage else 0
        return reported - self.accounted_tokens


try:  # Proper base class supplies the ignore_* protocol LangChain queries.
    from langchain_core.callbacks import BaseCallbackHandler as _BaseCB
except Exception:  # pragma: no cover - langchain not installed
    class _BaseCB:  # minimal stand-in
        pass


class LLMCallRecorder(_BaseCB):
    """
    LangChain callback that records **every** LLM call individually.

    `UsageMetadataCallbackHandler` aggregates across a whole graph run, which
    makes re-planning, retries, and per-turn context growth unobservable by
    construction. This handler fires per call instead, and additionally
    captures the message list so input context can be decomposed by role.

    Thread safety: spans are opened with an explicit `parent`, so parenting is
    deterministic even when callbacks fire off-thread (see tracer.start_span).
    """

    def __init__(self, agent: str, tracer=None, parent_span=None,
                 model: str = "", tool_schema_tokens: int = 0):
        self.agent = agent
        self.tracer = tracer
        self.parent_span = parent_span
        self.model = model
        self.tool_schema_tokens = tool_schema_tokens
        self.calls: List[LLMCallRecord] = []
        self._pending: Dict[Any, Dict] = {}

    # -- LangChain callback protocol ------------------------------------

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        """Capture the actual message list — the basis for context decomposition."""
        import time as _t
        flat = messages[0] if messages and isinstance(messages[0], list) else (messages or [])
        self._pending[run_id] = {"messages": flat, "start": _t.time()}

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        import time as _t
        self._pending.setdefault(run_id, {"messages": [], "start": _t.time()})

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        import time as _t
        pending = self._pending.pop(run_id, {})
        msgs = pending.get("messages", [])
        duration = (_t.time() - pending["start"]) * 1000 if pending.get("start") else 0.0

        usage = None
        gen = None
        try:
            gen = response.generations[0][0]
            usage = extract_usage(getattr(gen, "message", gen), self.model)
        except Exception:
            pass
        if usage is None:
            llm_out = getattr(response, "llm_output", None) or {}
            tu = llm_out.get("token_usage") or {}
            if tu:
                usage = Usage(
                    input_tokens=tu.get("prompt_tokens", 0),
                    output_tokens=tu.get("completion_tokens", 0),
                    cached_tokens=(tu.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
                    reasoning_tokens=(tu.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
                    source="api", tier="trusted", model=self.model,
                )

        rec = LLMCallRecord(
            index=len(self.calls), agent=self.agent, model=self.model,
            usage=usage, duration_ms=duration,
            context=self._decompose(msgs),
            n_messages=len(msgs),
            n_tool_messages=sum(1 for m in msgs if type(m).__name__ == "ToolMessage"),
        )
        self.calls.append(rec)
        self._emit_span(rec)

    def on_llm_error(self, error, *, run_id=None, **kwargs):
        self._pending.pop(run_id, None)

    # -- internals -------------------------------------------------------

    def _decompose(self, messages) -> Dict[str, int]:
        """
        Classify input context by message role (spec T4).

        Typed LangChain messages make this tractable even for the Navigator,
        whose per-turn prompt is assembled by LangGraph rather than by us.
        Tool-definition schemas are added from the caller-supplied count, since
        they travel in the request body rather than the message list.
        """
        def _count(t) -> int:
            if isinstance(t, list):
                t = " ".join(str(x) for x in t)
            try:
                import tiktoken
                return len(tiktoken.encoding_for_model("gpt-4").encode(str(t or "")))
            except Exception:
                return int(len(str(t or "").split()) * 1.3)

        buckets = {"system_tokens": 0, "task_tokens": 0, "tool_output_tokens": 0,
                   "history_tokens": 0, "tool_schema_tokens": self.tool_schema_tokens}
        seen_human = False
        for m in messages:
            kind = type(m).__name__
            n = _count(getattr(m, "content", ""))
            # Tool-call arguments travel in `.tool_calls`, not `.content`, and
            # are resent on every subsequent turn. Omitting them showed up as a
            # persistent positive residual (spec I4 working as intended).
            tc = getattr(m, "tool_calls", None)
            if tc:
                import json as _json
                # Count only what is actually transmitted (function name and
                # arguments), not LangChain's internal id/type bookkeeping.
                n += _count(_json.dumps(
                    [{"name": t.get("name"), "args": t.get("args")} for t in tc],
                    default=str))
            if kind == "SystemMessage":
                buckets["system_tokens"] += n
            elif kind == "ToolMessage":
                buckets["tool_output_tokens"] += n
            elif kind == "HumanMessage" and not seen_human:
                buckets["task_tokens"] += n
                seen_human = True
            else:
                # Prior turns being resent — grows with every ReAct iteration.
                buckets["history_tokens"] += n
        return buckets

    def _emit_span(self, rec: LLMCallRecord):
        if self.tracer is None or self.parent_span is None:
            return
        from .tracer import SPAN_NAMES, GEN_AI_ATTRIBUTES as A, SpanKind
        attrs = {
            A["AGENT_NAME"]: self.agent,
            A["CALL_INDEX"]: rec.index,
            "llm.duration_ms": round(rec.duration_ms, 1),
            "context.accounted_tokens": rec.accounted_tokens,
            "context.residual_tokens": rec.residual_tokens,
        }
        attrs.update({f"context.{k}": v for k, v in rec.context.items()})
        if rec.usage:
            attrs.update(rec.usage.to_span_attributes())
        span = self.tracer.start_span(
            SPAN_NAMES["LLM_CALL"], kind=SpanKind.CLIENT.value,
            attributes=attrs, parent=self.parent_span)
        self.tracer.end_span(span)

    # -- roll-up ---------------------------------------------------------

    def total_usage(self) -> Usage:
        total = Usage(input_tokens=0, output_tokens=0, source="api", model=self.model)
        for c in self.calls:
            if c.usage:
                total = total + c.usage
        return total

    def summary(self) -> Dict[str, Any]:
        t = self.total_usage()
        return {
            "agent": self.agent, "calls": len(self.calls), "model": self.model,
            "input_tokens": t.input_tokens, "output_tokens": t.output_tokens,
            "reasoning_tokens": t.reasoning_tokens, "cached_tokens": t.cached_tokens,
            "hidden_share": round(t.hidden_share, 3),
            "residual_tokens": sum(c.residual_tokens for c in self.calls),
        }


def tool_schema_tokens(tools) -> int:
    """
    Tokens consumed by tool/function definitions, which are resent on EVERY
    turn. Measured at 1,104 tokens for the 11 Mind2Web tools versus a 77-token
    system prompt — a 14x ratio that is invisible in standard accounting.
    """
    try:
        import json as _json, tiktoken
        from langchain_core.utils.function_calling import convert_to_openai_tool
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(_json.dumps([convert_to_openai_tool(t) for t in tools])))
    except Exception:
        return 0
