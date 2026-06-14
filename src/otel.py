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

from typing import Optional


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
