"""
OTel-compliant tracing for multi-agent execution.

Provides two layers:
  - ExecutionTrace / TracingManager : lightweight per-task trace (token, cost, latency)
  - OTelSpan / HierarchicalTracer  : full OpenTelemetry-compliant span tree

Both can be exported to OTLP JSON for any backend (Datadog, Splunk, Phoenix, Langfuse).
"""

import json
import time
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# OTel GenAI Semantic Convention constants
# ---------------------------------------------------------------------------

GEN_AI_ATTRIBUTES = {
    "SYSTEM":            "gen_ai.system",
    "REQUEST_MODEL":     "gen_ai.request.model",
    "OPERATION_NAME":    "gen_ai.operation.name",
    "INPUT_TOKENS":      "gen_ai.usage.input_tokens",
    "OUTPUT_TOKENS":     "gen_ai.usage.output_tokens",
    "AGENT_NAME":        "gen_ai.agent.name",
    "AGENT_ROLE":        "gen_ai.agent.role",
    "TOOL_NAME":         "gen_ai.tool.name",
    "EVAL_SCORE":        "gen_ai.evaluation.score",
    "EVAL_PASSED":       "gen_ai.evaluation.passed",
    "COST_USD":          "gen_ai.usage.cost_usd",
}

SPAN_NAMES = {
    "TASK_ROOT":       "task.execute",
    "AGENT_SUPERVISOR":"agent.supervisor.route",
    "AGENT_PLANNER":   "agent.planner.plan",
    "AGENT_NAVIGATOR": "agent.navigator.execute",
    "AGENT_VALIDATOR": "agent.validator.validate",
    "LLM_CALL":        "llm.chat.completion",
    "TOOL_CALL":       "tool.execute",
    "EVAL_TASK":       "evaluation.task_completion",
    "EVAL_TOOL":       "evaluation.tool_correctness",
    "EVAL_SAFETY":     "evaluation.safety_check",
}


# ---------------------------------------------------------------------------
# Lightweight per-task trace
# ---------------------------------------------------------------------------

@dataclass
class ExecutionTrace:
    """Lightweight trace captured per agent task."""
    task_id: int
    start_time: float
    end_time: Optional[float] = None

    reasoning_steps: List[str]       = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    decision_points: List[Dict]      = field(default_factory=list)
    errors: List[str]                = field(default_factory=list)

    agent_input_tokens: int  = 0
    agent_output_tokens: int = 0
    judge_input_tokens: int  = 0
    judge_output_tokens: int = 0

    agent_cost: float = 0.0
    judge_cost: float = 0.0
    total_cost: float = 0.0
    latency_ms: float = 0.0
    retries: int      = 0
    tokens_source: str = "estimated"   # "api" when counts come from real usage metadata

    def finish(self):
        self.end_time  = time.time()
        self.latency_ms = (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, output_dir: Path, timestamp: str = ""):
        suffix = f"_{timestamp}" if timestamp else ""
        filepath = output_dir / f"trace_{self.task_id}{suffix}.json"
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class TracingManager:
    """Manages a collection of ExecutionTraces."""

    def __init__(self):
        self.traces: List[ExecutionTrace] = []
        self.current_trace: Optional[ExecutionTrace] = None

    def start_trace(self, task_id: int) -> ExecutionTrace:
        trace = ExecutionTrace(task_id=task_id, start_time=time.time())
        self.current_trace = trace
        self.traces.append(trace)
        return trace

    def log_tool_call(self, tool_name: str, inputs: Any, outputs: Any):
        if self.current_trace:
            self.current_trace.tool_calls.append({
                "tool": tool_name,
                "inputs": str(inputs)[:200],
                "outputs": str(outputs)[:200],
                "timestamp": time.time(),
            })

    def log_error(self, error: str):
        if self.current_trace:
            self.current_trace.errors.append(error)

    def get_summary(self) -> Dict:
        if not self.traces:
            return {}
        return {
            "total_traces":      len(self.traces),
            "total_cost":        sum(t.total_cost for t in self.traces),
            "avg_latency_ms":    sum(t.latency_ms for t in self.traces) / len(self.traces),
            "total_tool_calls":  sum(len(t.tool_calls) for t in self.traces),
            "total_errors":      sum(len(t.errors) for t in self.traces),
        }

    def reset(self):
        self.traces = []
        self.current_trace = None


# ---------------------------------------------------------------------------
# OTel span data structures
# ---------------------------------------------------------------------------

class SpanKind(Enum):
    INTERNAL = "internal"
    CLIENT   = "client"
    SERVER   = "server"

class SpanStatus(Enum):
    UNSET = "unset"
    OK    = "ok"
    ERROR = "error"


@dataclass
class SpanEvent:
    name: str
    timestamp: float
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"name": self.name, "timestamp": self.timestamp, "attributes": self.attributes}


@dataclass
class OTelSpan:
    """OpenTelemetry-compliant span. Exportable to OTLP JSON."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    name: str = ""
    kind: str = SpanKind.INTERNAL.value
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    status: str = SpanStatus.UNSET.value
    error_message: Optional[str] = None

    def end(self, status: str = SpanStatus.OK.value, error: Optional[str] = None):
        self.end_time   = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status     = status
        self.error_message = error

    def add_event(self, name: str, attributes: Dict = None):
        self.events.append(SpanEvent(name=name, timestamp=time.time(), attributes=attributes or {}))

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def to_otlp_dict(self) -> Dict:
        def _val(v):
            if isinstance(v, bool):   return {"boolValue": v}
            if isinstance(v, int):    return {"intValue": v}
            if isinstance(v, float):  return {"doubleValue": v}
            return {"stringValue": str(v)}

        return {
            "traceId":            self.trace_id,
            "spanId":             self.span_id,
            "parentSpanId":       self.parent_span_id,
            "name":               self.name,
            "kind":               self.kind,
            "startTimeUnixNano":  int(self.start_time * 1_000_000_000),
            "endTimeUnixNano":    int(self.end_time * 1_000_000_000) if self.end_time else None,
            "durationMs":         self.duration_ms,
            "attributes":         [{"key": k, "value": _val(v)} for k, v in self.attributes.items()],
            "events":             [e.to_dict() for e in self.events],
            "status":             {"code": self.status, "message": self.error_message or ""},
        }

    def to_simple_dict(self) -> Dict:
        return {
            "trace_id":      self.trace_id,
            "span_id":       self.span_id,
            "parent_span_id":self.parent_span_id,
            "name":          self.name,
            "duration_ms":   self.duration_ms,
            "attributes":    self.attributes,
            "status":        self.status,
        }


# ---------------------------------------------------------------------------
# Hierarchical tracer
# ---------------------------------------------------------------------------

class HierarchicalTracer:
    """
    Manages parent-child OTel spans across multi-agent execution.
    Produces OTLP-compatible output for any observability backend.
    """

    def __init__(self):
        self.traces: Dict[str, List[OTelSpan]] = {}
        self.active_spans: List[OTelSpan]       = []
        self.current_trace_id: Optional[str]    = None

    def start_trace(self, root_name: str, attributes: Dict = None) -> OTelSpan:
        trace_id = str(uuid.uuid4())
        self.current_trace_id = trace_id
        self.traces[trace_id] = []

        root = OTelSpan(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            name=root_name,
            start_time=time.time(),
            attributes=attributes or {},
        )
        self.traces[trace_id].append(root)
        self.active_spans.append(root)
        return root

    def start_span(self, name: str, kind: str = SpanKind.INTERNAL.value,
                   attributes: Dict = None) -> OTelSpan:
        if not self.active_spans:
            raise RuntimeError("No active span. Call start_trace() first.")
        parent = self.active_spans[-1]
        span = OTelSpan(
            trace_id=parent.trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent.span_id,
            name=name,
            kind=kind,
            start_time=time.time(),
            attributes=attributes or {},
        )
        self.traces[parent.trace_id].append(span)
        self.active_spans.append(span)
        return span

    def end_span(self, span: OTelSpan, status: str = SpanStatus.OK.value,
                 error: Optional[str] = None, attributes: Dict = None):
        if attributes:
            span.attributes.update(attributes)
        span.end(status=status, error=error)
        if span in self.active_spans:
            self.active_spans.remove(span)

    def end_trace(self):
        while self.active_spans:
            self.end_span(self.active_spans[-1])
        self.current_trace_id = None

    def export_otlp(self, trace_id: Optional[str] = None) -> Dict:
        ids = [trace_id] if trace_id else list(self.traces.keys())
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name",    "value": {"stringValue": "multi-agent-otel-eval"}},
                            {"key": "service.version", "value": {"stringValue": "1.0"}},
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "genai.observability"},
                            "spans": [s.to_otlp_dict() for s in self.traces[tid]],
                        }
                    ],
                }
                for tid in ids
            ]
        }

    def save_all_traces(self, output_dir: Path, timestamp: str = "") -> Path:
        suffix = f"_{timestamp}" if timestamp else ""
        filepath = output_dir / f"all_otel_traces{suffix}.jsonl"
        with open(filepath, "w") as f:
            for tid in self.traces:
                f.write(json.dumps(self.export_otlp(tid), default=str) + "\n")
        return filepath

    def get_stats(self) -> Dict:
        all_spans = [s for spans in self.traces.values() for s in spans]
        if not all_spans:
            return {}
        return {
            "total_traces":        len(self.traces),
            "total_spans":         len(all_spans),
            "avg_spans_per_trace": len(all_spans) / len(self.traces),
            "errors":              sum(1 for s in all_spans if s.status == SpanStatus.ERROR.value),
        }

    def reset(self):
        self.traces = {}
        self.active_spans = []
        self.current_trace_id = None
