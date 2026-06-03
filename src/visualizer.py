"""
Visualization functions for evaluation results and OTel traces.

All functions accept a DataFrame of results and/or a HierarchicalTracer,
and return matplotlib Figure objects (or save them to disk).
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


_COLORS = {
    "primary":   "#2A9D8F",
    "secondary": "#8B5CF6",
    "warning":   "#F59E0B",
    "danger":    "#EF4444",
    "info":      "#3B82F6",
    "neutral":   "#64748B",
}


def _style_ax(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=11, fontweight="bold", color="#1E3A5F")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")


def plot_eval_dashboard(df: pd.DataFrame, threshold: float = 0.7,
                        save_path: Optional[Path] = None) -> plt.Figure:
    """
    4-panel evaluation dashboard:
    score distribution | pass/fail | tool F1 | cost/latency

    Args:
        threshold: pass threshold to display on the score and pass/fail panels.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Evaluation Results Dashboard", fontsize=14, fontweight="bold")

    # Panel 1: Score distribution
    ax = axes[0, 0]
    ax.hist(df["task_score"], bins=15, color=_COLORS["primary"], alpha=0.85, edgecolor="white")
    ax.axvline(df["task_score"].mean(), color=_COLORS["danger"], linestyle="--",
               label=f"Mean: {df['task_score'].mean():.2f}")
    ax.axvline(threshold, color=_COLORS["neutral"], linestyle=":",
               label=f"Threshold: {threshold:.2f}")
    ax.legend(fontsize=9)
    _style_ax(ax, "Task Score Distribution", "Score", "Count")

    # Panel 2: Pass/fail
    ax = axes[0, 1]
    passed = df["task_passed"].sum()
    failed = len(df) - passed
    ax.bar(["Pass", "Fail"], [passed, failed],
           color=[_COLORS["primary"], _COLORS["danger"]], alpha=0.85, edgecolor="white")
    ax.bar_label(ax.containers[0], fmt="%d")
    _style_ax(ax, f"Pass/Fail (threshold: {threshold:.2f})", ylabel="Tasks")

    # Panel 3: Tool F1
    ax = axes[1, 0]
    if "tool_f1" in df.columns:
        ax.hist(df["tool_f1"], bins=12, color=_COLORS["secondary"], alpha=0.85, edgecolor="white")
        ax.axvline(df["tool_f1"].mean(), color=_COLORS["warning"], linestyle="--",
                   label=f"Mean F1: {df['tool_f1'].mean():.2f}")
        ax.legend(fontsize=9)
    _style_ax(ax, "Tool Selection F1", "F1 Score", "Count")

    # Panel 4: Cost vs latency scatter
    ax = axes[1, 1]
    if "total_cost" in df.columns and "latency_ms" in df.columns:
        colors = [_COLORS["primary"] if p else _COLORS["danger"] for p in df["task_passed"]]
        ax.scatter(df["latency_ms"], df["total_cost"], c=colors, alpha=0.7, s=60)
        ax.set_xlabel("Latency (ms)", fontsize=10)
        ax.set_ylabel("Cost (USD)", fontsize=10)
        patches = [mpatches.Patch(color=_COLORS["primary"], label="Pass"),
                   mpatches.Patch(color=_COLORS["danger"],  label="Fail")]
        ax.legend(handles=patches, fontsize=9)
    _style_ax(ax, "Cost vs Latency")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_trace_tree(tracer, trace_id: str, save_path: Optional[Path] = None) -> plt.Figure:
    """Phoenix/Jaeger-style hierarchical span tree for one trace."""
    spans = tracer.traces.get(trace_id, [])
    if not spans:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No spans found", ha="center", va="center")
        return fig

    # Sort by start_time
    spans = sorted(spans, key=lambda s: s.start_time)
    root_start = spans[0].start_time

    fig, ax = plt.subplots(figsize=(12, max(4, len(spans) * 0.5)))

    # Build depth map
    depth_map: dict = {}
    for span in spans:
        if span.parent_span_id is None:
            depth_map[span.span_id] = 0
        else:
            parent_depth = depth_map.get(span.parent_span_id, 0)
            depth_map[span.span_id] = parent_depth + 1

    max_depth = max(depth_map.values(), default=0)
    palette = [_COLORS["primary"], _COLORS["secondary"], _COLORS["info"],
               _COLORS["warning"], _COLORS["danger"]]

    for i, span in enumerate(spans):
        depth = depth_map.get(span.span_id, 0)
        start = (span.start_time - root_start) * 1000  # ms
        dur   = max(span.duration_ms, 1)
        color = palette[depth % len(palette)]
        ax.barh(i, dur, left=start, height=0.5, color=color, alpha=0.85)
        label = f"{'  ' * depth}{span.name} ({dur:.0f}ms)"
        ax.text(start + dur + 2, i, label, va="center", fontsize=8)

    ax.set_xlabel("Time (ms from trace start)", fontsize=10)
    ax.set_yticks([])
    ax.set_title(f"Trace Tree — {trace_id[:8]}", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_waterfall(tracer, trace_id: str, save_path: Optional[Path] = None) -> plt.Figure:
    """Gantt/waterfall chart showing span timing."""
    return plot_trace_tree(tracer, trace_id, save_path)  # Same visual, alias for clarity


def plot_telemetry_dashboard(df: pd.DataFrame, tracer=None,
                             save_path: Optional[Path] = None) -> plt.Figure:
    """
    4-panel telemetry summary:
    token usage | cost breakdown | latency percentiles | health over time
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Telemetry Dashboard", fontsize=14, fontweight="bold")

    # Panel 1: Token usage
    ax = axes[0, 0]
    if "agent_tokens" in df.columns:
        ax.bar(df.index, df["agent_tokens"], color=_COLORS["primary"],  alpha=0.8, label="Agent")
        ax.bar(df.index, df.get("judge_tokens", 0), bottom=df["agent_tokens"],
               color=_COLORS["secondary"], alpha=0.8, label="Judge")
        ax.legend(fontsize=9)
    _style_ax(ax, "Tokens per Task", "Task #", "Tokens")

    # Panel 2: Cost per task
    ax = axes[0, 1]
    if "total_cost" in df.columns:
        ax.bar(df.index, df["total_cost"] * 1000, color=_COLORS["warning"], alpha=0.85)
    _style_ax(ax, "Cost per Task (m$)", "Task #", "Cost (m$)")

    # Panel 3: Latency percentiles
    ax = axes[1, 0]
    if "latency_ms" in df.columns:
        percs = [50, 75, 90, 95, 99]
        vals  = [df["latency_ms"].quantile(p / 100) for p in percs]
        ax.bar([str(p) for p in percs], vals, color=_COLORS["info"], alpha=0.85)
    _style_ax(ax, "Latency Percentiles", "Percentile", "ms")

    # Panel 4: Rolling pass rate
    ax = axes[1, 1]
    if "task_passed" in df.columns:
        rolling = df["task_passed"].astype(float).rolling(window=5, min_periods=1).mean()
        ax.plot(rolling, color=_COLORS["primary"], linewidth=2)
        ax.axhline(0.7, color=_COLORS["danger"], linestyle="--", alpha=0.7, label="Threshold 0.7")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
    _style_ax(ax, "Rolling Pass Rate (window=5)", "Task #", "Rate")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_dataset_overview(tasks: list, save_path: Optional[Path] = None) -> plt.Figure:
    """2×2 Mind2Web dataset overview: domains, websites, action counts, action types."""
    import pandas as pd
    df = pd.DataFrame(tasks)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Mind2Web Dataset Overview", fontsize=14, fontweight="bold", color="#1E3A5F")

    # Panel 1: Top 10 domains
    ax = axes[0, 0]
    if "domain" in df.columns:
        top = df["domain"].value_counts().head(10)
        ax.barh(range(len(top)), top.values, color=_COLORS["primary"], alpha=0.85)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([str(d)[:20] for d in top.index], fontsize=9)
        ax.invert_yaxis()
    _style_ax(ax, "Top 10 Domains", "Count")

    # Panel 2: Top 10 websites
    ax = axes[0, 1]
    if "website" in df.columns:
        top = df["website"].value_counts().head(10)
        ax.barh(range(len(top)), top.values, color=_COLORS["secondary"], alpha=0.85)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([str(w)[:20] for w in top.index], fontsize=9)
        ax.invert_yaxis()
    _style_ax(ax, "Top 10 Websites", "Count")

    # Panel 3: Action count distribution
    ax = axes[1, 0]
    if "action_reprs" in df.columns:
        counts = df["action_reprs"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        ax.hist(counts, bins=20, color=_COLORS["warning"], alpha=0.85, edgecolor="white")
        ax.axvline(counts.mean(), color=_COLORS["danger"], linestyle="--",
                   label=f"Mean: {counts.mean():.1f}")
        ax.legend(fontsize=9)
    _style_ax(ax, "Reference Actions per Task", "Actions", "Count")

    # Panel 4: Action type pie
    ax = axes[1, 1]
    action_types: dict = {}
    for actions in df.get("action_reprs", []):
        for action in (actions if isinstance(actions, list) else []):
            parts = str(action).split("->")
            if len(parts) > 1:
                atype = parts[1].strip().split(":")[0].split()[0] if parts[1].strip() else "OTHER"
                action_types[atype] = action_types.get(atype, 0) + 1
    if action_types:
        top5 = sorted(action_types.items(), key=lambda x: x[1], reverse=True)[:5]
        other = sum(c for _, c in sorted(action_types.items(), key=lambda x: x[1], reverse=True)[5:])
        labels = [t for t, _ in top5] + (["Other"] if other else [])
        sizes  = [c for _, c in top5]  + ([other]  if other else [])
        colors = list(_COLORS.values())[:len(labels)]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
               wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("Action Type Distribution", fontsize=11, fontweight="bold", color="#1E3A5F")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_baseline_vs_multi(df_baseline: pd.DataFrame, df_multi: pd.DataFrame,
                           save_path: Optional[Path] = None) -> plt.Figure:
    """
    3-panel single-agent vs. multi-agent comparison: pass rate, avg task score,
    and avg cost per task. Bars are annotated with the relative change so the
    quality/cost trade-off of orchestration is immediately visible.
    """
    labels = ["Single Agent", "Multi-Agent\n(Supervisor)"]
    colors = [_COLORS["neutral"], _COLORS["primary"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Single-Agent vs. Multi-Agent System", fontsize=14, fontweight="bold",
                 color="#1E3A5F")

    def _panel(ax, b_val, m_val, title, fmt, unit_strip="", higher_better=True):
        bars = ax.bar(labels, [b_val, m_val], color=colors, alpha=0.88,
                      edgecolor="white", linewidth=2, width=0.55)
        for bar, val in zip(bars, [b_val, m_val]):
            ax.text(bar.get_x() + bar.get_width() / 2, val, fmt.format(val),
                    ha="center", va="bottom", fontsize=12, fontweight="bold", color="#1E3A5F")
        _style_ax(ax, title)
        if b_val:
            pct = (m_val - b_val) / b_val * 100
            good = (pct >= 0) == higher_better
            ax.annotate(f"{pct:+.0f}%", xy=(0.5, 0.92), xycoords="axes fraction",
                        ha="center", fontsize=13, fontweight="bold",
                        color=_COLORS["primary"] if good else _COLORS["danger"])
        ax.margins(y=0.18)

    _panel(axes[0], df_baseline["task_passed"].mean() * 100,
           df_multi["task_passed"].mean() * 100, "Pass Rate (%)", "{:.0f}%")
    _panel(axes[1], df_baseline["task_score"].mean(),
           df_multi["task_score"].mean(), "Avg Task Score", "{:.2f}")
    _panel(axes[2], df_baseline["total_cost"].mean(),
           df_multi["total_cost"].mean(), "Avg Cost / Task ($)", "${:.4f}",
           higher_better=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
