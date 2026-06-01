"""
Cost tracking and agent health monitoring.

CostTracker  — logs token usage + cost per task, produces summary stats.
HealthMonitor — rolling-window health status (HEALTHY / DEGRADED / UNHEALTHY).
"""

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd


class CostTracker:
    """Log per-task token usage and compute cost summaries."""

    def __init__(self):
        self._records: List[Dict] = []

    def log(self, task_id: int, model: str,
            input_tokens: int, output_tokens: int, cost: float):
        self._records.append({
            "task_id":       task_id,
            "model":         model,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "total_tokens":  input_tokens + output_tokens,
            "cost":          cost,
            "timestamp":     datetime.now().isoformat(),
        })

    def get_summary(self) -> Dict:
        if not self._records:
            return {}
        df = pd.DataFrame(self._records)
        return {
            "total_cost":          df["cost"].sum(),
            "avg_cost_per_task":   df["cost"].mean(),
            "median_cost":         df["cost"].median(),
            "total_tokens":        int(df["total_tokens"].sum()),
            "cost_by_model":       df.groupby("model")["cost"].sum().to_dict(),
            "tokens_by_model":     df.groupby("model")["total_tokens"].sum().to_dict(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

    def reset(self):
        self._records = []


class HealthMonitor:
    """
    Rolling-window agent health monitor.

    Status thresholds:
      HEALTHY   : success_rate ≥ 90% and error_rate < 5%
      DEGRADED  : success_rate ≥ 70% and error_rate < 15%
      UNHEALTHY : below DEGRADED thresholds
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self._history: List[Dict] = []

    def log(self, task_id: int, success: bool, score: float,
            latency_ms: float, errors: int):
        self._history.append({
            "task_id":    task_id,
            "success":    bool(success),
            "score":      score,
            "latency_ms": latency_ms,
            "errors":     errors,
            "timestamp":  datetime.now(),
        })

    def get_status(self) -> Dict:
        if not self._history:
            return {"status": "NO_DATA"}

        recent = self._history[-self.window_size:]
        df = pd.DataFrame(recent)

        success_rate = df["success"].astype(bool).mean()
        error_rate   = (df["errors"] > 0).mean()

        if success_rate >= 0.9 and error_rate < 0.05:
            status = "HEALTHY"
        elif success_rate >= 0.7 and error_rate < 0.15:
            status = "DEGRADED"
        else:
            status = "UNHEALTHY"

        return {
            "status":        status,
            "success_rate":  success_rate,
            "avg_score":     df["score"].mean(),
            "error_rate":    error_rate,
            "p50_latency":   df["latency_ms"].quantile(0.50),
            "p95_latency":   df["latency_ms"].quantile(0.95),
            "p99_latency":   df["latency_ms"].quantile(0.99),
            "window_size":   len(recent),
            "total_logged":  len(self._history),
        }

    def check_drift(self, baseline_score: float, threshold: float = 0.1) -> Dict:
        if len(self._history) < self.window_size:
            return {"drift_detected": False, "reason": "insufficient_data"}
        recent_avg = np.mean([m["score"] for m in self._history[-self.window_size:]])
        drift = baseline_score - recent_avg
        return {
            "drift_detected": abs(drift) > threshold,
            "drift_amount":   drift,
            "baseline_score": baseline_score,
            "recent_score":   recent_avg,
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)

    def reset(self):
        self._history = []
