"""BenchmarkExecutionResultV1 的组装与落盘。

schema 分两层：
  **固定骨架** —— 所有 benchmark 都有（status / 计数 / 耗时 / 逐 episode 基础字段）
  **可扩展指标** —— `primary_metric` + `metrics`，由 adapter 决定装什么

`primary_metric` 是上层的统一入口：只读这一个字段就能出榜、排序、跨 benchmark 比，
不需要认识 CALVIN 的链长是什么。想深挖再读 `metrics`。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "benchmark_execution_result_v1"

# 恢复/重试相关字段：本节点不做恢复重试，一律 null，
# 留给 Forge/PAOS 上层在暴露给契约层时填充。
_RECOVERY_FIELDS = (
    "first_attempt_successes",
    "final_successes",
    "first_attempt_score",
    "assisted_final_score",
    "episodes_replanned",
    "recovered_after_replan",
)


@dataclass
class EpisodeRecord:
    episode_id: str
    task_id: int
    init_state_id: int
    run_index: int
    task_description: str
    success: bool
    termination: str  # success / partial / failed / timed_out
    num_steps: int
    return_value: float
    episode_seed: int | None = None
    mean_policy_latency_ms: float | None = None
    success_source: str = ""
    # benchmark 特有的逐 episode 成绩（如 CALVIN 的 chain_length）
    metrics: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "init_state_id": self.init_state_id,
            "run_index": self.run_index,
            "task_description": self.task_description,
            "success": self.success,
            "termination": self.termination,
            "num_steps": self.num_steps,
            "return_value": self.return_value,
            "episode_seed": self.episode_seed,
            "mean_policy_latency_ms": self.mean_policy_latency_ms,
            "success_source": self.success_source,
            "metrics": self.metrics,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


@dataclass
class BenchmarkResult:
    benchmark: str
    suite: str
    max_steps: int
    total_episodes: int
    seed: int = 0
    # 复现口径：benchmark 特有的、影响分数的配置（如 LIBERO 的 layout_mode/control_mode）
    reproducibility: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    episodes: list[EpisodeRecord] = field(default_factory=list)
    elapsed_s: float | None = None
    primary_metric: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def successes(self) -> int:
        return sum(1 for e in self.episodes if e.success)

    @property
    def num_steps(self) -> int:
        return sum(e.num_steps for e in self.episodes)

    @property
    def mean_policy_latency_ms(self) -> float | None:
        # 用 is not None 而不是真值判断：延迟恰好为 0.0 的 episode 不该被悄悄丢掉。
        vals = [
            e.mean_policy_latency_ms
            for e in self.episodes
            if e.mean_policy_latency_ms is not None
        ]
        return sum(vals) / len(vals) if vals else None

    def to_dict(self) -> dict[str, Any]:
        completed = len(self.episodes)
        denom = self.total_episodes or 1
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "benchmark": self.benchmark,
            "status": self.status,
            "suite": self.suite,
            "execution_mode": "target_native",
            "max_steps": self.max_steps,
            "seed": self.seed,
            "reproducibility": self.reproducibility,
            # 上层的统一入口
            "primary_metric": self.primary_metric,
            "metrics": self.metrics,
            # 计数（所有 benchmark 通用）
            "successes": self.successes,
            "total_episodes": self.total_episodes,
            "completed_episodes": completed,
            "valid_episodes": sum(1 for e in self.episodes if e.error_code is None),
            "success_rate": self.successes / denom,
            "num_steps": self.num_steps,
            "mean_policy_latency_ms": self.mean_policy_latency_ms,
            "elapsed_s": self.elapsed_s,
            "timed_out_episodes": sum(
                1 for e in self.episodes if e.termination == "timed_out"
            ),
            "episodes": [e.to_dict() for e in self.episodes],
        }
        payload.update({name: None for name in _RECOVERY_FIELDS})
        return payload

    def dump(self, result_dir: str, stamp: str) -> str:
        os.makedirs(result_dir, exist_ok=True)
        tag = self.suite or self.benchmark
        path = os.path.join(result_dir, f"{tag}_{stamp}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
        return path
