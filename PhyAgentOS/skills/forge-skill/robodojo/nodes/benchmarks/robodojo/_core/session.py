"""服务模式:一次 bench.run 的封装,以及与 bench_endpoint 的 JSON 通道。

⚠️ **本文件绝不 import forge_tool。** 它跑在 Isaac Sim 的 Python 3.11 里,而
forge_tool 用了 PEP 695 的 `type X = ...`,3.11 上是 SyntaxError(不是依赖问题,
装进去也一样炸)。Forge 协议整层由 3.12 的 bench_endpoint 节点负责,这边只发
最朴素的 JSON —— 形态和已有的 `benchmark_status` / `benchmark_result` 一致。

线协议常量在两个包里各有一份(这里 + bench_endpoint/wire.py)。**是刻意重复的**:
两个包跑在不同 Python 上,不可能共享依赖。改动时两边必须一起改,`WIRE_VERSION`
就是用来在忘记时把错误暴露出来的。
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any

import pyarrow as pa

from .config import BaseBenchmarkConfig, resolve_and_validate_plan
from .runner import BenchmarkRunner

WIRE_VERSION = 1

CMD_RUN = "run"
CMD_CANCEL = "cancel"
CMD_DESCRIBE = "describe"

REPORT_CAPABILITIES = "capabilities"
REPORT_ACCEPTED = "accepted"
REPORT_REJECTED = "rejected"
REPORT_PROGRESS = "progress"
REPORT_HEARTBEAT = "heartbeat"
REPORT_RESULT = "result"

# 回传的 episodes 上限。Tool Wire 有 DEFAULT_MAX_MESSAGE_BYTES,官方协议
# 150 集的明细会撞上;超了就只给 result_path,明细留在盘上。
MAX_EPISODES_IN_PAYLOAD = 64

HEARTBEAT_PERIOD_S = 1.0


class WireError(ValueError):
    """线上消息不合法。丢弃并记日志,不要让它冲出事件循环。"""


def encode(payload: dict[str, Any]) -> pa.Array:
    body = dict(payload)
    body.setdefault("v", WIRE_VERSION)
    return pa.array([json.dumps(body, ensure_ascii=False)])


def decode(value: Any) -> dict[str, Any]:
    raw = value
    if isinstance(raw, (pa.Array, pa.ChunkedArray)):
        items = raw.to_pylist()
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    if len(items) != 1:
        raise WireError(f"bench_cmd 每条消息只应有 1 个元素,收到 {len(items)}")
    text = items[0]
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8")
    if not isinstance(text, str):
        raise WireError(f"bench_cmd 载体必须是字符串,收到 {type(text).__name__}")
    try:
        body = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WireError(f"bench_cmd 不是合法 JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise WireError("bench_cmd 必须是 JSON 对象")
    if body.get("v") != WIRE_VERSION:
        raise WireError(f"线协议版本不匹配:本端 {WIRE_VERSION},收到 {body.get('v')!r}")
    if not isinstance(body.get("type"), str):
        raise WireError("bench_cmd 缺少 type 字段")
    return body


# ---- capabilities ----


def capabilities_payload(
    cfg: BaseBenchmarkConfig,
    adapter: Any,
    *,
    limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    """describe 的数据源。

    task 列表只有 adapter 知道(它在 3.11 侧),所以由本节点**主动推**给
    endpoint 缓存,而不是让 endpoint 按需来拉 —— 按需拉会要求 endpoint 去 await
    一条 dora 消息,而它的分发是同步跑到底的,必然死锁。
    """
    names = getattr(adapter, "tasks", None)
    tasks = []
    for task_id in range(adapter.num_tasks):
        entry = {
            "task_id": task_id,
            "num_init_states": int(adapter.num_init_states(task_id)),
        }
        if names is not None and task_id < len(names):
            entry["name"] = str(names[task_id])
        try:
            entry["language"] = str(adapter.task_language(task_id))
        except Exception:  # noqa: BLE001 拿不到语言指令不该挡住 describe
            entry["language"] = ""
        tasks.append(entry)

    return {
        "type": REPORT_CAPABILITIES,
        "benchmark": cfg.benchmark,
        "suite": cfg.suite,
        "policy_id": cfg.policy_id,
        "env_cfg_type": getattr(cfg, "env_cfg_type", ""),
        "action_type": getattr(cfg, "action_type", ""),
        "num_tasks": int(adapter.num_tasks),
        "tasks": tasks,
        "joint_names": list(adapter.state_joint_names()),
        "camera_ids": sorted(getattr(cfg, "camera_map", {}) or {}),
        "defaults": {
            "num_runs": cfg.num_runs,
            "seed": cfg.seed,
            "max_steps": cfg.max_steps,
        },
        "limits": dict(limits or {}),
    }


# ---- 一次 invocation ----


class BenchSession:
    """一次 bench.run。

    `BenchmarkRunner` 是**一次性**的 —— `__init__` 就把 `cfg.episode_plan()`
    绑死,`finished` 是不可逆的单向开关。所以每次 invocation 新建一个,并且用
    `replace()` 克隆 cfg,绝不改模板 cfg(改了会污染下一次)。
    """

    def __init__(
        self,
        node: Any,
        cfg_template: BaseBenchmarkConfig,
        adapter: Any,
        plan: dict[str, Any],
        invocation_id: str,
        log=print,
    ) -> None:
        self.invocation_id = invocation_id
        self._log = log
        self.cfg = replace(
            cfg_template,
            task_ids=[int(t) for t in plan["task_ids"]],
            init_state_ids=[int(i) for i in plan["init_state_ids"]],
            num_runs=int(plan["num_runs"]),
            seed=int(plan["seed"]),
            max_steps=int(plan["max_steps"]),
        )
        # 权威兜底校验。endpoint 侧拿 capabilities 也校验了一遍,但那份可能陈旧;
        # 这里抛出来就变成 rejected 回执,而不是"跑起来之后失败"。
        resolve_and_validate_plan(self.cfg, adapter)

        # **adapter 会回写 cfg,必须让它和 runner 读的是同一个对象。**
        # RoboDojo 的 adapter 在 reset_episode 里按环境的 step_lim 修正
        # max_steps(build_tower 是 300 → 1050)。它持有的是构造时传入的**模板**
        # cfg,而 runner 读的是上面这份克隆 —— 不指到同一个对象的话,修正落在
        # 模板上、runner 仍按 300 在第 300 步截断,**而且不报任何错**:
        # 结果是 timed_out(300 步),基线是 partial(1050 步),两边根本不是
        # 同一件事。adapter 自己的注释就警告过这个形态,L1-D 实测复现。
        # 判据是「有没有 cfg 这个属性」,不是「cfg 是不是 None」—— 后者会让
        # cfg 初始为 None 的 adapter 静默拿不到回写通道。
        if hasattr(adapter, "cfg"):
            adapter.cfg = self.cfg

        # result_tag 必须传:文件名的时间戳只到秒,同一秒内两次 invocation 会
        # 写到同一路径,前一次的结果被静默覆盖。
        self.runner = BenchmarkRunner(
            node, self.cfg, adapter, log=log, result_tag=invocation_id
        )
        self.started_at = time.time()
        self._last_progress = -1

    # ---- 生命周期 ----

    def start(self) -> None:
        self.runner.policy_command("start")
        self.runner.advance()

    def on_action(self, command: Any) -> None:
        self.runner.on_action(command)

    def republish_if_stalled(self, timeout_s: float) -> None:
        self.runner.republish_if_stalled(timeout_s)

    def fail_stalled(self, deadline_s: float) -> bool:
        if not self.runner.check_action_deadline(deadline_s):
            return False
        self.runner.stall(
            "policy_stalled", f"waiting for action > {deadline_s:.0f}s"
        )
        return True

    def cancel(self) -> None:
        self.runner.finish("cancelled")

    def fail(self, code: str, message: str) -> None:
        self.runner.fail_current(code, message)

    @property
    def finished(self) -> bool:
        return self.runner.finished

    # ---- 回执 ----

    def progress_report(self) -> dict[str, Any] | None:
        """episode 边界才发一条,不要每步发。"""
        index = self.runner.plan_index
        if index == self._last_progress:
            return None
        self._last_progress = index
        return {
            "type": REPORT_PROGRESS,
            "invocation_id": self.invocation_id,
            "episode_index": index,
            "total": len(self.runner.plan),
            "successes": self.runner.result.successes,
        }

    def result_report(self) -> dict[str, Any]:
        result = self.runner.result
        payload = result.to_dict()
        episodes = payload.get("episodes") or []
        truncated = len(episodes) > MAX_EPISODES_IN_PAYLOAD
        out = {
            "status": result.status,
            "primary_metric": result.primary_metric,
            "success_rate": payload.get("success_rate"),
            "successes": payload.get("successes"),
            "total_episodes": payload.get("total_episodes"),
            "completed_episodes": payload.get("completed_episodes"),
            "valid_episodes": payload.get("valid_episodes"),
            "timed_out_episodes": payload.get("timed_out_episodes"),
            "num_steps": payload.get("num_steps"),
            "mean_policy_latency_ms": payload.get("mean_policy_latency_ms"),
            "elapsed_s": payload.get("elapsed_s"),
            "obs_republished": (result.metrics or {}).get("obs_republished"),
            "result_path": getattr(self.runner, "result_path", None),
            "episodes": [] if truncated else [_slim(e) for e in episodes],
            "episodes_truncated": truncated,
        }
        return {
            "type": REPORT_RESULT,
            "invocation_id": self.invocation_id,
            "status": result.status,
            "payload": out,
        }


def _slim(episode: dict[str, Any]) -> dict[str, Any]:
    """只回传判分需要的字段。task_description 之类留在 result.json 里。"""
    return {
        "task_id": episode.get("task_id"),
        "init_state_id": episode.get("init_state_id"),
        "run_index": episode.get("run_index"),
        "seed": episode.get("episode_seed"),
        "success": episode.get("success"),
        "termination": episode.get("termination"),
        "error_code": episode.get("error_code"),
    }


__all__ = [
    "CMD_CANCEL",
    "CMD_DESCRIBE",
    "CMD_RUN",
    "HEARTBEAT_PERIOD_S",
    "MAX_EPISODES_IN_PAYLOAD",
    "REPORT_ACCEPTED",
    "REPORT_CAPABILITIES",
    "REPORT_HEARTBEAT",
    "REPORT_PROGRESS",
    "REPORT_REJECTED",
    "REPORT_RESULT",
    "WIRE_VERSION",
    "BenchSession",
    "WireError",
    "capabilities_payload",
    "decode",
    "encode",
]
