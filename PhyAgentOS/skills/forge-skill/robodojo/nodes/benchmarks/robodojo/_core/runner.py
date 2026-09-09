"""批次调度核心。通用逻辑，**不认识任何具体 benchmark**。

时序契约：action 驱动的 1:1 锁步。发布一次观测 → 等 policy 的 action → step →
再发布观测。policy 内部的 action chunk 由它自己排队，本节点按 1:1 消费，
`awaiting_action == False` 时到达的多余 action 直接丢弃。
"""

from __future__ import annotations

import inspect
import json
import time
from typing import Any

import numpy as np
import pyarrow as pa
from forge_msgs import Image, JointCommand, JointState, PolicyCommand

from .adapter import CHUNK_END_SENTINEL, default_aggregate
from .config import BaseBenchmarkConfig, derive_episode_seed
from .video import EpisodeRecorder
from .result import BenchmarkResult, EpisodeRecord


class BenchmarkRunner:
    def __init__(
        self,
        node,
        cfg: BaseBenchmarkConfig,
        adapter,
        log=print,
        result_tag: str | None = None,
    ) -> None:
        self.node = node
        self.cfg = cfg
        self.adapter = adapter
        self._log = log
        # 结果文件名的去重后缀。**服务模式必须传**：文件名是
        # `<suite>_<时间戳>.json` 而时间戳只到秒，同一秒内的两次 invocation
        # 会写到同一路径，前一次的结果被静默覆盖（autorun 一个进程一批，
        # 撞不上，所以这个坑只在服务模式里出现）。
        self.result_tag = result_tag
        self.plan = cfg.episode_plan()
        self.result = BenchmarkResult(
            benchmark=cfg.benchmark,
            suite=cfg.suite,
            max_steps=cfg.max_steps,
            total_episodes=len(self.plan),
            seed=cfg.seed,
            reproducibility=getattr(adapter, "reproducibility", lambda: {})(),
        )
        self.plan_index = -1
        self.started_at = time.time()

        self.step_idx = 0
        self.episode_return = 0.0
        self.episode_seed: int | None = None
        self.last_metrics: dict[str, Any] = {}
        self.awaiting_action = False
        self.obs_sent_at: float | None = None
        self.awaiting_action_since: float | None = None
        self.latencies: list[float] = []
        self.finished = False
        self._last_obs: dict[str, Any] | None = None
        self._republished = 0
        self.result_path: str | None = None

        self._joint_names = list(adapter.state_joint_names())
        # None = 不校验（老 adapter 的默认行为）
        _an = getattr(adapter, "action_component_names", None)
        self._action_names: list[str] | None = list(_an()) if callable(_an) and _an() else None
        self._step_takes_chunk_end = "chunk_end" in inspect.signature(adapter.step).parameters
        self.recorder = EpisodeRecorder(
            cfg.record_dir, fps=getattr(adapter, "RECORD_FPS", 20.0), log=self._log
        )

    # ---- policy 生命周期 ----

    def policy_command(self, command: str, inputs: dict[str, Any] | None = None) -> None:
        msg = PolicyCommand.from_inputs(
            policy_id=self.cfg.policy_id, command=command, inputs=inputs or {}
        )
        self.node.send_output("policy_command", msg.to_arrow())

    # ---- 观测发布 ----

    def republish_if_stalled(self, timeout_s: float) -> None:
        """等 action 等太久就把同一帧观测再发一次。

        为什么需要：`start_next_episode` 先发 `set_instruction` + `reset_scene`
        再发观测，但 `policy_command` 和 `proprio_state`/`image/*` 是**不同的
        dora 通道，跨通道到达顺序没有保证**。若观测先于 `reset_scene` 被 policy
        取出，policy 的 `reset_scene` 处理会把它清掉 → policy 等下一帧观测、
        本节点等 action，**双向死锁**，两边都不报错、GPU 占着不放。实测在
        pi05 上必现，在更快的 smolvla 上侥幸躲过 —— 属于间歇性故障，比稳定
        故障更难查。

        重发是幂等的：policy 拿到同一帧再推一次，动作可能略有不同但语义无损；
        而不重发的代价是整批挂死。所以宁可重发。
        """
        if self.finished or not self.awaiting_action or self._last_obs is None:
            return
        if self.obs_sent_at is None or (time.time() - self.obs_sent_at) < timeout_s:
            return
        self._republished += 1
        self._log(
            f"等 action 超过 {timeout_s:.0f}s（第 {self._republished} 次重发观测）"
            f"｜episode {self.plan_index + 1}/{len(self.plan)} step {self.step_idx}"
        )
        self._publish_obs(self._last_obs)

    def check_action_deadline(self, deadline_s: float) -> bool:
        """等 action 超过 deadline 就认为 policy 停滞（心跳/重发不刷新计时）。"""
        if self.finished or not self.awaiting_action or self.awaiting_action_since is None:
            return False
        return (time.time() - self.awaiting_action_since) >= deadline_s

    def _publish_obs(self, obs: dict[str, Any]) -> None:
        self._last_obs = obs
        state = self.adapter.to_state(obs)
        if state is not None:
            self.node.send_output(
                "proprio_state",
                JointState.from_np(
                    np.asarray(state, dtype=np.float64), self._joint_names, "position"
                ).to_arrow(),
            )
        frames = self.adapter.to_images(obs)
        for output_id, frame in frames.items():
            self.node.send_output(output_id, Image.from_numpy(frame, "rgb8").to_arrow())
        self.recorder.add(frames)

        self.obs_sent_at = time.time()
        if not self.awaiting_action:
            self.awaiting_action_since = time.time()
        self.awaiting_action = True

    def emit_status(self, text: str) -> None:
        self.node.send_output("benchmark_status", pa.array([text]))
        self._log(text)

    # ---- episode 推进 ----

    def advance(self) -> None:
        """推进到下一 episode。任何异常都收敛成"批次失败"，绝不冲出事件循环。"""
        if self.finished:
            return
        try:
            self.start_next_episode()
        except Exception as exc:  # noqa: BLE001
            self._log(f"启动下一 episode 失败：{type(exc).__name__}: {exc}")
            self.finish("failed")

    def start_next_episode(self) -> bool:
        self.plan_index += 1
        if self.plan_index >= len(self.plan):
            self.finish("succeeded")
            return False

        task_id, init_state_id, run_index = self.plan[self.plan_index]
        # 可选钩子：指令依赖整个 episode 身份而非仅 task 时用它（CALVIN 的每条
        # 序列有自己的首个子任务指令）。不实现就退回按 task 取。
        ep_instr = getattr(self.adapter, "episode_instruction", None)
        instruction = (
            ep_instr(task_id, init_state_id, run_index)
            if callable(ep_instr)
            else self.adapter.task_language(task_id)
        )
        self.episode_seed = derive_episode_seed(
            self.cfg.seed, task_id, init_state_id, run_index
        )

        # 指令必须逐 episode 下发：多数 benchmark 每个 task 指令不同，而 policy
        # 配置里那句是写死的。reset_scene 清上一局残留的队列与缓存（不能用 reset，
        # 那会把 policy 打回 idle 相）。
        if self.cfg.send_instruction:
            self.policy_command("set_instruction", {"instruction": instruction})
        self.policy_command("reset_scene")

        obs = self.adapter.reset_episode(
            task_id=task_id,
            init_state_id=init_state_id,
            run_index=run_index,
            episode_seed=self.episode_seed,
        )

        # 有些 benchmark 的指令是**场景相关**的，reset 之后才知道（RoboDojo 的
        # instruction 由 desc_manager 按当前场景生成）。上面那次 set_instruction
        # 只能发个占位值，这里拿真值补发一次，补发在首帧观测之前，policy 不会用错。
        real = getattr(self.adapter, "instruction_from_obs", None)
        if self.cfg.send_instruction and callable(real):
            true_instr = real(obs)
            if true_instr and true_instr != instruction:
                instruction = true_instr
                self.policy_command("set_instruction", {"instruction": instruction})

        self.step_idx = 0
        self.episode_return = 0.0
        self.last_metrics = {}
        self.latencies = []
        self.recorder.reset()
        self.emit_status(
            f"episode {self.plan_index + 1}/{len(self.plan)} 开始 "
            f"t{task_id}_i{init_state_id}_r{run_index} :: {instruction}"
        )
        self._publish_obs(obs)
        return True

    def on_action(self, command: JointCommand) -> None:
        if self.finished or not self.awaiting_action:
            return  # 非 1:1 的多余 action 直接丢弃

        action = np.asarray(command.position, dtype=np.float32).reshape(-1)
        if action.size == 0 or not np.all(np.isfinite(action)):
            self.fail_current("invalid_action", f"action 非法：size={action.size}")
            return

        # 分量名校验。只读 position 会让"policy 把分量顺序发错"这类 bug 完全静默：
        # 机器人照动、每步都有值、分数悄悄归零（RoboCasa 那轮就是这么丢掉一周的）。
        # adapter 声明了期望名字就逐字比对；返回 None 时保持老行为，向后兼容。
        if self._action_names is not None:
            got = list(command.name or [])
            if got != self._action_names:
                self.fail_current(
                    "action_name_mismatch",
                    f"动作分量名不匹配：期望 {self._action_names}，收到 {got or '(空)'}",
                )
                return

        if self.obs_sent_at is not None:
            self.latencies.append((time.time() - self.obs_sent_at) * 1000.0)
        self.awaiting_action = False
        self.awaiting_action_since = None

        # chunk 边界标记。有些仿真器（RoboDojo）的"是否结束"查询本身带副作用，
        # 官方 rollout 在每个 action chunk 走完时会额外查一次；我们严格 1:1 的
        # 循环天然少查那一次，会让成绩系统性偏低。policy 节点把 chunk 最后一个
        # 动作的 JointCommand.mode 填成 "chunk_end"，adapter 收到就补上。
        # 用 mode 而不是另开一条 dora 通道：跨通道到达顺序没有保证。
        if self._step_takes_chunk_end:
            eff = list(command.effort or [])
            chunk_end = bool(eff) and eff[0] == CHUNK_END_SENTINEL
            outcome = self.adapter.step(action, chunk_end=chunk_end)
        else:
            outcome = self.adapter.step(action)
        self.step_idx += 1
        self.episode_return += outcome.reward
        self.last_metrics = outcome.metrics or {}

        # 每步查一次判定，不是 episode 结束才查
        if outcome.success or outcome.done:
            if outcome.success:
                termination = "success"
            elif self.last_metrics:
                # 长程任务：环境说结束了但没全对 → 部分完成
                termination = "partial"
            else:
                termination = "failed"
            self._record_episode(success=outcome.success, termination=termination)
            self.advance()
        elif self.step_idx >= self.cfg.max_steps:
            self._record_episode(success=False, termination="timed_out")
            self.advance()
        else:
            # 链式任务的子任务边界：换指令并让 policy 丢掉上一子任务的动作队列
            if outcome.next_instruction:
                if self.cfg.send_instruction:
                    self.policy_command(
                        "set_instruction", {"instruction": outcome.next_instruction}
                    )
                self.policy_command("reset_scene")
                self.emit_status(f"  ↳ 子任务切换 :: {outcome.next_instruction}")
            self._publish_obs(outcome.obs)

    # ---- episode 收尾 ----

    def _record_episode(
        self,
        success: bool,
        termination: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """只记录，不推进。推进由调用方显式 advance()。"""
        task_id, init_state_id, run_index = self.plan[self.plan_index]
        record = EpisodeRecord(
            episode_id=f"{self.cfg.suite}_t{task_id}_i{init_state_id}_r{run_index}",
            task_id=task_id,
            init_state_id=init_state_id,
            run_index=run_index,
            task_description=self.adapter.task_language(task_id),
            success=success,
            termination=termination,
            num_steps=self.step_idx,
            return_value=self.episode_return,
            episode_seed=self.episode_seed,
            mean_policy_latency_ms=(
                sum(self.latencies) / len(self.latencies) if self.latencies else None
            ),
            success_source=getattr(self.adapter, "SUCCESS_SOURCE", ""),
            metrics=self.last_metrics,
            error_code=error_code,
            error_message=error_message,
        )
        video = self.recorder.save(
            f"ep_t{task_id}_i{init_state_id}_r{run_index}_{termination}"
        )
        if video:
            record.metrics = {**record.metrics, "video": video}

        self.result.episodes.append(record)
        self.awaiting_action = False
        self.awaiting_action_since = None

        done_n = len(self.result.episodes)
        self.emit_status(
            f"{record.episode_id} {termination} ({self.step_idx} 步) | "
            f"累计 {self.result.successes}/{done_n}"
        )

    def fail_current(self, code: str, message: str) -> None:
        """当前 episode 记为失败并推进。异常路径的唯一入口。"""
        if self.finished or not 0 <= self.plan_index < len(self.plan):
            return
        self._log(f"episode 失败：{code} {message}")
        self._record_episode(
            success=False, termination="failed", error_code=code, error_message=message
        )
        self.advance()

    def stall(self, code: str, message: str) -> None:
        """policy 停滞：记录当前 episode 为失败并立即结束整批为 failed。

        与 fail_current 不同：**不 advance**。policy 已经不产 action，推进到下一
        episode 只会再等一个 timeout；这里就地结束批次，status=failed，释放 session。
        """
        if self.finished or not 0 <= self.plan_index < len(self.plan):
            return
        self._log(f"policy 停滞：{code} {message}")
        self._record_episode(
            success=False, termination="failed", error_code=code, error_message=message
        )
        self.finish("failed")

    # ---- 批次收尾 ----

    def finish(self, status: str) -> None:
        if self.finished:
            return
        self.finished = True
        self.result.status = status
        self.result.elapsed_s = time.time() - self.started_at

        # 聚合：adapter 有 aggregate 就用它，否则默认成功率
        ep_metrics = [e.metrics for e in self.result.episodes]
        agg = getattr(self.adapter, "aggregate", None)
        try:
            if callable(agg):
                primary, extra = agg(ep_metrics)
            else:
                primary, extra = default_aggregate(
                    self.result.successes, self.result.total_episodes, ep_metrics
                )
        except Exception as exc:  # noqa: BLE001
            self._log(f"指标聚合失败，退回默认：{type(exc).__name__}: {exc}")
            primary, extra = default_aggregate(
                self.result.successes, self.result.total_episodes, ep_metrics
            )
        self.result.primary_metric = primary
        # 重发次数进结果文件：正常应为 0。非 0 说明链路上丢过观测，
        # 分数仍然有效（重发是幂等的），但值得看一眼是不是 policy 太慢。
        self.result.metrics = {**extra, "obs_republished": self._republished}

        payload = self.result.to_dict()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if self.result_tag:
            # 顺带的好处：result_path 里能直接看出是哪次 invocation 产的
            safe = "".join(c for c in str(self.result_tag) if c.isalnum() or c in "-_")[:32]
            if safe:
                stamp = f"{stamp}_{safe}"

        # 落盘优先：即使后续发消息/关 env 出问题，结果也必须先安全落地。
        path = None
        try:
            path = self.result.dump(self.cfg.result_dir, stamp)
        except Exception as exc:  # noqa: BLE001
            self._log(f"结果落盘失败：{type(exc).__name__}: {exc}")
        # 服务模式要把落盘路径回给 Tool 调用方（episodes 明细不上线，只给路径）
        self.result_path = path

        # **收尾里没有 adapter.close**：在服务模式下，Skill Runtime 的生命周期
        # 长于单次 invocation —— 关了 env 下一次 bench.run 就要重载整个场景
        # （几十秒）。真正关 env 的地方是 run_node 的 finally，进程退出时才走。
        # 独立跑（autorun）时也没问题：main.py 的 simulation_app.close() 会收干净。
        for action in (
            lambda: self.node.send_output(
                "benchmark_result", pa.array([json.dumps(payload, ensure_ascii=False)])
            ),
            lambda: self.policy_command("stop"),
        ):
            try:
                action()
            except Exception as exc:  # noqa: BLE001
                self._log(f"收尾步骤失败（忽略）：{type(exc).__name__}: {exc}")

        pm = self.result.primary_metric
        self._log(
            f"批次 {status}：{pm.get('name','?')} = {pm.get('value')} "
            f"（{self.result.successes}/{self.result.total_episodes} 成功），"
            f"耗时 {self.result.elapsed_s:.0f}s"
        )
        if path:
            self._log(f"结果已落盘：{path}")
