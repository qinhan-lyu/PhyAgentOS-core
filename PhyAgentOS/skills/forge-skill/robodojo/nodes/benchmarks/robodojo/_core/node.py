"""Dora 节点入口。各 benchmark 的 main.py 只需调用 run_node()。

两种模式:

- **autorun(旧行为,默认)**:起来就把 config.yaml 里的批次跑完然后退出。
  已有的 `dataflow.yaml` 走这条,行为一个字没变 —— 它是 L1 的数值对照基线,
  不能弄丢。
- **serve(服务模式)**:等 `bench_cmd`,收到一条跑一次,跑完继续等。
  由 config 里的 `serve: true` 打开,给 Forge Tool 用。

两条路共用同一个 BenchmarkRunner,判分逻辑完全一致。
"""

from __future__ import annotations

import time
from typing import Any, Callable

from dora import Node
from forge_msgs import JointCommand

from .config import BaseBenchmarkConfig, resolve_and_validate_plan
from .runner import BenchmarkRunner
from .session import (
    CMD_CANCEL,
    CMD_DESCRIBE,
    CMD_RUN,
    HEARTBEAT_PERIOD_S,
    REPORT_ACCEPTED,
    REPORT_HEARTBEAT,
    REPORT_REJECTED,
    BenchSession,
    WireError,
    capabilities_payload,
    decode,
    encode,
)

# 等 action 的看门狗周期（秒）。取值要**明显大于**最慢策略的单步推理耗时，
# 否则会在正常推理途中误重发：pi05(3B) 实测首帧约 2 s，这里留足余量。
_WATCHDOG_S = 20.0

# 等 action 的硬超时（秒）。到期认为 policy 停滞（崩溃/死锁/不再产 action），
# 记录当前 episode 为 failed 并结束整批，避免“tools ready 但评测永远 running”。
# 心跳与观测重发不刷新计时；取值须明显大于最慢策略的单步推理（G05 稳态约 2s，
# 首帧在 reset 之后，reset 的耗时不计入，因为 reset 期间事件循环被阻塞）。
_POLICY_STALL_S = 180.0

CMD_INPUT_ID = "bench_cmd"
REPORT_OUTPUT_ID = "bench_report"


def _is_timeout(event: Any) -> bool:
    """dora 把 next(timeout=) 的超时表达成 ERROR 事件而不是 None。"""
    if not isinstance(event, dict) or event.get("type") != "ERROR":
        return False
    return "timed out" in str(event.get("error", "")).lower()


def _limits(cfg: BaseBenchmarkConfig) -> dict[str, int]:
    return {
        "max_total_episodes": int(getattr(cfg, "max_total_episodes", 0) or 0),
        "max_max_steps": int(getattr(cfg, "max_max_steps", 0) or 0),
    }


def run_node(
    cfg: BaseBenchmarkConfig,
    adapter: Any,
    log: Callable[[str], None] = print,
) -> int:
    if getattr(cfg, "serve", False):
        return _run_serve(cfg, adapter, log)
    return _run_autorun(cfg, adapter, log)


# ======================================================================
# autorun：旧行为，逐字保留
# ======================================================================


def _run_autorun(
    cfg: BaseBenchmarkConfig,
    adapter: Any,
    log: Callable[[str], None],
) -> int:
    """事件循环。无论怎么退出，结果都必须落盘。"""
    # fail fast：越界的 task/init_state 必须在建 Node 之前炸掉
    resolve_and_validate_plan(cfg, adapter)

    node = Node()
    runner = BenchmarkRunner(node, cfg, adapter, log=log)
    log(
        f"benchmark={cfg.benchmark} suite={cfg.suite} tasks={cfg.task_ids} "
        f"init_states={len(cfg.init_state_ids)} num_runs={cfg.num_runs} "
        f"seed={cfg.seed} → {len(runner.plan)} episodes"
    )

    rc = 0
    try:
        runner.policy_command("start")
        runner.advance()

        while not runner.finished:
            event = node.next(timeout=_WATCHDOG_S)

            # 超时：要么 policy 真的很慢，要么这一份观测丢了/被清了。两种情况
            # 重发一次都无害（policy 只是拿到同一帧再推一次），不重发的话后者
            # 会双向死锁 —— 见 runner.republish_if_stalled 的注释。
            #
            # **坑**：dora 的 next(timeout=) 超时不返回 None，而是抛一个 ERROR
            # 事件 "Timeout event stream error: Receiver timed out"。照抄
            # 文档里的 `if event is None` 会让超时走进致命错误分支，把整批判失败
            # （实测就这么丢过一轮 20 集）。所以两种形态都要认。
            if event is None or _is_timeout(event):
                runner.republish_if_stalled(_WATCHDOG_S)
                continue

            if event["type"] == "INPUT":
                if event["id"] == "action":
                    try:
                        runner.on_action(JointCommand.from_arrow(event["value"]))
                    except Exception as exc:  # noqa: BLE001 单步失败不该打死整批
                        runner.fail_current("step_error", f"{type(exc).__name__}: {exc}")
            elif event["type"] == "STOP":
                runner.finish("cancelled")
                break
            elif event["type"] == "ERROR":
                log(f"dora error: {event.get('error')}")
                runner.finish("failed")
                rc = 1
                break
    finally:
        runner.finish("cancelled")
        # finish() 里不再关 adapter（服务模式要复用），autorun 的收尾放这里。
        adapter.close()

    return rc


# ======================================================================
# serve：等 bench_cmd，一次跑一批，跑完继续等
# ======================================================================


def _run_serve(
    cfg: BaseBenchmarkConfig,
    adapter: Any,
    log: Callable[[str], None],
) -> int:
    node = Node()
    caps = capabilities_payload(cfg, adapter, limits=_limits(cfg))

    session: BenchSession | None = None
    last_heartbeat = 0.0
    rc = 0

    def report(payload: dict[str, Any]) -> None:
        node.send_output(REPORT_OUTPUT_ID, encode(payload))

    log(
        f"serve 模式：benchmark={cfg.benchmark} suite={cfg.suite} "
        f"{caps['num_tasks']} 个 task，等待 bench_cmd"
    )

    try:
        while True:
            # **必须带 timeout**：`bench_cmd` 是低频输入，空闲时若无限阻塞，
            # 心跳就发不出去，endpoint 侧会把本节点判成掉线。
            event = node.next(timeout=HEARTBEAT_PERIOD_S)
            now = time.time()

            # capabilities 每个心跳周期重发一次（幂等）。endpoint 可能比本节点
            # 晚起，或者中途重启 —— 只推一次的话它永远等不到，tool 永远 not ready。
            if now - last_heartbeat >= HEARTBEAT_PERIOD_S:
                last_heartbeat = now
                report(caps)
                report(
                    {
                        "type": REPORT_HEARTBEAT,
                        "invocation_id": session.invocation_id if session else None,
                        "busy": session is not None,
                    }
                )

            if event is None or _is_timeout(event):
                if session is not None:
                    session.republish_if_stalled(_WATCHDOG_S)
                    if session.fail_stalled(_POLICY_STALL_S):
                        report(session.result_report())
                        log(f"invocation {session.invocation_id} 因 policy 停滞结束")
                        session = None
                continue

            etype = event.get("type")
            if etype == "STOP":
                if session is not None:
                    session.cancel()
                    report(session.result_report())
                break
            if etype == "ERROR":
                log(f"dora error: {event.get('error')}")
                rc = 1
                break
            if etype != "INPUT":
                continue

            eid = event.get("id")

            if eid == "tick":
                # tick 只用来兜住「长时间没有 action」。**绝不在 tick 上推进
                # episode** —— 推进只由 action 驱动。
                if session is not None:
                    session.republish_if_stalled(_WATCHDOG_S)
                    if session.fail_stalled(_POLICY_STALL_S):
                        report(session.result_report())
                        log(f"invocation {session.invocation_id} 因 policy 停滞结束")
                        session = None

            elif eid == CMD_INPUT_ID:
                session = _handle_command(
                    event["value"], node, cfg, adapter, session, report, log
                )

            elif eid == "action":
                if session is None:
                    continue  # 上一批的迟到动作，丢掉
                try:
                    session.on_action(JointCommand.from_arrow(event["value"]))
                except Exception as exc:  # noqa: BLE001 单步失败不该打死整批
                    session.fail("step_error", f"{type(exc).__name__}: {exc}")
                progress = session.progress_report()
                if progress is not None:
                    report(progress)
                if session.finished:
                    report(session.result_report())
                    log(f"invocation {session.invocation_id} 结束，回到等待")
                    session = None
    finally:
        if session is not None and not session.finished:
            session.cancel()
        # 真正关仿真器的地方只有这里：进程退出时。
        adapter.close()

    return rc


def _handle_command(
    value: Any,
    node: Node,
    cfg: BaseBenchmarkConfig,
    adapter: Any,
    session: BenchSession | None,
    report: Callable[[dict[str, Any]], None],
    log: Callable[[str], None],
) -> BenchSession | None:
    """处理一条 bench_cmd，返回新的 session（或原样返回）。"""
    try:
        message = decode(value)
    except WireError as exc:
        log(f"丢弃非法 bench_cmd：{exc}")
        return session

    kind = message.get("type")

    if kind == CMD_DESCRIBE:
        report(capabilities_payload(cfg, adapter, limits=_limits(cfg)))
        return session

    if kind == CMD_CANCEL:
        if session is not None and message.get("invocation_id") == session.invocation_id:
            log(f"收到 cancel：{session.invocation_id}")
            session.cancel()
            report(session.result_report())
            return None
        return session

    if kind != CMD_RUN:
        log(f"未知 bench_cmd 类型：{kind!r}")
        return session

    invocation_id = str(message.get("invocation_id") or "")
    plan = message.get("plan")
    if not invocation_id or not isinstance(plan, dict):
        log("bench_cmd run 缺少 invocation_id 或 plan")
        return session

    if session is not None:
        # endpoint 侧已经挡了并发，走到这里说明两边状态不同步 —— 必须回 rejected。
        # 静默忽略会让调用方永远等一个不会来的结果。
        report(
            {
                "type": REPORT_REJECTED,
                "invocation_id": invocation_id,
                "code": "FORGE_BUSY",
                "message": f"已有 invocation {session.invocation_id} 在跑",
            }
        )
        return session

    try:
        new_session = BenchSession(node, cfg, adapter, plan, invocation_id, log=log)
    except Exception as exc:  # noqa: BLE001 计划非法不该打死节点
        report(
            {
                "type": REPORT_REJECTED,
                "invocation_id": invocation_id,
                "code": "BENCH_INVALID_ARGUMENT",
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
        return None

    report({"type": REPORT_ACCEPTED, "invocation_id": invocation_id})
    log(
        f"invocation {invocation_id} 开跑：tasks={new_session.cfg.task_ids} "
        f"init_states={new_session.cfg.init_state_ids} "
        f"num_runs={new_session.cfg.num_runs} → {len(new_session.runner.plan)} episodes"
    )
    # **进长阻塞之前先补一拍心跳。** `start()` 里的 reset_episode 会加载 Isaac Sim
    # 场景，几十秒内事件循环完全停住、心跳发不出去；不在这里先发一条的话，
    # endpoint 那边的静默计时是从上一拍开始算的，白白少掉将近一个周期。
    report(
        {"type": REPORT_HEARTBEAT, "invocation_id": invocation_id, "busy": True}
    )
    new_session.start()
    report(
        {"type": REPORT_HEARTBEAT, "invocation_id": invocation_id, "busy": True}
    )
    return new_session
