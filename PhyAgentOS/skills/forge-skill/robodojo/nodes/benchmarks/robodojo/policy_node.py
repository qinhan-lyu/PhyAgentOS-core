#!/usr/bin/env python3
"""RoboDojo 的 forge policy 节点。

它在 dora 这一侧扮演 XPolicyLab 里 `model_client` 的角色，把观测喂给
`XPolicyLab.policy.<POLICY>.model.Model`，再把动作 chunk 逐个吐回 benchmark 节点。

**必须逐字复刻官方 rollout 的节奏**（XPolicyLab/policy/demo_policy/deploy.py）：

    model.reset()
    while not is_episode_end():
        model.update_obs(obs); actions = model.get_action()      # 一次一个 chunk
        for i, a in enumerate(actions):
            take_action(a)
            if is_episode_end() or i+1 == len(actions): break
            model.update_obs(next_obs)                            # chunk 内也每步喂

三条不能偷懒的地方：

1. **观测驱动，不是 tick 驱动。** benchmark 节点是严格 1:1 锁步，窗口外到达的
   动作会被直接丢弃。按 tick 无条件推理会发得远比消费快 —— RoboCasa 那轮实测
   丢掉 59% 的动作，把 chunk 抽稀成 41%，成绩腰斩。tick 在这里只当心跳。

2. **chunk 执行期间每一步都要 update_obs。** 上面内层那句不是冗余：有策略吃
   历史帧（Hy-Embodied 用 6 帧、每 20 步采一次）。只在需要动作时喂会静默劣化。

3. **chunk 最后一个动作要打 chunk_end 标记。** RoboDojo 的 `is_episode_end()`
   有副作用（会跑 `_final_check()`），官方每个 chunk 查 len+1 次，我们 1:1 天然
   只查 len 次。标记让 benchmark 侧补上那一次。详见 adapters/robodojo_adapter.py。
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
from dora import Node
from forge_msgs import Image, JointCommand, JointState, PolicyCommand

# chunk 边界标记塞在 effort 里 —— mode 是 Literal 标量塞不下自定义值，
# name/position 是契约本身不能动。详见 _core/adapter.py。
CHUNK_END_SENTINEL = 1.0

# 状态/动作的分量规格。**维度绝对不能写死** —— ARX X5 是 6 DOF 手臂
# (arm_dim=[6,6], ee_dim=[1,1]，总 14)，不是想当然的 7。一律走官方的
# get_robot_action_dim_info(env_cfg_type)，和 benchmark 侧同源。
# 两边算出来必须完全一致：不一致时 benchmark 的 action_component_names 校验
# 会直接把这一集判失败（这正是我们要的：宁可炸掉，也不要静默跑歪）。
def _build_obs_spec(env_cfg_type: str) -> tuple[tuple[str, int], ...]:
    """观测 state 的完整规格（含 ee_pose，28 维），按键名字典序 —— 必须与
    benchmark 侧 build_obs_spec() 完全一致。"""
    from XPolicyLab.utils.process_data import get_robot_action_dim_info

    info = get_robot_action_dim_info(env_cfg_type)
    arm_dims, ee_dims = list(info["arm_dim"]), list(info["ee_dim"])
    prefixes = [""] if len(arm_dims) == 1 else ["left_", "right_"]
    spec: list[tuple[str, int]] = []
    for p, ad, ed in zip(prefixes, arm_dims, ee_dims):
        spec.append((f"{p}arm_joint_state", int(ad)))
        spec.append((f"{p}ee_joint_state", int(ed)))
        spec.append((f"{p}ee_pose", 7))
    return tuple(sorted(spec, key=lambda kv: kv[0]))


def _build_state_spec(env_cfg_type: str, action_type: str) -> tuple[tuple[str, int], ...]:
    from XPolicyLab.utils.process_data import get_robot_action_dim_info

    info = get_robot_action_dim_info(env_cfg_type)
    arm_dims, ee_dims = list(info["arm_dim"]), list(info["ee_dim"])
    prefixes = [""] if len(arm_dims) == 1 else ["left_", "right_"]
    spec: list[tuple[str, int]] = []
    for p, ad, ed in zip(prefixes, arm_dims, ee_dims):
        spec.append((f"{p}arm_joint_state", int(ad)) if action_type == "joint" else (f"{p}ee_pose", 7))
        spec.append((f"{p}ee_joint_state", int(ed)))
    return tuple(spec)

CHANNEL_CAM = {
    "image/head": "cam_head",
    "image/left_wrist": "cam_left_wrist",
    "image/right_wrist": "cam_right_wrist",
}
OBS_PARTS = {"proprio_state", *CHANNEL_CAM.keys()}


def log(msg: str) -> None:
    print(f"[robodojo_policy] {msg}", flush=True)


class _Tracer:
    """把每帧观测压成指纹写 jsonl，字段与 XPolicyLab/policy/forge_probe 逐字一致。

    等价性验收就是拿这两份 jsonl 逐行 diff：官方那份由 forge_probe 在策略服务器
    侧录，我们这份在 policy 节点侧录，**两边看到的都是同一层的观测**，所以能直接比。
    """

    def __init__(self) -> None:
        # _build_model() 会 chdir 到 policy 目录（官方约定），所以相对路径必须
        # 在那之前就解析掉 —— 否则轨迹会写到 XPolicyLab/policy/<name>/ 里去。
        self.path = os.environ.get("FORGE_TRACE")
        if self.path and not os.path.isabs(self.path):
            self.path = os.path.abspath(self.path)
        self.episode = -1
        self.frame = 0
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
            open(self.path, "w").close()
            log(f"录制到 {self.path}")

    @staticmethod
    def _hash(a: Any) -> str:
        import hashlib

        return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]

    def dump(self, kind: str, obs: dict[str, Any] | None) -> None:
        if not self.path:
            return
        import json

        rec: dict[str, Any] = {"ep": self.episode, "frame": self.frame, "kind": kind}
        if obs is not None:
            st = obs.get("state") or {}
            rec["state"] = {k: np.asarray(v).ravel().tolist() for k, v in sorted(st.items())}
            vis = obs.get("vision") or {}
            rec["vision"] = {
                c: {"sha": self._hash(d.get("color")), "shape": list(np.asarray(d.get("color")).shape)}
                for c, d in sorted(vis.items())
                if d.get("color") is not None
            }
            rec["instruction"] = obs.get("instruction")
            self.frame += 1
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _build_model() -> Any:
    """按 XPolicyLab 的约定加载策略适配器。

    注意 CWD：官方 setup_eval_* 脚本要求 CWD 是 policy 目录（策略里有相对路径）。
    """
    xpl_root = os.environ.get("XPOLICYLAB_ROOT")
    if not xpl_root or not os.path.isdir(xpl_root):
        raise RuntimeError("必须设置 XPOLICYLAB_ROOT 指向 XPolicyLab 检出目录")
    parent = os.path.dirname(os.path.abspath(xpl_root))
    for p in (parent, xpl_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    policy = os.environ.get("POLICY_NAME", "demo_policy")
    import importlib

    from utils.load_file import load_yaml  # type: ignore

    deploy = load_yaml(os.path.join(xpl_root, "policy", policy, "deploy.yml")) or {}
    # **不能用 setdefault**：deploy.yml 里这些键是存在的、值为 null（demo_policy
    # 的模板就长这样），setdefault 不覆盖 None，结果 Model 拿到 env_cfg_type=None
    # 去找 `env_cfg/None.yml`。必须显式判空。
    for _k, _v in (
        ("env_cfg_type", os.environ.get("ENV_CFG_TYPE", "arx_x5")),
        ("action_type", os.environ.get("ACTION_TYPE", "joint")),
        ("bench_name", "RoboDojo"),
        ("task_name", os.environ.get("TASK_NAME")),
        ("seed", 0),
    ):
        if deploy.get(_k) is None and _v is not None:
            deploy[_k] = _v
    ckpt = os.environ.get("CKPT_NAME")
    if ckpt:
        deploy["ckpt_name"] = ckpt

    proc = os.environ.get("G05_PROCESSOR_PATH")
    if proc:
        overrides = list(deploy.get("hydra_overrides") or [])
        overrides.append("model.model_arch.hf_processor_path=" + proc)
        deploy["hydra_overrides"] = overrides
        log(f"G05_PROCESSOR_PATH 覆写 hf_processor_path -> {proc}")

    os.chdir(os.path.join(xpl_root, "policy", policy))
    mod = importlib.import_module(f"XPolicyLab.policy.{policy}.model")
    log(f"加载策略 {policy}（action_type={deploy['action_type']}, env_cfg={deploy['env_cfg_type']}）")
    return mod.Model(deploy)


def _flatten_action(action: dict[str, Any], spec) -> np.ndarray:
    parts = []
    for key, dim in spec:
        v = action.get(key)
        if v is None:
            raise KeyError(f"策略动作缺少 {key}；实得键={sorted(action)}")
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if arr.size != dim:
            raise ValueError(f"{key} 维度应为 {dim}，收到 {arr.size}")
        parts.append(arr)
    return np.concatenate(parts).astype(np.float32)


def main() -> int:
    node = Node()
    # Tracer 必须在 _build_model() 之前构造：后者会 chdir 到 policy 目录。
    tracer = _Tracer()
    model = _build_model()
    env_cfg_type = os.environ.get("ENV_CFG_TYPE", "arx_x5")
    action_type = os.environ.get("ACTION_TYPE", "joint")
    state_spec = _build_state_spec(env_cfg_type, action_type)   # 动作 14 维
    obs_spec = _build_obs_spec(env_cfg_type)                    # 观测 28 维
    component_names = [f"{k}.{i}" for k, d in state_spec for i in range(d)]
    log(f"动作 {len(component_names)} 维 / 观测 {sum(d for _, d in obs_spec)} 维")

    state: np.ndarray | None = None
    images: dict[str, np.ndarray] = {}
    instruction: str = ""
    fresh: set[str] = set()
    chunk: list[np.ndarray] = []
    chunk_pos = 0
    phase = "idle"
    sent = 0

    def send_action(vec: np.ndarray, chunk_end: bool) -> None:
        nonlocal sent
        cmd = JointCommand(
            name=list(component_names),
            position=[float(x) for x in vec],
            mode="position",
            # chunk 最后一个动作打标记，让 benchmark 侧补那次 is_episode_end()
            # forge_msgs 要求 effort 要么为空、要么与 name 等长，所以填满。
            effort=[CHUNK_END_SENTINEL] * len(component_names) if chunk_end else [],
        )
        node.send_output("action", cmd.to_arrow())
        sent += 1

    def build_obs() -> dict[str, Any]:
        """拼回 XPolicyLab 的标准观测格式。图像已是 RGB，**不做任何通道转换**。"""
        off = 0
        st: dict[str, np.ndarray] = {}
        for key, dim in obs_spec:
            st[key] = np.asarray(state[off : off + dim], dtype=np.float64)
            off += dim
        return {
            "instruction": instruction,
            "vision": {cam: {"color": img} for cam, img in images.items()},
            "state": st,
        }

    def on_observation() -> bool:
        """收齐一份观测后：先无条件 update_obs，再决定要不要新推一个 chunk。"""
        nonlocal chunk, chunk_pos
        obs = build_obs()
        tracer.dump("update_obs", obs)
        model.update_obs(obs)                  # 坑 2：每步都喂，不管要不要动作
        if chunk_pos >= len(chunk):
            actions = model.get_action()
            if not actions:
                log("策略返回空 chunk，跳过这一帧")
                return False
            chunk = [_flatten_action(a, state_spec) for a in actions]
            chunk_pos = 0
        send_action(chunk[chunk_pos], chunk_end=(chunk_pos + 1 == len(chunk)))
        chunk_pos += 1
        return True

    log("就绪，等 benchmark 节点的观测")
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        eid = event["id"]

        # **`set_instruction` / `reset_scene` 不是独立的 dora 通道。**
        # `_core/runner.py:155-156` 是用 `policy_command(...)` 发的，也就是
        # `policy_command` 通道上的 `PolicyCommand` 消息，`command` 字段区分类型、
        # payload 在 `inputs_json` 里。按独立通道去收**永远收不到，而且不报错**：
        # 指令一直是空串、`model.reset()` 一次都不会调。逐行 diff 才把它揪出来
        # （指令 1050 帧全不一致 + reset 记录 0 次 vs 官方 2 次）。
        if eid == "policy_command":
            cmd = PolicyCommand.from_arrow(event["value"])
            if cmd.command == "start":
                phase = "running"
                log("收到 start")
            elif cmd.command in ("stop", "cancel"):
                phase = "idle"
            elif cmd.command == "set_instruction":
                import json as _json

                payload = _json.loads(cmd.inputs_json or "{}")
                instruction = str(payload.get("instruction", ""))
                log(f"指令：{instruction}")
            elif cmd.command == "reset_scene":
                # 新 episode：清模型内部状态 + 清本地缓存。
                # 坑：reset_scene 与观测走**不同的 dora 通道，跨通道到达顺序没有
                # 保证**，所以这里清掉的可能是已经先到的观测。benchmark 侧的看门狗
                # 会重发，重发是幂等的（见 _core/runner.republish_if_stalled）。
                # 官方每集调两次 reset（env.reset() 里一次 + eval_one_episode 开头
                # 一次），对有内部状态的策略这个次数是有意义的，所以照抄。
                tracer.episode += 1
                tracer.frame = 0
                tracer.dump("reset", None)
                model.reset()
                tracer.dump("reset", None)
                model.reset()
                state, images, fresh = None, {}, set()
                chunk, chunk_pos = [], 0
            continue

        if eid == "proprio_state":
            state = np.asarray(JointState.from_arrow(event["value"]).position, dtype=np.float64)
            fresh.add(eid)
        elif eid in CHANNEL_CAM:
            # 必须用官方的 to_numpy()：`Image.data` 是 **bytes**，
            # `np.asarray(bytes, dtype=np.uint8)` 会把整个 bytes 当成一个标量去
            # int() → `ValueError: invalid literal for int() with base 10: b'\x84{h}...'`。
            # to_numpy() 还顺带处理 encoding/step，不用我们自己拼形状。
            images[CHANNEL_CAM[eid]] = np.ascontiguousarray(
                Image.from_arrow(event["value"]).to_numpy(), dtype=np.uint8
            )
            fresh.add(eid)
        else:
            continue        # tick 之类：只当心跳，绝不在这里推理（坑 1）

        if phase == "running" and state is not None and fresh >= OBS_PARTS:
            if on_observation():
                fresh.clear()

    log(f"退出，共发出 {sent} 个动作")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
