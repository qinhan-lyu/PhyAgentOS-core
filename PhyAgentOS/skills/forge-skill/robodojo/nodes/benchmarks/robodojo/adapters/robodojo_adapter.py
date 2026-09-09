"""RoboDojo adapter：把官方 eval env 包成 _core 认识的 BenchmarkAdapter。

设计原则和 RoboCasa 那版一致——**一切走官方封装层**，不自己拼动作、不自己解图像。
官方给的东西比 RoboCasa 好得多：

  * 动作是**带名字的 dict**（`left_arm_joint_state` 等），不是扁平向量；
    env 还自带 `validate_action_dict()`。所以"分量顺序错位"这类坑
    在这一层结构上不成立——只要我们不把它压扁再瞎还原。
  * 观测里图像已经是 **RGB**（XPolicyLab 的硬规矩：任何 `COLOR_BGR2RGB`
    都是 bug），且由官方解码好，`obs["vision"][cam]["color"]` 直接是数组。

下面三条是实测/读码挖出来的坑，改代码前先读：

坑 1 —— `create_eval_env()` **无条件构造 `WsModelClient`**（eval_env.py:188），
        `deploy_cfg["port"]` 还是必填，没有策略服务器就起不来。我们的动作从
        dora 来，不需要那个 WS 客户端，所以在调用前把该命名空间里的
        `WsModelClient` 换成 stub。**不 fork 上游代码**。

坑 2 —— `is_episode_end()` 不是纯查询（eval_env.py:841），**调用次数会影响成绩**。
        它每次都跑 `reward_manager.get_reward(final_check=...)`，其中会 `_mark_env_failed`、
        推进 trigger 状态；`final_check=True` 时还额外跑 `_final_check()`/`_final_score()`。
        （注意：`final_check` **并非恒为 True** —— `success` 初值是 True，所以
        `not success and not end_flag` 为假，只有到 step_lim 或环境已判失败才置位。
        即便如此 `get_reward` 仍有副作用，次数照样重要。）

        **官方每集的真实调用序列**（逐句抄自 eval_env.run_eval + demo_policy/deploy.py，
        关键是 `take_action()` 自己最后一句就是 `self.is_episode_end()`，见 eval_env.py:462）：

            env.reset(seed)            # 内含 model_client.call("reset")
            env.run_reward(); env.get_score()
            [外层] is_episode_end()                                   ← ①
              get_obs → update_obs → get_action
              i=0..N-2: take_action(){…内含 is_episode_end()} ; 显式 is_episode_end() ; get_obs
              i=N-1   : take_action(){…内含 is_episode_end()} ; 显式 is_episode_end() ; break
            [外层] is_episode_end()                                   ← 下一 chunk 的 ①

        每个 chunk：get_obs N 次、is_episode_end **2N+1** 次。
        我们严格 1:1，每个 action 天然 2 次，所以**每 chunk 要补 1 次**——由 policy 节点
        在 chunk 最后一个动作打标记（见 _core/adapter.py 的 CHUNK_END_SENTINEL）。
        补发位置在 chunk 末尾，等价于官方下一 chunk 开头的 ①；**唯独 episode 开头
        那一次 ① 没有对应物，必须在 reset_episode 里显式补**。

坑 3 —— Isaac Sim 的 `AppLauncher` 必须在 import 任何 isaaclab 相关模块**之前**
        跑完。所以 env 的构造推迟到 main.py 起完 app 之后，本文件顶层不 import
        任何 isaaclab / omni 的东西。
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from _core.adapter import StepOutcome, default_aggregate

# 键名取自 XPolicyLab 的标准观测/动作格式（README 的 Standard Data Formats），
# **不能自己起名**：这些键要原样进 take_action。
#
# **维度绝对不能写死。** ARX X5 是 6 DOF 手臂（arm_dim=[6,6], ee_dim=[1,1]，
# 总 14），不是想当然的 7 —— 我第一版就按 7 写死了。XPolicyLab 的 AGENTS.md
# 明文规定：动作维度一律走 `get_robot_action_dim_info(env_cfg_type)`。
# 换机器人只要换 env_cfg_type，这里不用动。
def build_state_spec(env_cfg_type: str, action_type: str) -> tuple[tuple[str, int], ...]:
    """按 env_cfg_type + action_type 推出 (键名, 维度) 序列。单臂/双臂都覆盖。

    键名规则抄自 XPolicyLab/policy/demo_policy/model.py 的 get_action()：
      action_type == "joint" → `{p}arm_joint_state`，维度 = arm_dim
      action_type == "ee"    → `{p}ee_pose`，固定 7 维 [x,y,z,qw,qx,qy,qz]
    夹爪都是 `{p}ee_joint_state`，维度 = ee_dim。

    **action_type 必须和策略的 deploy.yml 对上**：G05 是 joint，官方 launcher
    的默认值却是 ee。两边不一致 = 键名对不上 = 直接 KeyError（这是好事，
    比静默跑歪强）。
    """
    from XPolicyLab.utils.process_data import get_robot_action_dim_info

    info = get_robot_action_dim_info(env_cfg_type)
    arm_dims, ee_dims = list(info["arm_dim"]), list(info["ee_dim"])
    if len(arm_dims) != len(ee_dims):
        raise ValueError(f"arm_dim 与 ee_dim 数量不一致：{info}")
    if len(arm_dims) == 1:
        prefixes = [""]
    elif len(arm_dims) == 2:
        prefixes = ["left_", "right_"]
    else:
        raise NotImplementedError(f"不支持 {len(arm_dims)} 条手臂")
    if action_type not in ("joint", "ee"):
        raise ValueError(f'action_type 必须是 "joint" 或 "ee"，收到 {action_type!r}')
    spec: list[tuple[str, int]] = []
    for p, ad, ed in zip(prefixes, arm_dims, ee_dims):
        if action_type == "joint":
            spec.append((f"{p}arm_joint_state", int(ad)))
        else:
            spec.append((f"{p}ee_pose", 7))
        spec.append((f"{p}ee_joint_state", int(ed)))
    return tuple(spec)


def build_obs_spec(env_cfg_type: str) -> tuple[tuple[str, int], ...]:
    """观测里 state 的完整 (键名, 维度) 序列，**按键名字典序**。

    注意它比动作规格大：动作只有 arm + gripper（joint 模式 14 维），观测还多一份
    `{p}ee_pose`（各 7 维），实测 ARX X5 合计 **6 个键 / 28 维**。
    只把动作那 14 维塞进 proprio 通道，策略就再也看不到 ee_pose —— 而且不报错。
    字典序恰好给出 left_arm / left_ee_joint / left_ee_pose / right_... 的顺序，
    与官方观测字典一致。首帧会拿真实观测校验，不一致直接炸。
    """
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


def assert_real_robodojo() -> None:
    """守卫：确认 import 到的是真的 RoboDojo 包而不是同名目录造出的命名空间包。

    本仓库常被 clone 成 `robodojo`，与上游同名。父目录一旦进 sys.path，
    `import env` / `import task` 就会先撞上本仓库 → 变成没有 __init__ 的
    命名空间包 → 上游注册代码从不执行。症状是"Task not found"而不是
    ImportError，极难排查（RoboCasa 那次是 `Environment XXX not found`）。
    """
    import env as _env

    if getattr(_env, "__file__", None) is None and not getattr(_env, "__path__", None):
        raise ImportError(
            "import 到的 `env` 是空命名空间包，多半是 sys.path 里混进了本仓库的父目录。"
            "只把 RoboDojo 仓库根挂上 sys.path。"
        )


def _stub_model_client() -> None:
    """把 eval_env 里的 WsModelClient 换成 stub —— 见文件头坑 1。"""
    import src.eval_client.eval_env as _ee

    # 生命周期类调用要放行：`env.reset()` 自己就会调 `model_client.call("reset")`
    # （eval_env.py:245），不是只有 eval_one_episode 才碰 model_client。
    # 但**取动作**必须炸——那说明有人误用了官方 rollout，我们的动作从 dora 来。
    _LIFECYCLE = frozenset({"reset", "update_obs", "update_obs_batch",
                            "prepare_case", "trial_end"})

    class _NoPolicyClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def call(self, func_name: str | None = None, *a: Any, **k: Any) -> None:
            if func_name in _LIFECYCLE:
                return None
            raise RuntimeError(
                f"forge 节点不经过 WS 策略服务器，但收到了 {func_name!r}。"
                "动作从 dora 的 action 通道来；走到这里说明有人调用了官方的 "
                "eval_one_episode()，那条路我们不用。"
            )

        def close(self) -> None:
            pass

    _ee.WsModelClient = _NoPolicyClient


class RoboDojoAdapter:
    """一个 adapter 实例管一个 Isaac Sim env（num_envs 固定为 1）。"""

    RECORD_FPS = 30.0

    def __init__(self, cfg: Any, simulation_app: Any, log: Any = print) -> None:
        assert_real_robodojo()
        self._log = log
        self.cfg = cfg
        self._app = simulation_app
        self._env: Any = None
        self._env_task_id: int | None = None
        self._chunk_end_extra_calls = 0
        self._episode_end_calls = 0
        self._current_layout_id: int | None = None
        self._episode_started = False
        self._last_obs: dict[str, Any] | None = None

        from env.global_configs import BENCHMARK, ROOT_DIR

        self._benchmark = BENCHMARK
        self._root = ROOT_DIR
        self._state_spec = build_state_spec(cfg.env_cfg_type, cfg.action_type)   # 动作：14 维
        self._obs_spec = build_obs_spec(cfg.env_cfg_type)                        # 观测：28 维
        self._obs_spec_checked = False
        self.tasks: list[str] = self._resolve_tasks()

    # ---- 元信息 ----

    def _resolve_tasks(self) -> list[str]:
        """suite 支持两种写法：逗号分隔的任务名，或 `all`（走官方 inventory）。"""
        raw = (self.cfg.suite or "").strip()
        if raw and raw.lower() != "all":
            return [t.strip() for t in raw.split(",") if t.strip()]
        import importlib

        reg = importlib.import_module(f"task.{self._benchmark}.task_registry")
        listed = getattr(reg, "list_tasks", None)
        if not callable(listed):
            raise RuntimeError("task_registry 没有 list_tasks()，请在 suite 里显式列任务名")
        return list(listed())

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    def num_init_states(self, task_id: int) -> int:
        """init_state_id 映射成 seed_manager 里第几个**布局 ID**。

        **RoboDojo 的 "seed" 不是随机种子，是布局 ID。**
        `seed_manager.init_eval()` 去 `Assets/Eval_Layout/RoboDojo/<config>/<seed>/`
        枚举预生成的布局文件，`seed_list` 就是磁盘上那批 ID；`--seed 0/1/2` 选的
        是**哪一套布局目录**，不是 RNG 种子。同一个布局 ID → 场景逐位确定。

        所以绝不能用 `_core` 的 `derive_episode_seed()` 哈希 —— 那样和官方跑的
        根本不是同一批场景，逐位等价无从谈起（RoboCasa 那轮就是因为两边种子
        推导不同，只能比聚合值、没法逐集比对）。
        """
        return int(self.cfg.init_states_per_task)

    def _layout_ids(self, task_id: int) -> list[int]:
        env = self._ensure_env(task_id)
        ids = list(env.seed_manager.seed_list)
        if not ids:
            raise RuntimeError(
                f"任务 {self.tasks[task_id]} 在 Eval_Layout 下没有布局文件；"
                "多半是资产没下全或 seed 目录选错了"
            )
        return ids

    def task_language(self, task_id: int) -> str:
        """任务指令。权威来源是 reset 后的 `obs["instruction"]`；这里给的是
        批次展开时的预览值，reset 之后核心会用真值覆盖（见 reset_episode）。
        """
        return self._task_instruction_cache.get(self.tasks[task_id], self.tasks[task_id])

    _task_instruction_cache: dict[str, str] = {}

    # ---- episode 生命周期 ----

    def _build_env_cfg(self, task_name: str) -> Any:
        """逐字照抄官方 src/eval_client/main.py:253-341 的组装顺序。

        任何一步顺序换了都可能改变随机化结果 —— 这里不做"看起来等价"的重排。
        """
        from omegaconf import OmegaConf

        from env.global_configs import ENV_CONFIG_PATH
        from utils.load_file import load_yaml
        from utils.pipeline_utils import (
            process_config,
            process_randomization,
            resolve_random_task_num_envs,
        )

        import importlib

        task_registry = importlib.import_module(f"task.{self._benchmark}.task_registry")
        bench_path = os.path.join(self._root, "task", self._benchmark)

        eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, self.cfg.env_cfg_type + ".yml"))
        eval_cfg["task_name"] = task_name
        eval_cfg["num_envs"] = 1
        eval_cfg["device_id"] = int(self.cfg.device_id)
        eval_cfg["eval_batch"] = False          # 我们永远单环境，见 main.py:313
        eval_cfg["policy_name"] = self.cfg.policy_id
        eval_cfg["additional_info"] = self.cfg.additional_info
        eval_cfg["seed"] = int(self.cfg.seed)
        eval_cfg["physx_monitor_enabled"] = False

        # deploy_cfg 只是为了满足 create_eval_env 的必填校验；WsModelClient 已被
        # stub 掉，port 填什么都不会真的去连。
        deploy_cfg = {
            "policy_name": self.cfg.policy_id,
            "port": 0,
            "host": "localhost",
            "protocol": "ws",
            "policy_server_url": "ws://localhost:0",
            "evaluation_id": os.environ.get("ROBODOJO_RUN_ID", "forge"),
            "trial_id": f"{task_name}-forge",
            "action_case_id": f"{task_name}_case",
            "repeat_index": None,
        }

        env_cfg = OmegaConf.create(
            {
                "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml")),
                "scene": load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", eval_cfg["config"]["scene"] + ".yml")),
                "camera": load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", eval_cfg["config"]["camera"] + ".yml")),
                "robot": load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", eval_cfg["config"]["robot"] + ".yml")),
                "task_env": load_yaml(
                    task_registry.task_config_path(os.path.join(bench_path, "config"), task_name)
                ),
                "eval_cfg": eval_cfg,
                "deploy_cfg": deploy_cfg,
            }
        )
        num_envs = resolve_random_task_num_envs(task_name, 1, env_cfg.sim)
        OmegaConf.update(env_cfg, "sim.scene.num_envs", num_envs, force_add=True)
        OmegaConf.update(env_cfg, "eval_cfg.num_envs", num_envs, force_add=True)
        env_cfg = process_randomization(env_cfg)
        env_cfg, eval_num = process_config(env_cfg, task_name=task_name)
        OmegaConf.update(
            env_cfg,
            "camera.default_frequency",
            eval_cfg["observation"].get("collect_freq", 0),
            force_add=True,
        )
        env_cfg.sim.seed = [0]
        self._native_eval_num = int(eval_num)
        return env_cfg

    def _ensure_env(self, task_id: int) -> Any:
        """换 task 必须重建 env：场景资产和 reward 规则都是 task 绑定的。"""
        if self._env is not None and self._env_task_id == task_id:
            # 官方每批之间是 `env.close()` → 下一批 `env.reset(seed=...)`
            # （src/eval_client/main.py 的 while 循环）。close() 会
            # `obs_manager.reset()` + 收掉 video writer；不调的话跨 episode
            # 状态泄漏。跑 1 集看不出来，跑 50 集必然出问题。
            # 标志必须**先复位再关**：reset_episode 里 `_ensure_env` 之后紧接着
            # `_layout_ids` 又会调一次本函数，不复位就会连关两次。
            # 顺序照抄官方 main.py 的 while 体尾部：
            #   run_eval() → seed_manager.eval_step() → (若还有下一批) env.close()
            if self._episode_started:
                self._episode_started = False
                self._env.seed_manager.eval_step()
                self._env.close()
            return self._env
        if self._env is not None:
            self._env.close()
            self._env = None
        _stub_model_client()
        from src.eval_client.eval_env import create_eval_env

        task_name = self.tasks[task_id]
        self._env = create_eval_env(self._build_env_cfg(task_name), self._app)
        self._env_task_id = task_id
        return self._env

    def reset_episode(
        self, task_id: int, init_state_id: int, run_index: int, episode_seed: int
    ) -> dict[str, Any]:
        env = self._ensure_env(task_id)
        # **忽略 _core 传进来的 episode_seed**（那是哈希出来的随机种子）。
        # 这里必须按顺序取 seed_manager 的布局 ID，才和官方跑的是同一批场景。
        layout_ids = self._layout_ids(task_id)
        if init_state_id >= len(layout_ids):
            raise IndexError(
                f"init_state_id={init_state_id} 越界：任务 {self.tasks[task_id]} "
                f"只有 {len(layout_ids)} 个布局"
            )
        layout_id = int(layout_ids[init_state_id])
        self._current_layout_id = layout_id
        # 官方 reset 收的是一个列表，长度 = num_envs
        env.reset(seed=[layout_id])

        # **官方 run_eval() 的前置，一句都不能漏**（eval_client/eval_env.py:771）：
        #
        #     def run_eval(self):
        #         self.run_reward()                      # ← 注册本 task 的成功判定条件
        #         if hasattr(self, "get_score"): self.get_score()
        #         ...
        #         self.eval_one_episode()                # ← 只看这一句会漏掉上面两句
        #
        # `run_reward()` 是每个 task 自己实现的方法，作用是把该任务的判定条件
        # 灌进 reward_manager（上游 CLAUDE.md 的 PR 检查项："run_reward() 必须
        # 有意义地调用 self.reward_manager.check(...)"）。
        #
        # **不调它的后果**：check_list / final_check_list / trigger_check_list 全空
        # → `get_reward()` 直接返回 1.0 → `is_episode_end()` 判定
        # `reward > 1-1e-3` → 第 1 步就 end_flag=True, success=True。
        # 零动作策略"一步搭好塔"，报出 success_rate = 1.0。**假阳性且分数报高。**
        env.run_reward()
        if hasattr(env, "get_score"):
            env.get_score()

        # 官方外层 `while not is_episode_end():` 的第一次求值 ——
        # 它发生在**首帧 get_obs 之前**，我们必须补上，否则整条判定序列比官方
        # 少一次、且从第一步就错位。
        env.is_episode_end()

        # **max_steps 是调用方给定的真实上限，绝不静默覆盖。** RoboDojo 每个 task
        # 有原生 step_lim（build_tower=1050，put_bottles_into_dustbin=700），环境在
        # 达到 step_lim 且未成功时会自行 is_episode_end()（→ partial/failed），不会
        # 让 runner 越过去。这里**不把请求的 max_steps 改成 env_lim** —— 否则请求
        # 300 会被静默抬成 700，违反对外 schema 的“upper bound”约定（见 LIBERO
        # 公共协议：max_steps 是硬上限，`step_idx >= cfg.max_steps` 即 timed_out）。
        # profile 的 config.max_steps 应声明为该 task 的原生 horizon，让默认值允许
        # 跑完；调用方显式传更小值就是刻意限步（结果 timed_out，属“合法步数上限
        # 导致的未成功”，与工程错误区分）。这里只做**只读诊断**，不改 cfg。
        env_lim = int(getattr(env, "step_lim", 0) or 0)
        if env_lim and env_lim != self.cfg.max_steps:
            self._log(
                f"[diagnostic] 环境原生 step_lim={env_lim}；请求 max_steps={self.cfg.max_steps}。"
                f"max_steps 作为硬上限生效，环境在 step_lim 处自然终止。"
            )

        self._episode_end_calls = 0
        obs = env.get_obs()
        self._episode_started = True
        self._last_obs = obs
        instr = obs.get("instruction")
        if isinstance(instr, str) and instr:
            self._task_instruction_cache[self.tasks[task_id]] = instr
        return obs

    def _require_env(self) -> Any:
        if self._env is None:
            raise RuntimeError("env 还没建，reset_episode 必须先调用")
        return self._env

    def to_action_dict(self, flat: np.ndarray) -> dict[str, np.ndarray]:
        """扁平向量 → 官方动作 dict。切片边界与 self._state_spec 一致。"""
        out: dict[str, np.ndarray] = {}
        off = 0
        for key, dim in self._state_spec:
            out[key] = np.asarray(flat[off : off + dim], dtype=np.float32)
            off += dim
        return out

    def step(self, action: np.ndarray, chunk_end: bool = False) -> StepOutcome:
        env = self._require_env()
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        expected = sum(d for _, d in self._state_spec)
        if act.size != expected:
            raise ValueError(f"动作维度应为 {expected}，收到 {act.size}")

        env.take_action(self.to_action_dict(act))

        # 内层那一次（每个动作后必调，包括 chunk 最后一个）
        done = bool(env.is_episode_end())
        self._episode_end_calls += 1
        # 外层那一次：官方在 chunk 走完、episode 还没结束时会再调一次。
        # 少调这一次会让 _final_check() 的执行次数偏少 → 成绩系统性偏低。
        if chunk_end and not done:
            done = bool(env.is_episode_end())
            self._episode_end_calls += 1
            self._chunk_end_extra_calls += 1

        # **`env.success` 的语义是"还没失败"，不是"任务已完成"。**
        # 它在 reset 时被初始化成 True（eval_env.py:114/218），由
        # `reward_manager._mark_env_failed()` 才翻成 False。所以每步直接读它，
        # 第 1 步就恒为 True —— 我第一版就这么写的，零动作策略"一步搭好塔"，
        # 报出 success_rate = 1.0。**假阳性，而且是把分数报高的方向。**
        # 只有 episode 结束之后这个值才有意义。
        success = bool(done and env.success[0])
        # RoboDojo 的分档得分。口径与官方 eval_env.run_eval 逐字一致：
        # 成功记 1.0，否则取 reward_manager 的过程分再除以 100。
        # 只在 episode 结束时取 —— 中途取到的是尚未结算的中间态。
        episode_score = None
        if done:
            episode_score = 1.0 if success else 0.0
            if not success and hasattr(env, "get_score"):
                try:
                    process_scores = env.reward_manager.get_score()
                except Exception as exc:          # 不能因为取分失败就丢掉整集
                    self._log(f"取 process score 失败，本集按 0 记：{exc}")
                    process_scores = None
                if process_scores is not None:
                    episode_score = float(process_scores[0]) / 100.0
        # episode 已结束就不要再 get_obs()：官方在 break 之后不取观测，而
        # get_obs 内含 render() + _stream_vision()，多调一次就是多一帧。
        # （末帧官方是由 is_episode_end() 内部的 get_obs_batch(last_frame=True) 取的。）
        if not done:
            self._last_obs = env.get_obs()
        return StepOutcome(
            obs=self._last_obs or {},
            reward=1.0 if success else 0.0,
            done=done,
            success=success,
            metrics={
                "success": success,
                "episode_end_calls": self._episode_end_calls,
                # 官方 _result.json 的 details 里也是 layout_id，两边可直接对齐
                "layout_id": self._current_layout_id,
                "score": episode_score,
            },
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            self._env_task_id = None

    # ---- 观测转换 ----

    def to_state(self, obs: dict[str, Any]) -> np.ndarray | None:
        state = obs.get("state") or {}
        if not self._obs_spec_checked:
            want, got = {k for k, _ in self._obs_spec}, set(state)
            if want != got:
                raise KeyError(
                    "观测 state 的键集与推导出来的不一致，说明机器人配置变了：\n"
                    f"  推导={sorted(want)}\n  实得={sorted(got)}"
                )
            self._obs_spec_checked = True
        parts: list[np.ndarray] = []
        for key, dim in self._obs_spec:
            v = state.get(key)
            if v is None:
                raise KeyError(f"观测缺少 state.{key}；拿到的键={sorted(state)}")
            # **用 float64，不要转 float32。** dora 的 JointState 本来就是 float64；
            # 中间过一道 float32 会在第 7 位有效数字上引入舍入，逐位比对时表现为
            # 1e-11 量级的"差异"，白白污染等价性判据（实测就是这么来的）。
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            if arr.size != dim:
                raise ValueError(f"state.{key} 维度应为 {dim}，收到 {arr.size}")
            parts.append(arr)
        return np.concatenate(parts).astype(np.float64)

    def to_images(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """图像**已经是 RGB 且已解码**（XPolicyLab 的硬规矩），这里不做任何通道转换。

        任何 COLOR_BGR2RGB 都是 bug —— 上游 AGENTS.md 原话。
        """
        vision = obs.get("vision") or {}
        out: dict[str, np.ndarray] = {}
        for channel, cam in self.cfg.camera_map.items():
            entry = vision.get(cam)
            if entry is None:
                raise KeyError(f"观测里没有相机 {cam}；可用={sorted(vision)}")
            color = entry.get("color")
            if color is None:
                raise KeyError(f"相机 {cam} 没有 color")
            out[channel] = np.ascontiguousarray(np.asarray(color, dtype=np.uint8))
        return out

    # ---- 可选钩子 ----

    def instruction_from_obs(self, obs: dict[str, Any]) -> str | None:
        """从首帧观测里取真指令。

        RoboDojo 的指令由 `desc_manager.get_one_description()` 按**当前场景**生成
        （`obs_manager.py:89`），reset 之前根本不存在，所以 `_core` 那次先发的
        set_instruction 只能是任务名占位。核心会用这里的返回值补发一次。
        不实现这个钩子的话，策略拿到的是 "build_tower" 而不是
        "Build a tower using the wooden blocks and wooden boards." —— 对 VLA 是致命的，
        而且不报错。
        """
        v = obs.get("instruction")
        return str(v) if isinstance(v, str) and v else None

    def state_joint_names(self) -> list[str]:
        """观测 proprio 的分量名（28 个），不是动作的 14 个。"""
        return [f"{key}.{i}" for key, dim in self._obs_spec for i in range(dim)]

    def action_component_names(self) -> list[str] | None:
        """动作和 state 用同一套命名，policy 必须逐字回填到 JointCommand.name。"""
        return [f"{key}.{i}" for key, dim in self._state_spec for i in range(dim)]

    def aggregate(self, episode_metrics: list[dict[str, Any]]) -> tuple[dict, dict]:
        successes = sum(1 for m in episode_metrics if m.get("success"))
        primary, extra = default_aggregate(successes, len(episode_metrics), episode_metrics)
        extra["chunk_end_extra_calls"] = self._chunk_end_extra_calls
        return primary, extra
