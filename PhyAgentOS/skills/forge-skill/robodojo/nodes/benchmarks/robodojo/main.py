#!/usr/bin/env python3
"""RoboDojo benchmark 节点（Dora）。调度核心在 _core，这里只负责组装。

**启动顺序有硬约束**：Isaac Sim 的 `AppLauncher` 必须在 import 任何
isaaclab/omni 相关模块之前跑完，否则 Kit 起不来。所以本文件的 import 是分段的，
不要为了"整洁"把下半段的 import 提到顶上。官方 src/eval_client/main.py 同理。
"""

from __future__ import annotations

import argparse
import os
import sys

# 只把**本仓库根目录**挂上 sys.path，绝不挂它的父目录。
# 本仓库可能被 clone 成与上游同名的目录；父目录一旦进 sys.path，
# `import env` / `import task` 就会先撞上本仓库 → 命名空间包 → 上游注册代码
# 从不执行。症状是"Task not found"而不是 ImportError。守卫见
# adapters/robodojo_adapter.py 的 assert_real_robodojo。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _log(msg: str) -> None:
    print(f"[robodojo_bench] {msg}", flush=True)


def main() -> int:
    # **先把配置路径定死成绝对路径**：下面会 os.chdir 到 RoboDojo 仓库根，
    # 之后任何相对路径都会解析到那边去。dora 传进来的 BENCHMARK_CONFIG 通常写
    # `./config.yaml`，不先解析就会变成去 RoboDojo 仓库根找，报 FileNotFoundError。
    config_path = os.environ.get("BENCHMARK_CONFIG", "config.yaml")
    if not os.path.isabs(config_path):
        config_path = os.path.join(_HERE, config_path)

    # RoboDojo 仓库根：env/ task/ utils/ src/ 都在那儿，必须进 sys.path。
    dojo_root = os.environ.get("ROBODOJO_ROOT")
    if not dojo_root or not os.path.isdir(dojo_root):
        raise RuntimeError("必须设置 ROBODOJO_ROOT 指向 RoboDojo 仓库根目录")
    if dojo_root not in sys.path:
        sys.path.insert(0, dojo_root)
    # 官方代码里到处用相对路径（eval_result/ 等），CWD 必须是仓库根。
    os.chdir(dojo_root)

    from config import load_config  # noqa: E402  (本仓库的 config.py)

    cfg = load_config(config_path)
    # 可移植性：设备与输出目录允许用环境变量覆盖（发布后不需要改 profile 里
    # 的机器相关路径；不设则保持 profile 里的默认值，行为与旧版完全一致）。
    if (dev := os.environ.get("ROBODOJO_DEVICE_ID")) is not None:
        try:
            cfg.device_id = int(dev)
        except ValueError:
            raise RuntimeError(f"ROBODOJO_DEVICE_ID 必须是整数，收到 {dev!r}")
    for attr, var in (
        ("result_dir", "ROBODOJO_RESULT_DIR"),
        ("record_dir", "ROBODOJO_RECORD_DIR"),
    ):
        val = os.environ.get(var)
        if val:
            setattr(cfg, attr, val)
    # 输出路径同理：chdir 之后相对路径会落到 RoboDojo 仓库里去，一律先解析成绝对路径。
    for attr in ("record_dir", "result_dir"):
        v = getattr(cfg, attr, None)
        if isinstance(v, str) and v and not os.path.isabs(v):
            setattr(cfg, attr, os.path.normpath(os.path.join(_HERE, v)))

    # ---- 第一段结束：下面开始碰 Isaac Sim ----
    from isaaclab.app import AppLauncher  # noqa: E402

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    app_args = parser.parse_args([])
    app_args.headless = True
    app_args.device = f"cuda:{cfg.device_id}"
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    simulation_app = AppLauncher(app_args).app
    _log(f"Isaac Sim 已启动 (device=cuda:{cfg.device_id}, headless)")

    # ---- 第二段：Kit 起来了，现在才能 import 依赖它的东西 ----
    from _core.node import run_node  # noqa: E402
    from adapters.robodojo_adapter import RoboDojoAdapter  # noqa: E402

    adapter = RoboDojoAdapter(cfg, simulation_app, log=_log)
    _log(
        f"suite={cfg.suite} → {adapter.num_tasks} 个 task | "
        f"env_cfg={cfg.env_cfg_type} | 每 task {cfg.init_states_per_task} 个 seed"
    )
    try:
        return run_node(cfg, adapter, log=_log)
    finally:
        adapter.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
