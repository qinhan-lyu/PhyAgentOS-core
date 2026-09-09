"""RoboDojo 节点配置：共有字段来自 _core，这里只加 RoboDojo 特有的。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from _core.config import BaseBenchmarkConfig, fill_base, read_yaml

# dora 通道 → RoboDojo 相机名。
#
# 相机名取自 XPolicyLab 的标准观测格式（README 的 Standard Data Formats）：
# 双臂机器人是 cam_head / cam_left_wrist / cam_right_wrist。
# **不要照抄 RoboCasa 那套 main/second/third 到 camera1/2/3 的映射** —— 那次
# 官方文档和 checkpoint 的 train_config.json 自相矛盾，我们赔了两天。
# 这里原生相机名是唯一的，policy 侧直接按名字取，零歧义。
CAMERA_MAP: dict[str, str] = {
    "image/head": "cam_head",
    "image/left_wrist": "cam_left_wrist",
    "image/right_wrist": "cam_right_wrist",
}


@dataclass
class RoboDojoConfig(BaseBenchmarkConfig):
    benchmark: str = "robodojo"
    # 逗号分隔的任务名，或 "all" 走官方 inventory。
    # 主对照任务用 build_tower：官方榜上头部 79/53/49/47/33、中位数才 3，
    # 区分度最好。put_bottles_into_dustbin 分虽高但头部挤在 60-97，有饱和风险。
    suite: str = "build_tower"
    # env_cfg/<name>.yml，决定机器人/场景/相机/sim 四份子配置。ARX X5 双臂。
    env_cfg_type: str = "arx_x5"
    device_id: int = 0
    # 官方 native 是每任务 50 集（Generalization 类拆 25+25）。我们只为验证
    # 链路，默认取很小的值，别照抄官方协议。
    init_states_per_task: int = 2
    # 动作语义。必须与策略 deploy.yml 的 action_type 一致：
    #   joint → arm_joint_state（维度取自机器人配置）
    #   ee    → ee_pose（固定 7 维）
    # G05 是 joint；官方 launcher 不给就默认 ee，别想当然。
    action_type: str = "joint"
    policy_id: str = "demo_policy"
    additional_info: str = "forge"
    camera_map: dict = field(default_factory=lambda: dict(CAMERA_MAP))


def load_config(path: str | Path) -> RoboDojoConfig:
    data = read_yaml(path)
    cfg = RoboDojoConfig()
    fill_base(cfg, data)

    cfg.env_cfg_type = str(data.get("env_cfg_type", cfg.env_cfg_type))
    cfg.device_id = int(data.get("device_id", cfg.device_id))
    cfg.init_states_per_task = int(data.get("init_states_per_task", cfg.init_states_per_task))
    cfg.action_type = str(data.get("action_type", cfg.action_type))
    if cfg.action_type not in ("joint", "ee"):
        raise ValueError(f'action_type 必须是 "joint" 或 "ee"，收到 {cfg.action_type!r}')
    cfg.policy_id = str(data.get("policy_id", cfg.policy_id))
    cfg.additional_info = str(data.get("additional_info", cfg.additional_info))
    cam = data.get("camera_map")
    if cam:
        if not isinstance(cam, dict):
            raise ValueError("camera_map 必须是 {dora 通道: 相机名} 的映射")
        cfg.camera_map = dict(cam)
    if cfg.init_states_per_task < 1:
        raise ValueError("init_states_per_task 必须 ≥1")
    return cfg
