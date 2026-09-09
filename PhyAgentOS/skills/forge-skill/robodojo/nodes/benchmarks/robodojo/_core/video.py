"""episode 录像。

只依赖 OpenCV：120 上没有 ffmpeg，也没有 imageio，实测 cv2 的 `mp4v` 可写、
`avc1`(h264) 打不开。所以固定用 mp4v，失败再退 MJPG/avi。
录像失败**绝不能影响批次**——所有异常都吞掉并只记一行日志。
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np


def _tile(frames: dict[str, np.ndarray]) -> np.ndarray:
    """多路相机横向拼一张，便于人工复核。高度不同则补黑边。"""
    imgs = [np.asarray(v, dtype=np.uint8) for _, v in sorted(frames.items())]
    if not imgs:
        raise ValueError("没有可录的图像")
    h = max(im.shape[0] for im in imgs)
    padded = []
    for im in imgs:
        if im.shape[0] < h:
            pad = np.zeros((h - im.shape[0], im.shape[1], 3), dtype=np.uint8)
            im = np.concatenate([im, pad], axis=0)
        padded.append(im)
    return np.concatenate(padded, axis=1)


class EpisodeRecorder:
    """按 episode 攒帧、结束时落一个视频文件。"""

    def __init__(self, record_dir: str | None, fps: float = 20.0,
                 log: Callable[[str], None] = print) -> None:
        self.record_dir = record_dir
        self.fps = float(fps)
        self._log = log
        self._frames: list[np.ndarray] = []
        self.enabled = bool(record_dir)
        if self.enabled:
            try:
                os.makedirs(record_dir, exist_ok=True)
                import cv2  # noqa: F401
            except Exception as exc:  # noqa: BLE001
                self._log(f"录像不可用，已关闭：{type(exc).__name__}: {exc}")
                self.enabled = False

    def reset(self) -> None:
        self._frames = []

    def add(self, frames: dict[str, np.ndarray]) -> None:
        if not self.enabled:
            return
        try:
            self._frames.append(_tile(frames))
        except Exception as exc:  # noqa: BLE001
            self._log(f"录帧失败（忽略）：{type(exc).__name__}: {exc}")

    def save(self, name: str) -> str | None:
        """写文件并清空缓存。返回路径，失败返回 None。"""
        if not self.enabled or not self._frames:
            self._frames = []
            return None
        try:
            import cv2

            h, w = self._frames[0].shape[:2]
            for fourcc, ext in (("mp4v", ".mp4"), ("MJPG", ".avi")):
                path = os.path.join(self.record_dir, name + ext)
                writer = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc(*fourcc), self.fps, (w, h)
                )
                if not writer.isOpened():
                    writer.release()
                    continue
                for frame in self._frames:
                    writer.write(frame[:, :, ::-1])  # RGB → BGR
                writer.release()
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    return path
            self._log("所有编码器都写不出视频，跳过")
            return None
        except Exception as exc:  # noqa: BLE001
            self._log(f"写视频失败（忽略）：{type(exc).__name__}: {exc}")
            return None
        finally:
            self._frames = []
