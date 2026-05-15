#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目通用工具函数。

说明：
- 本文件集中放置日志、命令执行、图像处理、相机与渲染公共逻辑；
- 所有函数都尽量保持纯 Python + PyTorch，避免复杂编译依赖；
- 代码默认优先适配 CUDA（阿里云 A10-30G），同时提供 CPU 兜底。
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """设置随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，不存在则自动创建。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_str() -> str:
    """返回当前时间字符串，用于输出目录命名。"""
    return time.strftime("%Y%m%d_%H%M%S")


def create_logger(log_file: str | Path, logger_name: str = "aliyun_3dgs") -> logging.Logger:
    """创建同时输出到终端与文件的日志器。"""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def run_command(command: List[str], logger: logging.Logger, cwd: str | None = None) -> None:
    """
    执行系统命令并实时打印输出。
    出错时直接抛异常，便于上层脚本中断并给出清晰报错。
    """
    logger.info("执行命令: %s", " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.info(line.rstrip("\n"))
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"命令执行失败（exit_code={process.returncode}）: {' '.join(command)}")


def gpu_status_text(device: torch.device) -> str:
    """返回当前 GPU 显存占用信息文本。"""
    if device.type != "cuda":
        return "当前为 CPU 模式（未使用 CUDA）"
    alloc = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return f"GPU显存: 已分配 {alloc:.2f}GB / 预留 {reserved:.2f}GB / 总计 {total:.2f}GB"


def read_image_rgb(path: str | Path, resize_hw: Tuple[int, int] | None = None) -> np.ndarray:
    """读取 BGR 图并转为 RGB，返回 float32 [0,1]。"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize_hw is not None:
        h, w = resize_hw
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def save_rgb_image(path: str | Path, rgb: np.ndarray) -> None:
    """保存 RGB 图像到磁盘。"""
    img = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """将 [0,1] 概率值变换为 logits。"""
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


@dataclass
class CameraFrame:
    """单帧相机参数。"""

    image_path: str
    image_name: str
    w: int
    h: int
    fx: float
    fy: float
    cx: float
    cy: float
    c2w: np.ndarray  # [4, 4]

    @property
    def w2c(self) -> np.ndarray:
        """由 c2w 计算 w2c。"""
        return np.linalg.inv(self.c2w)


def load_transforms_json(path: str | Path) -> List[CameraFrame]:
    """读取 colmap_process.py 导出的 transforms.json。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames: List[CameraFrame] = []
    for fr in data["frames"]:
        frames.append(
            CameraFrame(
                image_path=fr["image_path"],
                image_name=fr["image_name"],
                w=int(fr["w"]),
                h=int(fr["h"]),
                fx=float(fr["fx"]),
                fy=float(fr["fy"]),
                cx=float(fr["cx"]),
                cy=float(fr["cy"]),
                c2w=np.asarray(fr["c2w"], dtype=np.float32),
            )
        )
    return frames


def render_gaussians_bilinear(
    xyz: torch.Tensor,
    rgb: torch.Tensor,
    opacity_logits: torch.Tensor,
    log_scales: torch.Tensor,
    camera: Dict[str, torch.Tensor],
    image_h: int,
    image_w: int,
    bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """
    轻量级可微渲染器（双线性 splat）。

    原理简述（Compact 思路的轻量近似）：
    1) 将 3D 高斯中心投影到 2D；
    2) 使用双线性权重分配到邻近 4 个像素；
    3) 使用深度衰减 + opacity 形成权重；
    4) 对颜色进行 scatter 累积，最后按权重归一化。

    优点：
    - 全流程可微分；
    - 纯 PyTorch，无需自定义 CUDA 编译；
    - 显存开销可控，适合 A10-30G + 4CPU 的自动化部署。
    """
    device = xyz.device

    w2c = camera["w2c"]  # [4, 4]
    fx = camera["fx"]
    fy = camera["fy"]
    cx = camera["cx"]
    cy = camera["cy"]

    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)  # [N, 4]
    cam_xyz_h = xyz_h @ w2c.t()  # [N, 4]
    cam_xyz = cam_xyz_h[:, :3]

    z = cam_xyz[:, 2]
    valid = z > 1e-4
    if valid.sum() == 0:
        bg = torch.tensor(bg_color, device=device).view(1, 1, 3).repeat(image_h, image_w, 1)
        return bg

    cam_xyz = cam_xyz[valid]
    z = z[valid]
    rgb = rgb[valid]
    opacity = torch.sigmoid(opacity_logits[valid]).squeeze(-1)
    scales = torch.exp(log_scales[valid]).squeeze(-1)

    u = fx * (cam_xyz[:, 0] / z) + cx
    v = fy * (cam_xyz[:, 1] / z) + cy

    in_view = (u >= 0.0) & (u <= image_w - 1.001) & (v >= 0.0) & (v <= image_h - 1.001)
    if in_view.sum() == 0:
        bg = torch.tensor(bg_color, device=device).view(1, 1, 3).repeat(image_h, image_w, 1)
        return bg

    u = u[in_view]
    v = v[in_view]
    z = z[in_view]
    rgb = rgb[in_view]
    opacity = opacity[in_view]
    scales = scales[in_view]

    x0 = torch.floor(u).long()
    y0 = torch.floor(v).long()
    dx = u - x0.float()
    dy = v - y0.float()

    # 使用 scale 影响 splat 强度，体现高斯“大小”在投影上的粗略影响。
    scale_gain = torch.clamp(scales * 80.0, 0.35, 2.5)
    depth_gain = torch.exp(-0.015 * z)
    base_weight = opacity * scale_gain * depth_gain

    all_x = torch.stack([x0, x0 + 1, x0, x0 + 1], dim=1)
    all_y = torch.stack([y0, y0, y0 + 1, y0 + 1], dim=1)
    all_w = torch.stack(
        [
            (1.0 - dx) * (1.0 - dy),
            dx * (1.0 - dy),
            (1.0 - dx) * dy,
            dx * dy,
        ],
        dim=1,
    )
    all_w = all_w * base_weight.unsqueeze(1)

    valid_px = (all_x >= 0) & (all_x < image_w) & (all_y >= 0) & (all_y < image_h)

    flat_idx = (all_y * image_w + all_x).view(-1)
    flat_w = all_w.view(-1)
    flat_valid = valid_px.view(-1)
    owner = torch.arange(rgb.shape[0], device=device).repeat_interleave(4)

    flat_idx = flat_idx[flat_valid]
    flat_w = flat_w[flat_valid]
    owner = owner[flat_valid]

    accum_color = torch.zeros((image_h * image_w, 3), device=device)
    accum_alpha = torch.zeros((image_h * image_w, 1), device=device)

    contrib_color = flat_w.unsqueeze(1) * rgb[owner]
    contrib_alpha = flat_w.unsqueeze(1)

    accum_color.index_add_(0, flat_idx, contrib_color)
    accum_alpha.index_add_(0, flat_idx, contrib_alpha)

    bg = torch.tensor(bg_color, device=device).view(1, 3)
    pred = accum_color / (accum_alpha + 1e-6)
    empty = (accum_alpha <= 1e-6).float()
    pred = pred * (1.0 - empty) + bg * empty
    pred = pred.view(image_h, image_w, 3)
    return pred.clamp(0.0, 1.0)


def camera_to_torch(frame: CameraFrame, device: torch.device) -> Dict[str, torch.Tensor]:
    """将 CameraFrame 转为 GPU/CPU Tensor 结构。"""
    w2c = torch.from_numpy(frame.w2c).to(device=device, dtype=torch.float32)
    return {
        "w2c": w2c,
        "fx": torch.tensor(frame.fx, device=device, dtype=torch.float32),
        "fy": torch.tensor(frame.fy, device=device, dtype=torch.float32),
        "cx": torch.tensor(frame.cx, device=device, dtype=torch.float32),
        "cy": torch.tensor(frame.cy, device=device, dtype=torch.float32),
    }


def resize_intrinsics(frame: CameraFrame, target_h: int, target_w: int) -> CameraFrame:
    """按目标分辨率缩放相机内参。"""
    sx = target_w / frame.w
    sy = target_h / frame.h
    return CameraFrame(
        image_path=frame.image_path,
        image_name=frame.image_name,
        w=target_w,
        h=target_h,
        fx=frame.fx * sx,
        fy=frame.fy * sy,
        cx=frame.cx * sx,
        cy=frame.cy * sy,
        c2w=frame.c2w.copy(),
    )


def hamming_distance(a: int, b: int) -> int:
    """计算两个 64bit 哈希的汉明距离。"""
    return int((a ^ b).bit_count())


def average_hash_8x8(gray: np.ndarray) -> int:
    """计算 8x8 average hash，用于去重。"""
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    avg = small.mean()
    bits = (small > avg).astype(np.uint8).flatten()
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def rotation_matrix_to_yaw_deg(rot_c2w: np.ndarray) -> float:
    """
    粗略计算 yaw（用于选择多角度对比图）。
    yaw = atan2(z, x)，这里以相机中心在世界坐标的方位角近似。
    """
    cam_pos = rot_c2w[:3, 3]
    yaw = math.degrees(math.atan2(cam_pos[2], cam_pos[0]))
    return float(yaw)
