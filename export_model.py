#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型导出工具：
1) 导出 .ply 点云；
2) 导出 .splat 二进制（轻量通用格式，包含 xyz/scale/rgba/quat）。
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from utils import ensure_dir


def load_checkpoint(path: str) -> Dict[str, np.ndarray]:
    """读取训练输出模型，并转为 numpy。"""
    ckpt = torch.load(path, map_location="cpu")
    xyz = ckpt["xyz"].detach().cpu().numpy().astype(np.float32)
    rgb = ckpt["rgb"].detach().cpu().numpy().astype(np.float32)
    opacity = torch.sigmoid(ckpt["opacity_logits"]).detach().cpu().numpy().astype(np.float32)
    scales = torch.exp(ckpt["log_scales"]).detach().cpu().numpy().astype(np.float32)
    return {"xyz": xyz, "rgb": rgb, "opacity": opacity, "scales": scales}


def export_ply(path: str | Path, xyz: np.ndarray, rgb: np.ndarray, opacity: np.ndarray) -> None:
    """导出 ASCII PLY。"""
    path = Path(path)
    ensure_dir(path.parent)
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    alpha_u8 = np.clip(opacity * 255.0, 0, 255).astype(np.uint8).reshape(-1)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("end_header\n")
        for i in range(xyz.shape[0]):
            x, y, z = xyz[i]
            r, g, b = rgb_u8[i]
            a = alpha_u8[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {int(a)}\n")


def export_splat(path: str | Path, xyz: np.ndarray, rgb: np.ndarray, opacity: np.ndarray, scales: np.ndarray) -> None:
    """
    导出自描述二进制 .splat。
    记录布局：
    - magic: 6 bytes = b'SPLAT1'
    - count: uint32
    - 每个点:
      * xyz: float32 * 3
      * scale: float32 * 3
      * rgba: uint8 * 4
      * quat(x,y,z,w): float32 * 4（默认单位四元数）
    """
    path = Path(path)
    ensure_dir(path.parent)
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    alpha_u8 = np.clip(opacity * 255.0, 0, 255).astype(np.uint8).reshape(-1)

    with open(path, "wb") as f:
        f.write(b"SPLAT1")
        f.write(struct.pack("<I", xyz.shape[0]))
        for i in range(xyz.shape[0]):
            x, y, z = xyz[i].tolist()
            sx, sy, sz = scales[i].tolist() if scales.ndim == 2 else [float(scales[i])] * 3
            r, g, b = rgb_u8[i].tolist()
            a = int(alpha_u8[i])
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
            f.write(struct.pack("<3f3f4B4f", x, y, z, sx, sy, sz, r, g, b, a, qx, qy, qz, qw))


def export_from_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    stem_name: str = "gaussians_final",
) -> Dict[str, str]:
    """供 train.py 调用：一次性导出 ply/splat。"""
    data = load_checkpoint(checkpoint_path)
    out_dir = ensure_dir(output_dir)
    ply_path = out_dir / f"{stem_name}.ply"
    splat_path = out_dir / f"{stem_name}.splat"

    export_ply(ply_path, data["xyz"], data["rgb"], data["opacity"])
    export_splat(splat_path, data["xyz"], data["rgb"], data["opacity"], data["scales"])
    return {"ply": str(ply_path.resolve()), "splat": str(splat_path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description="将训练模型导出为 .ply 与 .splat")
    parser.add_argument("--checkpoint", type=str, required=True, help="train.py 输出的 .pt 模型路径")
    parser.add_argument("--output_dir", type=str, default="output/models", help="导出目录")
    parser.add_argument("--stem_name", type=str, default="gaussians_final", help="导出文件名前缀")
    args = parser.parse_args()

    result = export_from_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        stem_name=args.stem_name,
    )
    print("导出完成:")
    print(result)


if __name__ == "__main__":
    main()
