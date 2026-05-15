#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
球谐 SH（degree 0–2）工具：与 3D Gaussian Splatting 常用约定一致。
用于视线相关颜色；DC 分量由 RGB 初始化，高阶分量初值为 0。
"""

from __future__ import annotations

import torch

# 实球谐基（degree 2）常数，与 Inria 3DGS / gsplat 常见实现一致
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2_0 = 1.0925484305930792
SH_C2_1 = -1.0925484305930792
SH_C2_2 = 0.31539156525252005
SH_C2_3 = -1.0925484305930792
SH_C2_4 = 0.5462742152960396


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    """将 RGB∈[0,1] 转为 SH 的 DC 系数（每通道），其余阶次由调用方置零。"""
    return (rgb - 0.5) / SH_C0


def sh0_to_rgb(sh0: torch.Tensor) -> torch.Tensor:
    """仅 DC 分量还原 RGB。"""
    return torch.clamp(sh0 * SH_C0 + 0.5, 0.0, 1.0)


def sh_basis_deg2(dirs: torch.Tensor) -> torch.Tensor:
    """
    计算 degree≤2 的 9 个基函数在方向 dirs 上的值。
    dirs: (N, 3) 单位向量（相机→点 或 点→相机，与训练/渲染一致即可）。
    返回: (N, 9)
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    return torch.stack(
        [
            SH_C0 * torch.ones_like(x),
            -SH_C1 * y,
            SH_C1 * z,
            -SH_C1 * x,
            SH_C2_0 * x * y,
            SH_C2_1 * y * z,
            SH_C2_2 * (3.0 * z * z - 1.0),
            SH_C2_3 * x * z,
            SH_C2_4 * (x * x - y * y),
        ],
        dim=-1,
    )


def eval_sh_deg2(dirs: torch.Tensor, sh: torch.Tensor) -> torch.Tensor:
    """
    dirs: (N, 3) 已单位化
    sh: (N, 9, 3) 每通道 9 个系数
    返回 RGB (N, 3) ∈[0,1]（clamp）
    """
    dirs = torch.nn.functional.normalize(dirs, dim=-1, eps=1e-6)
    bases = sh_basis_deg2(dirs)  # N,9
    rgb = torch.sum(bases[..., None] * sh, dim=-2) + 0.5
    return torch.clamp(rgb, 0.0, 1.0)


def camera_center_from_w2c(w2c: torch.Tensor) -> torch.Tensor:
    """由 world-to-cam 4x4 求相机中心在世界系下的坐标 (3,)。"""
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    # Xc = R Xw + t => Cw = -R^T t
    return -R.t() @ t
