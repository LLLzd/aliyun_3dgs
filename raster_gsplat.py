#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可选 gsplat 光栅化封装：若已安装 gsplat 且 --rasterizer gsplat，则走高质量 CUDA 路径。
未安装时给出明确安装提示。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

try:
    from gsplat.rendering import rasterization as _gsplat_rasterization
except ImportError:
    try:
        from gsplat import rasterization as _gsplat_rasterization  # type: ignore
    except ImportError:
        _gsplat_rasterization = None


def is_gsplat_available() -> bool:
    return _gsplat_rasterization is not None


def render_gsplat_rgb(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """
    单视角 RGB 渲染。viewmat: (4,4) world-to-cam；K: (3,3) 内参；colors: (N,3)。

    返回 (H, W, 3) float32 [0,1]。
    """
    if _gsplat_rasterization is None:
        raise RuntimeError(
            "未安装 gsplat。请在当前 CUDA/Torch 环境下执行: pip install gsplat\n"
            "若编译失败，请查阅 https://docs.gsplat.studio/ 与 PyTorch 版本匹配的 wheel。"
        )

    means2 = means.unsqueeze(0)
    quats2 = quats.unsqueeze(0)
    scales2 = scales.unsqueeze(0)
    opacities2 = opacities.unsqueeze(0)
    colors2 = colors.unsqueeze(0)
    viewmats = viewmat.unsqueeze(0)
    Ks = K.unsqueeze(0)

    try:
        out = _gsplat_rasterization(
            means2,
            quats2,
            scales2,
            opacities2,
            colors2,
            viewmats,
            Ks,
            width,
            height,
            render_mode="RGB",
        )
    except TypeError as e:
        raise RuntimeError(f"gsplat.rasterization 参数不匹配（请检查 gsplat 版本）: {e}") from e

    if isinstance(out, tuple):
        render_colors = out[0]
    else:
        render_colors = out
    # render_colors: (1, H, W, C)
    img = render_colors[0, ..., :3]
    return torch.clamp(img, 0.0, 1.0)


def pinhole_K(fx: torch.Tensor, fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor, device: torch.device) -> torch.Tensor:
    """构建 3x3 内参矩阵。"""
    z = torch.zeros((), device=device, dtype=torch.float32)
    o = torch.ones((), device=device, dtype=torch.float32)
    return torch.stack(
        [
            torch.stack([fx, z, cx]),
            torch.stack([z, fy, cy]),
            torch.stack([z, z, o]),
        ]
    )


def camera_dict_to_gsplat(
    camera: Dict[str, torch.Tensor], height: int, width: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """从 utils.camera_to_torch 字典得到 viewmat (4,4) 与 K (3,3)。"""
    w2c = camera["w2c"]
    fx, fy, cx, cy = camera["fx"], camera["fy"], camera["cx"], camera["cy"]
    K = pinhole_K(fx, fy, cx, cy, w2c.device)
    return w2c, K
