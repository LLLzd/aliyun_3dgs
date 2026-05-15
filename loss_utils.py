#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SSIM / 组合损失（纯 PyTorch，无额外依赖）。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)  # 1,1,H,W


def ssim_map(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    pred, gt: (B,3,H,W) 或 (3,H,W)；返回标量 SSIM ∈[0,1]（越大越相似）。
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
    c1, c2 = 0.01**2, 0.03**2
    device, dtype = pred.device, pred.dtype
    win = _gaussian_window(window_size, 1.5, device, dtype).to(dtype)
    win = win.expand(pred.shape[1], 1, window_size, window_size)

    mu_x = F.conv2d(pred, win, padding=window_size // 2, groups=pred.shape[1])
    mu_y = F.conv2d(gt, win, padding=window_size // 2, groups=gt.shape[1])
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, win, padding=window_size // 2, groups=pred.shape[1]) - mu_x2
    sigma_y2 = F.conv2d(gt * gt, win, padding=window_size // 2, groups=gt.shape[1]) - mu_y2
    sigma_xy = F.conv2d(pred * gt, win, padding=window_size // 2, groups=pred.shape[1]) - mu_xy

    ssim_n = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim = (ssim_n / (ssim_d + 1e-8)).mean()
    return ssim


def dssim_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """可微 D-SSIM 损失，越小越好。"""
    return 1.0 - ssim_map(pred, gt)
