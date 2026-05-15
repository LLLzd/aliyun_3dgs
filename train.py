#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Gaussian Splatting 训练主脚本（阿里云 A10-30G 适配版）。

设计目标：
- 端到端自动化：视频 -> 抽帧 -> COLMAP -> 训练 -> 导出 -> 渲染对比；
- 轻量化：参考 Compact-3DGS 的“点数控制 / 稀疏优化”思想，优先保证稳定与显存；
- 小白可运行：默认参数可直接跑，支持 quick/full 两种模式；
- 训练可每 N 步保存 checkpoint、断点续训（--resume），并在中间存档后可选生成对比渲染图。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import loss_utils
import raster_gsplat
import sh_utils
from colmap_process import run_colmap_pipeline
from export_model import export_from_checkpoint
from render import render_comparisons
from utils import (
    adjust_intrinsics_crop_resize,
    camera_to_torch,
    create_logger,
    ensure_dir,
    gpu_peak_status_text,
    gpu_status_text,
    load_transforms_json,
    now_str,
    read_image_rgb_crop_resize,
    render_gaussians_bilinear,
    set_seed,
)
from video2img import process_video


class GaussianModel(nn.Module):
    """
    高斯模型：球谐颜色 deg≤2（N,9,3）、不透明度、各向同性尺度。
    """

    def __init__(self, xyz: torch.Tensor, rgb: torch.Tensor):
        super().__init__()
        self.xyz = nn.Parameter(xyz.clone())
        dc = sh_utils.rgb_to_sh0(rgb)
        sh = torch.zeros(xyz.shape[0], 9, 3, device=xyz.device, dtype=torch.float32)
        sh[:, 0, :] = dc
        self.sh_coeffs = nn.Parameter(sh)
        self.opacity_logits = nn.Parameter(torch.zeros((xyz.shape[0], 1), dtype=torch.float32, device=xyz.device))
        self.log_scales = nn.Parameter(torch.full((xyz.shape[0], 1), -2.85, dtype=torch.float32, device=xyz.device))

    @property
    def rgb(self) -> torch.Tensor:
        """由 SH DC 还原的近似 RGB（用于导出/兼容）。"""
        return sh_utils.sh0_to_rgb(self.sh_coeffs[:, 0, :])

    def prune(self, keep_mask: torch.Tensor) -> None:
        """按 mask 裁剪高斯点。"""
        with torch.no_grad():
            self.xyz = nn.Parameter(self.xyz[keep_mask].detach())
            self.sh_coeffs = nn.Parameter(self.sh_coeffs[keep_mask].detach())
            self.opacity_logits = nn.Parameter(self.opacity_logits[keep_mask].detach())
            self.log_scales = nn.Parameter(self.log_scales[keep_mask].detach())

    def state_for_save(self) -> Dict[str, torch.Tensor]:
        """导出：含 sh_coeffs 与由 DC 近似的 rgb。"""
        return {
            "xyz": self.xyz.detach().cpu(),
            "sh_coeffs": self.sh_coeffs.detach().cpu(),
            "rgb": self.rgb.detach().cpu(),
            "opacity_logits": self.opacity_logits.detach().cpu(),
            "log_scales": self.log_scales.detach().cpu(),
        }

    @classmethod
    def from_saved_dict(cls, ckpt: Dict, device: torch.device) -> "GaussianModel":
        """从 torch.save 的 dict 恢复模型（支持含 sh_coeffs 或仅 rgb 的旧 ckpt）。"""
        if "sh_coeffs" in ckpt and ckpt["sh_coeffs"] is not None:
            m = cls.__new__(cls)
            nn.Module.__init__(m)
            m.xyz = nn.Parameter(ckpt["xyz"].to(device=device, dtype=torch.float32).contiguous())
            m.sh_coeffs = nn.Parameter(ckpt["sh_coeffs"].to(device=device, dtype=torch.float32).contiguous())
            m.opacity_logits = nn.Parameter(ckpt["opacity_logits"].to(device=device, dtype=torch.float32).contiguous())
            m.log_scales = nn.Parameter(ckpt["log_scales"].to(device=device, dtype=torch.float32).contiguous())
            return m
        xyz = ckpt["xyz"].to(device=device, dtype=torch.float32)
        rgb = ckpt["rgb"].to(device=device, dtype=torch.float32)
        return cls(xyz=xyz, rgb=rgb)


def build_training_checkpoint(
    model: GaussianModel,
    args: argparse.Namespace,
    global_step: int,
    optimizer: optim.Optimizer,
) -> Dict:
    """训练用完整 checkpoint（含路径与优化器，便于断点续训）。"""
    ck: Dict = dict(model.state_for_save())
    ck["train_h"] = int(args.train_h)
    ck["train_w"] = int(args.train_w)
    ck["iters"] = int(args.iters)
    ck["global_step"] = int(global_step)
    ck["mode"] = args.mode
    ck["rasterizer"] = args.rasterizer
    ck["transforms_path"] = str(Path(args.transforms_path).resolve())
    ck["points3d_path"] = str(Path(args.points3d_path).resolve())
    ck["optimizer"] = optimizer.state_dict()
    return ck


def sample_points_from_colmap(
    points3d_path: str,
    max_points: int,
    min_points: int = 8000,
) -> Tuple[np.ndarray, np.ndarray]:
    """从 COLMAP 稀疏点初始化高斯；点不足时自动补点。"""
    data = np.load(points3d_path)
    xyz = data["xyz"].astype(np.float32)
    rgb = data["rgb"].astype(np.float32) / 255.0

    if xyz.shape[0] == 0:
        # COLMAP 失败兜底：用随机球点保障流程不崩。
        xyz = np.random.uniform(-0.2, 0.2, size=(min_points, 3)).astype(np.float32)
        rgb = np.random.uniform(0.2, 0.8, size=(min_points, 3)).astype(np.float32)
        return xyz, rgb

    if xyz.shape[0] > max_points:
        idx = np.random.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]

    if xyz.shape[0] < min_points:
        need = min_points - xyz.shape[0]
        pick = np.random.choice(xyz.shape[0], size=need, replace=True)
        jitter = np.random.normal(scale=0.004, size=(need, 3)).astype(np.float32)
        xyz = np.concatenate([xyz, xyz[pick] + jitter], axis=0)
        rgb = np.concatenate([rgb, rgb[pick]], axis=0)
    return xyz.astype(np.float32), np.clip(rgb.astype(np.float32), 0.0, 1.0)


def load_training_data(
    transforms_path: str,
    target_h: int,
    target_w: int,
) -> Tuple[List, List[torch.Tensor]]:
    """加载相机与 GT 图像：中心裁剪到目标宽高比后降采样，与内参一致（无拉伸）。"""
    frames = load_transforms_json(transforms_path)
    resized_frames = [adjust_intrinsics_crop_resize(fr, target_h, target_w) for fr in frames]
    images: List[torch.Tensor] = []
    for fr, fr_orig in zip(resized_frames, frames):
        img = read_image_rgb_crop_resize(fr_orig.image_path, target_h, target_w, src_w=fr_orig.w, src_h=fr_orig.h)
        images.append(torch.from_numpy(img).float())
    return resized_frames, images


def make_optimizer(
    model: GaussianModel, lr_xyz: float, lr_sh: float, lr_opacity: float, lr_scale: float
) -> optim.Optimizer:
    """创建优化器（xyz / SH / 透明度 / 尺度）。"""
    return optim.Adam(
        [
            {"params": [model.xyz], "lr": lr_xyz},
            {"params": [model.sh_coeffs], "lr": lr_sh},
            {"params": [model.opacity_logits], "lr": lr_opacity},
            {"params": [model.log_scales], "lr": lr_scale},
        ],
        eps=1e-15,
    )


def maybe_prune(
    model: GaussianModel,
    min_points: int,
    opacity_threshold: float,
    max_points: int,
) -> bool:
    """
    透明度剪枝 + 上限裁剪；若按阈值会低于 min_points，则自动放宽阈值，避免长期卡死在下限。
    """
    with torch.no_grad():
        opacity = torch.sigmoid(model.opacity_logits).squeeze(-1)
        n = model.xyz.shape[0]

        for factor in (1.0, 0.5, 0.25, 0.1, 0.05):
            thr = opacity_threshold * factor
            keep = opacity > thr
            if int(keep.sum().item()) >= min_points:
                if int(keep.sum().item()) < n:
                    model.prune(keep)
                    return True
                break

        if n > max_points:
            k = max_points
            topk = torch.topk(opacity, k=k, largest=True).indices
            keep2 = torch.zeros_like(opacity, dtype=torch.bool)
            keep2[topk] = True
            model.prune(keep2)
            return True
    return False


def densify_and_clone(
    model: GaussianModel,
    xyz_grad_avg: torch.Tensor,
    grad_denom: float,
    max_points: int,
    grad_thresh: float,
    size_thresh: float,
) -> bool:
    """
    简易增密：高平均梯度 + 小尺度 -> clone；高平均梯度 + 大尺度 -> split（二分裂+缩小尺度）。
    返回是否修改了模型。
    """
    with torch.no_grad():
        if grad_denom <= 0:
            return False
        gavg = xyz_grad_avg / max(grad_denom, 1.0)
        scales = torch.exp(model.log_scales).squeeze(-1)
        mask_clone = (gavg > grad_thresh) & (scales < size_thresh)
        mask_split = (gavg > grad_thresh) & (scales >= size_thresh)

        n_before = model.xyz.shape[0]
        new_xyz_list = []
        new_sh_list = []
        new_op_list = []
        new_sc_list = []

        if mask_clone.any():
            idx = torch.nonzero(mask_clone, as_tuple=False).squeeze(-1)
            # 预算：不超过 max_points
            room = max_points - n_before
            if room > 0:
                idx = idx[:room]
                dup_xyz = model.xyz[idx] + torch.randn_like(model.xyz[idx]) * 1e-4
                new_xyz_list.append(dup_xyz)
                new_sh_list.append(model.sh_coeffs[idx].clone())
                new_op_list.append(model.opacity_logits[idx].clone())
                new_sc_list.append(model.log_scales[idx].clone())

        if mask_split.any():
            idx = torch.nonzero(mask_split, as_tuple=False).squeeze(-1)
            room = max_points - n_before - sum(x.shape[0] for x in new_xyz_list)
            if room > 1:
                idx = idx[: max(0, room // 2)]
                if idx.numel() > 0:
                    base = model.xyz[idx]
                    off = torch.randn_like(base) * scales[idx].unsqueeze(-1) * 0.08
                    split_xyz = torch.cat([base + off, base - off], dim=0)
                    sh_dup = model.sh_coeffs[idx]
                    split_sh = torch.cat([sh_dup, sh_dup], dim=0)
                    split_op = torch.cat([model.opacity_logits[idx], model.opacity_logits[idx]], dim=0)
                    split_sc = torch.cat(
                        [model.log_scales[idx] - 0.25, model.log_scales[idx] - 0.25],
                        dim=0,
                    )
                    new_xyz_list.append(split_xyz)
                    new_sh_list.append(split_sh)
                    new_op_list.append(split_op)
                    new_sc_list.append(split_sc)

        if not new_xyz_list:
            return False

        new_xyz = torch.cat(new_xyz_list, dim=0)
        new_sh = torch.cat(new_sh_list, dim=0)
        new_op = torch.cat(new_op_list, dim=0)
        new_sc = torch.cat(new_sc_list, dim=0)

        if n_before + new_xyz.shape[0] > max_points:
            # 截断新增
            allow = max_points - n_before
            new_xyz = new_xyz[:allow]
            new_sh = new_sh[:allow]
            new_op = new_op[:allow]
            new_sc = new_sc[:allow]

        model.xyz = nn.Parameter(torch.cat([model.xyz, new_xyz], dim=0))
        model.sh_coeffs = nn.Parameter(torch.cat([model.sh_coeffs, new_sh], dim=0))
        model.opacity_logits = nn.Parameter(torch.cat([model.opacity_logits, new_op], dim=0))
        model.log_scales = nn.Parameter(torch.cat([model.log_scales, new_sc], dim=0))
        return True


def resolve_train_point_chunk(args: argparse.Namespace, device: torch.device, logger) -> int:
    """
    训练前向使用的点数分块大小。
    大块/不分块可提高 GPU 利用率并减少 Python 循环次数；显存紧张时请增大分块或关闭自动策略。
    """
    if getattr(args, "train_full_point_chunk", False):
        logger.info("已指定 --train_full_point_chunk：训练前向不分块累加点（最吃显存、通常 GPU 更饱和）。")
        return 0
    if getattr(args, "no_auto_train_point_chunk", False):
        return int(args.render_point_chunk)
    if device.type != "cuda":
        return int(args.render_point_chunk)
    total = torch.cuda.get_device_properties(device).total_memory
    if int(args.render_point_chunk) > 0 and total >= 17 * (1024**3):
        logger.info(
            "检测到显存 ≥17GB 且未加 --no_auto_train_point_chunk：训练前向改为整幅点一次累加（等价 chunk=0），"
            "以提高 GPU 利用率与吞吐。若 OOM 可显式设置 --render_point_chunk 8192 等并加 --no_auto_train_point_chunk。"
        )
        return 0
    return int(args.render_point_chunk)


def append_train_metrics_csv(
    csv_path: Path,
    row: Dict[str, float | int],
    write_header: bool,
) -> None:
    fieldnames = [
        "iter",
        "loss",
        "l1",
        "dssim",
        "lpips",
        "gaussians",
        "lr_xyz",
        "lr_sh",
        "lr_opacity",
        "lr_scale",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def save_train_curves_png(
    out_path: Path,
    hist: Dict[str, List[float]],
    logger,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("未安装 matplotlib，跳过 train_curves.png（指标仍写入 CSV）。")
        return
    it = hist.get("iter") or []
    if len(it) < 2:
        return
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axs[0, 0].plot(it, hist["loss"], color="#1f77b4", lw=1.0)
    axs[0, 0].set_title("loss")
    axs[0, 0].set_xlabel("iter")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(it, hist["l1"], label="l1", color="#ff7f0e", lw=1.0)
    axs[0, 1].plot(it, hist["dssim"], label="dssim", color="#2ca02c", lw=1.0)
    axs[0, 1].set_title("l1 / d-ssim")
    axs[0, 1].legend(fontsize=8)
    axs[0, 1].set_xlabel("iter")
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].plot(it, hist["gaussians"], color="#9467bd", lw=1.0)
    axs[1, 0].set_title("gaussians")
    axs[1, 0].set_xlabel("iter")
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(it, hist["lr_xyz"], label="lr_xyz", color="#d62728", lw=1.0)
    axs[1, 1].plot(it, hist["lr_sh"], label="lr_sh", color="#8c564b", lw=1.0)
    axs[1, 1].set_title("learning rate")
    axs[1, 1].legend(fontsize=8)
    axs[1, 1].set_xlabel("iter")
    axs[1, 1].set_yscale("log")
    axs[1, 1].grid(True, alpha=0.3)

    fig.suptitle("training metrics (live)", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def forward_rgb_image(
    model: GaussianModel,
    cam: Dict[str, torch.Tensor],
    train_h: int,
    train_w: int,
    args: argparse.Namespace,
    train_point_chunk: int | None = None,
) -> torch.Tensor:
    """单视角前向，返回 (H,W,3)。"""
    pc = int(train_point_chunk) if train_point_chunk is not None else int(args.render_point_chunk)
    if args.rasterizer == "gsplat":
        if not raster_gsplat.is_gsplat_available():
            raise RuntimeError("已选择 --rasterizer gsplat 但未安装 gsplat，请 pip install gsplat 或改用 pytorch。")
        w2c, K = raster_gsplat.camera_dict_to_gsplat(cam, train_h, train_w)
        cc = sh_utils.camera_center_from_w2c(w2c)
        dirs = torch.nn.functional.normalize(cc.unsqueeze(0) - model.xyz, dim=-1, eps=1e-6)
        colors = sh_utils.eval_sh_deg2(dirs, model.sh_coeffs)
        quat = torch.zeros(model.xyz.shape[0], 4, device=model.xyz.device, dtype=model.xyz.dtype)
        quat[:, 0] = 1.0  # wxyz: 单位四元数
        sc = torch.exp(model.log_scales).expand(-1, 3)
        op = torch.sigmoid(model.opacity_logits).squeeze(-1)
        return raster_gsplat.render_gsplat_rgb(model.xyz, quat, sc, op, colors, w2c, K, train_w, train_h)
    return render_gaussians_bilinear(
        model.xyz,
        None,
        model.opacity_logits,
        model.log_scales,
        cam,
        train_h,
        train_w,
        sh_coeffs=model.sh_coeffs,
        point_chunk=pc,
    )


def run_training(args: argparse.Namespace) -> Dict[str, str]:
    """训练流程：多视角 batch、SSIM、余弦 LR、增密/剪枝、峰值显存日志。"""
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    run_dir = Path(args.run_dir)
    logs_dir = ensure_dir(run_dir / "logs")
    model_dir = ensure_dir(run_dir / "models")
    render_root = ensure_dir(run_dir / "renders")
    logger = create_logger(logs_dir / "train.log", logger_name="train")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if args.rasterizer == "gsplat" and not raster_gsplat.is_gsplat_available():
        raise RuntimeError("已选择 --rasterizer gsplat 但未安装 gsplat。请 pip install gsplat 后重试，或改用 --rasterizer pytorch。")

    resume_path = (getattr(args, "resume", "") or "").strip()
    start_iter = 0
    resume_pack: Dict | None = None
    if resume_path:
        rp = Path(resume_path).expanduser().resolve()
        if not rp.is_file():
            raise FileNotFoundError(f"--resume 不存在: {rp}")
        resume_pack = torch.load(rp, map_location=device)
        if "global_step" in resume_pack:
            start_iter = int(resume_pack["global_step"])
        else:
            logger.warning(
                "checkpoint 无 global_step，将从 iter=0 继续训练（权重已加载）。"
                "若只想渲染请勿使用 --resume；旧版 final 可重新跑满 iters 或换用带步数的中间 ckpt。"
            )
            start_iter = 0
        if resume_pack.get("train_h") and resume_pack.get("train_w"):
            args.train_h = int(resume_pack["train_h"])
            args.train_w = int(resume_pack["train_w"])
        if resume_pack.get("rasterizer"):
            args.rasterizer = str(resume_pack["rasterizer"])
        logger.info("断点续训: 从 %s 恢复，已完成 iter=%d，目标 iters=%d", rp, start_iter, args.iters)

    train_pc = resolve_train_point_chunk(args, device, logger)

    logger.info("训练设备: %s", device)
    logger.info(gpu_status_text(device) + gpu_peak_status_text(device))
    logger.info(
        "训练: iters=%d, start=%d, checkpoint_every=%d, views_per_step=%d, rasterizer=%s, max_gaussians=%d, min_gaussians=%d, "
        "train_point_chunk=%d (配置 render_point_chunk=%d), lr_hold_frac=%.3f, SSIM权重=%.3f",
        args.iters,
        start_iter,
        getattr(args, "checkpoint_every", 0),
        args.views_per_step,
        args.rasterizer,
        args.max_gaussians,
        args.min_gaussians,
        train_pc,
        int(args.render_point_chunk),
        float(getattr(args, "lr_hold_frac", 0.0)),
        args.ssim_weight,
    )

    frames, images = load_training_data(
        transforms_path=args.transforms_path,
        target_h=args.train_h,
        target_w=args.train_w,
    )
    num_views = len(frames)
    logger.info("训练图像数量: %d, 分辨率: %dx%d", num_views, args.train_w, args.train_h)

    if resume_pack is not None:
        model = GaussianModel.from_saved_dict(resume_pack, device).to(device)
        logger.info("恢复高斯点数: %d", model.xyz.shape[0])
        optimizer = make_optimizer(model, args.lr_xyz, args.lr_sh, args.lr_opacity, args.lr_scale)
        if "optimizer" in resume_pack:
            try:
                optimizer.load_state_dict(resume_pack["optimizer"])
                logger.info("已加载优化器状态。")
            except Exception as ex:  # noqa: BLE001
                logger.warning("优化器状态与当前模型不一致，已重置优化器: %s", ex)
    else:
        xyz_np, rgb_np = sample_points_from_colmap(
            points3d_path=args.points3d_path,
            max_points=args.max_gaussians,
            min_points=min(args.min_gaussians, args.max_gaussians // 4),
        )
        logger.info("初始化高斯点数: %d", xyz_np.shape[0])

        xyz = torch.from_numpy(xyz_np).to(device=device, dtype=torch.float32)
        rgb = torch.from_numpy(rgb_np).to(device=device, dtype=torch.float32)
        model = GaussianModel(xyz=xyz, rgb=rgb).to(device)
        optimizer = make_optimizer(model, args.lr_xyz, args.lr_sh, args.lr_opacity, args.lr_scale)

    base_lrs = {
        "xyz": args.lr_xyz,
        "sh": args.lr_sh,
        "opacity": args.lr_opacity,
        "scale": args.lr_scale,
    }

    lpips_fn = None
    if args.use_lpips:
        try:
            import lpips as _lpips  # type: ignore

            lpips_fn = _lpips.LPIPS(net="vgg").to(device)
            for p in lpips_fn.parameters():
                p.requires_grad = False
            logger.info("已启用 LPIPS 损失。")
        except ImportError:
            logger.warning("未安装 lpips，忽略 --use_lpips。可执行: pip install lpips")

    image_tensors = [img.to(device=device, dtype=torch.float32) for img in images]
    frame_cams = [camera_to_torch(fr, device=device) for fr in frames]

    metrics_csv = logs_dir / "train_metrics.csv"
    metrics_hist: Dict[str, List[float]] = {
        "iter": [],
        "loss": [],
        "l1": [],
        "dssim": [],
        "lpips": [],
        "gaussians": [],
        "lr_xyz": [],
        "lr_sh": [],
    }
    csv_need_header = (not metrics_csv.exists()) or metrics_csv.stat().st_size == 0
    _pe = int(getattr(args, "plot_curves_every", 0) or 0)
    plot_every = int(args.log_interval) if _pe <= 0 else _pe
    if getattr(args, "no_train_curves_plot", False):
        plot_every = 0

    n_pts = model.xyz.shape[0]
    xyz_grad_accum = torch.zeros(n_pts, device=device)
    grad_denom = 0.0

    if start_iter >= args.iters:
        logger.info("已完成 iter >= 目标 iters（%d >= %d），跳过训练循环。", start_iter, args.iters)
        ckpt_path = model_dir / "gaussians_final.pt"
        torch.save(build_training_checkpoint(model, args, start_iter, optimizer), ckpt_path)
        logger.info("模型保存: %s", ckpt_path)
        return {"checkpoint": str(ckpt_path.resolve())}

    pbar = tqdm(
        range(start_iter + 1, args.iters + 1),
        desc="Training",
        ncols=120,
    )
    for it in pbar:
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        l1_acc = 0.0
        dss_acc = 0.0
        lp_acc = 0.0

        for _ in range(args.views_per_step):
            idx = random.randint(0, num_views - 1)
            cam = frame_cams[idx]
            gt = image_tensors[idx]
            pred = forward_rgb_image(model, cam, args.train_h, args.train_w, args, train_point_chunk=train_pc)

            l1 = torch.mean(torch.abs(pred - gt))
            pred_b = pred.permute(2, 0, 1).unsqueeze(0)
            gt_b = gt.permute(2, 0, 1).unsqueeze(0)
            dss = loss_utils.dssim_loss(pred_b, gt_b)
            step_loss = (1.0 - args.ssim_weight) * l1 + args.ssim_weight * dss

            if lpips_fn is not None:
                lp = lpips_fn(pred_b * 2.0 - 1.0, gt_b * 2.0 - 1.0).mean()
                step_loss = step_loss + args.lpips_weight * lp
                lp_acc += float(lp.detach().item())

            opacity_reg = 3e-5 * torch.mean(torch.sigmoid(model.opacity_logits))
            scale_reg = 8e-5 * torch.mean(torch.exp(model.log_scales))
            step_loss = step_loss + opacity_reg + scale_reg

            total_loss = total_loss + step_loss
            l1_acc += float(l1.detach().item())
            dss_acc += float(dss.detach().item())

        total_loss = total_loss / args.views_per_step
        l1_m = l1_acc / args.views_per_step
        dss_m = dss_acc / args.views_per_step
        lp_m = lp_acc / args.views_per_step if lpips_fn is not None else 0.0

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if model.xyz.grad is not None:
            with torch.no_grad():
                g = model.xyz.grad.detach().abs().sum(dim=-1)
                if g.shape[0] == xyz_grad_accum.shape[0]:
                    xyz_grad_accum += g
                    grad_denom += 1.0

        optimizer.step()

        # 余弦学习率：可选 lr_hold_frac 比例的前段保持峰值，缓解后段过早衰减导致 loss 平台
        hold_end = int(max(0.0, min(1.0, float(getattr(args, "lr_hold_frac", 0.0)))) * args.iters)
        hold_end = min(hold_end, max(args.iters - 2, 0))
        if hold_end <= 0:
            t_lr = (it - 1) / max(args.iters - 1, 1)
            cosf = 0.5 * (1.0 + math.cos(math.pi * t_lr))
        elif it <= hold_end:
            cosf = 1.0
        else:
            span = max(args.iters - hold_end - 1, 1)
            t_lr = (it - hold_end - 1) / span
            cosf = 0.5 * (1.0 + math.cos(math.pi * t_lr))
        with torch.no_grad():
            optimizer.param_groups[0]["lr"] = args.lr_min + (base_lrs["xyz"] - args.lr_min) * cosf
            optimizer.param_groups[1]["lr"] = args.lr_min + (base_lrs["sh"] - args.lr_min) * cosf
            optimizer.param_groups[2]["lr"] = args.lr_min + (base_lrs["opacity"] - args.lr_min) * cosf
            optimizer.param_groups[3]["lr"] = args.lr_min + (base_lrs["scale"] - args.lr_min) * cosf

        loss_m = float(total_loss.detach().item())

        with torch.no_grad():
            model.log_scales.data.clamp_(-5.5, -0.75)

        if it % args.log_interval == 0 or it == start_iter + 1:
            lr0 = float(optimizer.param_groups[0]["lr"])
            lr1 = float(optimizer.param_groups[1]["lr"])
            near_cap = model.xyz.shape[0] >= int(0.93 * args.max_gaussians)
            cap_hint = " (接近 max_gaussians，增密空间小)" if near_cap else ""
            msg = (
                f"iter={it}/{args.iters} loss={loss_m:.6f} l1={l1_m:.6f} dssim={dss_m:.6f}"
                + (f" lpips={lp_m:.6f}" if lpips_fn is not None else "")
                + f" gaussians={model.xyz.shape[0]}{cap_hint} lr_xyz={lr0:.2e} {gpu_status_text(device)}{gpu_peak_status_text(device)}"
            )
            logger.info(msg)
            pbar.set_postfix(loss=f"{loss_m:.5f}", points=model.xyz.shape[0])

            row = {
                "iter": it,
                "loss": loss_m,
                "l1": l1_m,
                "dssim": dss_m,
                "lpips": lp_m if lpips_fn is not None else 0.0,
                "gaussians": model.xyz.shape[0],
                "lr_xyz": lr0,
                "lr_sh": float(optimizer.param_groups[1]["lr"]),
                "lr_opacity": float(optimizer.param_groups[2]["lr"]),
                "lr_scale": float(optimizer.param_groups[3]["lr"]),
            }
            append_train_metrics_csv(metrics_csv, row, write_header=csv_need_header)
            csv_need_header = False

            for k in metrics_hist:
                metrics_hist[k].append(float(row[k]))
            cap_hist = 10000
            if len(metrics_hist["iter"]) > cap_hist:
                over = len(metrics_hist["iter"]) - cap_hist
                for k in metrics_hist:
                    metrics_hist[k] = metrics_hist[k][over:]

            if plot_every > 0 and (it % plot_every == 0 or it == start_iter + 1):
                save_train_curves_png(logs_dir / "train_curves.png", metrics_hist, logger)

        # 增密（在剪枝前执行，便于点数增长）
        if (
            args.densify_interval > 0
            and it >= args.densify_from
            and it % args.densify_interval == 0
            and args.rasterizer == "pytorch"
        ):
            did = densify_and_clone(
                model=model,
                xyz_grad_avg=xyz_grad_accum,
                grad_denom=grad_denom,
                max_points=args.max_gaussians,
                grad_thresh=args.densify_grad_thresh,
                size_thresh=args.densify_size_thresh,
            )
            if did:
                optimizer = make_optimizer(model, args.lr_xyz, args.lr_sh, args.lr_opacity, args.lr_scale)
                # 重建 scheduler 步数对齐剩余迭代较复杂，此处仅重置优化器；scheduler 保持全局 T_max
                logger.info("增密后高斯点数: %d", model.xyz.shape[0])
            xyz_grad_accum = torch.zeros(model.xyz.shape[0], device=device)
            grad_denom = 0.0

        if it % args.prune_interval == 0 and it >= args.prune_start:
            pruned = maybe_prune(
                model=model,
                min_points=args.min_gaussians,
                opacity_threshold=args.opacity_prune_threshold,
                max_points=args.max_gaussians,
            )
            if pruned:
                optimizer = make_optimizer(model, args.lr_xyz, args.lr_sh, args.lr_opacity, args.lr_scale)
                logger.info("剪枝后高斯点数: %d", model.xyz.shape[0])
                xyz_grad_accum = torch.zeros(model.xyz.shape[0], device=device)
                grad_denom = 0.0

        checkpoint_every = int(getattr(args, "checkpoint_every", 0) or 0)
        if checkpoint_every > 0 and it % checkpoint_every == 0 and it < args.iters:
            mid_path = model_dir / f"gaussians_iter_{it:07d}.pt"
            payload = build_training_checkpoint(model, args, it, optimizer)
            torch.save(payload, mid_path)
            torch.save(payload, model_dir / "gaussians_latest.pt")
            logger.info("已保存中间 checkpoint: %s", mid_path)
            if not getattr(args, "no_render_on_checkpoint", False):
                out_r = ensure_dir(render_root / f"iter_{it:07d}")
                try:
                    render_comparisons(
                        checkpoint_path=str(mid_path),
                        transforms_path=args.transforms_path,
                        output_dir=str(out_r),
                        render_h=args.render_h,
                        render_w=args.render_w,
                        min_views=max(args.render_views, 8),
                        device_str=args.device,
                        log_file=str(logs_dir / f"render_iter_{it:07d}.log"),
                        render_point_chunk=args.render_point_chunk,
                    )
                    logger.info("中间对比渲染目录: %s", out_r)
                except Exception as ex:  # noqa: BLE001
                    logger.warning("中间渲染失败（训练继续）: %s", ex)

    if plot_every > 0 and len(metrics_hist.get("iter", [])) > 1:
        save_train_curves_png(logs_dir / "train_curves.png", metrics_hist, logger)
        logger.info("指标 CSV: %s ；曲线图: %s", metrics_csv, logs_dir / "train_curves.png")

    ckpt_path = model_dir / "gaussians_final.pt"
    torch.save(build_training_checkpoint(model, args, args.iters, optimizer), ckpt_path)
    logger.info("训练完成，模型保存: %s", ckpt_path)
    logger.info("最终高斯点数: %d", model.xyz.shape[0])
    logger.info(gpu_status_text(device) + gpu_peak_status_text(device))
    return {"checkpoint": str(ckpt_path.resolve())}


def pipeline(args: argparse.Namespace) -> Dict[str, str]:
    """端到端流程：抽帧 -> COLMAP -> 训练 -> 导出 -> 渲染；或 --resume 断点续训（跳过抽帧/COLMAP）。"""
    set_seed(args.seed)

    resume_path = (getattr(args, "resume", "") or "").strip()
    if resume_path:
        rp = Path(resume_path).expanduser().resolve()
        if not rp.is_file():
            raise FileNotFoundError(f"--resume 文件不存在: {rp}")
        ck_head = torch.load(rp, map_location="cpu")
        if rp.parent.name == "models":
            run_dir = ensure_dir(rp.parent.parent)
        else:
            run_dir = ensure_dir(rp.parent)
        logs_dir = ensure_dir(run_dir / "logs")
        tmp_dir = ensure_dir(run_dir / "tmp")
        frames_dir = ensure_dir(tmp_dir / "frames")
        colmap_dir = ensure_dir(tmp_dir / "colmap")
        render_dir = ensure_dir(run_dir / "renders")
        model_dir = ensure_dir(run_dir / "models")

        logger = create_logger(logs_dir / "pipeline.log", logger_name="pipeline")
        logger.info("断点续训：跳过抽帧/COLMAP。resume=%s 运行目录=%s", rp, run_dir.resolve())

        tp = ck_head.get("transforms_path")
        pp = ck_head.get("points3d_path")
        if not tp or not pp:
            raise RuntimeError(
                "checkpoint 中缺少 transforms_path / points3d_path，无法用该文件续训。"
                "请使用本次更新后保存的中间 checkpoint（gaussians_iter_*.pt 或 gaussians_latest.pt）。"
            )
        args.run_dir = str(run_dir)
        args.transforms_path = tp
        args.points3d_path = pp
        if ck_head.get("train_h") and ck_head.get("train_w"):
            args.train_h = int(ck_head["train_h"])
            args.train_w = int(ck_head["train_w"])
        if ck_head.get("rasterizer"):
            args.rasterizer = str(ck_head["rasterizer"])
        train_out = run_training(args)
    else:
        run_name = args.run_name if args.run_name else f"{args.mode}_{now_str()}"
        run_dir = ensure_dir(Path(args.output_root) / run_name)
        logs_dir = ensure_dir(run_dir / "logs")
        tmp_dir = ensure_dir(run_dir / "tmp")
        frames_dir = ensure_dir(tmp_dir / "frames")
        colmap_dir = ensure_dir(tmp_dir / "colmap")
        render_dir = ensure_dir(run_dir / "renders")
        model_dir = ensure_dir(run_dir / "models")

        logger = create_logger(logs_dir / "pipeline.log", logger_name="pipeline")
        logger.info("运行目录: %s", run_dir.resolve())
        logger.info("运行模式: %s", args.mode)
        logger.info("输入视频: %s", Path(args.video_path).resolve())

        # Step 1: 视频抽帧
        frame_stats = process_video(
            input_video=args.video_path,
            output_dir=str(frames_dir),
            target_frames=args.target_frames,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            blur_threshold=args.blur_threshold,
            hash_distance_threshold=args.hash_distance_threshold,
        )
        logger.info("抽帧完成: %s", frame_stats)
        with open(logs_dir / "frame_stats.json", "w", encoding="utf-8") as f:
            json.dump(frame_stats, f, ensure_ascii=False, indent=2)

        # Step 2: COLMAP 位姿估计
        colmap_res = run_colmap_pipeline(
            image_dir=str(frames_dir),
            workspace_dir=str(colmap_dir),
            use_gpu=1 if (args.device == "cuda" and torch.cuda.is_available()) else 0,
            matcher=args.colmap_matcher,
            logger_name="colmap",
            log_file=str(logs_dir / "colmap.log"),
        )
        logger.info("COLMAP 输出: %s", colmap_res)

        # Step 3: 训练
        args.run_dir = str(run_dir)
        args.transforms_path = colmap_res["transforms_path"]
        args.points3d_path = colmap_res["points3d_path"]
        train_out = run_training(args)

    checkpoint = train_out["checkpoint"]
    transforms_for_render = args.transforms_path

    # Step 4: 导出模型
    export_res = export_from_checkpoint(
        checkpoint_path=checkpoint,
        output_dir=str(model_dir),
        stem_name="gaussians_final",
    )
    logger.info("模型导出完成: %s", export_res)

    # Step 5: 渲染对比图
    render_res = render_comparisons(
        checkpoint_path=checkpoint,
        transforms_path=transforms_for_render,
        output_dir=str(render_dir),
        render_h=args.render_h,
        render_w=args.render_w,
        min_views=max(args.render_views, 8),
        device_str=args.device,
        log_file=str(logs_dir / "render.log"),
        render_point_chunk=args.render_point_chunk,
    )
    logger.info("渲染完成: %s", render_res)

    summary = {
        "run_dir": str(run_dir.resolve()),
        "checkpoint": checkpoint,
        "model_ply": export_res["ply"],
        "model_splat": export_res["splat"],
        "render_meta": render_res["render_meta"],
        "transforms": transforms_for_render,
    }
    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("全流程完成，摘要文件: %s", run_dir / "run_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阿里云 A10-30G 3DGS 端到端训练脚本")
    parser.add_argument("--mode", type=str, default="full", choices=["quick", "full"], help="quick=快速验环境, full=完整重建")
    parser.add_argument("--video_path", type=str, default="input/object.MOV", help="输入视频路径")
    parser.add_argument("--output_root", type=str, default="output", help="输出根目录")
    parser.add_argument("--run_name", type=str, default="", help="自定义运行名，不填则自动按时间生成")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="训练设备")

    # 抽帧参数
    parser.add_argument("--target_frames", type=int, default=120, help="目标抽帧数（自动换算 fps）")
    parser.add_argument("--min_frames", type=int, default=100, help="最小保留帧数")
    parser.add_argument("--max_frames", type=int, default=150, help="最大保留帧数")
    parser.add_argument("--blur_threshold", type=float, default=55.0, help="去模糊阈值")
    parser.add_argument("--hash_distance_threshold", type=int, default=6, help="去重复阈值")
    parser.add_argument("--colmap_matcher", type=str, default="sequential", choices=["sequential", "exhaustive"], help="COLMAP 匹配器")

    # 训练参数
    parser.add_argument(
        "--rasterizer",
        type=str,
        default="pytorch",
        choices=["pytorch", "gsplat"],
        help="pytorch=可微2x2 splat（默认）；gsplat=高质量CUDA光栅（需 pip install gsplat）",
    )
    parser.add_argument("--iters", type=int, default=50000, help="训练外层迭代步数（每步可含多视角）")
    parser.add_argument("--train_h", type=int, default=512, help="训练图像高度（中心正方形裁剪后缩放）")
    parser.add_argument("--train_w", type=int, default=512, help="训练图像宽度")
    parser.add_argument("--render_h", type=int, default=512, help="对比图渲染高度")
    parser.add_argument("--render_w", type=int, default=512, help="对比图渲染宽度")
    parser.add_argument("--render_views", type=int, default=8, help="对比图视角数量（至少 8）")
    parser.add_argument("--max_gaussians", type=int, default=200000, help="高斯点上限")
    parser.add_argument("--min_gaussians", type=int, default=12000, help="剪枝保留下限（配合增密）")
    parser.add_argument("--lr_xyz", type=float, default=0.00135, help="xyz 学习率峰值")
    parser.add_argument("--lr_sh", type=float, default=0.0025, help="球谐系数学习率峰值")
    parser.add_argument("--lr_opacity", type=float, default=0.0022, help="透明度学习率峰值")
    parser.add_argument("--lr_scale", type=float, default=0.00115, help="尺度学习率峰值")
    parser.add_argument("--lr_min", type=float, default=2e-5, help="余弦退火最小学习率")
    parser.add_argument("--views_per_step", type=int, default=4, help="每步随机采样的视角数（堆叠显存）")
    parser.add_argument("--render_point_chunk", type=int, default=65536, help="渲染时每批点数，0 表示不分块")
    parser.add_argument("--ssim_weight", type=float, default=0.25, help="D-SSIM 项权重（其余为 L1）")
    parser.add_argument("--lpips_weight", type=float, default=0.05, help="LPIPS 权重（需 --use_lpips）")
    parser.add_argument("--use_lpips", action="store_true", help="启用 LPIPS（需 pip install lpips）")
    parser.add_argument("--densify_from", type=int, default=500, help="开始增密的迭代")
    parser.add_argument("--densify_interval", type=int, default=350, help="增密间隔，0 关闭")
    parser.add_argument("--densify_grad_thresh", type=float, default=2e-4, help="平均位置梯度阈值")
    parser.add_argument("--densify_size_thresh", type=float, default=0.012, help="尺度判据：小于则 clone，否则 split")
    parser.add_argument("--prune_start", type=int, default=2200, help="开始剪枝迭代")
    parser.add_argument("--prune_interval", type=int, default=500, help="剪枝间隔")
    parser.add_argument("--opacity_prune_threshold", type=float, default=0.014, help="透明度剪枝基准阈值")
    parser.add_argument("--log_interval", type=int, default=50, help="训练日志间隔")
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10000,
        help="每隔 N iter 保存 models/gaussians_iter_XXXXXXX.pt 与 gaussians_latest.pt；0 关闭",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="从已有 .pt 断点续训（需含 global_step 与 transforms_path 等；pipeline 会跳过抽帧/COLMAP）",
    )
    parser.add_argument(
        "--no_render_on_checkpoint",
        action="store_true",
        help="中间 checkpoint 保存后不生成对比渲染（节省时间与显存）",
    )
    parser.add_argument(
        "--lr_hold_frac",
        type=float,
        default=0.22,
        help="余弦退火前该比例迭代保持峰值 LR，减轻后段过早衰减导致 loss 平台；0=全程标准余弦",
    )
    parser.add_argument(
        "--train_full_point_chunk",
        action="store_true",
        help="训练前向点数不分块（最吃显存、通常显著提高 GPU 利用率）",
    )
    parser.add_argument(
        "--no_auto_train_point_chunk",
        action="store_true",
        help="关闭「显存≥17GB 时训练自动不分块」策略",
    )
    parser.add_argument(
        "--plot_curves_every",
        type=int,
        default=0,
        help="刷新 logs/train_curves.png 的间隔；0 表示与 --log_interval 相同",
    )
    parser.add_argument(
        "--no_train_curves_plot",
        action="store_true",
        help="不写 train_curves.png（仍按 log_interval 追加 train_metrics.csv）",
    )
    return parser.parse_args()


def apply_mode_preset(args: argparse.Namespace) -> argparse.Namespace:
    """
    模式预设：
    - quick: 2 分钟级验证，适合先检查环境/流程；
    - full: 加强版默认（更高分辨率、更多迭代与点数），适合 A10-30G 最终重建。
    """
    if args.mode == "quick":
        args.iters = min(args.iters, 420)
        args.target_frames = min(args.target_frames, 80)
        args.min_frames = min(args.min_frames, 60)
        args.max_frames = min(args.max_frames, 90)
        args.train_h = min(args.train_h, 256)
        args.train_w = min(args.train_w, 256)
        args.render_h = min(args.render_h, 256)
        args.render_w = min(args.render_w, 256)
        args.max_gaussians = min(args.max_gaussians, 22000)
        args.min_gaussians = min(args.min_gaussians, 6000)
        args.prune_start = min(args.prune_start, 120)
        args.prune_interval = min(args.prune_interval, 80)
        args.log_interval = min(args.log_interval, 20)
        args.views_per_step = min(args.views_per_step, 1)
        args.render_point_chunk = min(args.render_point_chunk, 12000) if args.render_point_chunk else 12000
        args.densify_interval = 0
        args.rasterizer = "pytorch"
        args.lr_hold_frac = 0.0
    elif args.mode == "full":
        args.iters = max(args.iters, 60000)
        args.train_h = max(args.train_h, 512)
        args.train_w = max(args.train_w, 512)
        args.render_h = max(args.render_h, 512)
        args.render_w = max(args.render_w, 512)
        args.max_gaussians = max(args.max_gaussians, 450000)
        args.min_gaussians = max(args.min_gaussians, 8000)
        args.target_frames = max(args.target_frames, 128)
        args.min_frames = max(args.min_frames, 110)
        args.max_frames = max(args.max_frames, 150)
        args.opacity_prune_threshold = min(args.opacity_prune_threshold, 0.014)
        args.prune_start = max(args.prune_start, 2200)
        args.views_per_step = max(args.views_per_step, 4)
        args.densify_grad_thresh = min(args.densify_grad_thresh, 1.65e-4)
        if args.render_point_chunk <= 0:
            args.render_point_chunk = 65536

    # 训练/对比渲染统一为正方形，与「中心正方形裁剪 + 缩放」一致
    if args.train_h != args.train_w:
        s = min(args.train_h, args.train_w)
        args.train_h = args.train_w = s
    if args.render_h != args.render_w:
        s = min(args.render_h, args.render_w)
        args.render_h = args.render_w = s
    return args


def main() -> None:
    args = parse_args()
    args = apply_mode_preset(args)
    summary = pipeline(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
