#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Gaussian Splatting 训练主脚本（阿里云 A10-30G 适配版）。

设计目标：
- 端到端自动化：视频 -> 抽帧 -> COLMAP -> 训练 -> 导出 -> 渲染对比；
- 轻量化：参考 Compact-3DGS 的“点数控制 / 稀疏优化”思想，优先保证稳定与显存；
- 小白可运行：默认参数可直接跑，支持 quick/full 两种模式。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

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


def forward_rgb_image(
    model: GaussianModel,
    cam: Dict[str, torch.Tensor],
    train_h: int,
    train_w: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    """单视角前向，返回 (H,W,3)。"""
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
        point_chunk=args.render_point_chunk,
    )


def run_training(args: argparse.Namespace) -> Dict[str, str]:
    """训练流程：多视角 batch、SSIM、余弦 LR、增密/剪枝、峰值显存日志。"""
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    run_dir = Path(args.run_dir)
    logs_dir = ensure_dir(run_dir / "logs")
    model_dir = ensure_dir(run_dir / "models")
    logger = create_logger(logs_dir / "train.log", logger_name="train")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if args.rasterizer == "gsplat" and not raster_gsplat.is_gsplat_available():
        raise RuntimeError("已选择 --rasterizer gsplat 但未安装 gsplat。请 pip install gsplat 后重试，或改用 --rasterizer pytorch。")

    logger.info("训练设备: %s", device)
    logger.info(gpu_status_text(device) + gpu_peak_status_text(device))
    logger.info(
        "训练: iters=%d, views_per_step=%d, rasterizer=%s, max_gaussians=%d, min_gaussians=%d, "
        "point_chunk=%d, SSIM权重=%.3f",
        args.iters,
        args.views_per_step,
        args.rasterizer,
        args.max_gaussians,
        args.min_gaussians,
        args.render_point_chunk,
        args.ssim_weight,
    )

    frames, images = load_training_data(
        transforms_path=args.transforms_path,
        target_h=args.train_h,
        target_w=args.train_w,
    )
    num_views = len(frames)
    logger.info("训练图像数量: %d, 分辨率: %dx%d", num_views, args.train_w, args.train_h)

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

    n_pts = model.xyz.shape[0]
    xyz_grad_accum = torch.zeros(n_pts, device=device)
    grad_denom = 0.0

    pbar = tqdm(range(1, args.iters + 1), desc="Training", ncols=120)
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
            pred = forward_rgb_image(model, cam, args.train_h, args.train_w, args)

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

        # 余弦学习率（按全局 iter，不依赖 scheduler 对象，便于剪枝后重建优化器）
        t = (it - 1) / max(args.iters - 1, 1)
        cosf = 0.5 * (1.0 + math.cos(math.pi * t))
        with torch.no_grad():
            optimizer.param_groups[0]["lr"] = args.lr_min + (base_lrs["xyz"] - args.lr_min) * cosf
            optimizer.param_groups[1]["lr"] = args.lr_min + (base_lrs["sh"] - args.lr_min) * cosf
            optimizer.param_groups[2]["lr"] = args.lr_min + (base_lrs["opacity"] - args.lr_min) * cosf
            optimizer.param_groups[3]["lr"] = args.lr_min + (base_lrs["scale"] - args.lr_min) * cosf

        loss_m = float(total_loss.detach().item())

        with torch.no_grad():
            model.log_scales.data.clamp_(-5.5, -0.75)

        if it % args.log_interval == 0 or it == 1:
            msg = (
                f"iter={it}/{args.iters} loss={loss_m:.6f} l1={l1_m:.6f} dssim={dss_m:.6f}"
                + (f" lpips={lp_m:.6f}" if lpips_fn is not None else "")
                + f" gaussians={model.xyz.shape[0]} {gpu_status_text(device)}{gpu_peak_status_text(device)}"
            )
            logger.info(msg)
            pbar.set_postfix(loss=f"{loss_m:.5f}", points=model.xyz.shape[0])

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

    ckpt = model.state_for_save()
    ckpt["train_h"] = int(args.train_h)
    ckpt["train_w"] = int(args.train_w)
    ckpt["iters"] = int(args.iters)
    ckpt["mode"] = args.mode
    ckpt["rasterizer"] = args.rasterizer
    ckpt_path = model_dir / "gaussians_final.pt"
    torch.save(ckpt, ckpt_path)
    logger.info("训练完成，模型保存: %s", ckpt_path)
    logger.info("最终高斯点数: %d", model.xyz.shape[0])
    logger.info(gpu_status_text(device) + gpu_peak_status_text(device))
    return {"checkpoint": str(ckpt_path.resolve())}


def pipeline(args: argparse.Namespace) -> Dict[str, str]:
    """端到端流程：抽帧 -> COLMAP -> 训练 -> 导出 -> 渲染。"""
    set_seed(args.seed)

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
        use_gpu=1 if args.device == "cuda" else 0,
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
        transforms_path=colmap_res["transforms_path"],
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
        "transforms": colmap_res["transforms_path"],
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
    elif args.mode == "full":
        args.iters = max(args.iters, 60000)
        args.train_h = max(args.train_h, 512)
        args.train_w = max(args.train_w, 512)
        args.render_h = max(args.render_h, 512)
        args.render_w = max(args.render_w, 512)
        args.max_gaussians = max(args.max_gaussians, 300000)
        args.min_gaussians = max(args.min_gaussians, 8000)
        args.target_frames = max(args.target_frames, 128)
        args.min_frames = max(args.min_frames, 110)
        args.max_frames = max(args.max_frames, 150)
        args.opacity_prune_threshold = min(args.opacity_prune_threshold, 0.014)
        args.prune_start = max(args.prune_start, 2200)
        args.views_per_step = max(args.views_per_step, 4)
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
