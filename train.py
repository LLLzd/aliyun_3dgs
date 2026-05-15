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

from colmap_process import run_colmap_pipeline
from export_model import export_from_checkpoint
from render import render_comparisons
from utils import (
    adjust_intrinsics_crop_resize,
    camera_to_torch,
    create_logger,
    ensure_dir,
    gpu_status_text,
    inverse_sigmoid,
    load_transforms_json,
    now_str,
    read_image_rgb_crop_resize,
    render_gaussians_bilinear,
    set_seed,
)
from video2img import process_video


class GaussianModel(nn.Module):
    """
    轻量高斯模型：
    - xyz: 高斯中心；
    - rgb_logits: 颜色参数（通过 sigmoid 映射到 [0,1]）；
    - opacity_logits: 不透明度；
    - log_scales: 尺度（各向同性，压缩为 1 维）。
    """

    def __init__(self, xyz: torch.Tensor, rgb: torch.Tensor):
        super().__init__()
        self.xyz = nn.Parameter(xyz.clone())
        self.rgb_logits = nn.Parameter(inverse_sigmoid(rgb.clone()))
        self.opacity_logits = nn.Parameter(torch.zeros((xyz.shape[0], 1), dtype=torch.float32, device=xyz.device))
        self.log_scales = nn.Parameter(torch.full((xyz.shape[0], 1), -3.0, dtype=torch.float32, device=xyz.device))

    @property
    def rgb(self) -> torch.Tensor:
        return torch.sigmoid(self.rgb_logits)

    def prune(self, keep_mask: torch.Tensor) -> None:
        """按 mask 裁剪高斯点，降低显存并提高训练稳定性。"""
        with torch.no_grad():
            self.xyz = nn.Parameter(self.xyz[keep_mask].detach())
            self.rgb_logits = nn.Parameter(self.rgb_logits[keep_mask].detach())
            self.opacity_logits = nn.Parameter(self.opacity_logits[keep_mask].detach())
            self.log_scales = nn.Parameter(self.log_scales[keep_mask].detach())

    def state_for_save(self) -> Dict[str, torch.Tensor]:
        """导出保存字段。"""
        return {
            "xyz": self.xyz.detach().cpu(),
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
        jitter = np.random.normal(scale=0.0025, size=(need, 3)).astype(np.float32)
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


def make_optimizer(model: GaussianModel, lr_xyz: float, lr_color: float, lr_opacity: float, lr_scale: float) -> optim.Optimizer:
    """创建优化器（点位、颜色、透明度、尺度分别配置学习率）。"""
    return optim.Adam(
        [
            {"params": [model.xyz], "lr": lr_xyz},
            {"params": [model.rgb_logits], "lr": lr_color},
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
    """根据透明度和上限执行剪枝。"""
    with torch.no_grad():
        opacity = torch.sigmoid(model.opacity_logits).squeeze(-1)
        keep = opacity > opacity_threshold
        if keep.sum().item() < min_points:
            return False

        if int(keep.sum().item()) < model.xyz.shape[0]:
            model.prune(keep)
            return True

        if model.xyz.shape[0] > max_points:
            k = max_points
            topk = torch.topk(opacity, k=k, largest=True).indices
            keep2 = torch.zeros_like(opacity, dtype=torch.bool)
            keep2[topk] = True
            model.prune(keep2)
            return True
    return False


def run_training(args: argparse.Namespace) -> Dict[str, str]:
    """训练流程（仅训练阶段，输入数据应已准备完毕）。"""
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    run_dir = Path(args.run_dir)
    logs_dir = ensure_dir(run_dir / "logs")
    model_dir = ensure_dir(run_dir / "models")
    logger = create_logger(logs_dir / "train.log", logger_name="train")

    logger.info("训练设备: %s", device)
    logger.info(gpu_status_text(device))

    frames, images = load_training_data(
        transforms_path=args.transforms_path,
        target_h=args.train_h,
        target_w=args.train_w,
    )
    logger.info("训练图像数量: %d, 分辨率: %dx%d", len(frames), args.train_w, args.train_h)

    xyz_np, rgb_np = sample_points_from_colmap(
        points3d_path=args.points3d_path,
        max_points=args.max_gaussians,
        min_points=args.min_gaussians,
    )
    logger.info("初始化高斯点数: %d", xyz_np.shape[0])

    xyz = torch.from_numpy(xyz_np).to(device=device, dtype=torch.float32)
    rgb = torch.from_numpy(rgb_np).to(device=device, dtype=torch.float32)
    model = GaussianModel(xyz=xyz, rgb=rgb).to(device)
    optimizer = make_optimizer(model, args.lr_xyz, args.lr_color, args.lr_opacity, args.lr_scale)

    image_tensors = [img.to(device=device, dtype=torch.float32) for img in images]
    frame_cams = [camera_to_torch(fr, device=device) for fr in frames]

    pbar = tqdm(range(1, args.iters + 1), desc="Training", ncols=120)
    for it in pbar:
        optimizer.zero_grad(set_to_none=True)
        idx = random.randint(0, len(frame_cams) - 1)
        cam = frame_cams[idx]
        gt = image_tensors[idx]

        pred = render_gaussians_bilinear(
            xyz=model.xyz,
            rgb=model.rgb,
            opacity_logits=model.opacity_logits,
            log_scales=model.log_scales,
            camera=cam,
            image_h=args.train_h,
            image_w=args.train_w,
            bg_color=(1.0, 1.0, 1.0),
        )

        l1 = torch.mean(torch.abs(pred - gt))
        # 轻微正则：抑制过高 opacity，避免“涂抹式”填充。
        opacity_reg = 1e-4 * torch.mean(torch.sigmoid(model.opacity_logits))
        scale_reg = 1e-4 * torch.mean(torch.exp(model.log_scales))
        loss = l1 + opacity_reg + scale_reg
        loss.backward()

        # 防止异常梯度引发 NaN。
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 每轮限制尺度范围，防止爆炸。
        with torch.no_grad():
            model.log_scales.data.clamp_(-6.0, -1.0)

        if it % args.log_interval == 0 or it == 1:
            msg = (
                f"iter={it}/{args.iters} "
                f"loss={loss.item():.6f} l1={l1.item():.6f} "
                f"gaussians={model.xyz.shape[0]} "
                f"{gpu_status_text(device)}"
            )
            logger.info(msg)
            pbar.set_postfix(loss=f"{loss.item():.5f}", points=model.xyz.shape[0])

        if it % args.prune_interval == 0 and it >= args.prune_start:
            pruned = maybe_prune(
                model=model,
                min_points=args.min_gaussians,
                opacity_threshold=args.opacity_prune_threshold,
                max_points=args.max_gaussians,
            )
            if pruned:
                optimizer = make_optimizer(model, args.lr_xyz, args.lr_color, args.lr_opacity, args.lr_scale)
                logger.info("执行剪枝后高斯点数: %d", model.xyz.shape[0])

    ckpt = model.state_for_save()
    ckpt["train_h"] = int(args.train_h)
    ckpt["train_w"] = int(args.train_w)
    ckpt["iters"] = int(args.iters)
    ckpt["mode"] = args.mode
    ckpt_path = model_dir / "gaussians_final.pt"
    torch.save(ckpt, ckpt_path)
    logger.info("训练完成，模型保存: %s", ckpt_path)
    logger.info("最终高斯点数: %d", model.xyz.shape[0])
    logger.info(gpu_status_text(device))
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
    parser.add_argument("--iters", type=int, default=7000, help="训练迭代次数（full 默认约 20~40 分钟）")
    parser.add_argument("--train_h", type=int, default=360, help="训练图像高度")
    parser.add_argument("--train_w", type=int, default=640, help="训练图像宽度")
    parser.add_argument("--render_h", type=int, default=540, help="对比图渲染高度")
    parser.add_argument("--render_w", type=int, default=960, help="对比图渲染宽度")
    parser.add_argument("--render_views", type=int, default=8, help="对比图视角数量（至少 8）")
    parser.add_argument("--max_gaussians", type=int, default=65000, help="高斯点上限（防止显存溢出）")
    parser.add_argument("--min_gaussians", type=int, default=9000, help="高斯点下限（防止过度剪枝）")
    parser.add_argument("--lr_xyz", type=float, default=0.0012, help="xyz 学习率")
    parser.add_argument("--lr_color", type=float, default=0.0030, help="颜色学习率")
    parser.add_argument("--lr_opacity", type=float, default=0.0020, help="透明度学习率")
    parser.add_argument("--lr_scale", type=float, default=0.0010, help="尺度学习率")
    parser.add_argument("--prune_start", type=int, default=1200, help="开始剪枝迭代")
    parser.add_argument("--prune_interval", type=int, default=450, help="剪枝间隔")
    parser.add_argument("--opacity_prune_threshold", type=float, default=0.03, help="透明度剪枝阈值")
    parser.add_argument("--log_interval", type=int, default=50, help="训练日志间隔")
    return parser.parse_args()


def apply_mode_preset(args: argparse.Namespace) -> argparse.Namespace:
    """
    模式预设：
    - quick: 2 分钟级验证，适合先检查环境/流程；
    - full: 20~40 分钟级重建，适合最终结果。
    """
    if args.mode == "quick":
        args.iters = min(args.iters, 420)
        args.target_frames = min(args.target_frames, 80)
        args.min_frames = min(args.min_frames, 60)
        args.max_frames = min(args.max_frames, 90)
        args.train_h = min(args.train_h, 240)
        args.train_w = min(args.train_w, 426)
        args.render_h = min(args.render_h, 360)
        args.render_w = min(args.render_w, 640)
        args.max_gaussians = min(args.max_gaussians, 22000)
        args.min_gaussians = min(args.min_gaussians, 6000)
        args.prune_start = min(args.prune_start, 120)
        args.prune_interval = min(args.prune_interval, 80)
        args.log_interval = min(args.log_interval, 20)
    return args


def main() -> None:
    args = parse_args()
    args = apply_mode_preset(args)
    summary = pipeline(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
