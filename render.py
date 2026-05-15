#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
渲染对比脚本：
从训练模型与 COLMAP 相机中生成多角度“原始帧 vs 重建结果”对比图。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from utils import (
    camera_to_torch,
    create_logger,
    ensure_dir,
    load_transforms_json,
    read_image_rgb,
    render_gaussians_bilinear,
    resize_intrinsics,
    save_rgb_image,
)


def select_view_indices(frames: List, min_count: int = 8) -> List[Tuple[str, int]]:
    """选择至少 8 个代表角度（包含正侧背斜、顶、底）。"""
    centers = np.array([fr.c2w[:3, 3] for fr in frames], dtype=np.float32)
    center = centers.mean(axis=0, keepdims=True)
    rel = centers - center

    yaw = np.degrees(np.arctan2(rel[:, 2], rel[:, 0]))
    pitch = np.degrees(np.arctan2(rel[:, 1], np.sqrt(rel[:, 0] ** 2 + rel[:, 2] ** 2) + 1e-6))

    def nearest_yaw(target: float) -> int:
        diff = np.abs(((yaw - target + 180.0) % 360.0) - 180.0)
        return int(np.argmin(diff))

    picks: List[Tuple[str, int]] = [
        ("正面(0°)", nearest_yaw(0.0)),
        ("右侧(90°)", nearest_yaw(90.0)),
        ("背面(180°)", nearest_yaw(180.0)),
        ("左侧(-90°)", nearest_yaw(-90.0)),
        ("斜视(45°)", nearest_yaw(45.0)),
        ("斜视(-45°)", nearest_yaw(-45.0)),
        ("顶面(最高俯仰)", int(np.argmax(pitch))),
        ("底面(最低俯仰)", int(np.argmin(pitch))),
    ]

    # 去重后补足数量，避免某些角度映射到同一帧。
    used = set()
    dedup: List[Tuple[str, int]] = []
    for name, idx in picks:
        if idx not in used:
            used.add(idx)
            dedup.append((name, idx))
    if len(dedup) < min_count:
        remain = [i for i in np.argsort(yaw) if int(i) not in used]
        for i in remain:
            dedup.append((f"补充视角{len(dedup)+1}", int(i)))
            if len(dedup) >= min_count:
                break
    return dedup[:max(min_count, 8)]


def draw_compare(gt_rgb: np.ndarray, pred_rgb: np.ndarray, title: str, image_name: str) -> np.ndarray:
    """拼接对比图并添加中文标签。"""
    gt_u8 = np.clip(gt_rgb * 255.0, 0, 255).astype(np.uint8)
    pred_u8 = np.clip(pred_rgb * 255.0, 0, 255).astype(np.uint8)

    canvas = np.concatenate([gt_u8, pred_u8], axis=1)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    h, w = canvas_bgr.shape[:2]
    half = w // 2

    cv2.putText(canvas_bgr, "原始视角", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas_bgr, "重建渲染", (half + 20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas_bgr, f"角度: {title}", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas_bgr, f"帧: {image_name}", (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas_bgr


@torch.no_grad()
def render_comparisons(
    checkpoint_path: str,
    transforms_path: str,
    output_dir: str,
    render_h: int | None = None,
    render_w: int | None = None,
    min_views: int = 8,
    device_str: str = "cuda",
    log_file: str | None = None,
) -> Dict[str, str]:
    """对外主函数：根据模型输出对比图。"""
    out_dir = ensure_dir(output_dir)
    logger = create_logger(log_file or (out_dir / "render.log"), logger_name="render")

    frames = load_transforms_json(transforms_path)
    if len(frames) < 2:
        raise RuntimeError("可用相机帧数量不足，无法进行对比渲染。")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    xyz = ckpt["xyz"].to(device=device, dtype=torch.float32)
    rgb = ckpt["rgb"].to(device=device, dtype=torch.float32)
    opacity_logits = ckpt["opacity_logits"].to(device=device, dtype=torch.float32)
    log_scales = ckpt["log_scales"].to(device=device, dtype=torch.float32)

    # 分辨率以训练时记录为准；如未记录则采用用户指定或原图尺寸。
    train_h = ckpt.get("train_h", None)
    train_w = ckpt.get("train_w", None)
    if render_h is None:
        render_h = int(train_h) if train_h is not None else min(frames[0].h, 640)
    if render_w is None:
        render_w = int(train_w) if train_w is not None else int(render_h * frames[0].w / frames[0].h)

    selections = select_view_indices(frames, min_count=min_views)
    logger.info("将输出 %d 个视角对比图。", len(selections))

    save_paths: List[str] = []
    for idx, (view_name, frame_idx) in enumerate(selections, start=1):
        fr = resize_intrinsics(frames[frame_idx], target_h=render_h, target_w=render_w)
        cam = camera_to_torch(fr, device=device)
        pred = render_gaussians_bilinear(
            xyz=xyz,
            rgb=rgb,
            opacity_logits=opacity_logits,
            log_scales=log_scales,
            camera=cam,
            image_h=render_h,
            image_w=render_w,
            bg_color=(1.0, 1.0, 1.0),
        )
        pred_np = pred.detach().cpu().numpy()
        gt_np = read_image_rgb(fr.image_path, resize_hw=(render_h, render_w))
        compare_bgr = draw_compare(gt_np, pred_np, title=view_name, image_name=fr.image_name)

        out_path = out_dir / f"compare_{idx:02d}.png"
        cv2.imwrite(str(out_path), compare_bgr)
        save_paths.append(str(out_path.resolve()))
        logger.info("已保存: %s", out_path)

    meta = {
        "count": len(save_paths),
        "images": save_paths,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "transforms": str(Path(transforms_path).resolve()),
    }
    meta_path = out_dir / "render_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("渲染完成，元信息: %s", meta_path)
    return {"render_meta": str(meta_path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成多角度重建对比图（原始视角 vs 重建视角）")
    parser.add_argument("--checkpoint", type=str, required=True, help="train.py 输出的模型路径")
    parser.add_argument("--transforms", type=str, required=True, help="colmap_process.py 输出 transforms.json")
    parser.add_argument("--output_dir", type=str, default="output/renders", help="对比图输出目录")
    parser.add_argument("--render_h", type=int, default=0, help="渲染高度，0 表示自动")
    parser.add_argument("--render_w", type=int, default=0, help="渲染宽度，0 表示自动")
    parser.add_argument("--min_views", type=int, default=8, help="最少视角数量（至少 8）")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="渲染设备")
    parser.add_argument("--log_file", type=str, default="output/logs/render.log", help="日志文件")
    args = parser.parse_args()

    render_comparisons(
        checkpoint_path=args.checkpoint,
        transforms_path=args.transforms,
        output_dir=args.output_dir,
        render_h=args.render_h if args.render_h > 0 else None,
        render_w=args.render_w if args.render_w > 0 else None,
        min_views=max(args.min_views, 8),
        device_str=args.device,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
