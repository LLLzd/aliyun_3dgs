#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频抽帧脚本（自动读取 input/object.MOV）。

功能：
1) 自动从 MOV 抽帧（默认目标 120 张，范围可调）；
2) 对帧进行清晰度筛选（拉普拉斯方差去模糊）；
3) 使用感知哈希去重复；
4) 输出标准化命名帧，供 COLMAP 与训练直接使用。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from utils import (
    average_hash_8x8,
    create_logger,
    ensure_dir,
    hamming_distance,
)


def probe_video_duration(video_path: Path) -> float:
    """通过 ffprobe 获取视频时长（秒）。"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def extract_raw_frames(video_path: Path, raw_dir: Path, fps: float) -> None:
    """用 ffmpeg 先抽取原始帧。"""
    raw_pattern = raw_dir / "raw_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps:.4f},scale='min(1280,iw)':-2",
        "-q:v",
        "2",
        str(raw_pattern),
    ]
    subprocess.run(cmd, check=True)


def filter_frames(
    raw_dir: Path,
    out_dir: Path,
    min_frames: int,
    max_frames: int,
    blur_threshold: float,
    hash_distance_threshold: int,
) -> Dict[str, int]:
    """去模糊 + 去重复，输出标准化帧序列。"""
    raw_files = sorted(raw_dir.glob("raw_*.jpg"))
    if not raw_files:
        raise RuntimeError("ffmpeg 抽帧后没有得到任何图片，请检查视频是否损坏。")

    selected: list[Path] = []
    selected_hashes: list[int] = []
    blur_reject = 0
    dup_reject = 0

    for img_path in raw_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 拉普拉斯方差越大，通常越清晰。
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < blur_threshold:
            blur_reject += 1
            continue

        ahash = average_hash_8x8(gray)
        if selected_hashes:
            min_h = min(hamming_distance(ahash, h) for h in selected_hashes[-8:])
            if min_h < hash_distance_threshold:
                dup_reject += 1
                continue

        selected.append(img_path)
        selected_hashes.append(ahash)
        if len(selected) >= max_frames:
            break

    # 如果筛选后不足最小帧数，回退到“均匀抽样”确保流程可继续。
    if len(selected) < min_frames:
        indices = np.linspace(0, len(raw_files) - 1, num=min_frames, dtype=np.int32)
        selected = [raw_files[i] for i in indices]

    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(selected, start=1):
        dst = out_dir / f"frame_{idx:05d}.jpg"
        shutil.copy2(src, dst)

    return {
        "raw_count": len(raw_files),
        "selected_count": len(selected),
        "blur_reject": blur_reject,
        "dup_reject": dup_reject,
    }


def process_video(
    input_video: str,
    output_dir: str,
    target_frames: int = 120,
    min_frames: int = 100,
    max_frames: int = 150,
    blur_threshold: float = 55.0,
    hash_distance_threshold: int = 6,
) -> Dict[str, int]:
    """供外部脚本调用的视频预处理主函数。"""
    video_path = Path(input_video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"找不到输入视频: {video_path}")

    out_dir = ensure_dir(output_dir)
    raw_dir = ensure_dir(out_dir / "_raw_frames")

    duration = probe_video_duration(video_path)
    fps = max(2.0, min(12.0, target_frames / max(duration, 1e-6)))
    extract_raw_frames(video_path, raw_dir, fps=fps)

    stats = filter_frames(
        raw_dir=raw_dir,
        out_dir=out_dir,
        min_frames=min_frames,
        max_frames=max_frames,
        blur_threshold=blur_threshold,
        hash_distance_threshold=hash_distance_threshold,
    )
    stats["duration_sec"] = duration
    stats["extract_fps"] = fps
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="从 input/object.MOV 自动抽帧并清洗")
    parser.add_argument("--input_video", type=str, default="input/object.MOV", help="输入视频路径")
    parser.add_argument("--output_dir", type=str, default="output/tmp/frames", help="抽帧输出目录")
    parser.add_argument("--target_frames", type=int, default=120, help="期望抽帧数量（会自动换算 fps）")
    parser.add_argument("--min_frames", type=int, default=100, help="最少保留帧数")
    parser.add_argument("--max_frames", type=int, default=150, help="最多保留帧数")
    parser.add_argument("--blur_threshold", type=float, default=55.0, help="去模糊阈值（拉普拉斯方差）")
    parser.add_argument("--hash_distance_threshold", type=int, default=6, help="去重复阈值（汉明距离）")
    parser.add_argument("--log_file", type=str, default="output/logs/video2img.log", help="日志文件")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = create_logger(log_path, logger_name="video2img")

    logger.info("开始处理视频: %s", args.input_video)
    stats = process_video(
        input_video=args.input_video,
        output_dir=args.output_dir,
        target_frames=args.target_frames,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        blur_threshold=args.blur_threshold,
        hash_distance_threshold=args.hash_distance_threshold,
    )
    logger.info("抽帧完成，统计信息: %s", stats)

    meta_path = Path(args.output_dir) / "frame_stats.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("统计信息已写入: %s", meta_path)


if __name__ == "__main__":
    main()
