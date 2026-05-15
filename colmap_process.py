#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COLMAP 自动处理脚本。

功能：
1) 自动执行特征提取 / 特征匹配 / SfM 建图；
2) 自动导出 TXT 模型；
3) 解析 cameras/images/points3D，生成训练用 transforms.json 与 points3d.npz。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from utils import create_logger, ensure_dir, run_command


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP 四元数转旋转矩阵。qvec = [qw, qx, qy, qz]。"""
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def parse_cameras_txt(path: Path) -> Dict[int, Dict[str, float]]:
    """解析 cameras.txt，返回 camera_id -> 内参。"""
    cameras: Dict[int, Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            if model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
            elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                f, cx, cy = params[:3]
                fx, fy = f, f
            elif model == "OPENCV":
                fx, fy, cx, cy = params[:4]
            else:
                raise ValueError(f"暂不支持的相机模型: {model}")

            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            }
    return cameras


def parse_images_txt(path: Path) -> List[Dict]:
    """解析 images.txt（每两行为一组，第二行是 2D-3D 对应信息）。"""
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        cam_id = int(parts[8])
        image_name = parts[9]

        items.append(
            {
                "image_id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": cam_id,
                "image_name": image_name,
            }
        )
        i += 2
    return items


def parse_points3d_txt(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """解析 points3D.txt，返回 xyz/rgb/error。"""
    xyz_list = []
    rgb_list = []
    err_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            xyz = list(map(float, parts[1:4]))
            rgb = list(map(float, parts[4:7]))
            err = float(parts[7])
            xyz_list.append(xyz)
            rgb_list.append(rgb)
            err_list.append(err)

    if not xyz_list:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return (
        np.asarray(xyz_list, dtype=np.float32),
        np.asarray(rgb_list, dtype=np.float32),
        np.asarray(err_list, dtype=np.float32),
    )


def build_transforms(
    image_dir: Path,
    cameras: Dict[int, Dict[str, float]],
    images: List[Dict],
    output_path: Path,
) -> Dict:
    """合成训练所需 transforms.json。"""
    frames: List[Dict] = []

    for item in images:
        cam = cameras[item["camera_id"]]
        r = qvec2rotmat(item["qvec"])
        t = item["tvec"].reshape(3, 1)

        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = r
        w2c[:3, 3:4] = t
        c2w = np.linalg.inv(w2c)

        cam_pos = c2w[:3, 3]
        yaw_deg = math.degrees(math.atan2(cam_pos[2], cam_pos[0]))

        frame = {
            "image_name": item["image_name"],
            "image_path": str((image_dir / item["image_name"]).resolve()),
            "w": cam["width"],
            "h": cam["height"],
            "fx": cam["fx"],
            "fy": cam["fy"],
            "cx": cam["cx"],
            "cy": cam["cy"],
            "w2c": w2c.astype(np.float32).tolist(),
            "c2w": c2w.astype(np.float32).tolist(),
            "yaw_deg": yaw_deg,
        }
        frames.append(frame)

    data = {"num_frames": len(frames), "frames": frames}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def run_colmap_pipeline(
    image_dir: str,
    workspace_dir: str,
    use_gpu: int = 1,
    matcher: str = "sequential",
    logger_name: str = "colmap",
    log_file: str | None = None,
) -> Dict[str, str]:
    """COLMAP 全流程主函数，供 train.py 调用。"""
    ws = ensure_dir(workspace_dir)
    db_path = ws / "database.db"
    sparse_dir = ensure_dir(ws / "sparse")
    txt_dir = ensure_dir(ws / "txt")
    logger = create_logger(log_file or ws / "colmap.log", logger_name=logger_name)

    image_dir_p = Path(image_dir).resolve()
    logger.info("COLMAP 输入图片目录: %s", image_dir_p)
    logger.info("COLMAP 工作目录: %s", ws)

    feature_cmd = [
        "colmap",
        "feature_extractor",
        "--database_path",
        str(db_path),
        "--image_path",
        str(image_dir_p),
        "--ImageReader.single_camera",
        "1",
        "--SiftExtraction.use_gpu",
        str(use_gpu),
    ]
    run_command(feature_cmd, logger)

    if matcher == "exhaustive":
        match_cmd = ["colmap", "exhaustive_matcher", "--database_path", str(db_path), "--SiftMatching.use_gpu", str(use_gpu)]
    else:
        # 环绕小物体视频，顺序帧匹配更快，内存更稳。
        match_cmd = ["colmap", "sequential_matcher", "--database_path", str(db_path), "--SiftMatching.use_gpu", str(use_gpu)]
    run_command(match_cmd, logger)

    mapper_cmd = [
        "colmap",
        "mapper",
        "--database_path",
        str(db_path),
        "--image_path",
        str(image_dir_p),
        "--output_path",
        str(sparse_dir),
        "--Mapper.ba_global_function_tolerance",
        "1e-6",
    ]
    run_command(mapper_cmd, logger)

    # 默认读取 sparse/0；若不存在，寻找第一个子目录。
    model_dir = sparse_dir / "0"
    if not model_dir.exists():
        subs = sorted([p for p in sparse_dir.iterdir() if p.is_dir()])
        if not subs:
            raise RuntimeError("COLMAP mapper 没有输出可用模型，可能是特征匹配失败。")
        model_dir = subs[0]

    converter_cmd = [
        "colmap",
        "model_converter",
        "--input_path",
        str(model_dir),
        "--output_path",
        str(txt_dir),
        "--output_type",
        "TXT",
    ]
    run_command(converter_cmd, logger)

    cameras = parse_cameras_txt(txt_dir / "cameras.txt")
    images = parse_images_txt(txt_dir / "images.txt")
    xyz, rgb, err = parse_points3d_txt(txt_dir / "points3D.txt")

    transforms_path = ws / "transforms.json"
    build_transforms(
        image_dir=image_dir_p,
        cameras=cameras,
        images=images,
        output_path=transforms_path,
    )

    points_path = ws / "points3d.npz"
    np.savez_compressed(points_path, xyz=xyz, rgb=rgb, error=err)

    logger.info("COLMAP 解析完成，帧数=%d，稀疏点数=%d", len(images), xyz.shape[0])
    return {
        "workspace": str(ws.resolve()),
        "database_path": str(db_path.resolve()),
        "model_dir": str(model_dir.resolve()),
        "transforms_path": str(transforms_path.resolve()),
        "points3d_path": str(points_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="自动运行 COLMAP 并导出训练所需文件")
    parser.add_argument("--image_dir", type=str, default="output/tmp/frames", help="输入图像目录")
    parser.add_argument("--workspace_dir", type=str, default="output/tmp/colmap", help="COLMAP 工作目录")
    parser.add_argument("--use_gpu", type=int, default=1, help="COLMAP 是否使用 GPU（1/0）")
    parser.add_argument("--matcher", type=str, default="sequential", choices=["sequential", "exhaustive"], help="匹配器类型")
    parser.add_argument("--log_file", type=str, default="output/logs/colmap.log", help="日志文件")
    args = parser.parse_args()

    result = run_colmap_pipeline(
        image_dir=args.image_dir,
        workspace_dir=args.workspace_dir,
        use_gpu=args.use_gpu,
        matcher=args.matcher,
        logger_name="colmap",
        log_file=args.log_file,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
