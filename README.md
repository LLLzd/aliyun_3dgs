# aliyun_3dgs（A10-30G 一键 3D Gaussian Splatting）

本工程面向**阿里云 A10-30G GPU + 4CPU**环境，提供从 iPhone 视频到 3D 重建结果的端到端自动流程：

- 输入：`input/object.MOV`（约 20 秒环绕小物体视频）；
- 自动流程：抽帧 -> COLMAP 位姿估计 -> 轻量 3DGS 训练 -> 模型导出 -> 多角度渲染对比；
- 输出：`output/<run_name>/` 下的模型文件（`.pt/.ply/.splat`）、至少 8 张对比图、训练日志与运行摘要。

> 说明：默认训练为 **PyTorch 可微 splat + 球谐颜色 + 多视角 + SSIM**；可选 **`--rasterizer gsplat`** 使用 CUDA 光栅化（需单独安装 `gsplat`）。增密/剪枝与显存分块用于在 A10-30G 上拉高上限。

---

## 1. 项目结构（可直接运行）

```text
aliyun_3dgs/
├── input/                      # 输入视频目录（放 object.MOV）
│   └── .gitkeep
├── output/                     # 输出目录（自动生成每次运行子目录）
│   └── .gitkeep
├── utils.py                    # 公共工具：日志、命令执行、可微渲染、显存统计等
├── sh_utils.py                 # 球谐 SH deg≤2 与相机中心工具
├── loss_utils.py               # SSIM / D-SSIM 损失
├── raster_gsplat.py            # 可选 gsplat 光栅封装
├── video2img.py                # 视频抽帧：去模糊、去重复，输出 100~150 张
├── colmap_process.py           # COLMAP 自动 SfM，导出 transforms.json + points3d.npz
├── train.py                    # 主脚本：端到端一键流程（quick/full）
├── render.py                   # 多角度原始帧 vs 重建渲染对比图
├── export_model.py             # 导出 .ply / .splat
├── setup.sh                    # 一键环境部署（含 GPU/CUDA 校验、COLMAP/FFmpeg 安装）
├── run_all.sh                  # 一键运行入口（full 或 quick）
├── requirements.txt            # Python 依赖版本锁定（Py3.10）
├── .gitignore
└── README.md
```

---

## 2. 环境要求（阿里云 A10）

- 系统：Ubuntu 20.04 或 22.04（推荐）
- GPU：NVIDIA A10 30GB
- CUDA：11.8+（由驱动支持）
- Python：3.10
- CPU：4 核可运行（默认参数已控制负载）

---

## 3. 一键部署

在项目根目录执行：

```bash
bash setup.sh
```

`setup.sh` 会自动执行：

1. 系统检测（OS、GPU、CUDA）；
2. 安装系统依赖（`ffmpeg`、`colmap`、`python3.10-venv` 等）；
3. 直接使用系统 Python 环境（不创建 `.venv`）；
4. 安装 `requirements.txt` 中除 `torch/torchvision` 外的依赖；
5. 保留镜像预装 PyTorch，仅做 Torch CUDA 可用性校验；
6. 运行环境校验（`ffmpeg` 和 `colmap` 命令检测）。

---

## 4. 输入数据准备

把视频放到：

```text
input/object.MOV
```

要求建议：

- 物体基本保持在画面中心；
- 绕物体平稳环绕 1 圈左右；
- 光照尽量稳定，避免严重动态模糊；
- 时长 15~30 秒均可（默认按 20 秒左右优化）。

---

## 5. 一键运行（端到端）

### 5.1 完整重建（推荐，20~40 分钟）

```bash
python train.py --mode full --video_path input/object.MOV --output_root output
```

或：

```bash
bash run_all.sh full
```

### 5.2 快速验证（约 2 分钟）

```bash
python train.py --mode quick --video_path input/object.MOV --output_root output
```

或：

```bash
bash run_all.sh quick
```

---

## 6. 输出说明

单次运行输出目录示例：

```text
output/full_20260515_180000/
├── logs/
│   ├── pipeline.log
│   ├── train.log
│   ├── colmap.log
│   ├── render.log
│   └── frame_stats.json
├── models/
│   ├── gaussians_final.pt
│   ├── gaussians_final.ply
│   └── gaussians_final.splat
├── renders/
│   ├── compare_01.png
│   ├── compare_02.png
│   ├── ...（至少8张）
│   └── render_meta.json
├── tmp/
│   ├── frames/                 # 抽帧结果
│   └── colmap/                 # COLMAP 中间文件
└── run_summary.json
```

---

## 7. 核心脚本用法

### 7.1 `video2img.py`

```bash
python video2img.py \
  --input_video input/object.MOV \
  --output_dir output/tmp/frames \
  --target_frames 120 \
  --min_frames 100 \
  --max_frames 150
```

常用参数：

- `--target_frames`：目标抽帧数量（自动换算 fps）；
- `--blur_threshold`：去模糊阈值（越大越严格）；
- `--hash_distance_threshold`：去重复阈值（越小越容易删重复）。

### 7.2 `colmap_process.py`

```bash
python colmap_process.py \
  --image_dir output/tmp/frames \
  --workspace_dir output/tmp/colmap \
  --matcher sequential
```

常用参数：

- `--matcher sequential|exhaustive`：视频序列推荐 `sequential`（更快更稳）；
- `--use_gpu 1`：启用 COLMAP GPU 特征提取/匹配。

### 7.3 `train.py`

```bash
python train.py --mode full --video_path input/object.MOV --output_root output
```

关键参数：

- `--mode quick|full`：快速验证 / 完整训练（full 默认更长迭代、更多高斯、多视角与 SSIM）；
- `--rasterizer pytorch|gsplat`：`pytorch` 为默认可微 splat；`gsplat` 为高质量光栅（需 `pip install gsplat`）；
- `--iters`：外层迭代步数；`--views_per_step`：每步随机视角数（增大吃显存）；
- `--render_point_chunk`：渲染时每批高斯点数（分块降峰值显存，0 为不分块）；
- `--ssim_weight` / `--use_lpips` / `--lpips_weight`：感知损失组合；
- `--densify_interval` / `--densify_from` / `--densify_grad_thresh` / `--densify_size_thresh`：增密（仅 `pytorch` 后端）；
- `--max_gaussians` / `--min_gaussians`：点数上下限与剪枝；
- `--train_h --train_w`：训练输出边长（默认 **512×512**）；当二者相等时，对原图做**以画面中心为基准的正方形裁剪**（边长 `min(宽,高)`），再缩放到该边长，不拉伸；
- `--render_h --render_w`：对比图边长（默认 512）；
- `--render_views`：对比图视角数量（至少 8）。

### 7.4 `render.py`

```bash
python render.py \
  --checkpoint output/full_xxx/models/gaussians_final.pt \
  --transforms output/full_xxx/tmp/colmap/transforms.json \
  --output_dir output/full_xxx/renders \
  --min_views 8
```

### 7.5 `export_model.py`

```bash
python export_model.py \
  --checkpoint output/full_xxx/models/gaussians_final.pt \
  --output_dir output/full_xxx/models \
  --stem_name gaussians_final
```

会导出：

- `gaussians_final.ply`：点云（带 RGBA）；
- `gaussians_final.splat`：轻量二进制 splat 数据（`SPLAT1` 头）。

---

## 8. 渲染对比图解读

每张 `compare_xx.png` 左右对比：

- 左：原始视频帧（COLMAP 对应视角）；
- 右：重建渲染图；
- 底部标注角度（正面、侧面、背面、45°斜视角、顶面、底面等）。

用途：

- 快速判断几何轮廓是否重建正确；
- 检查颜色一致性与表面细节；
- 判断是否需要补拍或延长训练。

---

## 9. A10 环境专属建议

1. **优先使用 full 模式默认参数**：在 A10-30G 上通常可稳定运行，避免直接拉满分辨率；
2. **显存监控**：`train.log` 会周期输出显存与高斯点数；
3. **OOM 处理**：调小 `--max_gaussians`（如 50000）和正方形边长，例如 `--train_h 448 --train_w 448`；
4. **速度优化**：`--matcher sequential` + 合理抽帧（100~150）通常更快；
5. **画质优化**：增加 `--iters`（如 9000~12000）并保证视频清晰。

---

## 10. 常见问题排查（FAQ）

### Q1: `colmap: command not found`

- 先执行 `bash setup.sh`；
- 确认 `which colmap` 有输出。

### Q2: Torch 显示 CUDA 不可用

- 检查 `nvidia-smi` 是否正常；
- 重装 CUDA 对应轮子：
  ```bash
  pip install --upgrade --force-reinstall torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
  ```

### Q3: 报显存不足（OOM）

- 降低参数：
  - `--max_gaussians 45000`
  - `--train_h 448 --train_w 448`（须保持正方形；若只写一边，程序会取较小边统一为正方形）
- 或先用 `--mode quick` 验证流程。

### Q4: 重建模糊 / 漂浮点多

- 拍摄时降低模糊、保证环绕连续；
- 提高抽帧质量（适当提高 `--blur_threshold`）；
- 适当增加 `--iters`，并保证光照稳定。

---

## 11. 算法说明（简要）

本工程在工程可落地优先前提下，采用了 Compact 化思路：

- 使用 COLMAP 稀疏点初始化高斯，减少冷启动不稳定；
- 训练过程中按透明度与点数上限执行剪枝，避免显存爆炸；
- 使用可微双线性 splat 渲染器，纯 PyTorch 实现，无需额外编译；
- 持续输出显存占用与高斯点数量，便于 A10 环境稳定运行。

---

## 12. 最短路径（复制即跑）

```bash
cd /path/to/aliyun_3dgs
bash setup.sh
python train.py --mode full --video_path input/object.MOV --output_root output
```

完成后查看：

- 模型：`output/<run_name>/models/`
- 对比图：`output/<run_name>/renders/`
- 日志：`output/<run_name>/logs/`
