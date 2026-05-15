# aliyun_3dgs

面向 **NVIDIA A10 30GB / 4 vCPU** 等云端环境的 **3D Gaussian Splatting** 端到端流水线：从环绕物体视频到可导出模型与多视角对比渲染。默认使用 **PyTorch 可微光栅 + 球谐颜色（deg≤2）+ L1 / D-SSIM**，可选 **gsplat** 高质量 CUDA 光栅。

---

## 功能概览

| 模块 | 说明 |
|------|------|
| 抽帧 | 去模糊、感知哈希去重，输出约 100–150 张关键帧 |
| COLMAP | 自动 SfM，导出 `transforms.json` 与 `points3d.npz` |
| 训练 | 多视角随机采样、余弦 LR、可选 LPIPS、增密 / 剪枝（`pytorch` 后端） |
| 导出 | `.ply`（ASCII 点云）、`.splat`（二进制） |
| 渲染 | 至少 8 张「原图 vs 重建」对比图 |
| 断点 | 周期性 `.pt`、优化器状态、`--resume` 续训；中间存档可触发对比渲染 |
| 监控 | `train_metrics.csv` + 周期性更新的 `train_curves.png` |

---

## 环境要求

### 推荐（生产）

- **系统**：Ubuntu 20.04 / 22.04  
- **GPU**：NVIDIA A10 30GB（或其它 ≥16GB 显存的 CUDA GPU）  
- **驱动 / CUDA**：与 PyTorch cu118 轮子匹配（见 `setup.sh` / 官方说明）  
- **Python**：3.10  
- **系统工具**：FFmpeg、COLMAP（`setup.sh` 可安装）

### 本地试跑（macOS / 无 NVIDIA）

- **不要直接执行** `setup.sh`（脚本面向 Ubuntu + `nvidia-smi`）。  
- 自行安装：`brew install colmap ffmpeg`，按 [PyTorch 官网](https://pytorch.org) 安装 **macOS arm64** 的 torch。  
- 运行示例：`python train.py --mode quick --device cpu --video_path …`  
- 使用 **`--rasterizer pytorch`**；勿在 Mac 上依赖 **gsplat**（CUDA 向）。

---

## 仓库结构

```text
aliyun_3dgs/
├── train.py              # 主入口：pipeline（抽帧→COLMAP→训练→导出→渲染）
├── video2img.py          # 抽帧
├── colmap_process.py     # COLMAP 封装
├── utils.py              # 渲染、相机、日志、显存等
├── sh_utils.py           # SH deg≤2、视线相关颜色
├── loss_utils.py         # D-SSIM 等
├── raster_gsplat.py      # 可选 gsplat 光栅
├── render.py             # 独立对比渲染
├── export_model.py       # 独立导出 ply/splat
├── setup.sh              # Ubuntu + CUDA 环境一键准备
├── run_all.sh            # 调用 train.py（quick / full）
├── requirements.txt
├── input/                # 放置输入视频（如 object.MOV）
└── output/               # 每次运行生成子目录
```

---

## 快速开始

```bash
cd /path/to/aliyun_3dgs

# Ubuntu / A10：先准备环境
bash setup.sh

# 完整重建（默认 full 预设会抬高迭代、分辨率与点数上限）
python train.py --mode full --video_path input/object.MOV --output_root output

# 流程冒烟（约数分钟级，视机器而定）
python train.py --mode quick --video_path input/object.MOV --output_root output
```

或使用：

```bash
bash run_all.sh full
bash run_all.sh quick
```

---

## 运行模式：`quick` 与 `full`

| 项目 | `quick` | `full`（默认 `--mode full`） |
|------|---------|--------------------------------|
| 用途 | 验环境、通流程 | 正式重建 |
| 迭代 | 上限约 420 | ≥ 60000（可被 `--iters` 覆盖） |
| 分辨率 | 压至 256² | ≥ 512² |
| 高斯上限 | 约 2.2 万 | 预设下限 **45 万**（可被 `--max_gaussians` 再调高） |
| 增密 | 关闭 | 开启（仅 `pytorch` 光栅） |

正方形训练分辨率下，对输入图做**中心正方形裁剪**（边长 `min(宽,高)`）再缩放，**不拉伸**。

---

## 输出目录约定

```text
output/<run_name>/
├── logs/
│   ├── pipeline.log
│   ├── train.log
│   ├── colmap.log
│   ├── render.log
│   ├── frame_stats.json
│   ├── train_metrics.csv      # 按 log_interval 追加的训练指标
│   └── train_curves.png       # 与 CSV 同步刷新的曲线图（可关闭，见下文）
├── models/
│   ├── gaussians_final.pt
│   ├── gaussians_iter_*.pt    # 周期性 checkpoint（若开启）
│   ├── gaussians_latest.pt    # 最近一次中间存档
│   ├── gaussians_final.ply
│   └── gaussians_final.splat
├── renders/
│   ├── compare_*.png
│   ├── iter_*/                  # 中间 checkpoint 触发的对比图（若未 --no_render_on_checkpoint）
│   └── render_meta.json
├── tmp/
│   ├── frames/
│   └── colmap/
└── run_summary.json
```

---

## 训练与性能相关参数

### 设备与光栅

- `--device cuda|cpu`：无 CUDA 时训练会自动落到 CPU；**COLMAP 的 GPU 开关**仅在 `torch.cuda.is_available()` 时开启，避免 Mac 上误开 CUDA SIFT。  
- `--rasterizer pytorch|gsplat`：**gsplat** 需单独 `pip install gsplat` 且需 CUDA；增密 / 剪枝当前仅在 **pytorch** 路径启用。

### 显存与吞吐（A10 常见调优）

- **`render_point_chunk`**：>0 时按点数分块光栅，降低峰值显存；**0** 表示整幅一次累加。  
- **自动策略**：在 CUDA 且显存 **≥ 17GB** 时，若未加 `--no_auto_train_point_chunk`，训练前向会**自动改为不分块**（等价 chunk=0），以提高 GPU 利用率；OOM 时请加 `--no_auto_train_point_chunk` 并适当设 `--render_point_chunk`（如 32768）。  
- **`--train_full_point_chunk`**：强制训练不分块（最吃显存）。  
- **`--views_per_step`**：每优化步内随机采样的视角数；在单 backward 内会保留多视角计算图，**会抬高显存与单步耗时**，用于降低梯度方差。

### 学习率与损失

- **`--lr_hold_frac`**（默认 **0.22**）：前若干比例迭代保持**峰值学习率**，再在剩余区间做余弦衰减，减轻后段过早衰减导致的 loss 平台；`quick` 模式会置 **0**（全程标准余弦）。  
- **`--lr_min`**：余弦下端。  
- **`--ssim_weight`**：D-SSIM 与 L1 的混合权重。  
- **`--use_lpips` / `--lpips_weight`**：可选感知损失（需安装 `lpips`）。

### 点数与增密

- **`--max_gaussians` / `--min_gaussians`**：上限与剪枝保留下限；接近上限时增密空间变小，日志会提示。  
- **`--densify_*` / `--prune_*`**：增密与剪枝节奏（仅 pytorch 后端）。

### 周期性存档与续训

- **`--checkpoint_every N`**（默认 **10000**）：每 N iter 写入 `gaussians_iter_*.pt` 与 `gaussians_latest.pt`；**0** 关闭。checkpoint 内含 **`global_step`**、模型、`optimizer`、以及 **`transforms_path` / `points3d_path`** 等元数据。  
- **`--resume path/to.pt`**：跳过抽帧与 COLMAP，在同一 `run_dir` 下继续训练；**`--iters` 须大于 checkpoint 中的 `global_step`**。  
- **`--no_render_on_checkpoint`**：中间存档**不**跑对比渲染（省时间 / 显存）。

### 训练过程可视化

- **`logs/train_metrics.csv`**：在 **`--log_interval`**（默认 50）与第一步迭代时追加一行指标。  
- **`logs/train_curves.png`**：默认按与 `log_interval` 相同节奏刷新（可用 **`--plot_curves_every`** 单独指定；**0** 表示与 `log_interval` 一致）。  
- **`--no_train_curves_plot`**：仅关闭 PNG，**仍写 CSV**。

---

## 子脚本（进阶）

### `video2img.py`

抽帧与质量过滤；参数含 `--target_frames`、`--blur_threshold`、`--hash_distance_threshold` 等。

### `colmap_process.py`

`--matcher sequential|exhaustive`；`--use_gpu 0|1` 控制 COLMAP 内部 SIFT GPU（与 `train.py` 的 pipeline 逻辑独立时可手调）。

### `render.py` / `export_model.py`

对已存在的 `gaussians_final.pt`（或中间 `.pt`）单独渲染或导出；需传入对应 `transforms.json` 路径。

---

## 对比图说明

`compare_*.png`：**左**为 COLMAP 对应视角原图，**右**为模型渲染；底部为中文视角标签。用于快速检查几何、颜色与是否需要补拍或加长训练。

---

## 常见问题（FAQ）

### `colmap: command not found`

执行 `bash setup.sh`（Ubuntu），或在本机用包管理器安装 COLMAP。

### PyTorch 提示 CUDA 不可用

检查 `nvidia-smi` 与驱动；按 PyTorch 官网安装与驱动匹配的 `torch` / `torchvision` cu118 轮子。

### 训练 OOM

减小 `--max_gaussians`、降低正方形边长（如 `--train_h 448 --train_w 448`）、减小 `--views_per_step`，或加 **`--no_auto_train_point_chunk`** 并设置较大的 **`--render_point_chunk`**。

### loss 长期不降、高斯数不涨

常见原因：（1）**点数已接近 `max_gaussians`**，增密无空间；（2）**轻量 bilinear 光栅 + 分辨率** 的表达上限；（3）学习率已进入余弦尾段。可尝试：提高 **`--max_gaussians`**、调整 **`--lr_hold_frac`**、略增 **`--iters`**，或换 **`--rasterizer gsplat`**（在支持 CUDA 的机器上）。

### 旧 checkpoint 无法 `--resume`

仅带 **`global_step`** 与 **`transforms_path` / `points3d_path`** 的新版中间存档支持 pipeline 级续训；仅有 `gaussians_final.pt` 的旧文件请用于渲染 / 导出，勿依赖其作为进度续训。

---

## 算法与实现要点（简要）

- COLMAP 稀疏点初始化高斯，降低冷启动不稳定。  
- 球谐 deg≤2 视线相关颜色，多视角颜色更一致。  
- 透明度与点数上限驱动的剪枝；`pytorch` 路径下简易 clone / split 增密。  
- 训练日志中周期性输出显存与高斯数，便于在固定规格 GPU 上稳定排障。

---

## 最短命令备忘

```bash
bash setup.sh
python train.py --mode full --video_path input/object.MOV --output_root output
```

完成后查看 **`output/<run_name>/models/`**、**`renders/`**、**`logs/train_curves.png`** 与 **`run_summary.json`**。
