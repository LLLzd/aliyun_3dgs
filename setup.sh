#!/usr/bin/env bash
set -euo pipefail

# ============================================
# 阿里云 A10-30G + 4CPU 环境一键部署脚本
# - 安装系统依赖（COLMAP / FFmpeg 等）
# - 创建 Python venv
# - 安装 CUDA 11.8 对应 PyTorch
# - 安装项目依赖 requirements.txt
# ============================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
PIP_BIN=""

red()   { echo -e "\033[31m[ERROR]\033[0m $*"; }
green() { echo -e "\033[32m[INFO]\033[0m $*"; }
yellow(){ echo -e "\033[33m[WARN]\033[0m $*"; }

trap 'red "脚本执行失败，请查看上方日志定位问题。"' ERR

check_cmd() {
  command -v "$1" >/dev/null 2>&1
}

check_os() {
  if [[ ! -f /etc/os-release ]]; then
    red "无法识别系统类型（缺少 /etc/os-release），请使用 Ubuntu 20.04/22.04。"
    exit 1
  fi
  source /etc/os-release
  green "检测到系统: ${NAME} ${VERSION}"
}

check_gpu() {
  if ! check_cmd nvidia-smi; then
    red "未检测到 nvidia-smi，请先安装 NVIDIA 驱动。"
    exit 1
  fi
  green "GPU 信息："
  nvidia-smi || true

  local gpu_name
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)"
  if [[ -z "${gpu_name}" ]]; then
    red "无法读取 GPU 名称，请检查驱动状态。"
    exit 1
  fi
  green "检测到 GPU: ${gpu_name}"
}

check_cuda() {
  local cuda_version="unknown"
  if check_cmd nvcc; then
    cuda_version="$(nvcc --version | awk '/release/ {print $6}' | sed 's/,//')"
    green "检测到 NVCC CUDA 版本: ${cuda_version}"
  else
    yellow "未找到 nvcc，将仅依赖驱动运行 PyTorch CUDA。"
  fi

  # driver 侧 CUDA 能力检查
  local smi_cuda
  smi_cuda="$(nvidia-smi | awk '/CUDA Version/ {print $9}' | head -n 1 || true)"
  if [[ -z "${smi_cuda}" ]]; then
    yellow "无法从 nvidia-smi 获取 CUDA Version，请确认驱动是否正常。"
  else
    green "驱动声明 CUDA 版本: ${smi_cuda}"
  fi
}

install_system_deps() {
  green "安装系统依赖（需要 sudo 权限）..."
  sudo apt-get update
  sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ffmpeg \
    colmap \
    libgl1 \
    libglib2.0-0 \
    python3.10 \
    python3.10-venv \
    python3-pip
}

setup_venv() {
  if ! check_cmd "${PYTHON_BIN}"; then
    red "未找到 ${PYTHON_BIN}，请先安装 Python 3.10。"
    exit 1
  fi

  if [[ ! -d "${VENV_DIR}" ]]; then
    green "创建虚拟环境: ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  else
    green "复用已有虚拟环境: ${VENV_DIR}"
  fi

  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  PIP_BIN="${VENV_DIR}/bin/pip"
  python --version
  "${PIP_BIN}" install --upgrade pip setuptools wheel
}

install_python_deps() {
  green "安装 Python 依赖..."
  "${PIP_BIN}" install -r "${PROJECT_ROOT}/requirements.txt"

  green "安装 CUDA 11.8 版本 PyTorch（A10 推荐）..."
  "${PIP_BIN}" install --upgrade --force-reinstall \
    torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu118
}

verify_runtime() {
  green "执行运行时校验..."
  python - <<'PY'
import shutil
import torch

print("Torch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU 显存(GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

for cmd in ("ffmpeg", "colmap"):
    path = shutil.which(cmd)
    print(f"{cmd} 路径:", path)
    if path is None:
        raise SystemExit(f"缺少命令: {cmd}")
PY
}

print_next_steps() {
  green "环境部署完成。"
  cat <<EOF
后续使用方式：
1) 激活虚拟环境
   source "${VENV_DIR}/bin/activate"

2) 将视频放到:
   ${PROJECT_ROOT}/input/object.MOV

3) 一键运行完整重建:
   python "${PROJECT_ROOT}/train.py" --mode full --video_path "${PROJECT_ROOT}/input/object.MOV" --output_root "${PROJECT_ROOT}/output"

4) 快速验证（约 2 分钟）:
   python "${PROJECT_ROOT}/train.py" --mode quick --video_path "${PROJECT_ROOT}/input/object.MOV" --output_root "${PROJECT_ROOT}/output"
EOF
}

main() {
  check_os
  check_gpu
  check_cuda
  install_system_deps
  setup_venv
  install_python_deps
  verify_runtime
  print_next_steps
}

main "$@"
