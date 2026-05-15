#!/usr/bin/env bash
set -euo pipefail

# 一键重建入口：
# 自动调用 train.py 完成抽帧、COLMAP、训练、导出、渲染。

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-full}"  # full / quick

if [[ ! -f "${PROJECT_ROOT}/input/object.MOV" ]]; then
  echo "[ERROR] 未找到输入视频: ${PROJECT_ROOT}/input/object.MOV"
  echo "请将 iPhone 拍摄的环绕视频重命名为 object.MOV 后放入 input 目录。"
  exit 1
fi

python "${PROJECT_ROOT}/train.py" \
  --mode "${MODE}" \
  --video_path "${PROJECT_ROOT}/input/object.MOV" \
  --output_root "${PROJECT_ROOT}/output"
