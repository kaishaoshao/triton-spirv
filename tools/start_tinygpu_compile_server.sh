#!/usr/bin/env bash
set -euo pipefail

# 这个项目依赖 Triton 源码仓库中已经构建好的 TinyGPU Python 环境。
TRITON_ROOT="/Volumes/wsk/code/llvm-mlir/triton/triton"
TRITON_PYTHON="$TRITON_ROOT/.triton_tinygpu/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x "$TRITON_PYTHON" ]]; then
  echo "找不到 Triton Python 环境：$TRITON_PYTHON" >&2
  echo "请先在 $TRITON_ROOT 中完成 TinyGPU Triton 环境构建。" >&2
  exit 1
fi

echo "使用 Triton 根目录：$TRITON_ROOT"
echo "使用 Python：$TRITON_PYTHON"
exec "$TRITON_PYTHON" "$SCRIPT_DIR/tinygpu_compile_server.py" "$@"
