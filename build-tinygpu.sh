SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRITON_DIR="${TRITON_DIR:-$(cd "$SCRIPT_DIR/" && pwd)}"
LLVM_SRC_DIR="${LLVM_SRC_DIR:-$TRITON_DIR/llvm-project}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-$LLVM_SRC_DIR/build-triton-macos}"
VENV_DIR="${VENV_DIR:-$TRITON_DIR/.triton_tinygpu}"
TRITON_HOME_DIR="${TRITON_HOME_DIR:-/private/tmp/tritonhome}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
LLVM_PROJECTS="${LLVM_PROJECTS:-mlir;llvm;lld}"
LLVM_TARGETS="${LLVM_TARGETS:-host;NVPTX;AMDGPU}"
RUN_TESTS="${RUN_TESTS:-0}"
RUN_VERIFY="${RUN_VERIFY:-1}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 8)}"

# 可选环境变量：
#   PYTHON_BIN=python3
#   RUN_TESTS=1
#   RUN_VERIFY=1
#   JOBS=10

log() {
  printf '\n==> %s\n' "$1"
}

die() {
  printf 'error: %s\n' "$1" >&2
  exit 1
}

if [[ ! -d "$TRITON_DIR" ]]; then
  die "Triton 仓库不存在：$TRITON_DIR"
fi

if [[ ! -d "$LLVM_SRC_DIR/llvm" ]]; then
  die "LLVM 源码不存在：$LLVM_SRC_DIR"
fi

if [[ ! -f "$TRITON_DIR/python/requirements.txt" ]]; then
  die "看起来这不是 Triton 根目录：$TRITON_DIR"
fi

log "Triton 根目录: $TRITON_DIR"
log "LLVM 源码目录: $LLVM_SRC_DIR"
log "LLVM 构建目录: $LLVM_BUILD_DIR"
log "虚拟环境目录: $VENV_DIR"

log "检查 Python 可执行文件"
python3.13 --version

log "重新创建虚拟环境"
# rm -rf "$VENV_DIR"
# python3.13  -m venv "$VENV_DIR"

log "激活虚拟环境"
# source "$VENV_DIR/bin/activate.fish"
# python --version

log "升级基础打包工具"
# python -m pip install --upgrade pip setuptools wheel

log "安装 Python 依赖"
# python -m pip install -r "$TRITON_DIR/python/requirements.txt"
# python -m pip install -r "$TRITON_DIR/python/test-requirements.txt"

# if [[ ! -x "$LLVM_BUILD_DIR/bin/llvm-config" ]]; then
#   log "配置 LLVM/MLIR"
#   cmake -G Ninja \
#     -S "$LLVM_SRC_DIR/llvm" \
#     -B "$LLVM_BUILD_DIR" \
#     -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
#     -DLLVM_ENABLE_PROJECTS="$LLVM_PROJECTS" \
#     -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS" \
#     -DLLVM_ENABLE_ASSERTIONS=ON \
#     -DMLIR_ENABLE_BINDINGS_PYTHON=ON
# fi

# log "编译 LLVM/MLIR"
# ninja -C "$LLVM_BUILD_DIR" -j "$JOBS"

log "以 editable 方式安装 Triton"
PYTHONNOUSERSITE=1 \
TRITON_OFFLINE_BUILD=1 \
TRITON_BUILD_PROTON=OFF \
TRITON_HOME="$TRITON_HOME_DIR" \
LLVM_INCLUDE_DIRS="$LLVM_BUILD_DIR/include" \
LLVM_LIBRARY_DIR="$LLVM_BUILD_DIR/lib" \
LLVM_SYSPATH="$LLVM_BUILD_DIR" \
python -m pip install -e "$TRITON_DIR" --no-build-isolation -v

log "确认 Triton 安装位置"
python -m pip show triton
python - <<'PY'
import triton
print("triton imported from:", triton.__path__[0])
PY

python third_party/tinygpu/unittest/ch1-test01.py


