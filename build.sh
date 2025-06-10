export DEBUG=1
export LLVM_BUILD_DIR=/home/shaokai/Desktop/work/workpace/spirv/llvm-project-for-ztc/build
source ../../.venv/bin/activate
pip install -r python/requirements.txt
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
LLVM_SYSPATH=$LLVM_BUILD_DIR \
pip install -e  . --no-build-isolation