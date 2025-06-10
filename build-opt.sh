mkdir -p build-opt && cd build-opt
export LLVM_BUILD_DIR=/home/shaokai/Desktop/work/workpace/spirv/llvm-project-for-ztc/build
cmake -G Ninja  ..                             \
-DLLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include    \
-DLLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib         \
-DTRITON_CODEGEN_BACKENDS="nvidia;amd "        \
-DCMAKE_BUILD_TYPE=Debug                       \
-DBUILD_SPIRV_OPT=ON                           \
-DTEST_SPIRV_CC=ON  && ninja -j8