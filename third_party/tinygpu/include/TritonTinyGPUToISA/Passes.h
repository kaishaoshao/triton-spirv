#ifndef TRITON_TINYGPU_TRANSFORMS_PASS_H
#define TRITON_TINYGPU_TRANSFORMS_PASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::triton::tinygpu {

// 将TTGIR子集将为tinygpu.*方言
std::unique_ptr<OperationPass<ModuleOp>> createLowerTTGIRToTinyGPUIRPass();

// 将tinygpu.*方言降为 TinyGP汇编/二进制matadata
std::unique_ptr<OperationPass<ModuleOp>> createLowerTTGIRToTinyGPUPass();

} // namespace mlir::triton::tinygpu

#endif // TRITON_TINYGPU_TRANSFORMS_PASS_H
