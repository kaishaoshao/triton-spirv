#ifndef TRITON_TINYGPU_TRANSFORMS_PASS_H
#define TRITON_TINYGPU_TRANSFORMS_PASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::triton::tinygpu {

// 将受支持的TTGIR子集降为TinyGPU asm / bin metadata
std::unique_ptr<OperationPass<ModuleOp>> createLowerTTGIRToTinyGPUPass();
} // namespace mlir::triton::tinygpu

#endif // TRITON_TINYGPU_TRANSFORMS_PASS_H
