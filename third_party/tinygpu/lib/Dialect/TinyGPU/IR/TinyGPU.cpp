#include "Dialect/TinyGPU/IR/TinyGPU.h"

// Ops.cpp.inc 中使用 mlir::Builder、mlir::OpBuilder
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"


#define GET_DIALECT_DEFS
#include "Dialect/TinyGPU/IR/Dialect.cpp.inc"

void mlir::triton::tinygpu::TinyGPUDialect::initialize() {
  // TableGen 只生成操作类和操作列表； 方言注册仍由initialize() 完成。
  addOperations <
#define GET_OP_LIST
#include "Dialect/TinyGPU/IR/Ops.cpp.inc"
  >();;
}

#define GET_OP_CLASSES
#include "Dialect/TinyGPU/IR/Ops.cpp.inc"

