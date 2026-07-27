#ifndef TRITON_TINYGPU_IR_TINYGPU_H
#define TRITON_TINYGPU_IR_TINYGPU_H

// 生成的 Ops.h.inc 会使用 MLIR 的操作、属性、Region 和 Bytecode 接口。
// 这些头文件必须在包含 TableGen 生成文件之前引入，否则生成代码会出现
// Properties、RegionRange、BytecodeOpInterface 等类型未定义的级联错误。
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Bytecode/BytecodeImplementation.h"

#define GET_DIALECT_DECLS
#include "Dialect/TinyGPU/IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "Dialect/TinyGPU/IR/Ops.h.inc"

#endif // TRITON_TINYGPU_IR_TINYGPU_H
