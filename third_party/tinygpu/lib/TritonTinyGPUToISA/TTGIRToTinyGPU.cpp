// tinygpu.* 方言 -> TinyGPU ISA。
// 阶段 4 只负责消费方言，不再解析 TTGIR 的 arith/tt 操作。

#include "Dialect/TinyGPU/IR/TinyGPU.h"
#include "TritonTinyGPUToISA/Passes.h"
#include "TritonTinyGPUToISA/TinyGPUEmitter.h"
#include "mlir/IR/Diagnostics.h"

#include <llvm/ADT/DenseMap.h>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/Pass.h>

#include <triton/Dialect/Triton/IR/Dialect.h>

namespace mlir::triton::tinygpu {
namespace {

class LowerTTGIRToTinyGPUPass
    : public mlir::PassWrapper<LowerTTGIRToTinyGPUPass, OperationPass<ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTTGIRToTinyGPUPass)

    void runOnOperation() override {
      ModuleOp module = getOperation();
      TinyGPUEmitter emitter;
      llvm::DenseMap<Value, uint8_t> registers;

      for (FuncOp function : module.getOps<FuncOp>()) {
        for (Block &block : function.getBody()) {
          for (Operation &operation : block) {
            StringRef name = operation.getName().getStringRef();

            if (name == "tinygpu.base") {
              auto index = operation.getAttrOfType<IntegerAttr>("arg_index");
              if (!index || index.getInt() != 0) {
                operation.emitError("TinyGPU stage 4 only supports base argument 0");
                signalPassFailure();
                return;
              }
              registers[operation.getResult(0)] = 0;
              continue;
            }

            if (name == "tinygpu.const") {
              auto value = operation.getAttrOfType<IntegerAttr>("value");
              auto role = operation.getAttrOfType<StringAttr>("role");
              // TinyGPU 的立即数按 8-bit 无符号值编码，允许范围是 0 到 255。
              if (!value || !role || !value.getValue().isIntN(8)) {
                operation.emitError("TinyGPU const needs an 8-bit value and role");
                signalPassFailure();
                return;
              }
              // 阶段 4 只有一个地址常量和一个数据常量，暂时固定寄存器。
              uint8_t reg = role.getValue() == "address" ? 2 : 1;
              emitter.emitConstant(reg, static_cast<uint8_t>(value.getInt()));
              registers[operation.getResult(0)] = reg;
              continue;
            }

            if (name == "tinygpu.addptr") {
              auto base = registers.find(operation.getOperand(0));
              auto offset = registers.find(operation.getOperand(1));
              if (base == registers.end() || offset == registers.end()) {
                operation.emitError("TinyGPU addptr uses an unknown register value");
                signalPassFailure();
                return;
              }
              emitter.emitAdd(/*rd=*/2, base->second, offset->second);
              registers[operation.getResult(0)] = 2;
              continue;
            }

            if (name == "tinygpu.store") {
              auto address = registers.find(operation.getOperand(0));
              auto value = registers.find(operation.getOperand(1));
              if (address == registers.end() || value == registers.end()) {
                operation.emitError(
                    "TinyGPU store uses an unknown register value");
                signalPassFailure();
                return;
              }
              emitter.emitStore(address->second, value->second);
              continue;
            }

            if (name == "tinygpu.ret") {
              emitter.emitReturn();
              continue;
            }

            operation.emitError(
                "TinyGPU ISA lowering received an unknown operation");
            signalPassFailure();
            return;
          }
        }
      }

      // 产物先以module attribute传给Python 的 tinyasm/tinybin stage
      // 下一阶段会把这里替换为TinyGPU dialect 或结构化指令选择结果
      module->setAttr("tinygpu.asm",
                      StringAttr::get(module.getContext(), emitter.assembly()));
      module->setAttr("tinygpu.binary_hex",
                       StringAttr::get(module.getContext(),emitter.binaryHex()));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLowerTTGIRToTinyGPUPass() {
  return std::make_unique<LowerTTGIRToTinyGPUPass>();
}

} // namespace mlir::triton::tinygpu
