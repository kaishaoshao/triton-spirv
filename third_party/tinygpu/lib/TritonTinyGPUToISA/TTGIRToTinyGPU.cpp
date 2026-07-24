// 最小TTGIR -> TinyGPU lowering pass
//
// 该checkpoint 验证正式后端层次
// TTIR -> TTGIR -> TinyGPU pass -> tinyasm/tinybin
// 阶段3: 支持out + 常量偏移后的单个i8 store

#include "TritonTinyGPU/Transforms/Passes.h"
#include "TritonTinyGPUToISA/TinyGPUEmitter.h"

#include <llvm/ADT/DenseMap.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Pass/Pass.h>
#include <mlir/IR/Value.h>

#include <triton/Dialect/Triton/IR/Dialect.h>

namespace mlir::triton::tinygpu {
namespace {

// 阶段2: arith.constant i32 -> arith.trunci i8 产生的值
// 保留这条精准限制， 防止把任意TTGIR常量或类型转换错误地编码为 8-bit immediate。
static std::optional<uint8_t> getTruncatedI8Constant(Value value) {
  Operation *trunc = value.getDefiningOp();
  if (!trunc || trunc->getName().getStringRef() != "arith.trunci")
    return std::nullopt;

  auto resultType = dyn_cast<IntegerType>(value.getType());
  if(!resultType || resultType.getWidth() != 8)
    return std::nullopt;

  Operation *constant = trunc->getOperand(0).getDefiningOp();
  if (!constant ||
      constant->getName().getStringRef() != "arith.constant")
    return std::nullopt;

  auto valueAttr = constant->getAttrOfType<IntegerAttr> ("value");
  if (!valueAttr || !valueAttr.getValue().isIntN(8))
    return std::nullopt;
  return static_cast<uint8_t>(valueAttr.getInt());
}

// 读取addptr偏移使用的整型常量
// 阶段3只接受8-bit非负偏移，这样可以直接映射到TinyGPU的CONST指令
static std::optional<uint8_t> getI8IntegerConstant(Value value) {
  Operation *constant = value.getDefiningOp();
  if (!constant || constant->getName().getStringRef() != "arith.constant")
    return std::nullopt;

  auto valueAttr = constant->getAttrOfType<mlir::IntegerAttr>("value");
  if (!valueAttr || valueAttr.getInt() < 0 || !valueAttr.getValue().isIntN(8))
    return std::nullopt;
  return static_cast<uint8_t>(valueAttr.getInt());
}

class LowerTTGIRToTinyGPUPass
    : public mlir::PassWrapper<LowerTTGIRToTinyGPUPass, OperationPass<ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTTGIRToTinyGPUPass)

    void runOnOperation() override {
      ModuleOp module = getOperation();
      TinyGPUEmitter emitter;
      llvm::DenseMap<Value, uint8_t> valueRegisters;
      llvm::DenseMap<Value, uint8_t> addressRegisters;

      // TTIR -> TTGIR 的通用pass 会写入 ttg.num-warps. 以此拒绝错误的把
      // 原始 TTIR送入本pass， 保证TinyGPU lowering的输入层级固定
      if (!module->hasAttr("ttg.num-warps")) {
        module.emitError("TinyGPU pass requires TTGIR input; run "
                         "convert-triton-to-tritongpu first");
        signalPassFailure();
        return;
      }

      for (FuncOp function : module.getOps<FuncOp>()) {
        for (Block &block : function.getBody()) {
          for (Operation &operation : block) {
            if (operation.getName().getStringRef() == "arith.constant")
              continue;

            if (operation.getName().getStringRef() == "arith.trunci") {
              if (operation.getNumResults() != 1) {
                operation.emitError(
                    "TinyGPU store-constant expects one trunc result");
                signalPassFailure();
                return;
              }

              const auto constant =
                  getTruncatedI8Constant(operation.getResult(0));
              if (!constant || valueRegisters.count(operation.getResult(0))) {
                operation.emitError("TinyGPU expects an i8 constant value");
                signalPassFailure();
                return;
              }
              // R0是基地址；R1开始保存待写入的标量值。
              const uint8_t valueRegister = 1 + valueRegisters.size();
              emitter.emitConstant(valueRegister, *constant);
              valueRegisters[operation.getResult(0)] = valueRegister;
              continue;
            }

            if (operation.getName().getStringRef() == "tt.addptr") {
              if (operation.getNumOperands() != 2 ||
                  operation.getNumResults() != 1) {
                operation.emitError("TinyGPU addptr expects one pointer and one offset");
                signalPassFailure();
                return;
              }

              auto base = dyn_cast<BlockArgument>(operation.getOperand(0));
              const auto offset = getI8IntegerConstant(operation.getOperand(1));
              if (!base || base.getArgNumber() != 0 || !offset) {
                operation.emitError(
                    "inyGPU stage 3 only supports out + i8 constant");
                signalPassFailure();
                return;
              }

              // 用R2保存偏移后的地址： R2 = R0 + offset
              emitter.emitConstant(/*rd=*/2, *offset);
              emitter.emitAdd(/*rd=*/2, /*lhs=*/0, /*rhs=*/2);
              addressRegisters[operation.getResult(0)] = 2;
              continue;
            }

            if (operation.getName().getStringRef() == "tt.store") {
              // 仅支持 `tt.store %pointer, %constant`：无 mask、两个 operand。
              if (operation.getNumOperands() != 2) {
                operation.emitError(
                    "TinyGPU store-constant does not support masks");
                signalPassFailure();
                return;
              }

              auto pointer =
                  dyn_cast<mlir::BlockArgument>(operation.getOperand(0));
              const auto value = valueRegisters.find(operation.getOperand(1));
              uint8_t address = 0;

              if(pointer && pointer.getArgNumber() == 0)
                address = 0;
              else if (auto addressIt =
                           addressRegisters.find(operation.getOperand(0));
                       addressIt != addressRegisters.end())
                address = addressIt->second;
              else {
                operation.emitError("TinyGPU store expects out or out + constant");
                signalPassFailure();
                return;
              }

              if (value == valueRegisters.end()) {
              operation.emitError("TinyGPU store expects an i8 constant value");
              signalPassFailure();
              return;
              }
              emitter.emitStore(address, value->second);
              continue;
            }

            if (isa<ReturnOp>(operation)) {
              emitter.emitReturn();
              continue;
            }
            operation.emitError("TinyGPU stage 3 supports constants, addptr, "
                              "store and return only");
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
