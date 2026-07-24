// 最小TTGIR -> TinyGPU lowering pass
//
// 该checkpoint 验证正式后端层次
// TTIR -> TTGIR -> TinyGPU pass -> tinyasm/tinybin
// 当前只接受空TTGIR kernel， 并输出RET。下一阶段在此处实现TTGIR memory op

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

class LowerTTGIRToTinyGPUPass
    : public mlir::PassWrapper<LowerTTGIRToTinyGPUPass, OperationPass<ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTTGIRToTinyGPUPass)

    void runOnOperation() override {
      ModuleOp module = getOperation();
      TinyGPUEmitter emitter;
      llvm::DenseMap<Value, uint8_t> valueRegisters;
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
              if (!constant || !valueRegisters.empty()) {
                operation.emitError("TinyGPU only supports one i8 constant for "
                                    "the store-constant demo");
                signalPassFailure();
                return;
              }
              // R0 是第一个Kernel指针参数；
              // 阶段2 唯一临时值固定使用R1
              emitter.emitConstant(/*rd=*/1, *constant);
              valueRegisters[operation.getResult(0)] = 1;
              continue;
            }

            if (operation.getName().getStringRef() == "tt.store") {
              // 仅支持 “tt.store %arg0, %constant” : 无mask、两个operand,且
              // 指针必须是第一个函数参数。
              if (operation.getNumOperands() != 2) {
                operation.emitError(
                    "TinyGPU store-constant does not support masks");
                signalPassFailure();
                return;
              }

              auto pointer =
                  dyn_cast<mlir::BlockArgument>(operation.getOperand(0));
              const auto value = valueRegisters.find(operation.getOperand(1));
              if (!pointer || pointer.getArgNumber() != 0 ||
                  value == valueRegisters.end()) {
              operation.emitError("TinyGPU only supports tt.store of the i8 "
                                  "constant to the first pointer argument");
              signalPassFailure();
              return;
              }
              emitter.emitStore(/* address */ 0, value->second);
              continue;
            }

            if (isa<ReturnOp>(operation)) {
              emitter.emitReturn();
              continue;
            }
            operation.emitError(
                  "TinyGPU TTGIR checkpoint only supports an empty kernel");
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
