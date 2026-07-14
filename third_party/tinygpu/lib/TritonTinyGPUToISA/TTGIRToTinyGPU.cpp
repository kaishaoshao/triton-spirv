// 最小TTGIR -> TinyGPU lowering pass
//
// 该checkpoint 验证正式后端层次
// TTIR -> TTGIR -> TinyGPU pass -> tinyasm/tinybin
// 当前只接受空TTGIR kernel， 并输出RET。下一阶段在此处实现TTGIR memory op

#include "TritonTinyGPU/Transforms/Passes.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
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
            // 当时 checkpoint 不尝试伪造 load/store的结构。只允许 return，
            // 因而空kernel可以验证 TTIR -> TTGIR -> TinyGPU的完整接线
            if (!isa<ReturnOp>(operation)) {
              operation.emitError(
                  "TinyGPU TTGIR checkpoint only supports an empty kernel");
              signalPassFailure();
              return;
            }
          }
        }
      }

      // 产物先以module attribute传给Python 的 tinyasm/tinybin stage
      // 下一阶段会把这里替换为TinyGPU dialect 或结构化指令选择结果
      module->setAttr("tinygpu.asm",
                      StringAttr::get(module.getContext(), "RET\n"));
      module->setAttr("tinygpu.binary_hex", StringAttr::get(module.getContext(),"f000"));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLowerTTGIRToTinyGPUPass() {
  return std::make_unique<LowerTTGIRToTinyGPUPass>();
}

} // namespace mlir::triton::tinygpu
