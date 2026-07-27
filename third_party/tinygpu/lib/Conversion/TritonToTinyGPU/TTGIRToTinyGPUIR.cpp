#include "Dialect/TinyGPU/IR/TinyGPU.h"
#include "TritonTinyGPUToISA/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Value.h>
#include <llvm/ADT/StringRef.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Pass/Pass.h>

namespace mlir::triton::tinygpu {

namespace {

static std::optional<uint8_t> getConstant(Value value) {
  Operation *constant = value.getDefiningOp();
  if (!constant || constant->getName().getStringRef() != "arith.constant")
    return std::nullopt;
  auto attr = constant->getAttrOfType<IntegerAttr>("value");
  // TinyGPU 的立即数按 8-bit 无符号值编码，允许范围是 0 到 255。
  if (!attr || attr.getInt() < 0 || !attr.getValue().isIntN(8))
    return std::nullopt;
  return static_cast<uint8_t>(attr.getInt());
}


static std::optional<uint8_t> getTruncatedConstant(Value value) {
  Operation *trunc = value.getDefiningOp();
  if (!trunc || trunc->getName().getStringRef() != "arith.trunci")
    return std::nullopt;
  return getConstant(trunc->getOperand(0));
}

static Operation *createBase(mlir::OpBuilder &builder, Location loc) {
  OperationState state(loc, BaseOp::getOperationName());
  state.addTypes(builder.getI8Type());
  state.addAttribute("arg_index", builder.getI32IntegerAttr(0));
  return builder.create(state);
}

static Operation *createConstant(OpBuilder &builder, Location loc,
                                 uint8_t value, llvm::StringRef role) {
  OperationState state(loc, ConstOp::getOperationName());
  state.addTypes(builder.getI8Type());
  state.addAttribute("value", builder.getI8IntegerAttr(value));
  state.addAttribute("role", builder.getStringAttr(role));
  return builder.create(state);
}


static Operation *createAddPtr(OpBuilder &builder, Location loc, Value base,
                               Value offset) {
  OperationState state(loc, AddPtrOp::getOperationName());
  state.addOperands({base, offset});
  state.addTypes(builder.getI8Type());
  return builder.create(state);
}

static Operation *createStore(OpBuilder &builder, Location loc, Value address,
                              Value value) {
  OperationState state(loc, StoreOp::getOperationName());
  state.addOperands({address, value});
  return builder.create(state);
}

static Operation *createReturn(OpBuilder &builder, Location loc) {
  OperationState state(loc, ReturnOp::getOperationName());
  return builder.create(state);
}

class LowerTTGIRToTinyGPUIRPass
    : public PassWrapper<LowerTTGIRToTinyGPUIRPass, OperationPass<ModuleOp>> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTTGIRToTinyGPUIRPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module->hasAttr("ttg.num-warps")) {
      module.emitError("TinyGPU dialect lowering requires TTGIR input");
      signalPassFailure();
      return;
    }

    for (FuncOp function : module.getOps<FuncOp>())
    {
      Block &entry = function.getBody().front();
      OpBuilder entryBuilder(&entry, entry.begin());
      llvm::DenseMap<Value, Value> values;
      if (function.getNumArguments() > 1) {
        function.emitError(
            "TinyGPU stage 4 expects zero arguments for an empty kernel or "
            "one pointer argument for a store kernel");
        signalPassFailure();
        return;
      }

      Operation *base = nullptr;
      if (function.getNumArguments() == 1) {
        base = createBase(entryBuilder, function.getLoc());
        values[function.getArgument(0)] = base->getResult(0);
      }

      llvm::SmallVector<Operation *> original;
      for (Operation &operation : entry)
        if (&operation != base)
          original.push_back(&operation);

      for (Operation *operation : original) {
        OpBuilder builder(operation);
        StringRef name = operation->getName().getStringRef();

        if(name == "arith.constant")
          continue;

        if (name == "arith.trunci")
        {
          auto value = getTruncatedConstant(operation->getResult(0));
          if (!value) {
            operation->emitError("TinyGPU dialect expects an i8 constant value");
            signalPassFailure();
            return;
          }

          Operation *converted = createConstant(builder, operation->getLoc(),
                                                *value, StringRef("value"));
          values[operation->getResult(0)] = converted->getResult(0);
          continue;
        }

        if (name == "tt.addptr") {
          auto baseArgument = dyn_cast<BlockArgument>(operation->getOperand(0));
          auto offset = getConstant(operation->getOperand(1));
          if (!baseArgument || baseArgument.getArgNumber() != 0 || !offset) {
            operation->emitError(
                "TinyGPU dialect stage 4 only supports out + constant");
            signalPassFailure();
            return;
          }
          Operation *offsetValue = createConstant(
              builder, operation->getLoc(), *offset, StringRef("address"));
          Operation *address = createAddPtr(builder, operation->getLoc(),
                                            values[operation->getOperand(0)],
                                            offsetValue->getResult(0));
          values[operation->getResult(0)] = address->getResult(0);
          continue;
        }

        if (name == "tt.store") {
          if (operation->getNumOperands() != 2 ||
              !values.count(operation->getOperand(0)) ||
              !values.count(operation->getOperand(1))) {
            operation->emitError(
                "TinyGPU dialect stage 4 expects an unmasked scalar store");
            signalPassFailure();
            return;
            }
            createStore(builder, operation->getLoc(),
                        values[operation->getOperand(0)],
                        values[operation->getOperand(1)]);
            continue;
        }

        if (isa<mlir::triton::ReturnOp>(operation)) {
          createReturn(builder, operation->getLoc());
          continue;
        }

        operation->emitError("TinyGPU dialect stage 4 does not support this op");
        signalPassFailure();
        return;
      }

      // 必须逆序删除：arith.trunci 依赖 arith.constant，先删 constant
      // 会触发“operation destroyed but still has uses”。
      for (auto it = original.rbegin(); it != original.rend(); ++it)
        (*it)->erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
    createLowerTTGIRToTinyGPUIRPass() {
  return std::make_unique<LowerTTGIRToTinyGPUIRPass>();
}

} // namespace mlir::triton::tinygpu
