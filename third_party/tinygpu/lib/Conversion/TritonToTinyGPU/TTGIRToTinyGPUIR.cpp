#include "Dialect/TinyGPU/IR/TinyGPU.h"
#include "TritonTinyGPUToISA/Passes.h"

#include "mlir/IR/OperationSupport.h"
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

// 创建一个 TinyGPU 标量算术操作。
//
// 这里把 Triton/MLIR 的 arith.addi 等操作统一转换为 TinyGPU dialect
// 的 add/sub/mul/div。TinyGPU dialect 的结果类型固定为 i8，正好对应
// 当前 TinyGPU 标量寄存器模型。
static Operation *createBinary(OpBuilder &builder, Location loc, StringRef name,
                               Value lhs, Value rhs) {
  OperationState state(loc, name);
  state.addOperands({lhs, rhs});
  state.addTypes(builder.getI8Type());
  return builder.create(state);
}

static Operation *createAddPtr(OpBuilder &builder, Location loc, Value base,
                               Value offset) {
  OperationState state(loc, AddPtrOp::getOperationName());
  state.addOperands({base, offset});
  state.addTypes(builder.getI8Type());
  return builder.create(state);
}

static Operation *createLoad(OpBuilder &builder, Location loc, Value address) {
  OperationState state(loc, "tinygpu.load");
  state.addOperands(address);
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

// 将一个 TTIR/TTGIR SSA 值解析为 TinyGPU dialect SSA 值。
//
// 大多数值已经在 `values` 中完成映射，例如 kernel 参数、trunci 结果和
// 前面的 addptr 结果。对于尚未处理的 arith.constant，这里采用“按需
// materialize”的方式创建 tinygpu.const。这样既能支持 `x + 1`，又不会
// 为同一个 arith.constant 提前生成重复的 TinyGPU 常量。
static std::optional<Value>
getOrCreateValue(Operation *user, Value input, OpBuilder &builder,
                 llvm::DenseMap<Value, Value> &values) {
  if (auto it = values.find(input); it != values.end())
    return it->second;
  auto constant = getConstant(input);
  if (!constant) {
    user->emitError("TinyGPU Ch5 expects a scalar value or an 8-bit constant");
    return std::nullopt;
  }
  Operation *converted = createConstant(
      builder, input.getDefiningOp()->getLoc(), *constant, StringRef("value"));
  return converted->getResult(0);
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
            "TinyGPU Ch5 expects zero arguments for an empty kernel or "
            "one i8 pointer argument for a scalar kernel");
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

        // arith.constant 不直接生成 TinyGPU 操作。它会在被某个支持的
        // 标量操作使用时，由 getOrCreateValue() 按需转换。
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

        if (name == "arith.addi" || name == "arith.subi" ||
            name == "arith.muli" || name == "arith.divsi" ||
            name == "arith.divui") {
          if (operation->getNumOperands() != 2 ||
              operation->getNumResults() != 1) {
            operation->emitError(
                "TinyGPU Ch5 arithmetic expects two scalar operands");
            signalPassFailure();
            return;
          }

          auto lhs = getOrCreateValue(operation, operation->getOperand(0),
                                      builder, values);
          auto rhs = getOrCreateValue(operation, operation->getOperand(1),
                                      builder, values);

          if (!lhs || !rhs) {
            signalPassFailure();
            return;
          }

          StringRef tinygpuName = "tinygpu.add";
          if (name == "arith.subi")
            tinygpuName = "tinygpu.sub";
          else if (name == "arith.muli")
            tinygpuName = "tinygpu.mul";
          else if (name == "arith.divsi" || name == "arith.divui")
            tinygpuName = "tinygpu.div";

          Operation *converted = createBinary(builder, operation->getLoc(),
                                              tinygpuName, *lhs, *rhs);

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

        if (name == "tt.load") {
          // mask、边界检查和 cache policy 都留到后续阶段；Ch5 只接受
          // 一个标量地址操作数。
          if (operation->getNumOperands() != 1) {
            operation->emitError(
                "TinyGPU Ch5 only supports an unmasked scalar tt.load");
            signalPassFailure();
            return;
          }
          auto address = getOrCreateValue(operation, operation->getOperand(0),
                                          builder, values);
          if (!address) {
            signalPassFailure();
            return;
          }
          Operation *converted =
              createLoad(builder, operation->getLoc(), *address);
          values[operation->getResult(0)] = converted->getResult(0);
          continue;
        }

        if (name == "tt.store") {
          if (operation->getNumOperands() != 2 ) {
            operation->emitError(
                "TinyGPU Ch5 only supports an unmasked scalar tt.store");
            signalPassFailure();
            return;
          }

          auto address = getOrCreateValue(operation, operation->getOperand(0),
                                          builder, values);
          auto value = getOrCreateValue(operation, operation->getOperand(1),
                                        builder, values);
          if (!address || !value) {
            signalPassFailure();
            return;
          }
          createStore(builder, operation->getLoc(), *address, *value);
          continue;
        }

        if (isa<mlir::triton::ReturnOp>(operation)) {
          createReturn(builder, operation->getLoc());
          continue;
        }

        operation->emitError("TinyGPU Ch5 does not support this TTIR/TTGIR op");
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
