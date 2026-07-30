#include "Dialect/TinyGPU/IR/TinyGPU.h"
#include "TritonTinyGPUToISA/Passes.h"

#include <triton/Dialect/Triton/IR/Dialect.h>

#include <mlir/IR/Value.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/SmallVector.h>

namespace mlir::triton::tinygpu {

namespace {

static std::optional<uint8_t> getConstant(Value value) {
  Operation *constant = value.getDefiningOp();
  if (!constant || constant->getName().getStringRef() != "arith.constant")
    return std::nullopt;
  // 标量常量的形式：arith.constant 1 : i8。
  auto attr = constant->getAttrOfType<IntegerAttr>("value");
  // TinyGPU 的立即数按 8-bit 无符号值编码，允许范围是 0 到 255。
  if (attr) {
    // TinyGPU的立即数按8-bit无符号编码，允许范围是0-255
    if (attr.getInt() < 0 || !attr.getValue().isIntN(8))
      return std::nullopt;
    return static_cast<uint8_t>(attr.getInt());
  }

  // 向量常量的形式：arith.constant dense<1> : tensor<4xi8>。
  // 这是 Triton 在 value + 1 中实际生成的 IR：常量已经被前端广播，
  // 但 TinyGPU 每个 lane 只需要同一个标量立即数。
  auto dense = constant->getAttrOfType<DenseIntElementsAttr>("value");
  if (!dense || !dense.isSplat())
    return std::nullopt;
  APInt splatValue = dense.getSplatValue<APInt>();
  if (splatValue.isNegative() || !splatValue.isIntN(8))
    return std::nullopt;
  return static_cast<uint8_t>(splatValue.getZExtValue());
}

static std::optional<uint8_t> getTruncatedConstant(Value value) {
  Operation *trunc = value.getDefiningOp();
  if (!trunc || trunc->getName().getStringRef() != "arith.trunci")
    return std::nullopt;
  return getConstant(trunc->getOperand(0));
}

// 当前 TinyGPU lowering 只支持 4-lane 一维向量。
//
// TinyGPU IR 最终仍然使用一个标量 SSA 值表示“当前 lane 的元素”，但在
// TTGIR -> TinyGPUIR 过程中不能因此丢掉向量语义。这个检查帮助我们区分：
//
//   tensor<4x...>  -> 当前 lane 的标量值
//   scalar         -> 普通标量值
//
// 后续扩展通用 layout 时，会把这里的固定 4 替换为 layout 查询
static bool isSupportedVectorType(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && tensorType.getRank() == 1 &&
         tensorType.getDimSize(0) == 4;
}

static bool isVectorValue(Value value,
                          llvm::DenseMap<Value, bool> &vectorValues) {
  if (auto it = vectorValues.find(value); it != vectorValues.end())
    return it->second;
  return isSupportedVectorType(value.getType());
}

// 将当前唯一的 Triton 指针参数保留为 TinyGPU IR 中的参数引用。
// 多参数 ABI 会在 Ch8 单独引入，本阶段只验证 R0 这一条参数路径。
static Operation *createBase(mlir::OpBuilder &builder, Location loc) {
  OperationState state(loc, BaseOp::getOperationName());
  state.addTypes(builder.getI8Type());
  state.addAttribute("arg_index", builder.getI32IntegerAttr(0));
  return builder.create(state);
}

// 创建当前线程 lane id 的硬件值。
//
// TinyGPU 仿真器会在每个线程的 R15 中预置 threadIdx，因此这里不生成
// CONST 或其他算术指令，只在 TinyGPU IR 中留下一个有明确语义的 SSA 值。
// 下一层 ISA lowering 再把 tinygpu.thread_id 映射为固定寄存器 R15。
static Operation *createThreadId(OpBuilder &builder, Location loc) {
  OperationState state(loc, ThreadIdOp::getOperationName());
  state.addTypes(builder.getI8Type());
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

// 创建全局内存 load。
// 当前只接受无 mask 的 load，但它可以来自向量指针。由于 TinyGPU
// 采用 SPMD 执行方式，每个 lane 最终只需要一条标量 LDR 指令。
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
static std::optional<Value> getOrCreateValue(
                 Operation *user, Value input, OpBuilder &builder,
                 llvm::DenseMap<Value, Value> &values) {
  if (auto it = values.find(input); it != values.end())
    return it->second;
  auto constant = getConstant(input);
  if (!constant) {
    user->emitError(
        "TinyGPU expects a lowered scalar value or an 8-bit constant");
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
      // 记录 TTIR/TTGIR SSA 值是否代表 4-lane tensor。
      // values 只记录 TinyGPU 的标量 SSA 映射，vectorValues 保留上层语义。
      llvm::DenseMap<Value, bool> vectorValues;
      // 当前lowering 只使用一个指针参数，多参数ABI由后续实现负责
      if (function.getNumArguments() > 1) {
        function.emitError(
          "TinyGPU lowering supports only one pointer argument");
        signalPassFailure();
        return;
      }

      // BaseOp不发射机器指令，只表达唯一的kernel参数引用
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

        if (name == "arith.trunci") {
          // 向量make_range的元素类型通常先从i32截断到i8。 Ch6中
          // make_range 已经被表示成thread_id, 所以只需要传播同一个lane值
          // 不要再要求它必须是 arith.constant
          if (auto it = values.find(operation->getOperand(0));
              it != values.end()) {
            values[operation->getResult(0)] = it->second;
            vectorValues[operation->getResult(0)] =
                isVectorValue(operation->getOperand(0), vectorValues);
            continue;
          }
          auto value = getTruncatedConstant(operation->getResult(0));
          if (!value) {
            operation->emitError("TinyGPU dialect expects an i8 constant value");
            signalPassFailure();
            return;
          }

          Operation *converted = createConstant(builder, operation->getLoc(),
                                                *value, StringRef("value"));
          values[operation->getResult(0)] = converted->getResult(0);
          vectorValues[operation->getResult(0)] = false;
          continue;
        }

        if (name == "tt.make_range") {
          // Ch6 只支持 arange(0, 4)，它与 TinyGPU 的 4-lane 执行模型一一对应。
          // 每个 lane 使用自己的 threadIdx 作为该向量元素，因此不需要真的
          // 物化一个 tensor 或生成四份指令。
          auto start = operation->getAttrOfType<IntegerAttr>("start");
          auto end = operation->getAttrOfType<IntegerAttr>("end");
          if (!start || !end || start.getInt() != 0 || end.getInt() != 4 ||
              !isSupportedVectorType(operation->getResult(0).getType())) {
            operation->emitError(
                "TinyGPU only supports a tensor<4x...> make_range "
                "with start=0 and end=4");
            signalPassFailure();
            return;
          }
          Operation *threadId = createThreadId(builder, operation->getLoc());
          values[operation->getResult(0)] = threadId->getResult(0);
          vectorValues[operation->getResult(0)] = true;
          continue;
        }

        if (name == "tt.splat") {
          // tt.splat 把一个标量复制到所有lane。TinyGPU采用SPMD模型，
          // 每个线程本来就会独立执行同一条指令，所以只需要标量SSA值。
          if (operation->getNumOperands() != 1 ||
              operation->getNumResults() != 1  ||
              !isSupportedVectorType(operation->getResult(0).getType())) {
            operation->emitError(
                "TinyGPU expects a one-operand 4-lane tt.splat");
            signalPassFailure();
            return;
          }
          auto value = getOrCreateValue(operation, operation->getOperand(0),
                                        builder, values);
          if (!value) {
            signalPassFailure();
            return;
          }
          values[operation->getResult(0)] = *value;
          vectorValues[operation->getResult(0)] = true;
          continue;
        }

        if (name == "arith.addi" || name == "arith.subi" ||
            name == "arith.muli" || name == "arith.divsi" ||
            name == "arith.divui") {
          if (operation->getNumOperands() != 2 ||
              operation->getNumResults() != 1) {
            operation->emitError(
                "TinyGPU arithmetic expects two lowered scalar operands");
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

          // 当前允许两种形式：scalar + scalar，以及 4-lane vector 与
          // scalar/vector 的逐 lane 加法。结果的 tensor 形状必须与输入语义
          // 一致，不能把一个未广播的向量误当成普通标量。
          bool lhsVector =
              isVectorValue(operation->getOperand(0), vectorValues);
          bool rhsVector =
              isVectorValue(operation->getOperand(1), vectorValues);
          bool resultVector =
              isSupportedVectorType(operation->getResult(0).getType());

          if (resultVector != (lhsVector || rhsVector)) {
            operation->emitError(
                "TinyGPU arithmetic vector shape does not match operands");
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
          vectorValues[operation->getResult(0)] = resultVector;
          continue;
        }

        if (name == "tt.addptr") {
          // 标量 addptr 和向量 addptr
          // 都在这里汇合。上层向量值已经被折叠为当前 lane 的标量值，
          // 因此 TinyGPU IR 仍然只需要一个标量地址。
          if (operation->getNumOperands() != 2 ||
              operation->getNumResults() != 1) {
            operation->emitError("TinyGPU expects a binary tt.addptr");
            signalPassFailure();
            return;
          }

          auto base = getOrCreateValue(operation, operation->getOperand(0),
                                       builder, values);
          std::optional<Value> offset;
          Value offsetInput = operation->getOperand(1);

          if (auto it = values.find(offsetInput); it != values.end()) {
            // Ch6 的动态向量偏移已经是 thread_id，直接复用它。
            offset = it->second;
          } else if (auto constant = getConstant(offsetInput)) {
            // 标量常量偏移仍然必须标记为 address，保留 Ch3/Ch4 的
            // R2 优先寄存器约定，并避免把地址常量误当成普通数据常量。
            Operation *offsetValue =
                createConstant(builder, offsetInput.getDefiningOp()->getLoc(),
                               *constant, StringRef("address"));
            offset = offsetValue->getResult(0);
            values[offsetInput] = *offset;
          }

          if (!base || !offset) {
            signalPassFailure();
            return;
          }

          bool baseVector =
              isVectorValue(operation->getOperand(0), vectorValues);
          bool offsetVector =
              isVectorValue(operation->getOperand(1), vectorValues);
          bool resultVector =
              isSupportedVectorType(operation->getResult(0).getType());
          if (resultVector != (baseVector || offsetVector)) {
            operation->emitError(
                "TinyGPU addptr vector shape does not match operands");
            signalPassFailure();
            return;
          }

          Operation *address = createAddPtr(builder, operation->getLoc(),
                                            *base, *offset);
          values[operation->getResult(0)] = address->getResult(0);
          vectorValues[operation->getResult(0)] = resultVector;
          continue;
        }

        if (name == "tt.load") {
          // mask、边界检查和 cache policy 都留到后续阶段。向量
          // load 经过当前 lane 映射后，同样表现为一个标量地址操作数。
          if (operation->getNumOperands() != 1) {
            operation->emitError(
                "TinyGPU only supports an unmasked tt.load");
            signalPassFailure();
            return;
          }
          auto address = getOrCreateValue(operation, operation->getOperand(0),
                                          builder, values);
          if (!address) {
            signalPassFailure();
            return;
          }

          bool addressVector =
              isVectorValue(operation->getOperand(0), vectorValues);
          bool resultVector =
              isSupportedVectorType(operation->getResult(0).getType());
          if (addressVector != resultVector) {
            operation->emitError(
                "TinyGPU load result must match address vector shape");
            signalPassFailure();
            return;
          }

          Operation *converted =
              createLoad(builder, operation->getLoc(), *address);
          values[operation->getResult(0)] = converted->getResult(0);
          vectorValues[operation->getResult(0)] = resultVector;
          continue;
        }

        if (name == "tt.store") {
          if (operation->getNumOperands() != 2 ) {
            operation->emitError(
                "TinyGPU only supports an unmasked tt.store");
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

          bool addressVector =
              isVectorValue(operation->getOperand(0), vectorValues);
          bool valueVector =
              isVectorValue(operation->getOperand(1), vectorValues);
          if (addressVector != valueVector) {
            operation->emitError(
                "TinyGPU store address and value must have the same "
                "vector shape");
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

        operation->emitError("TinyGPU lowering does not support this op");
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
