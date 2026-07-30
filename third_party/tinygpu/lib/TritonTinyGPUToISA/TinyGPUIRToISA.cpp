// tinygpu.* 方言 -> TinyGPU ISA。
// 阶段 4 只负责消费方言，不再解析 TTGIR 的 arith/tt 操作。
// 阶段 5 覆盖完整的标量指令选择；这里仍然不直接解析 Triton TTIR/TTGIR。


#include "Dialect/TinyGPU/IR/TinyGPU.h"
#include "TritonTinyGPUToISA/Passes.h"
#include "TritonTinyGPUToISA/TinyGPUEmitter.h"
#include "mlir/IR/Diagnostics.h"

#include <llvm/ADT/DenseMap.h>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/Pass.h>

#include <triton/Dialect/Triton/IR/Dialect.h>

#include <array>
#include <optional>

namespace mlir::triton::tinygpu {
namespace {

static std::optional<uint8_t> getU8Attr(Operation &operation, StringRef name) {
  auto attr = operation.getAttrOfType<IntegerAttr>(name);
  if (!attr || attr.getInt() < 0 || attr.getInt() > 255)
    return std::nullopt;
  return static_cast<uint8_t>(attr.getInt());
}

static std::optional<uint8_t>
getRegister(Operation &operation, Value value,
            llvm::DenseMap<Value, uint8_t> &registers) {
  auto it = registers.find(value);
  if (it == registers.end()) {
    operation.emitError("TinyGPU ISA lowering uses an unknown SSA value");
    return std::nullopt;
  }
  return it->second;
}

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
          std::array<bool, 16> usedRegisters{};
          usedRegisters[0] = true;
          // R15 是 TinyGPU 仿真器预置的 threadIdx，Ch6 将它作为 lane id 使用。
          // 必须从通用寄存器分配池中排除，否则普通 SSA 值可能覆盖硬件值。
          usedRegisters[15] = true;

          auto claimRegister = [&](uint8_t reg) {
            usedRegisters[reg & 0xF] = true;
          };
          auto allocateRegister = [&]() -> std::optional<uint8_t> {
            for (uint8_t reg = 1; reg < 16; ++reg) {
              if (!usedRegisters[reg]) {
                usedRegisters[reg] = true;
                return reg;
              }
            }
            return std::nullopt;
          };

          for (Operation &operation : block) {
            StringRef name = operation.getName().getStringRef();

            if (name == "tinygpu.nop") {
              emitter.emitNop();
              continue;
            }

            if (name == "tinygpu.base") {
              auto index = operation.getAttrOfType<IntegerAttr>("arg_index");
              if (!index || index.getInt() != 0) {
                operation.emitError("TinyGPU stage 4 only supports base argument 0");
                signalPassFailure();
                return;
              }
              claimRegister(0);
              registers[operation.getResult(0)] = 0;
              continue;
            }

            if (name == "tinygpu.thread_id") {
              // thread_id 不发射机器指令；它只是把语义值绑定到硬件寄存器 R15。
              claimRegister(15);
              registers[operation.getResult(0)] = 15;
              continue;
            }

            if (name == "tinygpu.const") {
              auto value = operation.getAttrOfType<IntegerAttr>("value");
              auto role = operation.getAttrOfType<StringAttr>("role");
              // TinyGPU 的立即数按 8-bit 无符号值编码，允许范围是 0 到 255。
              if (!value || !role || value.getInt() < 0
                    || !value.getValue().isIntN(8)) {
                operation.emitError("TinyGPU const needs an 8-bit value and role");
                signalPassFailure();
                return;
              }
              // 保留 Ch4 的 R1/R2 约定；同类常量重复出现时再分配新寄存器。
              uint8_t preferred = role.getValue() == "address" ? 2 : 1;
              uint8_t reg = preferred;
              if (usedRegisters[reg]) {
                auto allocated = allocateRegister();
                if (!allocated) {
                  operation.emitError("TinyGPU ISA lowering ran out of registers");
                  signalPassFailure();
                  return;
                }
                reg = *allocated;
              } else {
                claimRegister(reg);
              }
              emitter.emitConstant(reg, static_cast<uint8_t>(value.getInt()));
              registers[operation.getResult(0)] = reg;
              continue;
            }

            if (name == "tinygpu.addptr") {
              auto base = getRegister(operation, operation.getOperand(0), registers);
              auto offset = getRegister(operation, operation.getOperand(1), registers);
              if (!base || !offset) {
                signalPassFailure();
                return;
              }
              emitter.emitAdd(/*rd=*/2, *base, *offset);
              claimRegister(2);
              registers[operation.getResult(0)] = 2;
              continue;
            }

            if (name == "tinygpu.add" || name == "tinygpu.sub" ||
                name == "tinygpu.mul" || name == "tinygpu.div") {
              auto lhs =
                  getRegister(operation, operation.getOperand(0), registers);
              auto rhs =
                  getRegister(operation, operation.getOperand(0), registers);
              auto result = allocateRegister();
              if (!lhs || !rhs || !result) {
                operation.emitError("TinyGPU arithmetic lowering failed");
                signalPassFailure();
                return;
              }
              if (name == "tinygpu.add")
                emitter.emitAdd(*result, *lhs, *rhs);
              else if (name == "tinygpu.sub")
                emitter.emitSub(*result, *lhs, *rhs);
              else if (name == "tinygpu.mul")
                emitter.emitMul(*result, *lhs, *rhs);
              else
                emitter.emitDiv(*result, *lhs, *rhs);
              registers[operation.getResult(0)] = *result;
              continue;
            }

            if (name == "tinygpu.cmp") {
              auto lhs =
                  getRegister(operation, operation.getOperand(0), registers);
              auto rhs =
                  getRegister(operation, operation.getOperand(1), registers);
              if (!lhs || !rhs) {
                signalPassFailure();
                return;
              }
              emitter.emitCompare(*lhs, *rhs);
              continue;
            }

            if (name == "tinygpu.branch" || name == "tinygpu.jump") {
              auto target = getU8Attr(operation, "target");
              if (!target) {
                operation.emitError("TinyGPU branch needs an 8-bit target");
                signalPassFailure();
                return;
              }
              if (name == "tinygpu.jump") {
                emitter.emitJump(*target);
                continue;
              }
              auto nzp = getU8Attr(operation, "nzp");
              if (!nzp || *nzp > 7) {
                operation.emitError("TinyGPU branch needs an NZP mask in [0, 7]");
                signalPassFailure();
                return;
              }
              emitter.emitBranch(*nzp, *target);
              continue;
            }

            if (name == "tinygpu.load" || name == "tinygpu.shared_load") {
              // 两种 load 的 IR 形状相同：一个地址输入、一个标量结果。
              // 只有最终访问的存储空间不同，因此在 emitter 选择时分支。
              auto address =
                  getRegister(operation, operation.getOperand(0), registers);
              auto result = allocateRegister();
              if (!address || !result) {
                operation.emitError("TinyGPU load lowering failed");
                signalPassFailure();
                return;
              }
              if (name == "tinygpu.load")
                emitter.emitLoad(*result, *address);
              else
                emitter.emitSharedLoad(*result, *address);
              registers[operation.getResult(0)] = *result;
              continue;
            }

            if (name == "tinygpu.store" || name == "tinygpu.shared_store") {
              // 两种 store 的 IR 形状也相同：地址和值两个输入、没有结果。
              // 统一解析寄存器后，只在最后选择 STR 或 SSTR。
              auto address =
                  getRegister(operation, operation.getOperand(0), registers);
              auto value =
                  getRegister(operation, operation.getOperand(1), registers);

              if (!address || !value) {
                operation.emitError("TinyGPU store lowering failed");
                signalPassFailure();
                return;
              }
              if (name == "tinygpu.store")
                emitter.emitStore(*address, *value);
              else
                emitter.emitSharedStore(*address, *value);
              continue;
            }

            if (name == "tinygpu.barrier") {
              emitter.emitBarrier();
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

std::unique_ptr<OperationPass<ModuleOp>> createLowerTinyGPUIRToISAPass() {
  return std::make_unique<LowerTTGIRToTinyGPUPass>();
}

} // namespace mlir::triton::tinygpu
