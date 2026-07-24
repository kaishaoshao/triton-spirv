#ifndef TRITON_TINYGPU_TO_ISA_TINYGPUEMITTER_H
#define TRITON_TINYGPU_TO_ISA_TINYGPUEMITTER_H

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::triton::tinygpu {

// 阶段1 ： 唯一的ISA发射器。 当前只需要RET
// 阶段2 :  只实现 store-constant demo 实际需要的三条指令
//          不提前声明尚不能由Triton 触发ADD/LDR/MUL等能力
class TinyGPUEmitter {
public:
  void emitReturn();
  void emitConstant(uint8_t rd, uint8_t immediate);
  void emitStore(uint8_t address, uint8_t value);
  void emitAdd(uint8_t rd, uint8_t lhs, uint8_t rhs);


  

  std::string assembly() const;
  std::string binaryHex() const;

private:
  struct EncodedInstruction {
    uint16_t word;
    std::string assembly;
  };

  std::vector<EncodedInstruction> instructions;
};

} // namespace mlir::triton::tinygpu

#endif // TRITON_TINYGPU_TO_ISA_TINYGPUEMITTER_H
