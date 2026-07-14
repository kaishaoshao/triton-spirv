#ifndef TRITON_TINYGPU_TO_ISA_TINYGPUEMITTER_H
#define TRITON_TINYGPU_TO_ISA_TINYGPUEMITTER_H

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::triton::tinygpu {

// 阶段1 ： 唯一的ISA发射器。 当前只需要RET

class TinyGPUEmitter {
public:
  void emitReturn();

  std::string assembly() const;
  std::string binaryHex() const;

private:
  std::vector<uint16_t> instructions;
};

} // namespace mlir::triton::tinygpu

#endif // TRITON_TINYGPU_TO_ISA_TINYGPUEMITTER_H
