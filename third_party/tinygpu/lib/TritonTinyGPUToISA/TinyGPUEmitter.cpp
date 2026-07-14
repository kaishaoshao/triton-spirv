#include <iomanip>
#include <sstream>

#include "TritonTinyGPUToISA/TinyGPUEmitter.h"

namespace mlir::triton::tinygpu {


void TinyGPUEmitter::emitReturn() {
  // TinyGPU ISA: RET = 0b1111 << 12 = 0xF000
  instructions.push_back(0xF000);
}

std::string TinyGPUEmitter::assembly() const {
  std::string result;
  for (uint16_t instruction : instructions) {
    if (instruction == 0xF000)
      result += "RET\n";
  }
  return result;
}


std::string TinyGPUEmitter::binaryHex() const {
  std::ostringstream result;
  result << std::hex << std::nouppercase << std::setfill('0');
  for (uint16_t instruction : instructions)
    result << std::setw(4) << instruction;
  return result.str();
}

} // namespace mlir::triton::tinygpu

