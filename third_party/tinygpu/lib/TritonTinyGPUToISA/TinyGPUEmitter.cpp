#include <iomanip>
#include <sstream>
#include <string>

#include "TritonTinyGPUToISA/TinyGPUEmitter.h"

namespace mlir::triton::tinygpu {

namespace {
std::string reg(uint8_t value) {
  return "R" + std::to_string(value);
}
} // namespace
void TinyGPUEmitter::emitReturn() {
  // TinyGPU ISA: RET = 0b1111 << 12 = 0xF000
  instructions.push_back({0xF000, "RET"});
}

void TinyGPUEmitter::emitConstant(uint8_t rd, uint8_t immediate) {
  // CONST 格式 ： opcode=1001、rd[11:8] 、imm8[7:0]
  const uint16_t word = 0x9000 | ((rd & 0xF) << 8)  | immediate;
  instructions.push_back({
    word, "CONST " + reg(rd) + ", #" + std::to_string(immediate)
  });
}

void TinyGPUEmitter::emitStore(uint8_t address, uint8_t value) {
  // STR 格式：opcode=1000、保留位、 地址寄存器 rs[7:4] 、值寄存器rt[3:0]
  const uint16_t word = 0x8000 | ((address & 0xF) << 4) | (value & 0xF);
  instructions.push_back({
    word, "STR [" + reg(address) + "], " + reg(value)
  });
}

void TinyGPUEmitter::emitAdd(uint8_t rd, uint8_t lhs, uint8_t rhs) {
  // ADD 格式： opcode=0011、rd[11:8] 、lhs[7:4]、rhs[3:0]
  const uint16_t word =
      0x3000 | ((rd & 0xF) << 8) | ((lhs & 0xF) << 4) | ((rhs & 0xF));
  instructions.push_back({
      word, "ADD " + reg(rd) + ", " +
                reg(lhs) + ", " + reg(rhs)
      });

}

std::string TinyGPUEmitter::assembly() const {
  std::string result;
  for (const EncodedInstruction &instruction : instructions)
    result += instruction.assembly + "\n";
  return result;
}


std::string TinyGPUEmitter::binaryHex() const {
  std::ostringstream result;
  result << std::hex << std::nouppercase << std::setfill('0');
  for (const EncodedInstruction &instruction : instructions)
    result << std::setw(4) << instruction.word;
  return result.str();
}

} // namespace mlir::triton::tinygpu

