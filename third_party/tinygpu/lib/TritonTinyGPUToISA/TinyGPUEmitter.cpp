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

void TinyGPUEmitter::emitNop() {
  // NOP 的 4-bit opcode 为 0000， 其余字段全部为0.
  instructions.push_back({0x0000, "NOP"});
}

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

void TinyGPUEmitter::emitLoad(uint8_t rd, uint8_t address) {
  // LDR 格式：opcode=0111、rd[11:8]、地址寄存器rs[7:4]
  const uint16_t word = 0x7000 | ((rd & 0xF) << 8) | ((address & 0xF) << 4);
  instructions.push_back({word, "LDR " + reg(rd) + ", [" + reg(address) + "]"});
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

void TinyGPUEmitter::emitSub(uint8_t rd, uint8_t lhs, uint8_t rhs) {
  // SUB 格式：opcode=0100、rd[11:8]、lhs[7:4]、rhs[3:0]。
  const uint16_t word = 0x4000 | ((rd & 0xF) << 8) | ((lhs & 0xF) << 4) | (rhs & 0xF);
  instructions.push_back({word, "SUB " + reg(rd) + ", " + reg(lhs) + ", "
                          + reg(rhs)});
}

void TinyGPUEmitter::emitMul(uint8_t rd, uint8_t lhs, uint8_t rhs) {
  // MUL 格式： opcode=0101、rd[11:8]、lhs[7:4] 、rhs[3:)]
  const uint16_t word = 0x5000 | ((rd & 0xF) << 8) | ((lhs & 0xF) << 4) | (rhs & 0xF);
  instructions.push_back({word, "MUL " + reg(rd) + ", " + reg(lhs) + ", "
                          + reg(rhs)});
}

void TinyGPUEmitter::emitDiv(uint8_t rd, uint8_t lhs, uint8_t rhs) {
  // DIV 格式：opcode=0110、rd[11:8]、lhs[7:4]、rhs[3:0]。
  const uint16_t word = 0x6000 | ((rd & 0xF) << 8) | ((lhs & 0xF) << 4) | (rhs & 0xF);
  instructions.push_back({word, "DIV " + reg(rd) + ", " + reg(lhs) + ", "
                          + reg(rhs)});
}

void TinyGPUEmitter::emitCompare(uint8_t lhs, uint8_t rhs) {
  // CMP 不写惠寄存器，只根据lhs-rhs更新NZP条件标志
  const uint16_t word = 0x2000 | ((lhs & 0xF) << 4) | (rhs & 0xF);
  instructions.push_back({word, "CMP " + reg(lhs) + ", " + reg(rhs)});
}

void TinyGPUEmitter::emitBranch(uint8_t nzp, uint8_t target) {
  // BRnzp 格式：opcode=0001、nzp[11:9]、保留位、target[7:0]。
  const uint16_t word = 0x1000 | ((nzp & 0x7) << 9) | target;
  instructions.push_back({word, "BRnzp " + std::to_string(nzp) + ", #" +
                         std::to_string(target)});
}

void TinyGPUEmitter::emitJump(uint8_t target) {
  // 无条件跳转是 nzp=111 的 BRnzp
  emitBranch(0x7, target);
}

void TinyGPUEmitter::emitSharedLoad(uint8_t rd, uint8_t address) {
  // SLDR 与 LDR 的寄存器字段相同，只是 opcode 指向 shared memory。
  // SLDR 格式：opcode=1010、rd[11:8]、地址寄存器 rs[7:4]。
  const uint16_t word = 0xA000 | ((rd & 0xF) << 8) |
                        ((address & 0xF) << 4);
  instructions.push_back({word, "SLDR " + reg(rd) + ", [S+" + reg(address) + "]"});
}

void TinyGPUEmitter::emitSharedStore(uint8_t address, uint8_t value) {
  // SSTR 格式：opcode=1011、地址寄存器 rs[7:4]、值寄存器 rt[3:0]。
  const uint16_t word = 0xB000 | ((address & 0xF) << 4) | (value & 0xF);
  instructions.push_back({word, "SSTR [S+" + reg(address) + "], " + reg(value)});
}

void TinyGPUEmitter::emitBarrier() {
  // BAR 只需要 opcode = 1100，其余字段保留为0.
  instructions.push_back({0xC000, "BAR"});
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

