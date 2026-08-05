#!/usr/bin/env python3
"""TinyGPU 的自包含 Python 仿真器。

本文件从 tiny-gpu-compiler 的仿真器迁移而来，但不依赖那个项目。它既是 Ch9
runtime 的执行核心，也可以直接作为终端工具运行。每条指令会经过
FETCH -> DECODE -> REQUEST -> WAIT -> EXECUTE -> UPDATE 六个可观察流水级，
因此 ``--verbose`` 输出也可以用来学习硬件状态如何变化。

示例：
  python tools/tinygpu_sim.py \\
    --bin 9201910a8021f000 --set 0=0 --expect 0=10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional


GLOBAL_MEMORY_SIZE = 256
SHARED_MEMORY_SIZE = 64
REGISTER_COUNT = 16


class Opcode:
    """TinyGPU 16-bit 指令字中高四位定义的 opcode。"""

    NOP = 0x0
    BRNZP = 0x1
    CMP = 0x2
    ADD = 0x3
    SUB = 0x4
    MUL = 0x5
    DIV = 0x6
    LDR = 0x7
    STR = 0x8
    CONST = 0x9
    SLDR = 0xA
    SSTR = 0xB
    BAR = 0xC
    RET = 0xF


OPCODE_NAMES = {
    Opcode.NOP: "NOP", Opcode.BRNZP: "BRnzp", Opcode.CMP: "CMP",
    Opcode.ADD: "ADD", Opcode.SUB: "SUB", Opcode.MUL: "MUL",
    Opcode.DIV: "DIV", Opcode.LDR: "LDR", Opcode.STR: "STR",
    Opcode.CONST: "CONST", Opcode.SLDR: "SLDR", Opcode.SSTR: "SSTR",
    Opcode.BAR: "BAR", Opcode.RET: "RET",
}


class PipelineStage(str, Enum):
    FETCH = "FETCH"
    DECODE = "DECODE"
    REQUEST = "REQUEST"
    WAIT = "WAIT"
    EXECUTE = "EXECUTE"
    UPDATE = "UPDATE"
    BARRIER = "BARRIER"
    DONE = "DONE"


@dataclass
class Instruction:
    """程序内存中的一个 16-bit 指令字。"""

    addr: int
    word: int


@dataclass
class DecodedInstruction:
    """一个线程在流水线中暂存的解码结果。"""

    opcode: int
    rd: int
    rs: int
    rt: int
    imm: int
    nzp_mask: int
    result: Optional[int] = None
    mem_addr: Optional[int] = None
    mem_data: Optional[int] = None
    mem_read: Optional[int] = None


@dataclass
class ThreadState:
    """每个模拟硬件线程拥有独立 PC、寄存器和条件码。"""

    thread_id: int
    block_id: int
    pc: int = 0
    registers: List[int] = field(default_factory=lambda: [0] * REGISTER_COUNT)
    nzp: int = 0b010
    stage: PipelineStage = PipelineStage.FETCH
    done: bool = False
    current_instruction: str = ""
    divergent: bool = False


class TinyGPUSim:
    """执行 TinyGPU 二进制；全局内存跨 block，共享内存每个 block 重置。"""

    def __init__(self, instructions: Iterable[Instruction], initial_memory: Iterable[int],
                 num_blocks: int = 1, threads_per_block: int = 4) -> None:
        if num_blocks <= 0 or threads_per_block <= 0:
            raise ValueError("num_blocks and threads_per_block must be positive")
        self.program = [instruction.word & 0xFFFF for instruction in instructions]
        self.memory = [0] * GLOBAL_MEMORY_SIZE
        for index, value in enumerate(initial_memory):
            if index == GLOBAL_MEMORY_SIZE:
                break
            self.memory[index] = value & 0xFF
        self.num_blocks = num_blocks
        self.threads_per_block = threads_per_block
        self.current_block = 0
        self.cycle = 0
        self.shared_memory = [0] * SHARED_MEMORY_SIZE
        self.threads: List[ThreadState] = []
        self.decoded: Dict[int, DecodedInstruction] = {}
        self.init_block(0)

    def init_block(self, block_id: int) -> None:
        """初始化新 block，并由硬件 ABI 写入 R13/R14/R15。"""

        self.current_block = block_id
        self.shared_memory = [0] * SHARED_MEMORY_SIZE
        self.decoded.clear()
        self.threads = []
        for thread_id in range(self.threads_per_block):
            registers = [0] * REGISTER_COUNT
            registers[13] = block_id & 0xFF
            registers[14] = self.threads_per_block & 0xFF
            registers[15] = thread_id & 0xFF
            self.threads.append(ThreadState(thread_id, block_id, registers=registers))

    def is_done(self) -> bool:
        return self.current_block >= self.num_blocks

    def state_snapshot(self) -> Dict[str, object]:
        """返回纯 Python 快照，供 CLI、网页对照测试和调试器使用。"""

        return {
            "cycle": self.cycle,
            "current_block": self.current_block,
            "total_blocks": self.num_blocks,
            "memory": list(self.memory),
            "shared_memory": list(self.shared_memory),
            "threads": [{
                "thread_id": thread.thread_id, "block_id": thread.block_id,
                "pc": thread.pc, "registers": list(thread.registers),
                "nzp": thread.nzp, "stage": thread.stage.value, "done": thread.done,
                "current_instruction": thread.current_instruction,
                "divergent": thread.divergent,
            } for thread in self.threads],
        }

    def step(self) -> Dict[str, object]:
        """推进所有未完成线程一个流水周期。"""

        if self.is_done():
            return self.state_snapshot()
        if all(thread.done for thread in self.threads):
            self.current_block += 1
            if not self.is_done():
                self.init_block(self.current_block)
            self.cycle += 1
            return self.state_snapshot()

        barrier_threads = [thread for thread in self.threads
                           if not thread.done and thread.stage == PipelineStage.BARRIER]
        if barrier_threads:
            active = [thread for thread in self.threads if not thread.done]
            if len(barrier_threads) == len(active):
                for thread in barrier_threads:
                    thread.stage = PipelineStage.FETCH
                    thread.pc += 1
            else:
                for thread in self.threads:
                    if not thread.done and thread.stage != PipelineStage.BARRIER:
                        self.execute_thread(thread)
            self.cycle += 1
            return self.state_snapshot()

        active_pcs = [thread.pc for thread in self.threads if not thread.done]
        majority_pc = max(set(active_pcs), key=active_pcs.count)
        for thread in self.threads:
            thread.divergent = not thread.done and thread.pc != majority_pc
            if not thread.done:
                self.execute_thread(thread)
        self.cycle += 1
        return self.state_snapshot()

    def execute_thread(self, thread: ThreadState) -> None:
        dispatch = {
            PipelineStage.FETCH: self.fetch, PipelineStage.DECODE: self.decode,
            PipelineStage.REQUEST: self.request, PipelineStage.WAIT: self.wait_stage,
            PipelineStage.EXECUTE: self.execute, PipelineStage.UPDATE: self.update,
        }
        handler = dispatch.get(thread.stage)
        if handler is not None:
            handler(thread)

    @staticmethod
    def _key(thread: ThreadState) -> int:
        return thread.thread_id + thread.block_id * 1000

    def fetch(self, thread: ThreadState) -> None:
        if thread.pc >= len(self.program):
            thread.done, thread.stage = True, PipelineStage.DONE
            return
        opcode = (self.program[thread.pc] >> 12) & 0xF
        thread.current_instruction = OPCODE_NAMES.get(opcode, f"UNK({opcode:04b})")
        thread.stage = PipelineStage.DECODE

    def decode(self, thread: ThreadState) -> None:
        word = self.program[thread.pc]
        self.decoded[self._key(thread)] = DecodedInstruction(
            opcode=(word >> 12) & 0xF, rd=(word >> 8) & 0xF,
            rs=(word >> 4) & 0xF, rt=word & 0xF, imm=word & 0xFF,
            nzp_mask=(word >> 9) & 0x7,
        )
        thread.stage = PipelineStage.REQUEST

    def request(self, thread: ThreadState) -> None:
        decoded, registers = self.decoded[self._key(thread)], thread.registers
        if decoded.opcode in (Opcode.LDR, Opcode.STR):
            decoded.mem_addr = registers[decoded.rs] & 0xFF
        if decoded.opcode == Opcode.STR:
            decoded.mem_data = registers[decoded.rt] & 0xFF
        if decoded.opcode in (Opcode.SLDR, Opcode.SSTR):
            decoded.mem_addr = registers[decoded.rs] & 0x3F
        if decoded.opcode == Opcode.SSTR:
            decoded.mem_data = registers[decoded.rt] & 0xFF
        thread.stage = PipelineStage.WAIT

    def wait_stage(self, thread: ThreadState) -> None:
        decoded = self.decoded[self._key(thread)]
        if decoded.opcode == Opcode.LDR:
            decoded.mem_read = self.memory[decoded.mem_addr or 0]
        elif decoded.opcode == Opcode.STR:
            self.memory[decoded.mem_addr or 0] = (decoded.mem_data or 0) & 0xFF
        elif decoded.opcode == Opcode.SLDR:
            decoded.mem_read = self.shared_memory[decoded.mem_addr or 0]
        elif decoded.opcode == Opcode.SSTR:
            self.shared_memory[decoded.mem_addr or 0] = (decoded.mem_data or 0) & 0xFF
        thread.stage = PipelineStage.EXECUTE

    def execute(self, thread: ThreadState) -> None:
        decoded, registers = self.decoded[self._key(thread)], thread.registers
        if decoded.opcode == Opcode.ADD:
            decoded.result = (registers[decoded.rs] + registers[decoded.rt]) & 0xFF
        elif decoded.opcode == Opcode.SUB:
            decoded.result = (registers[decoded.rs] - registers[decoded.rt]) & 0xFF
        elif decoded.opcode == Opcode.MUL:
            decoded.result = (registers[decoded.rs] * registers[decoded.rt]) & 0xFF
        elif decoded.opcode == Opcode.DIV:
            decoded.result = registers[decoded.rs] // registers[decoded.rt] if registers[decoded.rt] else 0
        elif decoded.opcode == Opcode.CMP:
            diff = registers[decoded.rs] - registers[decoded.rt]
            decoded.result = (4 if diff < 0 else 0) | (2 if diff == 0 else 0) | (1 if diff > 0 else 0)
        elif decoded.opcode == Opcode.CONST:
            decoded.result = decoded.imm
        elif decoded.opcode in (Opcode.LDR, Opcode.SLDR):
            decoded.result = decoded.mem_read or 0
        thread.stage = PipelineStage.UPDATE

    def update(self, thread: ThreadState) -> None:
        key, decoded = self._key(thread), self.decoded[self._key(thread)]
        if decoded.opcode in (Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV,
                              Opcode.CONST, Opcode.LDR, Opcode.SLDR):
            thread.registers[decoded.rd] = (decoded.result or 0) & 0xFF
            thread.pc += 1
        elif decoded.opcode == Opcode.CMP:
            thread.nzp = (decoded.result or 0) & 0x7
            thread.pc += 1
        elif decoded.opcode in (Opcode.NOP, Opcode.STR, Opcode.SSTR):
            thread.pc += 1
        elif decoded.opcode == Opcode.BAR:
            thread.stage = PipelineStage.BARRIER
            del self.decoded[key]
            return
        elif decoded.opcode == Opcode.BRNZP:
            thread.pc = decoded.imm if thread.nzp & decoded.nzp_mask else thread.pc + 1
        elif decoded.opcode == Opcode.RET:
            thread.done, thread.stage = True, PipelineStage.DONE
            del self.decoded[key]
            return
        del self.decoded[key]
        thread.stage = PipelineStage.FETCH


def instructions_from_hex(binary_hex: str) -> List[Instruction]:
    """将不含空格的 16-bit 大端十六进制串转为指令流。"""

    text = binary_hex.strip().removeprefix("0x")
    if not text or len(text) % 4 or any(char not in "0123456789abcdefABCDEF" for char in text):
        raise ValueError("--bin must be a non-empty sequence of 16-bit hexadecimal words")
    return [Instruction(index // 4, int(text[index:index + 4], 16))
            for index in range(0, len(text), 4)]


def _parse_memory_assignment(text: str) -> tuple[int, int]:
    try:
        address, value = (int(piece, 0) for piece in text.split("=", 1))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected ADDRESS=VALUE") from error
    if not 0 <= address < GLOBAL_MEMORY_SIZE or not 0 <= value <= 0xFF:
        raise argparse.ArgumentTypeError("address must be 0..255 and value must be 0..255")
    return address, value


def _parse_register_assignment(text: str) -> tuple[int, int]:
    """解析终端调试用的 R0-R12 初始值，禁止覆盖硬件 builtin。"""

    try:
        index, value = (int(piece, 0) for piece in text.split("=", 1))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected REGISTER=VALUE") from error
    if not 0 <= index <= 12 or not 0 <= value <= 0xFF:
        raise argparse.ArgumentTypeError(
            "register must be R0-R12 and value must be 0..255; R13-R15 are read-only builtins"
        )
    return index, value


def _print_snapshot(snapshot: Dict[str, object]) -> None:
    print(f"cycles = {snapshot['cycle']}")
    for thread in snapshot["threads"]:
        registers = " ".join(f"R{index}={value:02x}" for index, value in enumerate(thread["registers"]))
        print(f"thread {thread['thread_id']}: pc={thread['pc']} {thread['stage']} {registers}")


def main() -> int:
    parser = argparse.ArgumentParser(description="TinyGPU 自包含终端仿真器")
    parser.add_argument("--bin", required=True, help="连续的 16-bit 指令十六进制串，例如 9107f000")
    parser.add_argument("--set", action="append", default=[], type=_parse_memory_assignment,
                        help="初始化全局内存，格式 ADDRESS=VALUE，可重复")
    parser.add_argument("--reg", action="append", default=[], type=_parse_register_assignment,
                        help="初始化所有 lane 的 R0-R12，格式 REGISTER=VALUE，可重复")
    parser.add_argument("--expect", action="append", default=[], type=_parse_memory_assignment,
                        help="断言最终全局内存，格式 ADDRESS=VALUE，可重复")
    parser.add_argument("--threads", type=int, default=4, help="每个 block 的线程数，默认 4")
    parser.add_argument("--blocks", type=int, default=1, help="block 数，默认 1")
    parser.add_argument("--cycles", type=int, default=10000, help="最大周期数")
    parser.add_argument("--verbose", action="store_true", help="打印每个周期的寄存器状态")
    args = parser.parse_args()

    memory = [0] * GLOBAL_MEMORY_SIZE
    for address, value in args.set:
        memory[address] = value
    simulator = TinyGPUSim(instructions_from_hex(args.bin), memory, args.blocks, args.threads)
    # --reg 与 Ch9 ABI 对齐：每个 lane 看到相同的参数基址；R15 仍为各自 lane id。
    for thread in simulator.threads:
        for register, value in args.reg:
            thread.registers[register] = value
    snapshot = simulator.state_snapshot()
    while not simulator.is_done() and simulator.cycle < args.cycles:
        snapshot = simulator.step()
        if args.verbose:
            _print_snapshot(snapshot)
    if not simulator.is_done():
        raise RuntimeError(f"simulation did not finish within {args.cycles} cycles")
    for address, expected in args.expect:
        actual = simulator.memory[address]
        if actual != expected:
            print(f"FAIL memory[{address}] = {actual}, expected {expected}")
            return 2
        print(f"PASS memory[{address}] = {actual}")
    _print_snapshot(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
