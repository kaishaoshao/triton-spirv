''' TinyGPU Tritom runtime使用的仿真器桥接层
使用教程内 ``tinygpu_sim.py`` 的 TinyGPU 硬件模型。本文件只负责 Triton
tensor 与 TinyGPU 256-byte 全局内存之间的转换，以及 kernel ABI
    参数 0 -> R0
    参数 1 -> R1
    参数 2 -> R2
    lane id -> R15（由真实仿真器初始化）
当前范围刻意很小：一个 block、四个线程、最多三个连续的 CPU uint8 tensor，且
第一个参数是输出 tensor。完整调用链为 ``Triton launch -> 本 bridge ->
tools/tinygpu_sim.py -> tensor 回写``；多 block、其他 dtype、alias/view 和通用
内存分配留给后续阶段
'''

from __future__ import annotations
import sys
import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

GLOBAL_MEMORY_SIZE = 256
THREADS_PER_BLOCK = 4
MAX_POINTER_ARGUMENTS = 3

'''
一次 TinyGPU kernel launch 的可观察结果。
'''
@dataclass(frozen=True)
class SimulationResult:
    memory: tuple[int, ...]
    cycles: int

'''
从 ``tools/tinygpu_sim.py`` 加载硬件执行模型。
'''
@lru_cache(maxsize=1)
def _load_simulator() -> ModuleType:
    current = Path(__file__).resolve()
    candidates = [parent / "tools" / "tinygpu_sim.py" for parent in current.parents]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise RuntimeError(
            "TinyGPU simulator was not found. Copy tools/tinygpu_sim.py into "
            "the Triton project root"
        )

    spec = importlib.util.spec_from_file_location("triton_tinygpu_simulator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load TinyGPU simulator from {path}")
    module = importlib.util.module_from_spec(spec)
        # dataclass 需要模块先出现在 sys.modules 中，才能解析注解中的本模块类型。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

'''
支持的最小 host tensor 约束，并延迟导入 torch。
'''
def _require_tensor(argument: Any, index: int):
    try:
        import torch
    except ImportError as error: # pragma: no cover - Trion 通常已经依赖torch
        raise RuntimeError("TinyGPU runtime requires PyTorch CPU tensors") from error

    if not isinstance(argument, torch.Tensor):
        raise TypeError(
            f"TinyGPU argument {index} must be a torch.Tensor, got "
            f"{type(argument).__name__}"
        )
    if argument.device.type != "cpu":
        raise TypeError(
            f"TinyGPU argument {index} must be a CPU tensor, got {argument.device}"
        )
    if argument.dtype != torch.uint8:
        raise TypeError(
            f"TinyGPU argument {index} must have dtype torch.uint8, got "
            f"{argument.dtype}"
        )
    if not argument.is_contiguous():
        raise TypeError(
         f"TinyGPU argument {index} must be contiguous in Ch9"
    )
    return argument

'''
把tensor内容放入全局内存，并返回每个 ABI 参数的基址。
'''
def _pack_arguments(arguments: Sequence[Any]) -> tuple[list[int], list[int]]:
    if not 1 <= len(arguments) <= MAX_POINTER_ARGUMENTS:
        raise TypeError(
            "TinyGPU Ch9 supports one to three pointer tensor arguments; "
            f"received {len(arguments)}"
        )
    memory = [0] * GLOBAL_MEMORY_SIZE
    addresses: list[int] = []
    next_address = 0

    for index, argument in enumerate(arguments):
        tensor = _require_tensor(argument, index)
        values = [int(value) & 0xFF for value in tensor.reshape(-1).tolist()]
        if next_address + len(values) > GLOBAL_MEMORY_SIZE:
            raise RuntimeError(
                "TinyGPU global memory is limited to 256 bytes; tensor "
                f"arguments need {next_address + len(values)} bytes"
            )
        addresses.append(next_address)
        memory[next_address:next_address + len(values)] = values
        next_address += len(values)
    return memory, addresses

'''
将仿真器中的第一个参数内存段回写到 Triton 调用者的输出 tensor
'''
def _copy_output(output: Any, memory: Sequence[int], address: int) -> None:
    import torch

    values = memory[address:address + output.numel()]
    result = torch.tensor(values, dtype=torch.uint8, device="cpu").reshape(output.shape)
    output.copy_(result)


'''执行`tinybin`，并把第一个 tensor 参数视为输出缓冲区回写
  该函数是runtime 边界：上层 launcher 不需要理解内存布局，底层仿真器
  也不需要依赖 torch。未来增加设备内存、标量参数或多 block 时，应在这里扩展
  ABI，而不是把 Tensor 处理逻辑放入 driver 或 MLIR pass。
'''
def run_tinygpu_kernel(binary: bytes, grid: tuple[int, int, int],
                       arguments: Sequence[Any]) -> SimulationResult:
    if grid != (1, 1, 1):
        raise RuntimeError(
            "TinyGPU Ch9 currently supports exactly one block: use grid=(1, 1, 1)"
        )
    if len(binary) == 0 or len(binary) % 2 != 0:
        raise ValueError("TinyGPU binary must contain a non-empty sequence of 16-bit words")

    memory, addresses = _pack_arguments(arguments)
    simulator_module = _load_simulator()
    instructions = [
        simulator_module.Instruction(
        addr=index,
        word=int.from_bytes(binary[index * 2:index * 2 + 2], "big"),
        )
        for index in range(len(binary) // 2)
    ]
    simulator = simulator_module.TinyGPUSim(
        instructions,
        memory,
        num_blocks=1,
        threads_per_block=THREADS_PER_BLOCK,
    )

    # TinyGPUSim 已经负责初始化 R13/R14/R15。runtime 只填入 kernel ABI 的
    # 参数寄存器，因此每个线程看到相同的 out/a/b 基址和不同的 lane id。
    for thread in simulator.threads:
        for register, address in enumerate(addresses):
            thread.registers[register] = address

    max_cycles = 10_000
    while not simulator.is_done() and simulator.cycle < max_cycles:
        simulator.step()
    if not simulator.is_done():
        raise RuntimeError(f"TinyGPU simulator did not finish within {max_cycles} cycles")

    _copy_output(arguments[0], simulator.memory, addresses[0])
    return SimulationResult(memory=tuple(simulator.memory), cycles=simulator.cycle)
