"""Ch8：三个指针参数的 4-lane vector add 编译测试。

本阶段验证一个完整的多参数数据流：

    out = a + b

    offsets = [0, 1, 2, 3]
    lhs = load(a + offsets)
    rhs = load(b + offsets)
    store(out + offsets, lhs + rhs)

这里仍然是 compile-only 测试。脚本检查的是：

1. Triton 前端能够生成三个指针参数的 TTIR/TTGIR；
2. TTGIR lowering 为三个带 arg_index 的 tinygpu.base；
3. ISA lowering 将参数绑定到 R0/R1/R2；
4. 临时地址和数据值从 R3 开始分配，并且不会覆盖参数或 R15 lane id。

真正给 out、a、b 分配独立模拟器内存，并执行结果回读，放到下一阶段。
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


TARGET = GPUTarget("tinygpu", "sim", 4)

@triton.jit
def vector_add(out, a, b):
    # 四个 lane 分别处理偏移 0、1、2、3。
    offsets = tl.arange(0, 4)

    # 三个指针参数会在 TinyGPU ABI 中分别映射到 R0、R1、R2。
    lhs = tl.load(a + offsets)
    rhs = tl.load(b + offsets)
    tl.store(out + offsets, lhs + rhs)


def main():
    source = ASTSource(
        fn=vector_add,
        # ASTSource 的 signature 让 Triton 前端知道三个参数都是 i8 指针。
        signature={"out": "*i8", "a": "*i8", "b": "*i8"},
        constexprs={},
    )
    kernel = triton.compile(source, target=TARGET)

    assembly = kernel.asm["tinyasm"]
    lines = assembly.splitlines()

    # 参数寄存器固定后，三个 addptr 结果从 R3 开始分配。
    expected = (
        "ADD R3, R1, R15",  # a + lane
        "LDR R4, [R3]",
        "ADD R5, R2, R15",  # b + lane
        "LDR R6, [R5]",
        "ADD R7, R0, R15",  # out + lane
        "ADD R8, R4, R6",
        "STR [R7], R8",
        "RET",
    )
    for instruction in expected:
        assert instruction in assembly, (instruction, assembly)

    # 三条地址 ADD、一条数据 ADD、两条 LDR 和一条 STR 都必须存在。
    assert assembly.count("ADD ") == 4, assembly
    assert assembly.count("LDR ") == 2, assembly
    assert assembly.count("STR ") == 1, assembly
    assert lines[-1] == "RET", assembly

    # 当前 TinyGPU emitter 每条指令编码为一个 16-bit word。
    assert len(kernel.asm["tinybin"].hex()) == len(lines) * 4

    print("===== Ch8 three-argument vector add =====")
    print(assembly, end="")
    print("tinybin:", kernel.asm["tinybin"].hex())


if __name__ == "__main__":
    main()
