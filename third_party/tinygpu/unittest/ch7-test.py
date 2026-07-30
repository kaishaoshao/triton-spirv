"""Ch7：单指针 vector load/add/store 编译测试。

本阶段故意不引入多个 kernel 参数，也不引入正式 launch ABI。测试程序只有
一个指针参数，在四个 lane 上执行：

    offsets = [0, 1, 2, 3]
    value = load(ptr + offsets)
    store(ptr + offsets, value + 1)

Ch7 的新增能力只有数据流：

    vector addptr -> vector load -> vector add -> vector store

每个向量操作在 TinyGPU 中都被解释为“当前 lane 的一次标量操作”。
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


TARGET = GPUTarget("tinygpu", "sim", 4)


@triton.jit
def vector_load_add_store(ptr):
    # Ch6 已经把 arange(0, 4) 映射为当前 lane 的 thread_id。
    offsets = tl.arange(0, 4)

    # Ch7 新增：每个 lane 从自己的地址读取一个 i8 标量。
    value = tl.load(ptr + offsets)

    # 常量 1 会通过 tt.splat 复制到每个 lane，arith.addi 仍然逐 lane 执行。
    tl.store(ptr + offsets, value + 1)


def main():
    source = ASTSource(
        fn=vector_load_add_store,
        # Ch7 只有一个指针参数，因此继续使用 Ch6 的 R0 参数约定。
        signature={"ptr": "*i8"},
        constexprs={},
    )
    kernel = triton.compile(source, target=TARGET)

    assembly = kernel.asm["tinyasm"]
    lines = assembly.splitlines()

    # 地址计算、load、常量加法和 store 都必须存在。
    assert "ADD R2, R0, R15" in assembly, assembly
    assert "LDR" in assembly, assembly
    assert "CONST" in assembly, assembly
    assert "ADD" in assembly, assembly
    assert "STR [R2]" in assembly, assembly
    assert lines[-1] == "RET", assembly

    # 当前 TinyGPU emitter 每条指令编码为一个 16-bit word。
    assert len(kernel.asm["tinybin"].hex()) == len(lines) * 4

    print("===== Ch7 vector load/add/store =====")
    print(assembly, end="")
    print("tinybin:", kernel.asm["tinybin"].hex())


if __name__ == "__main__":
    main()
