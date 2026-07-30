"""Ch6 的最小 4-lane vector store 测试。

这个文件先固定 Ch6 的 Triton 输入：

    offsets = tl.arange(0, 4)
    tl.store(out + offsets, offsets)

每个 lane 最终应该把自己的 lane id 写入对应地址：

    lane 0: memory[0] = 0
    lane 1: memory[1] = 1
    lane 2: memory[2] = 2
    lane 3: memory[3] = 3

当前阶段仍然是 compile-only。脚本只打印 TinyGPU 汇编和二进制，不负责启动
仿真器；这样可以先分别观察 Triton lowering 和 ISA 生成是否正确。
"""

import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget


TARGET = GPUTarget("tinygpu", "sim", 4)


@triton.jit
def vector_store(out):
    # tl.arange(0, 4) 描述四个 lane 的逻辑偏移 [0, 1, 2, 3]。
    offsets = tl.arange(0, 4)
    # 向量指针和向量值配对，要求每个 lane 独立执行一次 store。
    tl.store(out + offsets, offsets)


def main():
    source = ASTSource(
        fn=vector_store,
        signature={"out": "*i8"},
        constexprs={},
    )
    kernel = triton.compile(source, target=TARGET)

    assembly = kernel.asm["tinyasm"]
    # Ch6 的关键断言：thread_id 使用 R15，地址由 R0 + R15 形成，值也来自 R15。
    # 这三个片段共同证明了“4-lane vector”已经降成“当前线程的一次标量执行”。
    for expected in ("ADD R2, R0, R15", "STR [R2], R15", "RET"):
        assert expected in assembly, (expected, assembly)

    print("===== Ch6 vector store =====")
    print(assembly, end="")
    print("tinybin:", kernel.asm["tinybin"].hex())


if __name__ == "__main__":
    main()
