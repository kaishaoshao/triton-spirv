"""验证第一条原生 TinyGPU 编译链。"""

import triton
from triton.backends.compiler import GPUTarget


@triton.jit
def empty_kernel():
    # 第一阶段仅验证空 kernel 可通过完整 backend 流水线。
    pass

def main():
    source = triton.compiler.ASTSource(
        fn=empty_kernel,
        signature={},
        constexprs={},
    )
    kernel = triton.compile(source, target=GPUTarget("tinygpu", "sim", 8))
    assert kernel.asm["tinybin"] == bytes([0xF0, 0x00])
    print(kernel.asm["tinyasm"])
    print(kernel.asm["tinybin"].hex())


if __name__ == "__main__":
    main()
