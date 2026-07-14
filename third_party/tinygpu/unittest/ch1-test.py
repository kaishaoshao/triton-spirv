"""验证第一条原生 TinyGPU 编译链。"""

import triton
from triton.backends.compiler import GPUTarget


@triton.jit
def empty_kernel():
    # TTGIR checkpoint：先验证正式管线，下一阶段才 lower memory op。
    pass


def main():
    source = triton.compiler.ASTSource(
        fn=empty_kernel,
        signature={},
        constexprs={},
    )
    kernel = triton.compile(source, target=GPUTarget("tinygpu", "sim", 4))
    assert kernel.asm["tinyasm"] == "RET\n"
    assert kernel.asm["tinybin"] == bytes([0xF0, 0x00])
    print(kernel.asm["tinyasm"], end="")
    print(kernel.asm["tinybin"].hex())


if __name__ == "__main__":
    main()
