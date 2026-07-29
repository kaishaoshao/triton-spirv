"""阶段 4 手动验收：确认 TTGIR 先经过 tinygpu.* 方言再发射 ISA。"""

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def store_with_offset(out):
    # 复用阶段 3 的输入，阶段 4 只改变中间层级，不增加新指令。
    tl.store(out + 1, 9)


def main():
    source = triton.compiler.ASTSource(
        fn=store_with_offset,
        signature={"out": "*i8"},
        constexprs={},
    )
    kernel = triton.compile(source, target=GPUTarget("tinygpu", "sim", 4))

    tinygpuir = kernel.asm["tinygpuir"]
    for operation in (
        "tinygpu.base",
        "tinygpu.const",
        "tinygpu.addptr",
        "tinygpu.store",
        "tinygpu.ret",
    ):
        assert operation in tinygpuir, (operation, tinygpuir)

    assert kernel.asm["tinyasm"] == (
        "CONST R2, #1\n"
        "ADD R2, R0, R2\n"
        "CONST R1, #9\n"
        "STR [R2], R1\n"
        "RET\n"
    )
    assert kernel.asm["tinybin"] == bytes.fromhex("9201320291098021f000")
    print(kernel.asm["ttir"])
    print(kernel.asm["ttgir"])
    print(tinygpuir)
    print(kernel.asm["tinyasm"], end="")
    print(kernel.asm["tinybin"].hex())


if __name__ == "__main__":
    main()
