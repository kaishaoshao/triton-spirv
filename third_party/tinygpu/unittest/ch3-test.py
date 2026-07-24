"""阶段 3 手动验收：tl.store(out + 1, 9) 生成地址计算和 store。"""

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def store_with_offset(out):
    # out 是 R0；这里验证字节指针的常量偏移，而不是向量寻址。
    tl.store(out + 1, 9)


def main():
    source = triton.compiler.ASTSource(
        fn=store_with_offset,
        signature={"out": "*i8"},
        constexprs={},
    )
    kernel = triton.compile(source, target=GPUTarget("tinygpu", "sim", 4))

    expected_asm = (
        "CONST R2, #1\n"
        "ADD R2, R0, R2\n"
        "CONST R1, #9\n"
        "STR [R2], R1\n"
        "RET\n"
    )
    expected_binary = bytes.fromhex("9201320291098021f000")
    assert kernel.asm["tinyasm"] == expected_asm
    assert kernel.asm["tinybin"] == expected_binary
    print(kernel.asm["tinyasm"], end="")
    print(kernel.asm["tinybin"].hex())


if __name__ == "__main__":
    main()
