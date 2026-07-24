"""阶段 2 手动验收：Triton 的 tl.store(out, 7) 生成 CONST/STR/RET。"""

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def store_constant(out):
    # 当前 ABI：第一个 i8* 参数 out 位于 TinyGPU 的 R0。
    tl.store(out, 7)


def main():
    source = triton.compiler.ASTSource(
        fn=store_constant,
        signature={"out": "*i8"},
        constexprs={},
    )
    kernel = triton.compile(source, target=GPUTarget("tinygpu", "sim", 4))

    assert kernel.asm["tinyasm"] == "CONST R1, #7\nSTR [R0], R1\nRET\n"
    assert kernel.asm["tinybin"] == bytes.fromhex("91078001f000")
    print(kernel.asm["tinyasm"], end="")
    print(kernel.asm["tinybin"].hex())

if __name__ == "__main__":
    main()
