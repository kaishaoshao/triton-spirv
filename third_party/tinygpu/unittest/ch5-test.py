"""Ch5 总测试：完整标量 ISA、Triton 标量 lowering 和仿真器输入。

为了减少教程文件数量，本文件把原来分散在多个 .tinygpuir 和 Python 文件
中的测试合并在一起。测试内容没有减少，仍然覆盖：

1. 全部标量 TinyGPU opcode 的汇编和 16-bit 编码；
2. shared store/load、barrier 和 global store 的仿真器输入；
3. ADD/SUB/MUL/DIV 的仿真器输入；
4. CMP/BRnzp/NOP/RET 的仿真器输入；
5. 真正的 Triton Python 标量 load、算术和 store lowering。

TinyGPU IR 仍然通过 triton.compile() 编译。它们只是临时写入 .tinygpuir
文件，因为 Triton 的 IRSource API 需要从文件路径读取 MLIR 文本。
"""

from __future__ import annotations

import os
import tempfile
import textwrap

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


TARGET = GPUTarget("tinygpu", "sim", 4)


def compile_tinygpu_ir(ir_text: str, name: str):
    """把内嵌的 TinyGPU IR 临时保存，再从 tinygpuir stage 启动编译。"""
    content = textwrap.dedent(ir_text).strip() + "\n"
    path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tinygpuir", prefix=f"{name}-", delete=False
        ) as file:
            file.write(content)
            path = file.name
        return triton.compile(path, target=TARGET)
    finally:
        if path is not None:
            os.unlink(path)


def check_all_scalar_opcodes():
    """检查每一个标量 opcode 都能经过 TinyGPU IR -> ISA lowering。"""
    kernel = compile_tinygpu_ir(
        r"""
        module {
          tt.func public @arithmetic() {
            %c10 = "tinygpu.const"() <{role = "value", value = 10 : i8}> : () -> i8
            %c3 = "tinygpu.const"() <{role = "value", value = 3 : i8}> : () -> i8
            %add = "tinygpu.add"(%c10, %c3) : (i8, i8) -> i8
            %sub = "tinygpu.sub"(%c10, %c3) : (i8, i8) -> i8
            %mul = "tinygpu.mul"(%c10, %c3) : (i8, i8) -> i8
            %div = "tinygpu.div"(%c10, %c3) : (i8, i8) -> i8
            "tinygpu.ret"() : () -> ()
          }

          tt.func public @memory() {
            %addr = "tinygpu.const"() <{role = "address", value = 0 : i8}> : () -> i8
            %value = "tinygpu.const"() <{role = "value", value = 7 : i8}> : () -> i8
            %loaded = "tinygpu.load"(%addr) : (i8) -> i8
            "tinygpu.store"(%addr, %loaded) : (i8, i8) -> ()
            %shared = "tinygpu.shared_load"(%addr) : (i8) -> i8
            "tinygpu.shared_store"(%addr, %shared) : (i8, i8) -> ()
            "tinygpu.barrier"() : () -> ()
            "tinygpu.ret"() : () -> ()
          }

          tt.func public @control() {
            %c1 = "tinygpu.const"() <{role = "value", value = 1 : i8}> : () -> i8
            %c2 = "tinygpu.const"() <{role = "value", value = 2 : i8}> : () -> i8
            "tinygpu.cmp"(%c2, %c1) : (i8, i8) -> ()
            "tinygpu.branch"() <{nzp = 2 : i32, target = 7 : i32}> : () -> ()
          }

          tt.func public @jump() {
            "tinygpu.nop"() : () -> ()
            "tinygpu.jump"() <{target = 0 : i32}> : () -> ()
          }
        }
        """,
        "ch5-all-opcodes",
    )

    assembly = kernel.asm["tinyasm"]
    required = (
        "NOP", "CONST", "ADD", "SUB", "MUL", "DIV", "CMP", "BRnzp",
        "LDR", "STR", "SLDR", "SSTR", "BAR", "RET",
    )
    for opcode in required:
        assert opcode in assembly, (opcode, assembly)
    assert len(kernel.asm["tinybin"].hex()) == len(assembly.splitlines()) * 4

    print("===== Ch5 all scalar opcodes =====")
    print(assembly, end="")
    print("tinybin:", kernel.asm["tinybin"].hex())


def compile_simulator_inputs():
    """生成三组独立 tinybin，分别交给 TinyGPUSim 做执行验证。"""
    shared = compile_tinygpu_ir(
        r"""
        module {
          tt.func public @shared_roundtrip() {
            %addr = "tinygpu.const"() <{role = "address", value = 0 : i8}> : () -> i8
            %value = "tinygpu.const"() <{role = "value", value = 10 : i8}> : () -> i8
            "tinygpu.shared_store"(%addr, %value) : (i8, i8) -> ()
            "tinygpu.barrier"() : () -> ()
            %loaded = "tinygpu.shared_load"(%addr) : (i8) -> i8
            "tinygpu.store"(%addr, %loaded) : (i8, i8) -> ()
            "tinygpu.ret"() : () -> ()
          }
        }
        """,
        "ch5-shared",
    )

    arithmetic = compile_tinygpu_ir(
        r"""
        module {
          tt.func public @arithmetic_results() {
            %addr0 = "tinygpu.const"() <{role = "address", value = 0 : i8}> : () -> i8
            %addr1 = "tinygpu.const"() <{role = "address", value = 1 : i8}> : () -> i8
            %addr2 = "tinygpu.const"() <{role = "address", value = 2 : i8}> : () -> i8
            %addr3 = "tinygpu.const"() <{role = "address", value = 3 : i8}> : () -> i8
            %lhs = "tinygpu.const"() <{role = "value", value = 10 : i8}> : () -> i8
            %rhs = "tinygpu.const"() <{role = "value", value = 3 : i8}> : () -> i8
            %add = "tinygpu.add"(%lhs, %rhs) : (i8, i8) -> i8
            %sub = "tinygpu.sub"(%lhs, %rhs) : (i8, i8) -> i8
            %mul = "tinygpu.mul"(%lhs, %rhs) : (i8, i8) -> i8
            %div = "tinygpu.div"(%lhs, %rhs) : (i8, i8) -> i8
            "tinygpu.store"(%addr0, %add) : (i8, i8) -> ()
            "tinygpu.store"(%addr1, %sub) : (i8, i8) -> ()
            "tinygpu.store"(%addr2, %mul) : (i8, i8) -> ()
            "tinygpu.store"(%addr3, %div) : (i8, i8) -> ()
            "tinygpu.ret"() : () -> ()
          }
        }
        """,
        "ch5-arithmetic",
    )

    branch = compile_tinygpu_ir(
        r"""
        module {
          tt.func public @branch_taken() {
            %lhs = "tinygpu.const"() <{role = "value", value = 2 : i8}> : () -> i8
            %rhs = "tinygpu.const"() <{role = "value", value = 1 : i8}> : () -> i8
            "tinygpu.cmp"(%lhs, %rhs) : (i8, i8) -> ()
            "tinygpu.branch"() <{nzp = 2 : i32, target = 4 : i32}> : () -> ()
          }

          tt.func public @branch_target() {
            "tinygpu.nop"() : () -> ()
            "tinygpu.ret"() : () -> ()
          }
        }
        """,
        "ch5-branch",
    )

    print("===== Ch5 simulator inputs =====")
    print("shared tinybin:", shared.asm["tinybin"].hex())
    print("arithmetic tinybin:", arithmetic.asm["tinybin"].hex())
    print("branch tinybin:", branch.asm["tinybin"].hex())


@triton.jit
def add_in_place(ptr):
    value = tl.load(ptr)
    tl.store(ptr, value + 1)


@triton.jit
def sub_in_place(ptr):
    value = tl.load(ptr)
    tl.store(ptr, value - 1)


@triton.jit
def mul_in_place(ptr):
    value = tl.load(ptr)
    tl.store(ptr, value * 2)


@triton.jit
def div_in_place(ptr):
    value = tl.load(ptr)
    tl.store(ptr, value // 2)


def check_triton_scalar_lowering():
    """验证真正的 Triton Python 能生成四种标量算术。"""
    kernels = {
        "add": (add_in_place, "ADD"),
        "sub": (sub_in_place, "SUB"),
        "mul": (mul_in_place, "MUL"),
        "div": (div_in_place, "DIV"),
    }
    for name, (fn, opcode) in kernels.items():
        source = triton.compiler.ASTSource(
            fn=fn,
            signature={"ptr": "*i8"},
            constexprs={},
        )
        kernel = triton.compile(source, target=TARGET)
        assembly = kernel.asm["tinyasm"]
        for expected in ("LDR", opcode, "STR", "RET"):
            assert expected in assembly, (name, expected, assembly)
        print(f"PASS Triton scalar {name}: LDR/{opcode}/STR/RET")


def main():
    check_all_scalar_opcodes()
    compile_simulator_inputs()
    check_triton_scalar_lowering()


if __name__ == "__main__":
    main()
