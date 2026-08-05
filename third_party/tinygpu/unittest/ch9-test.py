"""Ch9：通过 Triton kernel launch 调用 TinyGPU 仿真器。

这个测试不再手动调用 triton.compile() 后复制 tinybin。它验证完整路径：

    vector_add[(1,)](out, a, b)
        -> TinyGPUDriver
        -> TinyGPULauncher
        -> 教程内置的 TinyGPUSim
        -> out CPU tensor

Ch9 只支持一个 4-thread block、连续 CPU uint8 tensor，且第一个参数是输出。
这些限制让 runtime ABI 可以与 Ch8 的 R0/R1/R2/R15 定义一一对应。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.backends.tinygpu.driver import TinyGPUDriver
from triton.runtime.driver import driver


@triton.jit
def vector_add(out, a, b):
    # Ch8 的 4-lane 程序：每个 TinyGPU thread 负责一个数组元素。
    offsets = tl.arange(0, 4)
    lhs = tl.load(a + offsets)
    rhs = tl.load(b + offsets)
    tl.store(out + offsets, lhs * rhs)


def main():
    # 不依赖机器上是否存在 CUDA/HIP。显式设置 active driver 后，JITFunction 会
    # 从 TinyGPUDriver 取得 target，并使用 TinyGPULauncher 执行已编译的 tinybin。
    driver.set_active(TinyGPUDriver())

    # Ch9 runtime 把三个连续 tensor 复制到 256-byte TinyGPU 全局内存，并传入：
    # R0=out 基址，R1=a 基址，R2=b 基址。R15 仍由仿真器按 lane 初始化为 0..3。
    a = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
    b = torch.tensor([10, 20, 30, 40], dtype=torch.uint8)
    out = torch.zeros(4, dtype=torch.uint8)

    # 这是 Ch9 的关键验收点：调用路径从 Triton launch 直接进入仿真器，而不是
    # 从测试中读取 kernel.asm["tinybin"] 后再手工运行一个外部命令。
    vector_add[(1,)](out, a, b)

    expected = torch.tensor([10, 42, 90, 160], dtype=torch.uint8)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    print("===== Ch9 Triton runtime vector add =====")
    print("a:  ", a.tolist())
    print("b:  ", b.tolist())
    print("out:", out.tolist())


if __name__ == "__main__":
    main()
