"""Triton 的最小 TinyGPU driver。

这个 driver 故意保持“不激活”状态。
这是 in-tree 后端第一版最安全的做法，因为：

- Triton 不会自动把它当成默认 runtime driver
- compile-only 验证仍然可以通过显式 target 完成
- 后面接仿真器时，不需要一开始就复制 CUDA/HIP 的运行时行为
"""

from __future__ import annotations

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


class TinyGPUDriver(DriverBase):
    """compile-only driver 桩实现。

    Triton 发现一个 backend 时，需要同时看到 compiler 类和 driver 类。
    即使我们现在不做真实 launch，也要把这个类补齐，这样后端结构才完整。
    """

    @classmethod
    def is_active(cls):
        """返回 `False`，避免 Triton 自动把它选成活动 runtime。"""

        return False

    def get_current_target(self):
        """为显式 compile-only 流程返回一个默认 TinyGPU target。"""

        return GPUTarget("tinygpu", "sim", 8)

    def get_active_torch_device(self):
        """为了满足接口而保留，虽然现在并没有真实设备。

        对 compile-only 场景来说，返回 `0` 没问题，因为这个 driver 本来就不该
        成为活动 runtime driver。
        """

        return 0

    def get_benchmarker(self):
        """返回一个假的 benchmarker。

        Triton 希望这里返回一个可调用对象，并且结果看起来像 timing 数据。
        这里纯粹是桩实现，避免因为接口不完整而报错。
        """

        def bench(*args, **kwargs):
            del args, kwargs
            return [0.0, 0.0, 0.0]

        return bench
