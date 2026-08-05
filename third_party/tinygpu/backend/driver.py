"""将 Triton kernel launch 接到 ``tools/tinygpu_sim.py`` 的执行模型。"""

from __future__ import annotations
import os

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase

from .simulator import THREADS_PER_BLOCK, run_tinygpu_kernel

class TinyGPUUtils:
    """满足 CompiledKernel 初始化所需的最小 device-utils 接口。"""
    @staticmethod
    def get_device_properties(device):
        del device
        # 仿真器没有正式的硬件资源限制：shared memory的正式资源留给后续教程
        return {"max_shared_mem": 64}

    @staticmethod
    def load_binary(name, binary, shared, device):
        # 把 tinybin 直接作为 CompiledKernel.function 返回给 launcher
        del name, shared, device
        # CompiledKernel 只要求 module 为非 None 来表示初始化完成。function 保存
        # 原始 tinybin，TinyGPULauncher 收到后直接传给仿真器桥接层。
        return object(), bytes(binary), 0, 0, THREADS_PER_BLOCK


class TinyGPULauncher:
    """Triton 的统一 launcher 调用约定到 TinyGPU runtime 的适配器。"""

    def __init__(self, src, metadata):
        del src, metadata

    def __call__(self, grid_x, grid_y, grid_z, stream, function, packed_metadata,
                 launch_metadata, launch_enter_hook, launch_exit_hook, *arguments):
        del stream, packed_metadata
        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)

        run_tinygpu_kernel(function, (grid_x, grid_y, grid_z), arguments)

        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)


class TinyGPUDriver(DriverBase):
    """compile-only driver 桩实现。

    Triton 发现一个 backend 时，需要同时看到 compiler 类和 driver 类。
    即使我们现在不做真实 launch，也要把这个类补齐，这样后端结构才完整。
    """

    def __init__(self):
        self.utils = TinyGPUUtils()
        self.launcher_cls = TinyGPULauncher

    @classmethod
    def is_active(cls):
        """
        默认不抢占 CUDA/HIP driver。教程测试通过 driver.set_active() 显式选择
        TinyGPU；需要自动发现时，用户可设置此环境变量
        """
        return os.environ.get("TRITON_TINYGPU_SIM") == "1"

    def get_current_target(self):
        """为显式 compile-only 流程返回一个默认 TinyGPU target。"""
        return GPUTarget("tinygpu", "sim", 4)

    def get_active_torch_device(self):
        """为了满足接口而保留，虽然现在并没有真实设备。
        仿真器使用 CPU tensor 作为 host memory 的输入与输出。
        """
        import torch

        return torch.device("cpu")

    def get_current_device(self):
        return 0

    def set_current_device(self, device):
        if device != 0:
            raise ValueError("TinyGPU simulator exposes only device 0")

    def get_current_stream(self, device):
        if device != 0:
            raise ValueError("TinyGPU simulator exposes only device 0")
        return None

    def get_benchmarker(self):
        """返回一个假的 benchmarker。

        Triton 希望这里返回一个可调用对象，并且结果看起来像 timing 数据。
        这里纯粹是桩实现，避免因为接口不完整而报错。
        """

        def bench(*args, **kwargs):
            del args, kwargs
            return [0.0, 0.0, 0.0]

        return bench
