"""TinyGPU 的 Triton 编译后端。
当前教程验证 TTIR -> TTGIR -> TinyGPU dialect -> TinyGPU ISA 的两段 lowering，
并支持无 mask 的 4-lane vector load/add/store 和三参数 compile-only ABI。
"""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict
import hashlib

from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import tinygpu, ir, passes

@dataclass(frozen=True)
class TinyGPUOptions:
  """
  后端选项对象，会参与 Triton 的编译缓存 key。

    这一版只保留最少几个字段。
    后面你可以继续往里扩，比如：
    - block size
    - shared memory toggle
    - simulator/debug flags
    - ISA revision
    - shared memory 开关
    - simulator/debug 选项
    - ISA
  """

  num_warps:        int = 1
  num_ctas:         int = 1
  num_stages:       int = 1
  threads_per_warp: int = 4
  debug:           bool = False
  # Triton 前端在整数加减乘除时会读取这个选项，决定是否插入溢出检查。
  # TinyGPU 第一版不实现额外的溢出处理，但必须提供该字段以兼容前端接口。
  # Triton 前端在整数加减乘除时会读取这个选项，决定是否插入溢出检查。
  # 当前 TinyGPU 只支持 8-bit 标量算术，暂时关闭检查，避免前端生成
  # arith.extsi/cmpi/andi 等 Ch5 尚未实现的辅助操作。
  sanitize_overflow: bool = False

  """Triton 要求后端选项能生成稳定的 hash。"""
  def hash(self) -> str:
    key = "_".join([f"{name}-{value}" for name, value in sorted(self.__dict__.items())])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


"""后端：能接 Triton 输入，并输出占位产物。

    This backend is compile-only:
    这个后端目前只做 compile-only：
    - 支持显式传 `target=GPUTarget("tinygpu", "sim", 8)`
    - 输出占位的汇编和二进制文件
    - 不尝试做真实设备 launch

    The later real work should happen in:
    后面真正的工作主要会落在：

    - `make_tinyasm()` for TTIR/TTGIR -> TinyGPU assembly lowering
    - `make_tinybin()` for TinyGPU assembly -> 16-bit instruction encoding
    - `make_tinyasm()`：TTIR/TTGIR -> TinyGPU 汇编
    - `make_tinybin()`：TinyGPU 汇编 -> 16 位指令编码
"""
class TinyGPUBackend(BaseBackend):
  # `CompiledKernel` 会根据这个字段判断哪个产物算“最终二进制”。
  binary_ext = "tinybin"

  @staticmethod
  def supports_target(target: GPUTarget):
    #  告诉Triton，这个backend 能不能处理当前target
    return target.backend == "tinygpu"

  def  __init__(self, target: GPUTarget) -> None:
    super().__init__(target)

  def parse_options(self, opts) -> Any:
    # 把Triton传进来的原始dict选项转成类型化对象
    args = {
      name: opts[name]
      for name in TinyGPUOptions.__dataclass_fields__
      if name in opts and opts[name] is not None
    }
    return TinyGPUOptions(**args)

  def pack_metadata(self, metadata):
    """
    返回后端自己的launch metadata 打包结果
    - CUDA/HIP 会在这里准备运行期 launch 所需的信息。
    - 我们这个后端不会真的 launch，所以先返回空 tuple 就够了。
    """
    del metadata
    return ()

  def get_codegen_implementation(self, options):
    """
    返回 `ast_to_ttir` 可能需要的 target 专属 codegen helper。
    这一版TinyGPU暂时没有额外的语言helper, 所以直接返回空字典。
    """
    del options
    return {}

  def get_module_map(self) -> Dict[str, ModuleType]:
    """
    返回 AST lowering 时会用到的设备相关 Python 模块映射。
    CUDA 和 AMD 会在这里接 libdevice 一类的模块。
    TinyGPU 这一版还没有，所以返回空映射。
    """
    return {}

  def load_dialects(self, context):
    """
    如果后端需要额外 MLIR dialect，就在这里加载。
    当前最小版本只复用 Triton 已经加载好的核心 dialect。
    """
    # 阶段 4 注册 tinygpu.*，让 Python 创建的 MLIRContext 能识别方言。
    tinygpu.load_dialects(context)

  @staticmethod
  def make_ttir(mod, metadata):
    """
    后端拿到 Triton 前端结果后的第一阶段。
    对真实后端来说，这里通常会做 backend 自己的 TTIR 清理、规范化
    或一些轻量变换。

    但对这个最小骨架来说，我们先完全不改 TTIR，只专注验证后端接线。
    """
    del metadata
    return mod


  @staticmethod
  def make_ttgir(mod, metadate, options):
    # 将TTIR转为匹配TinyGPU 4线程block的TTGIR
    del metadate
    pm = ir.pass_manager(mod.context)
    # 复用Triton的通用TTIR -> TTGIR 转换； 后端只提供target名称和
    # 执行模型参数，不使用NVIDIA专属 ttnvgpuir pass
    passes.ttir.add_convert_to_ttgpuir(
      pm,
      f"tinygpu:{options.threads_per_warp}",
      options.num_warps,
      options.threads_per_warp,
      options.num_ctas,
    )
    pm.run(mod)
    return mod

  @staticmethod
  def make_tinygpuir(mod, metadata):
    # 运行TTGIR的 TinyGPU C++ pass

    del metadata
    pm = ir.pass_manager(mod.context)
    # 该绑定由 triton_tinygpu.cc 注册，最终会创建C++的
    # 第一阶段只生成 tinygpu.* 方言，便于单独查看 TinyGPU IR。
    tinygpu.passes.ttgpuir.add_lower_ttgir_to_tinygpuir(pm)
    pm.run(mod)
    return mod



  @staticmethod
  def _set_metadata(metadata):
    # 当前没有runtime launcher； 先补齐CompiledKernel读取的最小metadata
    metadata["name"] = "tinygpu_kernel"
    metadata["shared"] = 0
    metadata["cluster_dims"] = (1, 1, 1)
    metadata["num_warps"] = 1

  def make_tinyasm(self, mod, metadata):
    """
    输出一个可读的占位 TinyGPU 汇编产物。
    同时，这里也会补齐 Triton 的 `CompiledKernel` 后续会依赖的 metadata。

    我把 TTIR 文本直接嵌进输出里，是为了方便调试：
    这样你能很直观看到 TinyGPU backend 确实收到了 Triton 的 IR。

    等你开始做真实 lowering 时，这个函数就是最自然的切入点：
    在这里读取 TTIR，然后一步步翻译成 TinyGPU 指令。
    """
    self._set_metadata(metadata)
    pm = ir.pass_manager(mod.context)
    # 消费tinygpu.* 方言并生成tinyasm/tinybin metadate
    tinygpu.passes.ttgpuir.add_lower_tinygpuir_to_isa(pm)
    pm.run(mod)
    # C++ pass 把文本产物写入module attribute, Python stage只负责取出并
    # 交给 Triton 缓存，不在这里重新解析或生成指令
    # ir.module 没有暴露get_str_attr(); 该函数只绑定普通的 ir.operation。
    # 因此由TinyGPU的pybind接口读取ModuleOp的StringAttr, 避免修改 Triton
    # 核心 python bingding，也避免把IR文本解析逻辑放到Python。
    assembly, binary_hex = tinygpu.passes.ttgpuir.get_outputs(mod)
    metadata["tinygpu_binary"] = bytes.fromhex(binary_hex)
    return assembly

  @classmethod
  def make_tinybin(self, assembly, metadata):
    """
    输出一个占位的 16 位 TinyGPU 指令流。

    按现有 TinyGPU 例子，`RET` 指令通常编码成 `0xF000`。
    这里先用大端字节序写成：
    - `0xF0`
    - `0x00`
    等真实指令发射逻辑补上之后，这里就应该消费前面的汇编或结构化中间结果，
    然后返回仿真器真正需要的指令字节流。
    """
    self._set_metadata(metadata)
    # tinyasm 与 tinybin 由同一个 C++ pass 生成， 避免二者指令序列不一致
    del assembly
    return metadata.pop("tinygpu_binary")

  def add_stages(self, stages, options):
    """
     注册这个 backend 的顺序编译阶段。
        这里有个很关键的细节：
        - Triton 的 `ASTSource` 是从 `ttir` 开始接后端的
        - 所以第一阶段通常就应该叫 `ttir`

        当前最小流水线是：
        - `ttir`.    ：保留 Triton 生成的 TTIR
        - `ttgir`.   ：保留 Triton 生成的 TTGIR
        - `tinygpuir`：中间形态
        - `tinyasm`  ：输出占位的 TinyGPU 汇编
        - `tinybin`  ：输出占位的 TinyGPU 二进制
    """
    stages["ttir"]      = self.make_ttir
    stages["ttgir"]     = lambda mod, metadata : self.make_ttgir(mod, metadata, options)
    stages["tinygpuir"] = self.make_tinygpuir
    stages["tinyasm"]   = self.make_tinyasm
    stages["tinybin"]   = self.make_tinybin

  def hash(self) -> str:
    # 返回backend 身份字符串， 参与Triton的缓存key。
    return f"tinygpu-{self.target.arch}-{self.target.warp_size}"



