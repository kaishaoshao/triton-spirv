#!/usr/bin/env python3
"""导出 Triton -> TinyGPU 的完整编译流水线，供网页仿真器导入。

示例：
    python tools/export_tinygpu_trace.py \
      --script third_party/tinygpu/unittest/ch8-test01-vector-add.py \
      --kernel vector_add \
      --signature out:*i8,a:*i8,b:*i8 \
      --output /tmp/vector-add-tinygpu-trace.json

网页端只负责展示这些真实产物并执行 TinyBIN；Triton 编译必须在 Python/本地后端环境中完成。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

STAGES = ("ttir", "ttgir", "tinygpuir", "tinyasm", "tinybin")


def load_module(path: Path):
    module_name = f"tinygpu_trace_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 Python 测试文件：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_signature(text: str) -> dict[str, str]:
    signature: dict[str, str] = {}
    for item in text.split(","):
        name, separator, type_name = item.partition(":")
        if not separator or not name or not type_name:
            raise ValueError(f"签名必须形如 name:*i8,name2:i32，收到：{item}")
        signature[name.strip()] = type_name.strip()
    return signature


def artifact_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.hex()
    if value is None:
        raise RuntimeError("编译结果缺少某个阶段产物")
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, type=Path, help="包含 @triton.jit kernel 的 Python 文件")
    parser.add_argument("--kernel", required=True, help="Python 文件中的 kernel 变量名，例如 vector_add")
    parser.add_argument("--signature", required=True, help="ASTSource signature，例如 out:*i8,a:*i8,b:*i8")
    parser.add_argument("--output", type=Path, help="JSON 输出路径；不传则写到 stdout")
    args = parser.parse_args()

    # 延迟导入，让 `--help` 和静态语法检查不要求当前 shell 已安装 Triton。
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    module = load_module(args.script.resolve())
    function = getattr(module, args.kernel)
    target = getattr(module, "TARGET", GPUTarget("tinygpu", "sim", 4))
    source = ASTSource(fn=function, signature=parse_signature(args.signature), constexprs={})
    kernel = triton.compile(source, target=target)

    stages = {name: artifact_text(kernel.asm.get(name)) for name in STAGES}
    trace = {
        "version": 1,
        "source": {"script": str(args.script), "kernel": args.kernel},
        "target": str(target),
        "stages": stages,
        # tinybin 单独放一份，方便网页直接启动 TinyGPU 仿真。
        "tinybin": stages["tinybin"],
    }
    output = json.dumps(trace, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"已写入 {args.output}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
