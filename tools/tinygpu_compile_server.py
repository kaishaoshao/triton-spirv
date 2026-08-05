#!/usr/bin/env python3
"""本地 TinyGPU Triton 编译服务。

只监听 localhost，供 tools/tinygpu-simulator-web 在开发时调用。请求中的 code 会在
当前 Python 进程中执行，因此这个服务只适合本机教程环境，不要绑定到公网地址。
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import linecache
import sys
from typing import Any
import traceback


STAGES = ("ttir", "ttgir", "tinygpuir", "tinyasm", "tinybin")
MAX_BODY_BYTES = 2 * 1024 * 1024


def verify_triton_environment() -> None:
    """在监听端口前确认服务运行在已经安装 TinyGPU 后端的 Python 中。"""
    try:
        import triton
    except ModuleNotFoundError as error:
        raise SystemExit(
            "当前 Python 环境没有 triton。不要直接执行 tinygpu_compile_server.py；\n"
            "请从教程根目录执行：\n"
            "  bash tools/start_tinygpu_compile_server.sh --host 127.0.0.1 --port 8000\n"
            "该脚本会使用 /Volumes/wsk/code/llvm-mlir/triton/triton/"
            ".triton_tinygpu/bin/python。"
        ) from error

    print(f"Python executable: {sys.executable}")
    print(f"Triton package: {triton.__file__}")


def artifact_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.hex()
    if value is None:
        raise RuntimeError("编译结果缺少某个阶段产物")
    return str(value)


def compile_request(request: dict[str, Any]) -> dict[str, Any]:
    code = request.get("code")
    kernel_name = request.get("kernel")
    signature = request.get("signature")
    constexprs = request.get("constexprs", {})

    if not isinstance(code, str) or not code.strip():
        raise ValueError("code 必须是非空 Triton Python 源码")
    if len(code.encode("utf-8")) > MAX_BODY_BYTES:
        raise ValueError(f"源码不能超过 {MAX_BODY_BYTES // 1024} KiB")
    if kernel_name is not None and (not isinstance(kernel_name, str) or not kernel_name.isidentifier()):
        raise ValueError("kernel 必须是 Python 标识符")
    if signature is not None and (not isinstance(signature, dict) or not signature):
        raise ValueError("signature 必须是非空对象，例如 {\"out\": \"*i8\"}")
    if isinstance(signature, dict) and not all(isinstance(key, str) and isinstance(value, str) for key, value in signature.items()):
        raise ValueError("signature 的参数名和值都必须是字符串")
    if not isinstance(constexprs, dict):
        raise ValueError("constexprs 必须是 JSON 对象")

    # 延迟导入，使服务的 --help 和错误提示不依赖 Triton 已经成功加载。
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    # Triton 的 @triton.jit 会通过 inspect.getsourcelines() 读取 kernel 源码。
    # 浏览器代码没有真实 .py 文件，因此先登记到 linecache，保留诊断信息和行号。
    source_filename = "<tinygpu-web-editor>"
    linecache.cache[source_filename] = (
        len(code),
        None,
        code.splitlines(keepends=True),
        source_filename,
    )
    namespace: dict[str, Any] = {"__name__": "__tinygpu_web__", "__file__": source_filename}
    exec(compile(code, source_filename, "exec"), namespace, namespace)
    if kernel_name is None:
        # 网页默认编译源码中唯一的 @triton.jit kernel，省去重复填写函数名。
        candidates = [
            (name, value) for name, value in namespace.items()
            if isinstance(name, str) and getattr(value, "arg_names", None) is not None
        ]
        if len(candidates) != 1:
            names = ", ".join(name for name, _ in candidates) or "无"
            raise ValueError(
                "源码必须包含唯一的 @triton.jit kernel；"
                f"当前找到：{names}"
            )
        kernel_name, function = candidates[0]
    else:
        function = namespace.get(kernel_name)
        if function is None:
            raise ValueError(f"源码中没有找到 kernel：{kernel_name}")

    # 当前 TinyGPU 教程 ABI 只支持 i8 指针参数。网页不应要求用户重复填写
    # kernel 的参数名，因此没有显式 signature 时按 JITFunction 的声明顺序推断。
    argument_names = list(getattr(function, "arg_names", ()))
    if not argument_names:
        raise ValueError(f"{kernel_name} 不是可供 Triton 编译的 @triton.jit kernel")
    if signature is None:
        signature = {name: "*i8" for name in argument_names}
    elif set(signature) != set(argument_names):
        raise ValueError(
            f"signature 参数必须与 kernel 参数一致：期望 {', '.join(argument_names)}"
        )
    else:
        # 以 kernel 定义的顺序重建字典，避免调用方 JSON 的键顺序改变 ABI 顺序。
        signature = {name: signature[name] for name in argument_names}

    target = GPUTarget("tinygpu", "sim", 4)
    source = ASTSource(fn=function, signature=signature, constexprs=constexprs)
    compiled = triton.compile(source, target=target)
    stages = {name: artifact_text(compiled.asm.get(name)) for name in STAGES}
    return {
        "version": 1,
        "source": {"kernel": kernel_name, "mode": "local-http", "signature": signature},
        "target": str(target),
        "stages": stages,
        "tinybin": stages["tinybin"],
    }


class CompileHandler(BaseHTTPRequestHandler):
    server_version = "TinyGPUCompileServer/0.1"

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "http://localhost:5173")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._send_json(HTTPStatus.NO_CONTENT, {})

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/api/health":
            self._send_json(HTTPStatus.OK, {"ok": True, "service": "tinygpu-compiler"})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "只支持 /api/health 和 /api/compile"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/api/compile":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "只支持 POST /api/compile"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAX_BODY_BYTES:
                raise ValueError("请求体为空或超过 2 MiB")
            request = json.loads(self.rfile.read(content_length))
            if not isinstance(request, dict):
                raise ValueError("请求体必须是 JSON 对象")
            trace = compile_request(request)
            self._send_json(HTTPStatus.OK, trace)
        except Exception as error:  # 编译错误需要返回网页显示，而不是断开连接。
            detail = "".join(traceback.format_exception_only(type(error), error)).strip()
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": detail})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[tinygpu-compile] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认只监听本机")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    args = parser.parse_args()
    verify_triton_environment()
    server = ThreadingHTTPServer((args.host, args.port), CompileHandler)
    print(f"TinyGPU Triton compile server: http://{args.host}:{args.port}")
    print("按 Ctrl-C 停止。源码会在当前 Python 进程中执行，请仅用于本机教程环境。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止 TinyGPU 编译服务。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
