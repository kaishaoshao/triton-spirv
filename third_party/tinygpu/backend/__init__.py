"""
Triton 的 TinyGPU 外部后端包。

Triton 会通过 entry point 发现后端，并直接导入：
- `triton.backends.tinygpu.compiler`
- `triton.backends.tinygpu.driver`

"""
