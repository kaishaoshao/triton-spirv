# 第一版 TinyGPU 先只补一个最小 pybind 入口：
#
# - 提供 `init_triton_tinygpu(pybind11::module_ &&m)`
# - 让 `libtriton.so` 能稳定加载 tinygpu 子模块
# - 暴露一个“TTIR 文本 -> 占位 TinyGPU asm”的简单翻译函数
#
# 这一版暂时不引入 TinyGPU 专属 dialect/pass/runtime。

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(
    TritonTinyGPU
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_tinygpu.cc
    )
  target_link_libraries(
    TritonTinyGPU
    PRIVATE Python3::Module pybind11::headers)
endif()
