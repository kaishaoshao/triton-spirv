# 第一版 TinyGPU 先只补一个最小 pybind 入口：
# 这一版暂时不引入 TinyGPU 专属 dialect/pass/runtime。
# - 提供 `init_triton_tinygpu(pybind11::module_ &&m)`
# - 让 `libtriton.so` 能稳定加载 tinygpu 子模块
# - TTGIR -> TinyGPU 的 lowering
# - 放在独立 MLIR pass 中，避免在 pybind 入口或 Python 中解析 IR 文本。



include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(lib)

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(
    TritonTinyGPU
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_tinygpu.cc

    LINK_LIBS
    TritonTinyGPUToISA
    )
  target_link_libraries(
    TritonTinyGPU
    PRIVATE
    Python3::Module
    pybind11::headers
    )
endif()
