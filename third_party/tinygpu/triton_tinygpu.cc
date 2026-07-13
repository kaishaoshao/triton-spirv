// TinyGPU 的第一阶段 pybind 入口。
//
// Triton 的 python/src/main.cc 会调用：
//   init_triton_tinygpu(m.def_submodule("tinygpu"));
//
// 第一阶段只验证原生后端接入是否正确，不在这里实现 TTIR lowering。
// 真正的 lowering pass 会在第二阶段单独放到 lib/Conversion 中。

#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_triton_tinygpu(py::module &&m) {
  m.def("is_first_stage", [](){return true;});
}

