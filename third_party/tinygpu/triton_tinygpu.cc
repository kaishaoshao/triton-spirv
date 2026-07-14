// TinyGPU 的第一阶段 pybind 入口。
//
// Triton 的 python/src/main.cc 会调用：
//   init_triton_tinygpu(m.def_submodule("tinygpu"));
//
// 第一阶段只验证原生后端接入是否正确，不在这里实现 TTIR lowering。
// 真正的 lowering pass 会在第二阶段单独放到 lib/Conversion 中。

#include "TritonTinyGPU/Transforms/Passes.h"

#include <pybind11/pybind11.h>
#include <mlir/Pass/PassManager.h>

namespace py = pybind11;

void init_triton_tinygpu(py::module &&m) {
  // python 侧访问路径：triton._C.libtriton.tinygpu.passes.ttgpuir
  auto passes = m.def_submodule("passes");
  auto ttgpuir = passes.def_submodule("ttgpuir");

  ttgpuir.def("add_to_tinygpu", [](mlir::PassManager &pm) {
    // python 只把 pass 加入 pass manager，真正的TTGIR lowering 保持独立
    // C++文件中，便于后续继续添加 instructions selection 和 寄存器分配。
    pm.addPass(mlir::triton::tinygpu::createLowerTTGIRToTinyGPUPass());
  });

  ttgpuir.def("get_outputs", [](mlir::ModuleOp module) {
    auto assembly = module->getAttrOfType<mlir::StringAttr>("tinygpu.asm");
    auto binaryHex =
        module->getAttrOfType<mlir::StringAttr>("tinygpu.binary_hex");
    if (!assembly || !binaryHex)
      throw std::runtime_error("TinyGPU lowering pass did not produce output");

    // 返回Python可以直接解包的(assembly, binary_hex)二元组
    return py::make_tuple(assembly.getValue().str(), binaryHex.getValue().str());
  });


}

