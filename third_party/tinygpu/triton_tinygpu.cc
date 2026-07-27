// TinyGPU 的第一阶段 pybind 入口。
//
// Triton 的 python/src/main.cc 会调用：
//   init_triton_tinygpu(m.def_submodule("tinygpu"));
//
// 具体 lowering 位于 lib/Conversion 和 lib/TritonTinyGPUToISA；
// 这个文件只负责把 pass 注册给 Python，结构与 Triton 的 NVIDIA 后端保持一致。

#include "TritonTinyGPUToISA/Passes.h"
#include "Dialect/TinyGPU/IR/TinyGPU.h"
#include "include/TritonTinyGPUToISA/TinyGPUEmitter.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include <pybind11/pybind11.h>
#include <mlir/Pass/PassManager.h>

namespace py = pybind11;

void init_triton_tinygpu(py::module &&m) {
  // python 侧访问路径：triton._C.libtriton.tinygpu.passes.ttgpuir
  auto passes = m.def_submodule("passes");
  auto ttgpuir = passes.def_submodule("ttgpuir");

  ttgpuir.def("add_lower_ttgir_to_tinygpuir", [](mlir::PassManager &pm) {
    // 负责 TTGIR -> tinygpu.*，不直接发射 ISA。
    pm.addPass(mlir::triton::tinygpu::createLowerTTGIRToTinyGPUIRPass());
  });

  ttgpuir.def("add_lower_tinygpuir_to_isa", [](mlir::PassManager &pm) {
    //  pass 只消费 tinygpu.* 方言并生成 ISA metadata。
    pm.addPass(mlir::triton::tinygpu::createLowerTTGIRToTinyGPUPass());
  });

  ttgpuir.def("get_outputs", [](mlir::ModuleOp module) {
    // Triton 的 ir.module Python binding 目前只提供 get_int_attr()；
    // StringAttr 由 TinyGPU 自己读取，避免为了后端私有产物修改 Triton 核心 API。
    auto assembly = module->getAttrOfType<mlir::StringAttr>("tinygpu.asm");
    auto binaryHex =
        module->getAttrOfType<mlir::StringAttr>("tinygpu.binary_hex");
    if (!assembly || !binaryHex)
      throw std::runtime_error("TinyGPU lowering pass did not produce output");

    // 返回Python可以直接解包的(assembly, binary_hex)二元组
    return py::make_tuple(assembly.getValue().str(), binaryHex.getValue().str());
  });

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::tinygpu::TinyGPUDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

}

