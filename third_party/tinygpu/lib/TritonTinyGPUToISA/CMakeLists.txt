add_triton_library(TritonTinyGPUToISA
  TTGIRToTinyGPU.cpp
  TinyGPUEmitter.cpp

  LINK_LIBS PUBLIC
  TritonIR
  MLIRIR
  MLIRPass
)
