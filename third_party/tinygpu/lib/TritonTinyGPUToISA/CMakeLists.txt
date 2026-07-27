add_triton_library(TritonTinyGPUToISA
  TTGIRToTinyGPU.cpp
  TinyGPUEmitter.cpp

  LINK_LIBS PUBLIC
  TritonTinyGPUIR
  TritonIR
  MLIRIR
  MLIRPass
)
