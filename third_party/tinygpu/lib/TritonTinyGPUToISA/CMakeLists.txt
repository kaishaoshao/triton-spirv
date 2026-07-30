add_triton_library(TritonTinyGPUToISA
  TinyGPUIRToISA.cpp
  TinyGPUEmitter.cpp

  LINK_LIBS PUBLIC
  TritonTinyGPUIR
  TritonIR
  MLIRIR
  MLIRPass
)
