add_triton_library(TritonTinyGPUIR
  TinyGPU.cpp

  LINK_LIBS PUBLIC
  TritonIR
  MLIRIR
)

add_dependencies(TritonTinyGPUIR TritonTinyGPUTableGen)
