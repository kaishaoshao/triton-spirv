add_triton_library(TritonTinyGPUConversion
  TTGIRToTinyGPUIR.cpp

  LINK_LIBS PUBLIC
  TritonTinyGPUIR
  TritonIR
  MLIRIR
  MLIRPass
)
