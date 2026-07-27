#ifndef TRITON_TINYGPU_OPS
#define TRITON_TINYGPU_OPS

include "mlir/IR/OpBase.td"

def TinyGPU_Dialect : Dialect {
  let name = "tinygpu";
  let cppNamespace = "::mlir::triton::tinygpu";
  let description = [{ Minimal TinyGPU intermediate representation. }];
}

class TinyGPU_Op<string mnemonic, list<Trait> traits = []> :
  Op<TinyGPU_Dialect, mnemonic, traits>;

// 第一个 kernel 指针参数，对应TinyGPU 的 R0
def TinyGPU_BaseOp : TinyGPU_Op<"base"> {
  let arguments = (ins I32Attr:$arg_index);
  let results = (outs I8:$result);
}

// role=address 表示地址偏移， role=value 表示待写入的数据
def TinyGPU_ConstOp : TinyGPU_Op<"const"> {
  let arguments = (ins I8Attr:$value, StrAttr:$role);
  let results = (outs I8:$result);
}

def TinyGPU_AddPtrOp : TinyGPU_Op<"addptr"> {
  let arguments = (ins I8:$base, I8:$offset);
  let results = (outs I8:$result);
}

def TinyGPU_StoreOp : TinyGPU_Op<"store"> {
  let arguments = (ins I8:$address, I8:$value);
}

def TinyGPU_ReturnOp : TinyGPU_Op<"ret", [Terminator]>;

#endif // TRITON_TINYGPU_OPS
