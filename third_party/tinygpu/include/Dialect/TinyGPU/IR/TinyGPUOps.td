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

// 无操作指令，对应NOP
def TinyGPU_NopOp : TinyGPU_Op<"nop">;

// role=address 表示地址偏移， role=value 表示待写入的数据
def TinyGPU_ConstOp : TinyGPU_Op<"const"> {
  let arguments = (ins I8Attr:$value, StrAttr:$role);
  let results = (outs I8:$result);
}

def TinyGPU_AddPtrOp : TinyGPU_Op<"addptr"> {
  let arguments = (ins I8:$base, I8:$offset);
  let results = (outs I8:$result);
}

// 标量算术操作最终分别对应 ADD/SUB/MUL/DIV
class TinyGPU_BinaryOp<string mnemonic> : TinyGPU_Op<mnemonic> {
  let arguments = (ins I8:$lhs, I8:$rhs);
  let results = (outs I8:$result);
}
def TinyGPU_AddOp : TinyGPU_BinaryOp<"add">;
def TinyGPU_SubOp : TinyGPU_BinaryOp<"sub">;
def TinyGPU_MulOp : TinyGPU_BinaryOp<"mul">;
def TinyGPU_DivOp : TinyGPU_BinaryOp<"div">;

// CMP 修改 TinyGPU的NZP条件标志，不产生普通SSA结果
def TinyGPU_CmpOp : TinyGPU_Op<"cmp"> {
  let arguments = (ins I8:$lhs, I8:$rhs);
}

// 使用已经编码好的低级跳转目标；结构化CFG留到后续阶段
def TinyGPU_BranchOp : TinyGPU_Op<"branch", [Terminator]> {
  let arguments = (ins I32Attr:$nzp, I32Attr:$target);
}

def TinyGPU_JumpOp : TinyGPU_Op<"jump", [Terminator]> {
  let arguments = (ins I32Attr:$target);
}

def TinyGPU_ReturnOp : TinyGPU_Op<"ret", [Terminator]>;

// 全局内存和 shared memory 的标量访问
def TinyGPU_LoadOp : TinyGPU_Op<"load"> {
  let arguments = (ins I8:$address);
  let results = (outs I8:$result);
}

def TinyGPU_StoreOp : TinyGPU_Op<"store"> {
  let arguments = (ins I8:$address, I8:$value);
}

def TinyGPU_SharedLoadOp : TinyGPU_Op<"shared_load"> {
  let arguments = (ins I8:$address);
  let results = (outs I8:$result);
}

def TinyGPU_SharedStoreOp : TinyGPU_Op<"shared_store"> {
  let arguments = (ins I8:$address, I8:$value);
}

// 每个线程先写 shared memory。
// 在 tinygpu.barrier 等待其他线程。
// 所有线程到达后，才能执行 shared load。
// 确保读取到其他线程已经写入的数据。
def TinyGPU_BinaryOp : TinyGPU_Op<"barrier">;

#endif // TRITON_TINYGPU_OPS
