# Triton `third_party/tinygpu` 教程

这份文档说明你现在采用的这条路线：

```text
直接把 TinyGPU 后端放进 Triton 仓库内：
triton/third_party/tinygpu
```

这条路线和外部插件方案都可以工作，但定位不同：

- `triton-tinygpu-backend/`：偏教学、偏实验、偏外部插件
- `triton/third_party/tinygpu/`：偏仓库内嵌、偏真正集成

如果你已经决定把代码放进 `third_party/tinygpu`，那后面就按这份文档走。

## 第 1 步：明确当前阶段目标

当前不要一上来就做完整 TinyGPU 后端。

最小目标应该是：

1. Triton 能发现 `tinygpu` 后端
2. `triton.compile(..., target=GPUTarget("tinygpu", "sim", 8))` 能成功
3. 能产出 `tinyasm` 和 `tinybin`
4. `tinyasm` 里能看到 TTIR 内容，证明后端接线是通的

这一步叫“compile-only bring-up”。

## 第 2 步：in-tree 后端目录应该长什么样

当前最小目录结构如下：

```text
third_party/tinygpu/
  README.md
  CMakeLists.txt
  backend/
    __init__.py
    name.conf
    compiler.py
    driver.py
    driver.c
```

这一阶段真正起作用的是：

- `backend/name.conf`
- `backend/compiler.py`
- `backend/driver.py`
- `setup.py` 里的后端注册列表

## 第 3 步：为什么现在这种做法是可行的

这是可行的，而且比外部插件更接近最终集成形态。

原因是 Triton 的内建后端本来就是按这个结构放在 `third_party/` 下的，比如：

- `third_party/nvidia`
- `third_party/amd`

所以你把 TinyGPU 做成：

```text
third_party/tinygpu/backend/compiler.py
third_party/tinygpu/backend/driver.py
```

方向是对的。

你现在这条路线和外部插件路线最大的区别是：

- 外部插件靠 `TRITON_PLUGIN_DIRS`
- 仓库内嵌靠 `setup.py` 的内建 backend 列表

## 第 4 步：让 Triton 真正注册 `tinygpu`

要让 in-tree 后端生效，最关键的是这一行：

[`/Volumes/wsk/code/llvm-mlir/triton/triton/setup.py`](/Volumes/wsk/code/llvm-mlir/triton/triton/setup.py:592)

现在它已经改成包含：

```python
backends = [*BackendInstaller.copy(["nvidia", "amd", "tinygpu"]), *BackendInstaller.copy_externals()]
```

这意味着重新安装 Triton 时，它会把 `third_party/tinygpu/backend` 当成正式后端一起装进去。

## 第 5 步：重新安装本地 Triton

激活虚拟环境：

```bash
cd /Volumes/wsk/code/llvm-mlir/triton/triton
source .triton_riscv/bin/activate
```

重新 editable 安装：

```bash
PYTHONNOUSERSITE=1 \
TRITON_OFFLINE_BUILD=1 \
TRITON_BUILD_PROTON=OFF \
LLVM_INCLUDE_DIRS=/Volumes/wsk/code/llvm-mlir/triton/triton/llvm-project/build-triton-macos/include \
LLVM_LIBRARY_DIR=/Volumes/wsk/code/llvm-mlir/triton/triton/llvm-project/build-triton-macos/lib \
LLVM_SYSPATH=/Volumes/wsk/code/llvm-mlir/triton/triton/llvm-project/build-triton-macos \
python -m pip install -e /Volumes/wsk/code/llvm-mlir/triton/triton --no-build-isolation -v
```

## 第 6 步：确认后端是否被发现

执行：

```bash
source /Volumes/wsk/code/llvm-mlir/triton/triton/.triton_riscv/bin/activate
python - <<'PY'
import triton.backends
print(sorted(triton.backends.backends.keys()))
PY
```

如果成功，输出里应该包含：

```text
tinygpu
```

## 第 7 步：做最小 compile-only 验证

你可以直接在命令行里跑这段：

```bash
source /Volumes/wsk/code/llvm-mlir/triton/triton/.triton_riscv/bin/activate
python - <<'PY'
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

@triton.jit
def tiny_kernel(x_ptr):
    offs = tl.arange(0, 1)
    val = tl.load(x_ptr + offs)
    tl.store(x_ptr + offs, val)

src = triton.compiler.ASTSource(
    fn=tiny_kernel,
    signature={"x_ptr": "*fp32"},
    constexprs={},
)

k = triton.compile(src, target=GPUTarget("tinygpu", "sim", 8))
print(sorted(k.asm.keys()))
print(k.asm["tinyasm"])
print(k.asm["tinybin"])
PY
```

如果这一步通了，说明：

- backend 注册通了
- `compiler.py` 接口通了
- `driver.py` 结构通了
- `CompiledKernel` 能识别你的输出了

## 第 8 步：为什么当前 driver 要保持“不激活”

当前 `driver.py` 里 `is_active()` 返回 `False`，这是故意的。

原因：

1. 你现在的目标是编译链路，不是运行期链路
2. Triton 的默认 runtime 假设是 CUDA/HIP 风格
3. TinyGPU 更适合先走“显式 target 编译 + 单独仿真器执行”的路线

所以现在最合理的方式是：

- 编译：走 Triton backend
- 运行：后面单独接 tiny-gpu 仿真器

## 第 9 步：完成后端注册以后，下一步往哪里走

后端“支持”本身只是第一个里程碑。后面建议你按下面顺序推进。

### 阶段 A：看懂 TTIR 输入

先不要急着生成真实 TinyGPU 指令。

你应该先在 `make_tinyasm()` 里观察：

- Triton 给你的 TTIR 长什么样
- `tl.load` / `tl.store` / `arange` / 指针加法在 TTIR 里怎么表示

第一版建议只支持这些语义：

- `program_id(0)`
- `arange`
- pointer arithmetic
- `tl.load`
- `tl.store`
- `+`、`-`、`*`

### 阶段 B：先做一个真实 kernel 的 lowering

建议第一个只做 `vector_add`。

原因：

- 地址计算简单
- 与 TinyGPU 的既有样例最接近
- 只需要最基础的取数、加法、回写

目标链路：

```text
Triton kernel -> TTIR -> TinyGPU asm -> TinyGPU 16位指令
```

### 阶段 C：接 tiny-gpu 仿真器

等你能生成真实 `tinybin` 后，再单独写一个 runner：

1. 把 `tinybin` 装进 program memory
2. 把输入数据装进 data memory
3. 设置 thread count
4. 启动 tiny-gpu 仿真
5. 读取输出
6. 和参考结果比较

这一步最好不要一开始就强塞进 Triton 的 runtime driver。

## 第 10 步：当前这条 in-tree 路线和外部插件路线怎么配合

建议你这样理解：

- `triton-tinygpu-backend/`：留作教程和试验场
- `third_party/tinygpu/`：作为你真正推进的集成版本

也就是说：

- 教学和试错可以参考外部插件版
- 真正准备纳入 Triton 仓库内结构时，以 `third_party/tinygpu` 为主

## 第 11 步：接下来最值得做的事

如果你现在已经完成到“后端能被发现、能 compile-only”这一步，
那下一步最值得做的是：

1. 固定一个 `vector_add` Triton kernel
2. 在 `make_tinyasm()` 里打印并分析 TTIR
3. 写出第一版“TTIR 到 TinyGPU 指令”的手工 lowering 规则
4. 再把结果接到 tiny-gpu 仿真器

这会比一开始就碰 matmul、shared memory、barrier 更稳很多。
