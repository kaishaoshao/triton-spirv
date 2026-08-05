import { ChangeEvent, useState } from 'react';
import { CompilationPipeline } from './components/CompilationPipeline';
import { GPUSimulator } from './components/GPUSimulator';
import type { CompilationTrace, Instruction } from './simulator/types';

const DEFAULT_BIN = '91078001f000';
const DEFAULT_SOURCE = `import triton
import triton.language as tl

@triton.jit
def kernel(out):
    tl.store(out + 1, 9)`;

function binary(text: string): Instruction[] {
  const source = text.replace(/\s+/g, '').replace(/^0x/i, '');
  if (!source || source.length % 4 || /[^0-9a-f]/i.test(source)) {
    throw new Error('tinybin 必须是连续的 16-bit 十六进制字。');
  }
  return Array.from({ length: source.length / 4 }, (_, addr) => ({
    addr,
    hex: source.slice(addr * 4, addr * 4 + 4),
    asm: '',
    bits: '',
  }));
}

function memory(text: string): number[] {
  const result = Array(256).fill(0);
  if (!text.trim()) return result;
  text.split(/[\s,]+/).filter(Boolean).forEach((entry) => {
    const [addressText, valueText] = entry.split('=');
    const address = Number(addressText);
    const value = Number(valueText);
    if (!Number.isInteger(address) || !Number.isInteger(value) || address < 0 || address > 255 || value < 0 || value > 255) {
      throw new Error(`无效内存项 ${entry}`);
    }
    result[address] = value;
  });
  return result;
}

function normalizeBinary(text: string): string {
  return text.replace(/\s+/g, '').replace(/^0x/i, '').toLowerCase();
}

function parseTrace(text: string): CompilationTrace {
  const value = JSON.parse(text) as Partial<CompilationTrace> & { artifacts?: Record<string, unknown> };
  const sourceStages = value.stages && !Array.isArray(value.stages) ? value.stages : value.artifacts;
  if (!sourceStages || typeof sourceStages !== 'object') throw new Error('trace JSON 缺少 stages 对象。');

  const stages: CompilationTrace['stages'] = {};
  for (const key of ['ttir', 'ttgir', 'tinygpuir', 'tinyasm', 'tinybin'] as const) {
    const stage = (sourceStages as Record<string, unknown>)[key];
    if (stage !== undefined && stage !== null) stages[key] = typeof stage === 'string' ? stage : String(stage);
  }
  const tinybin = value.tinybin ?? stages.tinybin;
  if (!tinybin) throw new Error('trace JSON 缺少 tinybin，无法启动仿真。');
  const normalized = normalizeBinary(tinybin);
  binary(normalized);
  stages.tinybin = normalized;
  return { ...value, stages, tinybin: normalized };
}

export default function App() {
  const [hex, setHex] = useState(DEFAULT_BIN);
  const [mem, setMem] = useState('0=0');
  const [regs, setRegs] = useState(['0', '0', '0']);
  const [program, setProgram] = useState(() => binary(DEFAULT_BIN));
  const [initialMemory, setInitialMemory] = useState(() => memory('0=0'));
  const [initialRegisters, setInitialRegisters] = useState([0, 0, 0]);
  const [trace, setTrace] = useState<CompilationTrace | null>(null);
  const [traceText, setTraceText] = useState('');
  const [source, setSource] = useState(DEFAULT_SOURCE);
  const [compiling, setCompiling] = useState(false);
  const [error, setError] = useState('');

  function loadBinary(binaryText = hex, keepTrace = false) {
    try {
      const values = regs.map(Number);
      if (values.some((value) => !Number.isInteger(value) || value < 0 || value > 255)) throw new Error('R0/R1/R2 必须为 0..255。');
      setProgram(binary(binaryText));
      setHex(normalizeBinary(binaryText));
      setInitialMemory(memory(mem));
      setInitialRegisters(values);
      if (!keepTrace) setTrace(null);
      setError('');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  function loadTrace(text = traceText) {
    try {
      const parsed = parseTrace(text);
      setTrace(parsed);
      setTraceText(text);
      loadBinary(parsed.tinybin, true);
      setError('');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  function readTraceFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => loadTrace(String(reader.result ?? ''));
    reader.onerror = () => setError('读取 trace 文件失败。');
    reader.readAsText(file);
  }

  async function compileKernel() {
    setCompiling(true);
    setError('');
    try {
      const response = await fetch('/api/compile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: source }),
      });
      const payload = await response.json() as { error?: string };
      if (!response.ok) throw new Error(payload.error ?? `编译服务返回 HTTP ${response.status}`);
      const parsed = parseTrace(JSON.stringify(payload));
      setTrace(parsed);
      setTraceText(JSON.stringify(parsed, null, 2));
      loadBinary(parsed.tinybin, true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setCompiling(false);
    }
  }

  return (
    <main>
      <header>
        <p>TRITON TINYGPU BACKEND</p>
        <h1>TinyGPU 网页仿真器</h1>
        <span>上方编译并逐级 lowering，下方加载 TinyBIN 后进行逐周期仿真。</span>
      </header>

      <section className="compiler-workspace">
        <section className="input-panel compiler-panel">
          <div className="section-heading"><div><span className="eyebrow">01 · COMPILE</span><h2>编写 Triton 并编译</h2></div><span className="pipeline-state">LOCAL API</span></div>
          <p className="help-text">源码发送到 Triton 本地编译服务。成功后自动更新五级产物，并将 TinyBIN 回填到下方仿真器。</p>
          <div className="source-meta"><small className="signature-hint">网页自动识别源码中唯一的 <code>@triton.jit</code> kernel，并将全部参数推断为 <code>*i8</code> 指针。</small></div>
          <label>Triton Python 源码<textarea className="source-editor" value={source} onChange={(event) => setSource(event.target.value)} spellCheck={false} /></label>
          <div className="compile-actions"><button onClick={compileKernel} disabled={compiling}>{compiling ? '编译中…' : '编译并运行'}</button><small>/api → 127.0.0.1:8000</small></div>
        </section>

        <details className="trace-import">
          <summary>导入已有编译 trace（可选）</summary>
          <div className="trace-import-body">
            <p className="help-text">如果不启动本地编译服务，可以导入后端生成的 <code>tinygpu-trace.json</code>。</p>
            <textarea className="trace-input" value={traceText} onChange={(event) => setTraceText(event.target.value)} placeholder={'{"stages":{"ttir":"...","ttgir":"...","tinygpuir":"...","tinyasm":"...","tinybin":"..."}}'} spellCheck={false} />
            <div className="trace-actions"><button onClick={() => loadTrace()}>导入 trace 并仿真</button><label className="file-button">选择 JSON 文件<input type="file" accept="application/json,.json" onChange={readTraceFile} /></label></div>
          </div>
        </details>

        {error && <strong className="error">{error}</strong>}
        <section className="pipeline-workspace"><span className="eyebrow">COMPILE PIPELINE</span><CompilationPipeline trace={trace} /></section>
      </section>

      <section className="simulator-workspace">
        <div className="section-heading">
          <div><span className="eyebrow">02 · TINYBIN SIMULATOR</span><h2>二进制逐周期仿真</h2></div>
          <span className="pipeline-state">4 LANES · 256B MEM</span>
        </div>
        <section className="input-panel runtime-panel">
          <label>tinybin<textarea className="tinybin-input" value={hex} onChange={(event) => setHex(event.target.value)} spellCheck={false} /></label>
          <label>全局内存初始化<input value={mem} onChange={(event) => setMem(event.target.value)} placeholder="0=0, 4=1" /></label>
          <div className="abi">{regs.map((value, index) => <label key={index}>R{index}<input value={value} type="number" min="0" max="255" onChange={(event) => setRegs((old) => old.map((item, i) => i === index ? event.target.value : item))} /></label>)}</div>
          <button onClick={() => loadBinary()}>加载并运行当前 TinyBIN</button>
          <small>R0/R1/R2 是 kernel 参数基址；R13/R14/R15 由仿真器写入 builtin。</small>
        </section>
        <div className="simulator-layout">
          <GPUSimulator instructions={program} initialMemory={initialMemory} initialRegisters={initialRegisters} numBlocks={1} threadsPerBlock={4} />
        </div>
      </section>
    </main>
  );
}
