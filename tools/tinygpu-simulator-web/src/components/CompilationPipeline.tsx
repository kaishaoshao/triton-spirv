import { useState } from 'react';
import type { CompilationTrace } from '../simulator/types';

const STAGES = [
  { key: 'ttir', label: 'TTIR', kind: 'MLIR', description: 'Triton 前端产生的 Triton IR' },
  { key: 'ttgir', label: 'TTGIR', kind: 'MLIR', description: '按 TinyGPU 4-lane 执行模型转换后的 Triton GPU IR' },
  { key: 'tinygpuir', label: 'TinyGPU IR', kind: 'MLIR', description: '教程自定义 tinygpu.* 方言' },
  { key: 'tinyasm', label: 'TinyASM', kind: 'ASM', description: 'TinyGPU 汇编指令' },
  { key: 'tinybin', label: 'TinyBIN', kind: 'BIN', description: '每条指令一个 16-bit word 的十六进制字节流' },
] as const;

export function CompilationPipeline({ trace }: { trace: CompilationTrace | null }) {
  const [selected, setSelected] = useState('ttir');

  if (!trace) {
    return (
      <section className="pipeline empty-pipeline">
        <div className="section-heading">
          <div><span className="eyebrow">COMPILE PIPELINE</span><h2>逐级降低关系</h2></div>
          <span className="pipeline-state">等待导入 trace</span>
        </div>
        <p>点击上方“编译并运行”，或导入已有 trace；成功后这里会显示 TTIR → TTGIR → TinyGPU IR → TinyASM → TinyBIN。</p>
      </section>
    );
  }

  const available = STAGES.filter(({ key }) => trace.stages[key]);
  const current = available.find(({ key }) => key === selected) ?? available[0];
  const source = typeof trace.source === 'string' ? trace.source : trace.source?.script ?? trace.source?.kernel;

  return (
    <section className="pipeline">
      <div className="section-heading">
        <div><span className="eyebrow">COMPILE PIPELINE</span><h2>Triton → TinyGPU</h2></div>
        <span className="pipeline-state">{available.length}/5 个阶段</span>
      </div>
      {source && <p className="trace-source">来源：{source}{typeof trace.source === 'object' && trace.source.kernel ? ` · ${trace.source.kernel}` : ''}</p>}
      <div className="pipeline-flow" aria-label="编译逐级降低流程">
        {STAGES.map((stage, index) => {
          const present = Boolean(trace.stages[stage.key]);
          return (
            <div className="pipeline-node-wrap" key={stage.key}>
              <button className={`pipeline-node ${present ? 'present' : 'missing'} ${selected === stage.key ? 'selected' : ''}`} onClick={() => present && setSelected(stage.key)} disabled={!present}>
                <span className="pipeline-index">{index + 1}</span>
                <strong>{stage.label}</strong>
                <small>{stage.kind}</small>
              </button>
              {index < STAGES.length - 1 && <span className="pipeline-arrow" aria-hidden="true">→</span>}
            </div>
          );
        })}
      </div>
      {current && (
        <div className="stage-viewer">
          <div className="stage-viewer-heading"><div><h3>{current.label}</h3><p>{current.description}</p></div><span>{current.kind}</span></div>
          <pre>{trace.stages[current.key] ?? '该阶段没有导出。'}</pre>
        </div>
      )}
    </section>
  );
}
