import { useEffect, useRef, useState } from 'react';
import { Instruction, SimulationState, ThreadState } from '../simulator/types';
import { TinyGPUSim } from '../simulator/TinyGPUSim';

interface Props {
  instructions: Instruction[];
  initialMemory: number[];
  initialRegisters: number[];
  numBlocks: number;
  threadsPerBlock: number;
}

const COLORS: Record<string, string> = {
  FETCH: '#e06c75',
  DECODE: '#d19a66',
  REQUEST: '#e5c07b',
  WAIT: '#98c379',
  EXECUTE: '#61afef',
  UPDATE: '#c678dd',
  BARRIER: '#4ec9b0',
  DONE: '#555',
};

export function GPUSimulator({ instructions, initialMemory, initialRegisters, numBlocks, threadsPerBlock }: Props) {
  const [history, setHistory] = useState<SimulationState[]>([]);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    const sim = new TinyGPUSim(instructions, initialMemory, numBlocks, threadsPerBlock, initialRegisters);
    setHistory(sim.runToEnd());
    setStep(0);
    setPlaying(false);
    // 默认展开 lane 0 的寄存器，避免首次加载时状态区出现空白。
    setSelected(0);
  }, [instructions, initialMemory, initialRegisters, numBlocks, threadsPerBlock]);

  useEffect(() => {
    if (!playing || !history.length) return undefined;
    timer.current = window.setInterval(() => setStep((value) => {
      if (value >= history.length - 1) {
        setPlaying(false);
        return value;
      }
      return value + 1;
    }), 250);
    return () => {
      if (timer.current !== null) window.clearInterval(timer.current);
    };
  }, [playing, history.length]);

  const state = history[step];
  if (!state) return <p className="empty">没有可执行指令。</p>;
  const thread = state.threads.find((item) => item.threadId === selected);
  // 当前教程的一个 block 固定为四个 lane；R0 是第一个参数，即输出缓冲区基址。
  const outputBase = initialRegisters[0] ?? 0;
  const output = state.memory.slice(outputBase, outputBase + threadsPerBlock);

  return (
    <section className="simulator">
      <div className="simulator-run-panel">
        <div className="sim-controls">
          <button onClick={() => { setStep(0); setPlaying(false); }}>|&lt;</button>
          <button onClick={() => { setStep((value) => Math.max(0, value - 1)); setPlaying(false); }}>&lt;</button>
          <button className="play" onClick={() => setPlaying((value) => !value)}>{playing ? '暂停' : '播放'}</button>
          <button onClick={() => { setStep((value) => Math.min(history.length - 1, value + 1)); setPlaying(false); }}>&gt;</button>
          <button onClick={() => { setStep(history.length - 1); setPlaying(false); }}>&gt;|</button>
          <span>Cycle {state.cycle}/{history.length - 1}</span>
          <span>Block {state.currentBlock}/{state.totalBlocks}</span>
        </div>
        <input className="scrubber" type="range" min="0" max={history.length - 1} value={step} onChange={(event) => { setStep(Number(event.target.value)); setPlaying(false); }} />
        <div className="thread-grid">
          {state.threads.map((item) => <ThreadCard key={item.threadId} thread={item} selected={selected === item.threadId} onClick={() => setSelected((value) => value === item.threadId ? null : item.threadId)} />)}
        </div>
        <div className="output-panel"><h3>Output Buffer (R0 = {outputBase})</h3><code>[{output.join(', ')}]</code><small>显示当前周期中 R0 起始的 {threadsPerBlock} 个字节。</small></div>
      </div>
      {thread && <Registers thread={thread} />}
      <aside className="simulator-global-memory">
        <Memory title="Global Memory (256 bytes)" memory={state.memory} />
      </aside>
      <aside className="simulator-shared-memory">
        <Memory title="Shared Memory (64 bytes)" memory={state.sharedMemory} />
      </aside>
    </section>
  );
}

function ThreadCard({ thread, selected, onClick }: { thread: ThreadState; selected: boolean; onClick: () => void }) {
  const color = COLORS[thread.stage];
  return <button className={`thread-card ${selected ? 'selected' : ''} ${thread.divergent ? 'divergent' : ''}`} style={{ borderColor: color }} onClick={onClick}><header><b>Lane {thread.threadId}</b><span style={{ background: color }}>{thread.stage}</span></header><p>PC {thread.pc} | {thread.currentInstruction || '-'}</p><div className="compact-registers">{thread.registers.slice(0, 13).map((value, index) => <i key={index} title={`R${index}=${value}`}>{value || '·'}</i>)}</div></button>;
}

function Registers({ thread }: { thread: ThreadState }) {
  return <div className="register-file"><h3>Lane {thread.threadId} Registers</h3><div>{thread.registers.map((value, index) => <span className={index >= 13 ? 'builtin' : value ? 'nonzero' : ''} key={index}><b>{index === 13 ? 'BID' : index === 14 ? 'BDM' : index === 15 ? 'TID' : `R${index}`}</b> {value}</span>)}</div><p>NZP: {thread.nzp.toString(2).padStart(3, '0')}</p></div>;
}

function Memory({ title, memory }: { title: string; memory: number[] }) {
  return <div className="memory-panel"><h3>{title}</h3><div className="memory">{memory.map((value, address) => <span className={value ? 'nonzero' : ''} key={address} title={`[${address}]=${value}`}>{value}</span>)}</div></div>;
}
