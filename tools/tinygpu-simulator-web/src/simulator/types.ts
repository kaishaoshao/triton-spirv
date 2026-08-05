/** 教程 TinyGPU 网页仿真器的 16-bit ISA 数据模型。 */
export interface Instruction { addr: number; hex: string; asm: string; bits: string; }
export enum Opcode { NOP=0, BRNZP=1, CMP=2, ADD=3, SUB=4, MUL=5, DIV=6, LDR=7, STR=8, CONST=9, SLDR=10, SSTR=11, BAR=12, RET=15 }
export const OPCODE_NAMES: Record<number,string> = { 0:'NOP',1:'BRnzp',2:'CMP',3:'ADD',4:'SUB',5:'MUL',6:'DIV',7:'LDR',8:'STR',9:'CONST',10:'SLDR',11:'SSTR',12:'BAR',15:'RET' };

/** 后端导出的“编译流水线”文件。值来自 CompiledKernel.asm，而不是浏览器重新编译。 */
export interface CompilationTrace {
  version?: number;
  source?: { script?: string; kernel?: string } | string;
  target?: string;
  stages: Partial<Record<'ttir' | 'ttgir' | 'tinygpuir' | 'tinyasm' | 'tinybin', string>>;
  tinybin?: string;
}

export type PipelineStage = 'FETCH'|'DECODE'|'REQUEST'|'WAIT'|'EXECUTE'|'UPDATE'|'BARRIER'|'DONE';
export interface ThreadState { threadId:number; blockId:number; pc:number; registers:number[]; nzp:number; stage:PipelineStage; done:boolean; currentInstruction:string; divergent:boolean; }
export interface SimulationState { cycle:number; threads:ThreadState[]; memory:number[]; sharedMemory:number[]; currentBlock:number; totalBlocks:number; }
