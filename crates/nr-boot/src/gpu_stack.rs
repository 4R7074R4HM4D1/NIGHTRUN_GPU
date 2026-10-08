//! Stage 7: UEFI owns all 16 decode layers. CUDA process is a temporary driver.
//! One PCI RUN per token, GPU-resident hidden state through all 336 kernels.
extern crate alloc;
use alloc::boxed::Box;
use core::arch::asm;
use core::ptr::{read_volatile,write_volatile};
use core::sync::atomic::{fence,Ordering,AtomicU32,AtomicUsize};
use core::time::Duration;
use nr_model::infer::{InferCtx,StackBackend,QMat};
use nr_gpu::stack_plan;
use nr_gpu::GpuCommand;
#[path = "gpu_kernel_image.rs"] mod gpu_kernel_image; // NIGHTRUN_STAGE9A_UEFI_OWNED_CUBIN_V1
use uefi::boot;

// Reuse Stage6B's shared request region. Stage7 or Stage6B attaches, not both.
const REGION:usize=0xE000_0000;
const CAP:usize=0x0100_0000;
const MAGIC:u32=0x364c_504e;
const STATUS:usize=4; const SEQ:usize=8; const ACK:usize=12;
const ERROR:usize=16; const OP:usize=20; const POS:usize=24;
const CTX:usize=28; const DIM:usize=32; const FFN:usize=36;
const HD:usize=40; const HEADS:usize=44; const KV_HEADS:usize=48;
const KV_DIM:usize=52; const EPS:usize=56; const RESIDUAL:usize=60;
const ATTN_SCALE:usize=64; const SERVICE_US:usize=68; const GPU_US:usize=72;
const SYNC_START:usize=76; const SYNC_ROWS:usize=80;
const CMD_COUNT:usize=84; const CMD_ABI:usize=88; const SYNC_LAYER:usize=96;
const CMD_DATA:usize=0x9000; const FREQ:usize=0x1000; const INPUT:usize=0x2000;
// 16 layers x 512 fp16 = 16 KiB of per-token K and 16 KiB of V.
const OUT_K:usize=0x5000; const OUT_V:usize=0xD000; const DATA:usize=0x20000;
const READY:u32=0; const REQUEST:u32=1; const DONE:u32=2; const FAILED:u32=3;
const INIT:u32=1; const SYNC:u32=2; const RUN:u32=3; const DIAG:u32=4; const SNAPSHOT_KV:u32=5;
// Test layer index is chosen by UEFI, never by the CUDA executor.
const DIAG_LAYER:usize=4;
const DIAG_SELECTOR:usize=100;const DIAG_VALID:usize=104;
const DIAG_DATA:usize=0x0060_0000;const DIAG_SLOT:usize=0x10000;
const LAYERS:usize=16;
// Stage 8B protocol. All offsets are inside the 16 MiB ivshmem region.
// 16 * 2048 FP32 embeddings; 16 layers * 16 tokens * 512 FP16 per K/V.
const BATCH_COUNT:usize=108;
const BATCH_INPUT:usize=0x100000;
const BATCH_KEYS:usize=0x180000;
const BATCH_VALS:usize=0x1c0000;
const RUN_BATCH:u32=6;
// NIGHTRUN_STAGE8C_GPU_FINAL_V1. Protocol remains pointer-free.
const RUN_FINAL:u32=7; const FINAL_VOCAB:usize=112;
// NIGHTRUN_STAGE8D_GPU_EMBED_Q8_V1: UEFI-authored Q8 embedding lookup command.
const RUN_TOKEN_EMBED:u32=8;const RUN_BATCH_EMBED:u32=9;
// NIGHTRUN_STAGE8E_GPU_SAMPLER_V1: UEFI-owned sampling request protocol.
const RUN_SAMPLE:u32=10;
// NIGHTRUN_STAGE9A_UEFI_OWNED_CUBIN_V1 - offline NVIDIA ELF images owned by UEFI.
const LOAD_CUBIN:u32=11;
const MODULE_ID:usize=132; const MODULE_BYTES:usize=136; const MODULE_CRC:usize=140;
const MODULE_DATA:usize=0x800000; const MODULE_MAX:usize=0x400000;
// NIGHTRUN_STAGE9C_GENERIC_GPU_LAUNCH_V1: UEFI's pointer-free kernel launch descriptors.
const RUNTIME_CAP:usize=0x3a02a4;const RUNTIME_ABI:usize=0x3a02a0;
const RUNTIME_CAP_MAGIC:u32=0x314c524e; // "NRL1" (little endian)
const DIAG_WINDOW_ORDINAL:usize=0x3a02a8; // decode-relative diagnostic number
const RUNTIME_DESCS:usize=0x003a_0000;
const RUNTIME_DESC_BYTES:usize=224;
fn upload_runtime_launch(base:usize,idx:usize,desc:&nr_gpu::runtime_launch::KernelLaunch) {
    let off=RUNTIME_DESCS+idx*RUNTIME_DESC_BYTES;
    put(base,off,desc.module_id);
    for i in 0..3 {put(base,off+4+i*4,desc.grid[i]);put(base,off+16+i*4,desc.block[i]);}
    put(base,off+28,desc.shared_bytes);put(base,off+32,desc.arg_count);
    put(base,off+36,desc.reserved);
    for i in 0..12 {
        let s=i*4;
        let v=u32::from_le_bytes([desc.symbol[s],desc.symbol[s+1],desc.symbol[s+2],desc.symbol[s+3]]);
        put(base,off+40+s,v);
    }
    put(base,off+88,desc.padding[0]);put(base,off+92,desc.padding[1]);
    for i in 0..8 {
        let arg=desc.args[i]; let at=off+96+i*16;
        put(base,at,arg.kind);put(base,at+4,arg.value);
        put(base,at+8,arg.span);put(base,at+12,arg.offset);
    }
}

const SAMPLE_TEMP:usize=116;const SAMPLE_TOPP:usize=120;
const SAMPLE_SCALE:usize=124;const SAMPLE_EXPECT:usize=128;
const SAMPLE_OUT:usize=0x390000;
static SAMPLE_BASE:AtomicUsize=AtomicUsize::new(0);
static SAMPLE_VOCAB:AtomicU32=AtomicU32::new(0);
static SAMPLE_SCALE_BITS:AtomicU32=AtomicU32::new(0);
static COMMAND_SEQ:AtomicU32=AtomicU32::new(0);
static SAMPLE_READY_SEQ:AtomicU32=AtomicU32::new(0);
static SAMPLE_CALLS:AtomicU32=AtomicU32::new(0);
const EMBED_CMDS:usize=0x380000;
const FINAL_LOGITS:usize=0x240000; const FINAL_CMDS:usize=0x360000;
const FINAL_MAX_VOCAB:usize=262144;
#[inline]
fn pci_read(bus:u8,dev:u8,fun:u8,reg:u8)->u32 {
    let a=0x8000_0000 | ((bus as u32)<<16) | ((dev as u32)<<11)
        | ((fun as u32)<<8) | ((reg as u32)&0xfc);
    let v:u32;
    unsafe{
        asm!("out dx, eax",in("dx")0xcf8u16,in("eax")a,options(nostack,preserves_flags));
        asm!("in eax, dx",in("dx")0xcfcu16,out("eax")v,options(nostack,preserves_flags));
    }
    v
}
fn locate()->Option<usize>{
    for bi in 0..=255u16 {for dev in 0..32u8 {for fun in 0..8u8 {
        let bus=bi as u8;
        if pci_read(bus,dev,fun,0)!=0x1110_1af4 || pci_read(bus,dev,fun,4)&2==0 {continue;}
        let low=pci_read(bus,dev,fun,0x18);
        if low&1!=0 {continue;}
        let hi=match (low>>1)&3 {2=>pci_read(bus,dev,fun,0x1c),0=>0,_=>continue};
        let phys=((hi as u64)<<32)|((low&!0xf) as u64);
        if phys==0 || phys>usize::MAX as u64-(REGION+CAP) as u64 {continue;}
        let base=phys as usize+REGION;
        if unsafe{read_volatile(base as *const u32)}==MAGIC {
            serial_println!("[gpu-stack] PCI {:02x}:{:02x}.{} BAR2={:#x}",bus,dev,fun,phys);
            return Some(base);
        }
    }}}
    None
}
#[inline] fn get(base:usize,off:usize)->u32 {let r=unsafe{read_volatile((base+off) as *const u32)};fence(Ordering::Acquire);r}
#[inline] fn put(base:usize,off:usize,v:u32) {unsafe{write_volatile((base+off) as *mut u32,v)};fence(Ordering::Release);}
fn upload_command(base:usize,i:usize,c:&GpuCommand) {
    let off=CMD_DATA+i*48;
    put(base,off,c.opcode);
    put(base,off+4,c.version as u32|((c.flags as u32)<<16));
    put(base,off+8,c.dst);put(base,off+12,c.src0);
    put(base,off+16,c.src1);put(base,off+20,c.aux);
    put(base,off+24,c.rows);put(base,off+28,c.cols);
    put(base,off+32,c.scalar0_bits);put(base,off+36,c.scalar1_bits);
    put(base,off+40,c.reserved[0]);put(base,off+44,c.reserved[1]);
}
fn upload_command_at(base:usize,off:usize,c:&GpuCommand) {
    put(base,off,c.opcode);
    put(base,off+4,c.version as u32|((c.flags as u32)<<16));
    put(base,off+8,c.dst);put(base,off+12,c.src0);
    put(base,off+16,c.src1);put(base,off+20,c.aux);
    put(base,off+24,c.rows);put(base,off+28,c.cols);
    put(base,off+32,c.scalar0_bits);put(base,off+36,c.scalar1_bits);
    put(base,off+40,c.reserved[0]);put(base,off+44,c.reserved[1]);
}
fn write_f32(base:usize,off:usize,x:&[f32]) {for (i,&v) in x.iter().enumerate(){put(base,off+4*i,v.to_bits());}}
fn write_u16(base:usize,off:usize,x:&[u16]) {for (i,&v) in x.iter().enumerate(){unsafe{write_volatile((base+off+2*i) as *mut u16,v)};}fence(Ordering::Release);}

// GPU_KV_HISTORY_PROBE_V1: IEEE binary16 to binary32, no external crate or std.
fn kv_probe_f16_to_f32(h: u16) -> f32 {
    let sign = (u32::from(h & 0x8000)) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let fraction = u32::from(h & 0x03ff);
    let bits = if exp == 0 {
        if fraction == 0 { sign }
        else {
            let mut m = fraction;
            let mut e = 113u32; // 127 - 14; normalize subnormal mantissa
            while (m & 0x400) == 0 { m <<= 1; e -= 1; }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (fraction << 13)
    } else {
        sign | ((exp + 112) << 23) | (fraction << 13)
    };
    f32::from_bits(bits)
}

struct KvHistoryStats {
    mismatch: usize,
    first_pos: usize,
    first_element: usize,
    first_cpu: u16,
    first_gpu: u16,
    max_abs: f32,
    max_pos: usize,
    max_element: usize,
}
impl KvHistoryStats {
    const fn new() -> Self {
        Self { mismatch: 0, first_pos: 0, first_element: 0,
            first_cpu: 0, first_gpu: 0, max_abs: 0.0,
            max_pos: 0, max_element: 0 }
    }
    fn record(&mut self, cpu: u16, gpu: u16, pos: usize, col: usize) {
        if cpu != gpu {
            if self.mismatch == 0 {
                self.first_pos = pos; self.first_element = col;
                self.first_cpu = cpu; self.first_gpu = gpu;
            }
            self.mismatch += 1;
        }
        let a = kv_probe_f16_to_f32(cpu);
        let b = kv_probe_f16_to_f32(gpu);
        let diff = if a.is_finite() && b.is_finite() { (a-b).abs() }
                   else if cpu == gpu { 0.0 } else { f32::INFINITY };
        if diff > self.max_abs {
            self.max_abs = diff;
            self.max_pos = pos;
            self.max_element = col;
        }
    }
}

// Converts a Q8_0 FP16 scale exactly (including subnormal half values).
// Diagnostic-only FP32 fused multiply-add on x86-64 UEFI (no_std).
// Invoke only when the virtual CPU exposes FMA (QEMU: -cpu host).
#[target_feature(enable = "fma")]
unsafe fn gpu_gemv_fma_hw(a:f32,b:f32,c:f32)->f32 {
    use core::arch::x86_64::{_mm_set_ss,_mm_fmadd_ss,_mm_cvtss_f32};
    _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a),_mm_set_ss(b),_mm_set_ss(c)))
}

fn gemv_half_to_f32(bits:u16)->f32 {
    let sign=((bits as u32)&0x8000)<<16;
    let e=((bits>>10)&31) as u32;
    let m=(bits&0x03ff) as u32;
    if e==0 {
        let x=(m as f32)*f32::from_bits(0x3380_0000); // 2^-24
        if sign!=0 {-x} else {x}
    } else if e==31 {
        f32::from_bits(sign|0x7f80_0000|(m<<13))
    } else {
        f32::from_bits(sign|((e+112)<<23)|(m<<13))
    }
}
struct Bridge {base:usize,seq:u32,next_pos:usize,ctx:usize,kv:usize,dim:usize,calls:u32,trace_base_calls:u32,last_prefill_batch:bool,final_vocab:usize,embed_request:Option<(u32,f32)>,gpu_embed_batch:bool}
impl Bridge {
    // NIGHTRUN_STAGE9B_DECODE_TRACE_WINDOW_V1
    // Decode-only trace window; global calls also includes prompt prefill.
    #[inline]
    fn in_decode_trace_window(&self) -> bool {
        let n = self.calls.saturating_sub(self.trace_base_calls);
        n >= 1 && n <= 3
    }
    // Used at the beginning of compute(), BEFORE calls increments.
    #[inline]
    fn before_decode_trace_window(&self) -> bool {
        self.calls.saturating_sub(self.trace_base_calls) < 3
    }

    // NIGHTRUN_STAGE9A_UEFI_OWNED_CUBIN_V1: Upload compiled kernels from NightRun's own EFI image.
    // The temporary bridge only calls CUDA Driver API cuModuleLoadData.
    fn load_uefi_module(&mut self, id:u32, image:&[u8])->bool {
        if !(1..=3).contains(&id) || image.len()<64 || image.len()>MODULE_MAX ||
           image.get(..4)!=Some(&b"\x7fELF"[..]) {return false;}
        put(self.base,MODULE_ID,id);
        put(self.base,MODULE_BYTES,image.len() as u32);
        let mut crc=0xffff_ffffu32;
        for &byte in image {
            crc^=byte as u32;
            for _ in 0..8 {crc=(crc>>1)^((0u32.wrapping_sub(crc&1))&0xedb8_8320);}
        }
        put(self.base,MODULE_CRC,!crc);
        for (i,&byte) in image.iter().enumerate(){
            unsafe { write_volatile((self.base+MODULE_DATA+i) as *mut u8,byte); }
        }
        fence(Ordering::SeqCst);
        if !self.send(LOAD_CUBIN){return false;}
        serial_println!("[gpu-kernels] UEFI module={} bytes={} crc={:#010x} loaded",id,image.len(),!crc);
        true
    }
    fn send(&mut self,op:u32)->bool {
        if get(self.base,STATUS)!=READY {return false;}
        self.seq=COMMAND_SEQ.fetch_add(1,Ordering::AcqRel).wrapping_add(1);
        put(self.base,OP,op);put(self.base,SEQ,self.seq);
        fence(Ordering::Release);put(self.base,STATUS,REQUEST);fence(Ordering::SeqCst);
        for _ in 0..30000 {
            let s=get(self.base,STATUS);
            if s==FAILED {
                serial_println!("[gpu-stack] host FAILED op={} err={}",op,get(self.base,ERROR));
                return false;
            }
            if s==DONE && get(self.base,ACK)==self.seq {
                put(self.base,STATUS,READY);fence(Ordering::SeqCst);return true;
            }
            boot::stall(Duration::from_micros(100));
        }
        serial_println!("[gpu-stack] timeout op={} seq={}; CPU fallback",op,self.seq);
        false
    }
    // GPU_KV_HISTORY_PROBE_V1: compare the full historical K/V contents.
    // Only NightRun UEFI interprets the results. The temporary CUDA executor
    // does a read-only GPU -> PCI-buffer transfer; it does NOT compare tensors.
    fn inspect_kv_history(&mut self, layer: usize, pos: usize,
                          keys: &[u16], vals: &[u16]) -> bool {
        if layer >= LAYERS || pos > self.ctx { return false; }
        let mut ks = KvHistoryStats::new();
        let mut vs = KvHistoryStats::new();
        let mut begin = 0usize;
        const MAX_ROWS: usize = 256; // 256 * 512 * 2 bytes * K/V = 512 KiB.
        while begin < pos {
            let rows = (pos - begin).min(MAX_ROWS);
            let words = rows * self.kv;
            let bytes = words * 2;
            // The same request offsets as SYNC, but SNAPSHOT_KV never writes VRAM.
            put(self.base, SYNC_LAYER, layer as u32);
            put(self.base, SYNC_START, begin as u32);
            put(self.base, SYNC_ROWS, rows as u32);
            if !self.send(SNAPSHOT_KV) {
                serial_println!("[gpu-kv-history] snapshot unavailable layer={} pos={} begin={}",
                    layer,pos,begin);
                return false;
            }
            let base_idx = (layer * self.ctx + begin) * self.kv;
            for i in 0..words {
                let row = begin + i / self.kv;
                let col = i % self.kv;
                let got_k = unsafe { read_volatile((self.base+DATA+i*2) as *const u16) };
                let got_v = unsafe { read_volatile((self.base+DATA+bytes+i*2) as *const u16) };
                ks.record(keys[base_idx+i],got_k,row,col);
                vs.record(vals[base_idx+i],got_v,row,col);
            }
            begin += rows;
        }
        serial_println!("[gpu-kv-history] layer={} before_pos={} rows={} keys_mismatch={} values_mismatch={} total_each={}",
            layer,pos,pos,ks.mismatch,vs.mismatch,pos*self.kv);
        serial_println!("[gpu-kv-history] layer={} keys first_pos={} first_col={} cpu_bits={:#06x} gpu_bits={:#06x} max_abs={} max_pos={} max_col={}",
            layer,ks.first_pos,ks.first_element,ks.first_cpu,ks.first_gpu,
            ks.max_abs,ks.max_pos,ks.max_element);
        serial_println!("[gpu-kv-history] layer={} values first_pos={} first_col={} cpu_bits={:#06x} gpu_bits={:#06x} max_abs={} max_pos={} max_col={}",
            layer,vs.first_pos,vs.first_element,vs.first_cpu,vs.first_gpu,
            vs.max_abs,vs.max_pos,vs.max_element);
        true
    }
    fn sync_history(&mut self,keys:&[u16],vals:&[u16],position:usize)->bool {
        let rows_per_chunk=(1_048_576/(self.kv*4)).max(1);
        for layer in 0..LAYERS {
            let mut start=0usize;
            while start<position {
                let rows=(position-start).min(rows_per_chunk);
                let a=(layer*self.ctx+start)*self.kv;
                let b=a+rows*self.kv;
                write_u16(self.base,DATA,&keys[a..b]);
                write_u16(self.base,DATA+rows*self.kv*2,&vals[a..b]);
                put(self.base,SYNC_LAYER,layer as u32);
                put(self.base,SYNC_START,start as u32);put(self.base,SYNC_ROWS,rows as u32);
                if !self.send(SYNC){return false;}
                start+=rows;
            }
            if position==0 {
                put(self.base,SYNC_LAYER,layer as u32);
                put(self.base,SYNC_START,0);put(self.base,SYNC_ROWS,0);
                if !self.send(SYNC){return false;}
            }
        }
        self.next_pos=position;true
    }
}
impl StackBackend for Bridge {
    fn compute_token_q8(&mut self,token:u32,vocab:usize,scale:f32,x:&mut[f32],
        pos:usize,keys:&mut[u16],vals:&mut[u16])->bool {
        if vocab!=self.final_vocab || !scale.is_finite() || token as usize>=vocab ||
            self.embed_request.is_some() {return false;}
        self.embed_request=Some((token,scale));
        let ok=self.compute(x,pos,keys,vals);
        self.embed_request=None;
        if ok && (self.calls<=3 || self.calls%64==0) {
            serial_println!("[gpu-embed] mode=DECODE pos={} token={} resident=1",pos,token);
        }
        ok
    }
    fn compute_prefill_tokens_q8(&mut self,tokens:&[u32],vocab:usize,scale:f32,
        start:usize,last:&mut[f32],keys:&mut[u16],vals:&mut[u16])->bool {
        if tokens.is_empty() || tokens.len()>16 || vocab!=self.final_vocab ||
            !scale.is_finite() || self.gpu_embed_batch {return false;}
        for (i,&token) in tokens.iter().enumerate() {
            let Ok(cmd)=GpuCommand::q8_embedding_row(token,vocab as u32,self.dim as u32,scale)
                else {return false;};
            upload_command_at(self.base,EMBED_CMDS+i*48,&cmd);
        }
        self.gpu_embed_batch=true;
        let ok=self.compute_prefill_batch(&[],start,tokens.len(),last,keys,vals);
        self.gpu_embed_batch=false;
        if ok {serial_println!("[gpu-embed] mode=PREFILL_BATCH start={} count={} resident=1",
            start,tokens.len());}
        ok
    }

    // NIGHTRUN_STAGE8C_GPU_FINAL_V1: never dispatch model logic from CUDA.
    fn compute_final(&mut self,hidden:&[f32],logits:&mut[f32])->bool {
        SAMPLE_READY_SEQ.store(0,Ordering::Release);
        if self.final_vocab==0 || logits.len()!=self.final_vocab ||
           hidden.len()!=self.dim || !hidden.iter().all(|v|v.is_finite()) {return false;}
        // Explicit copy handles parity replay modifying temporary GPU scratch.
        write_f32(self.base,INPUT,hidden);
        if !self.send(RUN_FINAL){return false;}
        for (i,out) in logits.iter_mut().enumerate() {
            let v=f32::from_bits(get(self.base,FINAL_LOGITS+i*4));
            if !v.is_finite(){return false;}
            *out=v;
        }
        // Bind one GPU candidate request to precisely this successful output head.
        SAMPLE_READY_SEQ.store(self.seq,Ordering::Release);
        if self.calls<=3 || self.calls%64==0 {
            serial_println!("[gpu-final] resident_norm_q8_vocab ACTIVE vocab={} GPU={}us service={}us",
                self.final_vocab,get(self.base,GPU_US),get(self.base,SERVICE_US));
        }
        true
    }

    // NIGHTRUN_STAGE8B_UEFI_16TOKEN_PREFILL_BATCH_V1:
    // The 336-op UEFI command list is already resident on the GPU executor.
    // UEFI supplies every input vector and a bounded contiguous token range.
    fn compute_prefill_batch(&mut self,embeddings:&[f32],start:usize,count:usize,
            last:&mut[f32],keys:&mut[u16],vals:&mut[u16])->bool {
        if count==0 || count>16 || start.checked_add(count).map_or(true,|end|end>self.ctx) ||
           (!self.gpu_embed_batch && embeddings.len()!=count*self.dim) || last.len()!=self.dim ||
           keys.len()!=LAYERS*self.ctx*self.kv || vals.len()!=keys.len() ||
           (!self.gpu_embed_batch && embeddings.iter().any(|x|!x.is_finite())) {return false;}
        if start!=self.next_pos && !self.sync_history(keys,vals,start){return false;}
        if !self.gpu_embed_batch {write_f32(self.base,BATCH_INPUT,embeddings);}
        put(self.base,POS,start as u32);
        put(self.base,BATCH_COUNT,count as u32);
        let op=if self.gpu_embed_batch {RUN_BATCH_EMBED} else {RUN_BATCH};
        if !self.send(op){return false;}
        // Validate all device results *before* committing ANY CPU KV row.
        for i in 0..self.dim {
            if !f32::from_bits(get(self.base,INPUT+i*4)).is_finite(){return false;}
        }
        for layer in 0..LAYERS {for t in 0..count {for i in 0..self.kv {
            let off=((layer*count+t)*self.kv+i)*2;
            let k=unsafe{read_volatile((self.base+BATCH_KEYS+off) as *const u16)};
            let v=unsafe{read_volatile((self.base+BATCH_VALS+off) as *const u16)};
            if k&0x7c00==0x7c00 || v&0x7c00==0x7c00 {return false;}
        }}}
        for (i,x) in last.iter_mut().enumerate() {
            *x=f32::from_bits(get(self.base,INPUT+i*4));
        }
        for layer in 0..LAYERS {for t in 0..count {for i in 0..self.kv {
            let src=((layer*count+t)*self.kv+i)*2;
            let dst=(layer*self.ctx+start+t)*self.kv+i;
            keys[dst]=unsafe{read_volatile((self.base+BATCH_KEYS+src) as *const u16)};
            vals[dst]=unsafe{read_volatile((self.base+BATCH_VALS+src) as *const u16)};
        }}}
        self.next_pos=start+count;
        self.calls+=count as u32;
        self.last_prefill_batch=true;
        serial_println!("[gpu-prefill-batch] start={} count={} end={} stack_calls={} GPU={}us service={}us",
            start,count,start+count,self.calls,get(self.base,GPU_US),get(self.base,SERVICE_US));
        true
    }

    fn compute(&mut self,x:&mut[f32],pos:usize,keys:&mut[u16],vals:&mut[u16])->bool {
        if pos>=self.ctx || x.len()!=self.dim || keys.len()!=LAYERS*self.ctx*self.kv ||
           vals.len()!=keys.len() || x.iter().any(|v|!v.is_finite()) {return false;}
        if pos!=self.next_pos && !self.sync_history(keys,vals,pos){return false;}
        // Only the first three live decode attempts, after any CPU-prefill sync,
        // and before RUN mutates this token's GPU KV entries.
        if self.before_decode_trace_window() {
            for li in [2usize,4usize] {
                if !self.inspect_kv_history(li,pos,keys,vals){return false;}
            }
        }
        let op=if let Some((token,scale))=self.embed_request {
            let Ok(cmd)=GpuCommand::q8_embedding_row(
                token,self.final_vocab as u32,self.dim as u32,scale)
                else {return false;};
            upload_command_at(self.base,EMBED_CMDS,&cmd);
            RUN_TOKEN_EMBED
        } else {
            write_f32(self.base,INPUT,x);
            RUN
        };
        put(self.base,POS,pos as u32);
        if !self.send(op){return false;}
        for i in 0..x.len() {if !f32::from_bits(get(self.base,INPUT+i*4)).is_finite(){return false;}}
        for layer in 0..LAYERS {for i in 0..self.kv {
            let off=layer*self.kv+i;
            let k=unsafe{read_volatile((self.base+OUT_K+2*off) as *const u16)};
            let v=unsafe{read_volatile((self.base+OUT_V+2*off) as *const u16)};
            if k&0x7c00==0x7c00 || v&0x7c00==0x7c00 {return false;}
        }}
        // Transactional commit: only after every layer's output was validated.
        for (i,y) in x.iter_mut().enumerate(){*y=f32::from_bits(get(self.base,INPUT+i*4));}
        for layer in 0..LAYERS {for i in 0..self.kv {
            let off=layer*self.kv+i;
            keys[(layer*self.ctx+pos)*self.kv+i]=unsafe{read_volatile((self.base+OUT_K+2*off) as *const u16)};
            vals[(layer*self.ctx+pos)*self.kv+i]=unsafe{read_volatile((self.base+OUT_V+2*off) as *const u16)};
        }}
        self.next_pos=pos+1;self.calls+=1;
        if self.calls<=3 || self.calls%64==0 {
            serial_println!("[gpu-stack] 16-layer GPU #{} pos={} GPU={}us service={}us",
                self.calls,pos,get(self.base,GPU_US),get(self.base,SERVICE_US));
        }
        true
    }
    // NIGHTRUN_STAGE8A_UEFI_GPU_PREFILL_V1: observability only.
    fn on_prefill_complete(&mut self,start:usize,tokens:usize) {
        self.trace_base_calls = self.calls;
        serial_println!("[gpu-prefill] mode={} start={} tokens={} end={} stack_calls={}",
            if self.last_prefill_batch {"GPU_BATCH16"} else {"GPU_SEQUENTIAL"},
            start,tokens,start+tokens,self.calls);
        self.last_prefill_batch=false;
    }
    fn on_prefill_failure(&mut self,pos:usize,gpu_tokens:usize) {
        serial_println!("[gpu-prefill] mode=GPU_SEQUENTIAL HARD_FALLBACK pos={} gpu_tokens={} reason=GPU_EXECUTION_OR_PROTOCOL_FAILURE",
            pos,gpu_tokens);
    }
    fn on_parity_check(&mut self,ok:bool,error:f32) {
        serial_println!("[gpu-stack] 16-layer CPU parity {} max_abs_error={}",
            if ok {"PASS"} else {"FAIL; disabling GPU stack"},error);
    }
    fn trace_begin_layer(&mut self,layer:usize,pos:usize,input:&[f32],
                         keys:&[u16],vals:&[u16])->bool {
        if !self.in_decode_trace_window() || layer!=DIAG_LAYER ||
           pos>=self.ctx || input.len()!=self.dim ||
           keys.len()!=self.ctx*self.kv || vals.len()!=keys.len() ||
           pos*self.kv*4 > DIAG_DATA-DATA {return false;}
        // Give the diagnostic replay the EXACT CPU reference KV history.
        // The temporary driver saves and restores the original GPU cache.
        write_u16(self.base,DATA,&keys[..pos*self.kv]);
        write_u16(self.base,DATA+pos*self.kv*2,&vals[..pos*self.kv]);
        put(self.base,DIAG_SELECTOR,layer as u32);
        put(self.base,DIAG_VALID,0);
        put(self.base,DIAG_WINDOW_ORDINAL,
            self.calls.saturating_sub(self.trace_base_calls));
        write_f32(self.base,INPUT,input);
        put(self.base,POS,pos as u32);
        let ok=self.send(DIAG) && get(self.base,DIAG_VALID)==1;
        put(self.base,DIAG_WINDOW_ORDINAL,0);
        serial_println!("[gpu-opdiag] same-input layer={} pos={} {}",
            layer,pos,if ok {"READY"} else {"UNAVAILABLE"});
        ok
    }
    // GPU_Q8_BYTE_PROBE_V1: exact packed activation Q8 parity.
    // GPU_DYNAMIC_Q8_ORIGIN_V1: no inference changes, read only existing diagnostic snapshots.
    fn trace_compare_q8(&mut self,layer:usize,stage:usize,
        cpu:&[u8],pre_quant:&[f32]) {
        if !self.in_decode_trace_window() || layer!=DIAG_LAYER ||
           !(stage==1 || stage==10 || stage==14 || stage==18) ||
           get(self.base,DIAG_VALID)!=1 || cpu.is_empty() ||
           cpu.len()>DIAG_SLOT || cpu.len()%34!=0 {return;}
        let mut scale_blocks=0usize;
        let mut quant_blocks=0usize;
        let mut quant_bytes=0usize;
        let mut first_off=0usize;
        let mut first_cpu=0u8;
        let mut first_gpu=0u8;
        let mut first_seen=false;
        let off=DIAG_DATA+stage*DIAG_SLOT;
        for (block,b) in cpu.chunks_exact(34).enumerate() {
            let mut bad_scale=false;
            let mut bad_q=false;
            for (j,&expected) in b.iter().enumerate() {
                let actual=unsafe{read_volatile((self.base+off+block*34+j) as *const u8)};
                if actual!=expected {
                    if !first_seen {
                        first_seen=true;first_off=block*34+j;
                        first_cpu=expected;first_gpu=actual;
                    }
                    if j<2 {bad_scale=true;} else {bad_q=true;quant_bytes+=1;}
                }
            }
            if bad_scale {scale_blocks+=1;}
            if bad_q {quant_blocks+=1;}
        }
        serial_println!("[gpu-q8-bytes] layer={} stage={} blocks={} scale_blocks={} q_blocks={} q_bytes={} first_offset={} cpu_byte={:#04x} gpu_byte={:#04x}",
            layer,stage,cpu.len()/34,scale_blocks,quant_blocks,quant_bytes,
            first_off,first_cpu,first_gpu);
        // GPU_DYNAMIC_Q8_ORIGIN_V1: inspect the FIRST changed scale (or Q8
        // payload if scales agree), with matching pre-quantization FP32 values.
        if (scale_blocks!=0 || quant_blocks!=0) &&
           pre_quant.len()==(cpu.len()/34)*32 {
            let mut selected=first_off/34;
            let mut scale_changed=false;
            // Prioritize the earliest changed FP16 scale over payload changes.
            for block in 0..cpu.len()/34 {
                let a=block*34;
                let cp=u16::from_le_bytes([cpu[a],cpu[a+1]]);
                let gl=unsafe{read_volatile((self.base+off+a) as *const u8)};
                let gh=unsafe{read_volatile((self.base+off+a+1) as *const u8)};
                if cp!=u16::from_le_bytes([gl,gh]) {
                    selected=block;scale_changed=true;break;
                }
            }
            let a=selected*34;
            let cscale=u16::from_le_bytes([cpu[a],cpu[a+1]]);
            let gl=unsafe{read_volatile((self.base+off+a) as *const u8)};
            let gh=unsafe{read_volatile((self.base+off+a+1) as *const u8)};
            let gscale=u16::from_le_bytes([gl,gh]);
            let prev=match stage {1=>0usize,10=>9,14=>13,18=>17,_=>return};
            let foff=DIAG_DATA+prev*DIAG_SLOT;
            let mut cmax=0f32;let mut gmax=0f32;
            let mut cidx=0usize;let mut gidx=0usize;
            let mut largest_delta=0f32;let mut input_delta_count=0usize;
            for j in 0..32usize {
                let idx=selected*32+j;
                let c=pre_quant[idx];
                let g=f32::from_bits(get(self.base,foff+idx*4));
                if !c.is_finite() || !g.is_finite() {return;}
                if c.abs()>cmax {cmax=c.abs();cidx=idx;}
                if g.abs()>gmax {gmax=g.abs();gidx=idx;}
                let delta=(c-g).abs();
                if delta>largest_delta {largest_delta=delta;}
                if c.to_bits()!=g.to_bits() {input_delta_count+=1;}
            }
            serial_println!("[gpu-q8-origin] layer={} stage={} block={} kind={} cpu_scale_bits={:#06x} gpu_scale_bits={:#06x} cpu_scale={} gpu_scale={} cpu_max={} gpu_max={} cpu_max_bits={:#010x} gpu_max_bits={:#010x} cpu_max_idx={} gpu_max_idx={} pre_scale_cpu={} pre_scale_gpu={} source_float_mismatches={} max_float_delta={}",
                layer,stage,selected,if scale_changed {"scale"} else {"payload"},
                cscale,gscale,gemv_half_to_f32(cscale),gemv_half_to_f32(gscale),
                cmax,gmax,cmax.to_bits(),gmax.to_bits(),cidx,gidx,
                cmax/127f32,gmax/127f32,input_delta_count,largest_delta);
            if !scale_changed && first_off%34>=2 {
                let idx=(first_off/34)*32+(first_off%34-2);
                let cp=pre_quant[idx];
                let gp=f32::from_bits(get(self.base,foff+idx*4));
                serial_println!("[gpu-q8-origin-value] stage={} idx={} cpu_fp32={} gpu_fp32={} cpu_bits={:#010x} gpu_bits={:#010x} cpu_quant_byte={:#04x} gpu_quant_byte={:#04x}",
                    stage,idx,cp,gp,cp.to_bits(),gp.to_bits(),first_cpu,first_gpu);
            }
        }
        // GPU_Q8_SCALE_BOUNDARY_DIAG_V1: exact half-scale bits in altered Q8 blocks.
        // Read-only, never changes the active model, weights, or cache.
        if stage==1 && layer==4 && self.in_decode_trace_window() {
            let mut shown=0usize;
            for (block,b) in cpu.chunks_exact(34).enumerate() {
                let qoff=off+block*34;
                let lo=unsafe{read_volatile((self.base+qoff) as *const u8)};
                let hi=unsafe{read_volatile((self.base+qoff+1) as *const u8)};
                let cp=u16::from_le_bytes([b[0],b[1]]);
                let gp=u16::from_le_bytes([lo,hi]);
                if cp!=gp {
                    serial_println!("[gpu-q8-boundary] layer={} stage=1 block={} cpu_scale_bits={:#06x} gpu_scale_bits={:#06x} cpu_scale={} gpu_scale={} scale_delta={}",
                        layer,block,cp,gp,gemv_half_to_f32(cp),
                        gemv_half_to_f32(gp),
                        (gemv_half_to_f32(cp)-gemv_half_to_f32(gp)).abs());
                    shown+=1;
                    if shown==4 {break;}
                }
            }
        }

    }
    // GPU_GEMV_SCALAR_REFERENCE_V1: compute exact Q8 block products in UEFI.
    // This is a diagnostic oracle, not a replacement inference engine.
    fn trace_compare_gemv(&mut self,layer:usize,stage:usize,
        weights:&[u8],activation:&[u8],cpu:&[f32]) {
        if !self.in_decode_trace_window() || layer!=DIAG_LAYER ||
           !(stage==11 || stage==15 || stage==16 || stage==19) ||
           get(self.base,DIAG_VALID)!=1 || cpu.is_empty() || cpu.len()>8192 ||
           activation.is_empty() || activation.len()%34!=0 ||
           weights.len()!=cpu.len()*activation.len() {return;}
        let slot=DIAG_DATA+stage*DIAG_SLOT;
        let mut row=0usize;let mut maxerr=-1f32;let mut gpu=0f32;
        for (i,&c) in cpu.iter().enumerate() {
            let g=f32::from_bits(get(self.base,slot+i*4));
            if !g.is_finite() {return;}
            let err=(g-c).abs();
            if err>maxerr {maxerr=err;row=i;gpu=g;}
        }
        // Q8 block: [little-endian half scale (2), 32 signed integer bytes].
        let blocks=activation.len()/34;
        let row_weights=&weights[row*activation.len()..(row+1)*activation.len()];
        let mut serial=0f32; let mut fma=0f32;let mut paired=0f32;
        let mut wide=0f64;
        let mut quad=[0f32;4];
        for b in 0..blocks {
            let a=&activation[b*34..b*34+34];
            let w=&row_weights[b*34..b*34+34];
            let sx=gemv_half_to_f32(u16::from_le_bytes([a[0],a[1]]));
            let sw=gemv_half_to_f32(u16::from_le_bytes([w[0],w[1]]));
            let mut dot=0i32;
            for k in 0..32 {
                dot+=(a[k+2] as i8 as i32)*(w[k+2] as i8 as i32);
            }
            let d=dot as f32;
            let p=(d*sw)*sx;
            serial+=p;
            fma=unsafe { gpu_gemv_fma_hw(d*sw,sx,fma) };
            paired+=d*(sw*sx);
            wide+=(dot as f64)*(sw as f64)*(sx as f64);
            quad[b%4]+=p;
        }
        let quad_sum=((quad[0]+quad[1])+quad[2])+quad[3];
        serial_println!("[gpu-gemv] layer={} stage={} row={} blocks={} cpu={} gpu={} serial={} fma={} paired={} quad={} fp64={} cpu_gpu_err={} gpu_serial_err={} cpu_serial_err={}",
            layer,stage,row,blocks,cpu[row],gpu,serial,fma,paired,quad_sum,wide,
            maxerr,(gpu-serial).abs(),(cpu[row]-serial).abs());
    }
    fn trace_compare_stage(&mut self,layer:usize,stage:usize,cpu:&[f32]) {
        if !self.in_decode_trace_window() || layer!=DIAG_LAYER || stage>=21 ||
           get(self.base,DIAG_VALID)!=1 || cpu.len()>8192 {return;}
        let off=DIAG_DATA+stage*DIAG_SLOT;
        let mut max=0.0f32;let mut idx=0usize;let mut sample_cpu=0.0f32;
        let mut sample_gpu=0.0f32;let mut strict=0usize;let mut broad=0usize;
        for (i,&expected) in cpu.iter().enumerate() {
            let actual=f32::from_bits(get(self.base,off+i*4));
            let err=if actual.is_finite() {(expected-actual).abs()} else {f32::INFINITY};
            if err>max {max=err;idx=i;sample_cpu=expected;sample_gpu=actual;}
            if err>0.001+0.002*expected.abs() {strict+=1;}
            if err>0.05+0.01*expected.abs() {broad+=1;}
        }
        serial_println!("[gpu-opdiag] layer={} stage={} max_abs={} idx={} cpu={} gpu={} strict={} broad={}",
            layer,stage,max,idx,sample_cpu,sample_gpu,strict,broad);
        // GPU_Q8_SCALE_BOUNDARY_DIAG_V1: exact RMSNorm pre-quantization
        // maxima in Q8 block 22, whose FP16 scale differed at pos 63.
        if layer==4 && stage==0 && self.calls<=3 && cpu.len()>=736 {
            let start=22usize*32;
            let mut cm=0f32;let mut gm=0f32;
            let mut ci=0usize;let mut gi=0usize;
            for j in 0..32usize {
                let cv=cpu[start+j].abs();
                let gv=f32::from_bits(get(self.base,off+(start+j)*4)).abs();
                if cv>cm {cm=cv;ci=j;}
                if gv>gm {gm=gv;gi=j;}
            }
            serial_println!("[gpu-q8-boundary] layer=4 stage=0 block=22 cpu_max={} gpu_max={} cpu_max_bits={:#010x} gpu_max_bits={:#010x} cpu_index={} gpu_index={} cpu_pre_scale={} gpu_pre_scale={}",
                cm,gm,cm.to_bits(),gm.to_bits(),start+ci,start+gi,cm/127f32,gm/127f32);
        }

    }
    fn trace_compare_cache(&mut self,layer:usize,stage:usize,cpu:&[u16]) {
        if !self.in_decode_trace_window() || layer!=DIAG_LAYER ||
           !(stage==7 || stage==8) || get(self.base,DIAG_VALID)!=1 ||
           cpu.len()!=self.kv {return;}
        let off=DIAG_DATA+stage*DIAG_SLOT;
        let mut bad=0usize;let mut first=0usize;
        for (i,&expected) in cpu.iter().enumerate() {
            let actual=unsafe {read_volatile((self.base+off+i*2) as *const u16)};
            if actual!=expected {if bad==0 {first=i;}bad+=1;}
        }
        serial_println!("[gpu-opdiag] layer={} stage={} fp16_kv_mismatch={} of={} first_idx={}",
            layer,stage,bad,cpu.len(),first);
    }
    fn trace_compare_layer(&mut self,layer:usize,cpu:&[f32]) {
        // The temporary driver records 2048 floats after each layer on the
        // first three RUNs. NightRun performs ALL comparisons, not the host.
        const TRACE:usize=0x0040_0000;
        if !self.in_decode_trace_window() || layer>=LAYERS || cpu.len()!=self.dim {return;}
        let mut max_err=0.0f32;
        let mut max_idx=0usize;
        let mut expected=0.0f32;
        let mut actual=0.0f32;
        let mut violations=0usize;
        for (i,&reference) in cpu.iter().enumerate() {
            let got=f32::from_bits(get(self.base,TRACE+(layer*self.dim+i)*4));
            let error=if got.is_finite() {(reference-got).abs()} else {f32::INFINITY};
            if error>max_err {max_err=error;max_idx=i;expected=reference;actual=got;}
            if error>0.05+0.01*reference.abs() {violations+=1;}
        }
        serial_println!("[gpu-stack-trace] layer={} max_abs_error={} idx={} cpu={} gpu={} violations={}",
            layer,max_err,max_idx,expected,actual,violations);
    }
}
pub fn attach(infer:&mut InferCtx<'_>)->bool {
    if !crate::gpu_resident::ready() {return false;}
    let d=infer.dims;
    if d.n_layers!=LAYERS||d.dim!=2048||d.att_dim!=2048||d.ffn_dim!=8192||
       d.head_dim!=64||d.n_heads!=32||d.n_kv_heads!=8||d.kv_dim!=512||
       d.ctx==0||d.ctx>16384{return false;}
    for w in &infer.weights.layers {
        if !matches!(w.wq,QMat::Q8(_))||!matches!(w.wk,QMat::Q8(_))||
           !matches!(w.wv,QMat::Q8(_))||!matches!(w.wo,QMat::Q8(_))||
           !matches!(w.w_gate,QMat::Q8(_))||!matches!(w.w_up,QMat::Q8(_))||
           !matches!(w.w_down,QMat::Q8(_))||w.q_norm.is_some()||w.k_norm.is_some() {return false;}
    }
    let Some(base)=locate() else {return false;};
    if get(base,STATUS)!=READY {serial_println!("[gpu-stack] executor not ready");return false;}
    let (eps,residual)=infer.mlp_params();
    let frequencies=infer.layer0_rope_frequencies();
    if frequencies.len()!=d.head_dim/2 {return false;}
    let Ok(plan)=stack_plan::decoder_stack(d.dim as u32,d.ffn_dim as u32,d.kv_dim as u32,
         d.n_heads as u32,d.n_kv_heads as u32,d.head_dim as u32,
         eps,residual,infer.attention_score_scale()) else {return false;};
    let mut bridge=Bridge{base,seq:0,next_pos:0,ctx:d.ctx,kv:d.kv_dim,dim:d.dim,calls:0,trace_base_calls:0,last_prefill_batch:false,final_vocab:0,embed_request:None,gpu_embed_batch:false};
    // NIGHTRUN_STAGE9A_UEFI_OWNED_CUBIN_V1: fail closed if any NightRun-owned module cannot be loaded.
    if !bridge.load_uefi_module(1,gpu_kernel_image::MLP) ||
       !bridge.load_uefi_module(2,gpu_kernel_image::ATTENTION) ||
       !bridge.load_uefi_module(3,gpu_kernel_image::ROPE_KV) {
        serial_println!("[gpu-kernels] GPU module initialization FAILED; CPU fallback");
        return false;
    }
    serial_println!("[gpu-kernels] all 3 NightRun-owned sm_89 CUBINs resident");
    put(base,CTX,d.ctx as u32);put(base,DIM,d.dim as u32);put(base,FFN,d.ffn_dim as u32);
    put(base,HD,d.head_dim as u32);put(base,HEADS,d.n_heads as u32);
    put(base,KV_HEADS,d.n_kv_heads as u32);put(base,KV_DIM,d.kv_dim as u32);
    put(base,EPS,eps.to_bits());put(base,RESIDUAL,residual.to_bits());
    put(base,ATTN_SCALE,infer.attention_score_scale().to_bits());
    write_f32(base,FREQ,frequencies);
    for (i,cmd) in plan.as_slice().iter().enumerate(){upload_command(base,i,cmd);}
    put(base,CMD_COUNT,plan.as_slice().len() as u32);
    put(base,CMD_ABI,nr_gpu::ABI_VERSION as u32);
    serial_println!("[gpu-stack] NightRun UEFI submitted {} commands for {} layers",
        plan.as_slice().len(),LAYERS);
    // NIGHTRUN_STAGE8C_GPU_FINAL_V1: output operations are UEFI authored.
    // The existing Q8 classifier remains the fallback for other formats.
    if matches!(infer.weights.output,QMat::Q8(_)) && d.vocab<=FINAL_MAX_VOCAB {
        let Ok(final_plan)=stack_plan::output_head(d.dim as u32,d.vocab as u32,eps)
            else {return false;};
        for (i,cmd) in final_plan.as_slice().iter().enumerate() {
            upload_command_at(base,FINAL_CMDS+i*48,cmd);
        }
        put(base,FINAL_VOCAB,d.vocab as u32);
        bridge.final_vocab=d.vocab;
        serial_println!("[gpu-final] NightRun UEFI authored 3 resident output commands vocab={}",d.vocab);
    } else { put(base,FINAL_VOCAB,0); }
    // NIGHTRUN_STAGE9C_GENERIC_GPU_LAUNCH_V1: UEFI explicitly owns symbol,
    // grid/block and typed CUDA arguments. Legacy final head remains fallback.
    put(base,RUNTIME_ABI,0);
    if bridge.final_vocab>0 && get(base,RUNTIME_CAP)==RUNTIME_CAP_MAGIC {
        if let Ok(launches)=nr_gpu::runtime_launch::output_head_launches(
            d.dim as u32,d.vocab as u32,eps) {
            for (i,c) in launches.iter().enumerate(){upload_runtime_launch(base,i,c);}
            fence(Ordering::SeqCst);
            put(base,RUNTIME_ABI,nr_gpu::runtime_launch::GENERIC_LAUNCH_ABI);
            serial_println!("[gpu-runtime] UEFI-authored generic output launches=3 module=1 ABI=1");
        }
    } else {serial_println!("[gpu-runtime] generic runtime unavailable; legacy output path");}
    if !bridge.send(INIT){serial_println!("[gpu-stack] attach failed; Stage6B fallback");return false;}
    // Single UEFI request sequence, shared with the non-model sampling primitive.
    COMMAND_SEQ.store(bridge.seq,Ordering::Release);
    SAMPLE_READY_SEQ.store(0,Ordering::Release);
    SAMPLE_VOCAB.store(bridge.final_vocab as u32,Ordering::Release);
    SAMPLE_SCALE_BITS.store(infer.gpu_sampler_logit_scale().to_bits(),Ordering::Release);
    SAMPLE_BASE.store(base,Ordering::Release);
    infer.set_stack_backend(Box::leak(Box::new(bridge)));
    serial_println!("[gpu-stack] all 16 resident transformer layers ACTIVE (CPU parity enabled)");
    true
}

// NIGHTRUN_STAGE8E_GPU_SAMPLER_V1. Called only from the UEFI inference thread.
// CUDA produces candidate IDs and weights; UEFI owns RNG and the token decision.
// GPU failure returns None BEFORE the RNG is advanced.
pub fn try_sample(sampler:&mut nr_model::Sampler,logits:&[f32])->Option<u32> {
    let prior=SAMPLE_READY_SEQ.swap(0,Ordering::AcqRel);
    if prior==0 || sampler.temperature<=0.0 ||
       !sampler.temperature.is_finite() || !sampler.top_p.is_finite() ||
       !(0.0<sampler.top_p && sampler.top_p<=1.0) {return None;}
    let base=SAMPLE_BASE.load(Ordering::Acquire);
    let vocab=SAMPLE_VOCAB.load(Ordering::Acquire) as usize;
    let scale=f32::from_bits(SAMPLE_SCALE_BITS.load(Ordering::Acquire));
    if base==0 || vocab==0 || logits.len()!=vocab ||
       COMMAND_SEQ.load(Ordering::Acquire)!=prior ||
       get(base,STATUS)!=READY || !scale.is_finite() || scale<=0.0 {return None;}
    put(base,SAMPLE_TEMP,sampler.temperature.to_bits());
    put(base,SAMPLE_TOPP,sampler.top_p.to_bits());
    put(base,SAMPLE_SCALE,scale.to_bits());
    put(base,SAMPLE_EXPECT,prior);
    // This is the same UEFI-owned command sequencer used by Bridge::send.
    let seq=COMMAND_SEQ.fetch_add(1,Ordering::AcqRel).wrapping_add(1);
    put(base,OP,RUN_SAMPLE);put(base,SEQ,seq);
    fence(Ordering::Release);put(base,STATUS,REQUEST);fence(Ordering::SeqCst);
    let mut completed=false;
    for _ in 0..30000 {
        let status=get(base,STATUS);
        if status==FAILED {break;}
        if status==DONE && get(base,ACK)==seq {
            put(base,STATUS,READY);fence(Ordering::SeqCst);
            completed=true;break;
        }
        boot::stall(Duration::from_micros(100));
    }
    if !completed {
        serial_println!("[gpu-sample] GPU preparation unavailable; CPU sampler fallback");
        return None;
    }
    let n=get(base,SAMPLE_OUT) as usize;
    let mass=f32::from_bits(get(base,SAMPLE_OUT+4));
    if n==0 || n>64 || !mass.is_finite() || mass<=0.0 {return None;}
    let mut ids=[0u32;64];let mut weights=[0f32;64];
    for i in 0..n {
        ids[i]=get(base,SAMPLE_OUT+8+4*i);
        weights[i]=f32::from_bits(get(base,SAMPLE_OUT+264+4*i));
        if ids[i] as usize>=vocab || !weights[i].is_finite() ||
           weights[i]<0.0 {return None;}
        // Reject candidates from any stale output: current UEFI-visible logits
        // must agree with the same output-head result in the shared buffer.
        let original=f32::from_bits(get(base,FINAL_LOGITS+4*ids[i] as usize));
        let scaled=original*scale;
        let expected=logits[ids[i] as usize];
        if !scaled.is_finite() || !expected.is_finite() ||
           (scaled-expected).abs()>0.0001*(1.0+expected.abs()) {return None;}
    }
    let token=sampler.sample_prepared(&ids,&weights,n,mass)?;
    let count=SAMPLE_CALLS.fetch_add(1,Ordering::AcqRel).wrapping_add(1);
    if count<=3 || count%64==0 {
        serial_println!("[gpu-sample] ACTIVE sample_calls={} topk=64 nucleus={} token={} GPU={}us service={}us",
            count,n,token,get(base,GPU_US),get(base,SERVICE_US));
    }
    Some(token)
}
