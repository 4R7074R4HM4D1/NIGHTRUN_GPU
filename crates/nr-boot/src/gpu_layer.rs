//! Stage 6B full layer-0 forward pass through NightRun's GPU command executor.
//! The UEFI guest owns the inference schedule and a CPU KV mirror for fallback.
extern crate alloc;
use alloc::boxed::Box;
use core::arch::asm;
use core::ptr::{read_volatile,write_volatile};
use core::sync::atomic::{fence,Ordering};
use core::time::Duration;
use nr_model::infer::{InferCtx,LayerBackend,QMat};
use nr_gpu::layer_plan;
use nr_gpu::GpuCommand;
use uefi::boot;

// Private PCI BAR region, separate from classifier, Stage3, resident, MLP and attention.
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
const CMD_COUNT:usize=84; const CMD_ABI:usize=88;
const CMD_DATA:usize=0x9000; // 48-byte records, no physical/GPU pointers
const FREQ:usize=0x1000; const INPUT:usize=0x2000;
const OUT_K:usize=0x5000; const OUT_V:usize=0x5800; const DATA:usize=0x10000;
const READY:u32=0; const REQUEST:u32=1; const DONE:u32=2; const FAILED:u32=3;
const INIT:u32=1; const SYNC:u32=2; const RUN:u32=3;
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
            serial_println!("[gpu-layer] PCI {:02x}:{:02x}.{} BAR2={:#x}",bus,dev,fun,phys);
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
    put(base,off+8,c.dst); put(base,off+12,c.src0);
    put(base,off+16,c.src1); put(base,off+20,c.aux);
    put(base,off+24,c.rows); put(base,off+28,c.cols);
    put(base,off+32,c.scalar0_bits); put(base,off+36,c.scalar1_bits);
    put(base,off+40,c.reserved[0]); put(base,off+44,c.reserved[1]);
}
fn write_f32(base:usize,off:usize,x:&[f32]) {for (i,&v) in x.iter().enumerate(){put(base,off+4*i,v.to_bits());}}
fn write_u16(base:usize,off:usize,x:&[u16]) {for (i,&v) in x.iter().enumerate(){unsafe{write_volatile((base+off+2*i) as *mut u16,v)};}fence(Ordering::Release);}

struct Bridge {base:usize,seq:u32,next_pos:usize,ctx:usize,kv:usize,dim:usize,calls:u32}
impl Bridge {
    fn send(&mut self,op:u32)->bool {
        if get(self.base,STATUS)!=READY {return false;}
        self.seq=self.seq.wrapping_add(1);
        put(self.base,OP,op);put(self.base,SEQ,self.seq);
        fence(Ordering::Release);put(self.base,STATUS,REQUEST);fence(Ordering::SeqCst);
        for _ in 0..30000 {
            let s=get(self.base,STATUS);
            if s==FAILED {
                serial_println!("[gpu-layer] host FAILED op={} err={}",op,get(self.base,ERROR));
                return false;
            }
            if s==DONE && get(self.base,ACK)==self.seq {
                put(self.base,STATUS,READY);fence(Ordering::SeqCst);return true;
            }
            boot::stall(Duration::from_micros(100));
        }
        serial_println!("[gpu-layer] timeout op={} seq={}; CPU fallback",op,self.seq);
        false
    }
    fn sync_history(&mut self,keys:&[u16],vals:&[u16],position:usize)->bool {
        let rows_per_chunk=(1_048_576/(self.kv*4)).max(1);
        let mut start=0usize;
        while start<position {
            let rows=(position-start).min(rows_per_chunk);
            let a=start*self.kv;let b=a+rows*self.kv;
            write_u16(self.base,DATA,&keys[a..b]);
            write_u16(self.base,DATA+rows*self.kv*2,&vals[a..b]);
            put(self.base,SYNC_START,start as u32);put(self.base,SYNC_ROWS,rows as u32);
            if !self.send(SYNC){return false;}
            start+=rows;
        }
        if position==0 {
            put(self.base,SYNC_START,0);put(self.base,SYNC_ROWS,0);
            if !self.send(SYNC){return false;}
        }
        self.next_pos=position;true
    }
}
impl LayerBackend for Bridge {
    fn compute(&mut self,x:&mut[f32],pos:usize,keys:&mut[u16],vals:&mut[u16])->bool {
        if pos>=self.ctx||x.len()!=self.dim||keys.len()!=self.ctx*self.kv||vals.len()!=keys.len()
            ||x.iter().any(|v|!v.is_finite()) {return false;}
        if pos!=self.next_pos && !self.sync_history(keys,vals,pos){return false;}
        write_f32(self.base,INPUT,x);put(self.base,POS,pos as u32);
        if !self.send(RUN){return false;}
        for i in 0..x.len(){if !f32::from_bits(get(self.base,INPUT+i*4)).is_finite(){return false;}}
        for i in 0..self.kv {
            let k=unsafe{read_volatile((self.base+OUT_K+2*i) as *const u16)};
            let v=unsafe{read_volatile((self.base+OUT_V+2*i) as *const u16)};
            if k&0x7c00==0x7c00 || v&0x7c00==0x7c00 {return false;}
        }
        // Only after validating all results: commit hidden state and KV mirror.
        for (i,y) in x.iter_mut().enumerate(){*y=f32::from_bits(get(self.base,INPUT+i*4));}
        for i in 0..self.kv {
            keys[pos*self.kv+i]=unsafe{read_volatile((self.base+OUT_K+2*i) as *const u16)};
            vals[pos*self.kv+i]=unsafe{read_volatile((self.base+OUT_V+2*i) as *const u16)};
        }
        self.next_pos=pos+1;self.calls+=1;
        if self.calls<=3||self.calls%64==0 {
            serial_println!("[gpu-layer] layer-0 full GPU #{} pos={} GPU={}us service={}us",
                self.calls,pos,get(self.base,GPU_US),get(self.base,SERVICE_US));
        }
        true
    }
    fn on_parity_check(&mut self,ok:bool,error:f32) {
        serial_println!("[gpu-layer] full-layer CPU parity {} max_abs_error={}",
            if ok {"PASS"} else {"FAIL; disabling full-layer GPU"},error);
    }
}
pub fn attach(infer:&mut InferCtx<'_>)->bool {
    if !crate::gpu_resident::ready() {return false;}
    let d=infer.dims;
    if d.dim!=2048||d.att_dim!=2048||d.ffn_dim!=8192||d.head_dim!=64||
       d.n_heads!=32||d.n_kv_heads!=8||d.kv_dim!=512||d.ctx==0||d.ctx>16384 {return false;}
    let w=&infer.weights.layers[0];
    if !matches!(w.wq,QMat::Q8(_))||!matches!(w.wk,QMat::Q8(_))||
       !matches!(w.wv,QMat::Q8(_))||!matches!(w.wo,QMat::Q8(_))||
       !matches!(w.w_gate,QMat::Q8(_))||!matches!(w.w_up,QMat::Q8(_))||
       !matches!(w.w_down,QMat::Q8(_)) {return false;}
    let Some(base)=locate() else {serial_println!("[gpu-layer] executor unavailable; Stage6A fallback");return false;};
    if get(base,STATUS)!=READY {serial_println!("[gpu-layer] executor not ready");return false;}
    let (eps,residual)=infer.mlp_params();
    let frequencies=infer.layer0_rope_frequencies();
    if frequencies.len()!=d.head_dim/2 {return false;}
    let mut bridge=Bridge{base,seq:0,next_pos:0,ctx:d.ctx,kv:d.kv_dim,dim:d.dim,calls:0};
    put(base,CTX,d.ctx as u32);put(base,DIM,d.dim as u32);put(base,FFN,d.ffn_dim as u32);
    put(base,HD,d.head_dim as u32);put(base,HEADS,d.n_heads as u32);
    put(base,KV_HEADS,d.n_kv_heads as u32);put(base,KV_DIM,d.kv_dim as u32);
    put(base,EPS,eps.to_bits());put(base,RESIDUAL,residual.to_bits());
    put(base,ATTN_SCALE,infer.attention_score_scale().to_bits());
    write_f32(base,FREQ,frequencies);
    // The entire execution order is authored and serialized by UEFI NightRun.
    let Ok(plan)=layer_plan::layer0_decode(d.dim as u32,d.ffn_dim as u32,d.kv_dim as u32,
        d.n_heads as u32,d.n_kv_heads as u32,d.head_dim as u32,
        eps,residual,infer.attention_score_scale()) else {return false;};
    for (i,cmd) in plan.as_slice().iter().enumerate() {upload_command(base,i,cmd);}
    put(base,CMD_COUNT,plan.as_slice().len() as u32);
    put(base,CMD_ABI,nr_gpu::ABI_VERSION as u32);
    serial_println!("[gpu-layer] NightRun UEFI submitted {} ordered kernel commands",plan.as_slice().len());
    if !bridge.send(INIT) {serial_println!("[gpu-layer] init failed; Stage6A fallback");return false;}
    infer.set_layer_backend(Box::leak(Box::new(bridge)));
    serial_println!("[gpu-layer] full resident layer-0 ACTIVE; GPU QKV/RoPE/GQA/O/MLP (CPU parity)");
    true
}
