//! Stage 6A: NightRun's live layer-0 GQA attention using resident GPU KV.
//! NightRun remains the inference scheduler and CPU fallback owner.
extern crate alloc;
use alloc::boxed::Box;
use core::arch::asm;
use core::ptr::{read_volatile,write_volatile};
use core::sync::atomic::{fence,Ordering};
use core::time::Duration;
use nr_model::infer::{AttentionBackend,InferCtx};
use uefi::boot;

const REGION:usize=0xC000_0000; // Separate +3GiB ivshmem region
const CAP:usize=0x0040_0000;
const MAGIC:u32=0x3641_504e;
const STATUS:usize=4; const SEQ:usize=8; const ACK:usize=12;
const ERROR:usize=16; const OP:usize=20; const POS:usize=24;
const CTX:usize=28; const HEADS:usize=32; const KV_HEADS:usize=36;
const HEAD_DIM:usize=40; const KV_DIM:usize=44;
const SERVICE_US:usize=48; const GPU_US:usize=52;
const SYNC_START:usize=56; const SYNC_ROWS:usize=60;
const SCALE:usize=64;
const QUERY:usize=0x1000; const KEY:usize=0x5000;
const VAL:usize=0x6000; const OUTPUT:usize=0x8000;
const DATA:usize=0x10000;
const READY:u32=0; const REQUEST:u32=1; const DONE:u32=2;
const FAILED:u32=3; const INIT:u32=1; const SYNC:u32=2; const RUN:u32=3;

#[inline]
fn pci_read(b:u8,d:u8,f:u8,r:u8)->u32 {
    let addr=0x8000_0000 | ((b as u32)<<16) | ((d as u32)<<11)
        | ((f as u32)<<8) | ((r as u32)&0xfc);
    let v:u32;
    unsafe {
        asm!("out dx, eax",in("dx") 0xcf8u16,in("eax") addr,options(nostack,preserves_flags));
        asm!("in eax, dx",in("dx") 0xcfcu16,out("eax") v,options(nostack,preserves_flags));
    }
    v
}
fn locate()->Option<usize>{
    for bus in 0..=255u16 { for dev in 0..32u8 {for fun in 0..8u8 {
        let b=bus as u8;
        if pci_read(b,dev,fun,0)!=0x1110_1af4 || pci_read(b,dev,fun,4)&2==0 {continue;}
        let low=pci_read(b,dev,fun,0x18);
        if low&1!=0 {continue;}
        let high=match (low>>1)&3 {2=>pci_read(b,dev,fun,0x1c),0=>0,_=>continue};
        let phys=((high as u64)<<32)|(low&!0xf) as u64;
        if phys==0 || phys>usize::MAX as u64-(REGION+CAP) as u64 {continue;}
        let base=phys as usize+REGION;
        if unsafe{read_volatile(base as *const u32)}==MAGIC {
            serial_println!("[gpu-attn] PCI {:02x}:{:02x}.{} BAR2={:#x}",b,dev,fun,phys);
            return Some(base);
        }
    }}}
    None
}
#[inline]
fn put(base:usize,off:usize,value:u32) {unsafe{write_volatile((base+off) as *mut u32,value)};}
#[inline]
fn get(base:usize,off:usize)->u32 {unsafe{read_volatile((base+off) as *const u32)}}
fn stage_u16(base:usize,off:usize,words:&[u16]) {
    for (i,&v) in words.iter().enumerate(){unsafe{write_volatile((base+off+2*i) as *mut u16,v)};}
    fence(Ordering::Release);
}
fn stage_f32(base:usize,off:usize,words:&[f32]){
    for (i,&v) in words.iter().enumerate(){put(base,off+4*i,v.to_bits());}
    fence(Ordering::Release);
}
struct PciAttn {base:usize,seq:u32,next_pos:usize,ctx:usize,kv:usize,heads:usize,
    head_dim:usize,calls:usize}
impl PciAttn {
    fn send(&mut self,op:u32)->bool {
        if get(self.base,STATUS)!=READY {return false;}
        self.seq=self.seq.wrapping_add(1);
        put(self.base,OP,op);put(self.base,SEQ,self.seq);
        fence(Ordering::Release);put(self.base,STATUS,REQUEST);fence(Ordering::SeqCst);
        for _ in 0..20_000 {
            let st=get(self.base,STATUS);fence(Ordering::Acquire);
            if st==FAILED {
                serial_println!("[gpu-attn] host failure op={} err={}",op,get(self.base,ERROR));
                return false;
            }
            if st==DONE && get(self.base,ACK)==self.seq {
                put(self.base,STATUS,READY);fence(Ordering::SeqCst);
                return true;
            }
            boot::stall(Duration::from_micros(100));
        }
        serial_println!("[gpu-attn] host timeout op={}; CPU fallback",op);
        false
    }
    fn sync_history(&mut self,keys:&[u16],vals:&[u16],position:usize)->bool {
        // Rehydrate GPU KV cache from NightRun CPU cache after prefill/reset.
        // A bounded chunk prevents creating large BAR staging buffers.
        let chunk_rows=(1_048_576/(self.kv*4)).max(1);
        let mut start=0usize;
        while start<position {
            let rows=(position-start).min(chunk_rows);
            let a=start*self.kv;let b=a+rows*self.kv;
            stage_u16(self.base,DATA,&keys[a..b]);
            stage_u16(self.base,DATA+rows*self.kv*2,&vals[a..b]);
            put(self.base,SYNC_START,start as u32);
            put(self.base,SYNC_ROWS,rows as u32);
            if !self.send(SYNC){return false;}
            start+=rows;
        }
        if position==0 {
            // The host must discard its previous sequence on context reset.
            put(self.base,SYNC_START,0);
            put(self.base,SYNC_ROWS,0);
            if !self.send(SYNC) { return false; }
        }
        self.next_pos=position;
        true
    }
}
impl AttentionBackend for PciAttn {
    fn compute(&mut self,query:&[f32],keys:&[u16],vals:&[u16],pos:usize,out:&mut[f32])->bool {
        if pos>=self.ctx || query.len()!=self.heads*self.head_dim ||
            out.len()!=query.len() || keys.len()!=self.ctx*self.kv ||
            vals.len()!=keys.len() || query.iter().any(|x|!x.is_finite()) {return false;}
        if self.next_pos!=pos && !self.sync_history(keys,vals,pos){return false;}
        stage_f32(self.base,QUERY,query);
        let row=pos*self.kv;
        stage_u16(self.base,KEY,&keys[row..row+self.kv]);
        stage_u16(self.base,VAL,&vals[row..row+self.kv]);
        put(self.base,POS,pos as u32);
        if !self.send(RUN){return false;}
        for i in 0..out.len() {
            if !f32::from_bits(get(self.base,OUTPUT+4*i)).is_finite(){return false;}
        }
        for (i,o) in out.iter_mut().enumerate(){*o=f32::from_bits(get(self.base,OUTPUT+4*i));}
        self.next_pos=pos+1;self.calls+=1;
        if self.calls<=3 || self.calls%64==0 {
            serial_println!("[gpu-attn] layer-0 #{} pos={} GPU={}us service={}us",
                self.calls,pos,get(self.base,GPU_US),get(self.base,SERVICE_US));
        }
        true
    }
    fn on_parity_check(&mut self,ok:bool,max_error:f32){
        serial_println!("[gpu-attn] CPU parity {} max_abs_error={}",
            if ok {"PASS"} else {"FAIL - CPU fallback"},max_error);
    }
}
pub fn attach(infer:&mut InferCtx<'_>){
    if !crate::gpu_resident::ready(){
        serial_println!("[gpu-attn] model not resident; CPU attention retained");return;
    }
    let d=infer.dims;
    if d.dim!=2048 || d.att_dim!=2048 || d.head_dim!=64 || d.n_heads!=32 ||
        d.n_kv_heads!=8 || d.ctx==0 || d.ctx>16384 || d.kv_dim!=512 {
        serial_println!("[gpu-attn] unsupported shape; CPU attention retained");return;
    }
    let Some(base)=locate() else {
        serial_println!("[gpu-attn] executor unavailable; CPU attention retained");return;
    };
    if get(base,STATUS)!=READY {serial_println!("[gpu-attn] executor busy");return;}
    let mut backend=PciAttn{base,seq:0,next_pos:0,ctx:d.ctx,kv:d.kv_dim,
        heads:d.n_heads,head_dim:d.head_dim,calls:0};
    let scale=infer.attention_score_scale();
    put(base,CTX,d.ctx as u32);put(base,HEADS,d.n_heads as u32);
    put(base,KV_HEADS,d.n_kv_heads as u32);put(base,HEAD_DIM,d.head_dim as u32);
    put(base,KV_DIM,d.kv_dim as u32);put(base,SCALE,scale.to_bits());
    if !backend.send(INIT){serial_println!("[gpu-attn] init rejected; CPU fallback");return;}
    infer.set_attention_backend(Box::leak(Box::new(backend)));
    serial_println!("[gpu-attn] real layer-0 GPU attention ACTIVE (KV resident; CPU parity enabled)");
}
