//! Stage 4: run NightRun's REAL layer-0 MLP during autoregressive inference.
//! CUDA is only a QEMU host-side launch mechanism. NightRun owns the model,
//! layer scheduling, Q8 weights, command buffer, activation and CPU fallback.
extern crate alloc;
use alloc::boxed::Box;
use core::arch::asm;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};
use core::time::Duration;
use nr_gpu::{BufferId, CommandList, GpuCommand};
use nr_model::infer::{InferCtx, MlpBackend, QMat};
use uefi::boot;

const REGION: usize = 0x8000_0000; // 2 GiB into existing 8 GiB ivshmem BAR2
const CAPACITY: usize = 0x0800_0000;
const MAGIC: u32 = 0x3447_504e; // "NPG4"
const STATUS: usize=4;
const REQUEST_SEQ: usize=8;
const COMPLETED_SEQ: usize=12;
const ERROR_CODE: usize=16;
const DIM: usize=20;
const FFN: usize=24;
const SERVICE_US: usize=28;
const GPU_US: usize=32;
const EPS_BITS: usize=36;
const SCALE_BITS: usize=40;
const CMD_COUNT: usize=44;
const RESIDENT_MODE: usize=48; // Stage 5B: use tensor IDs, no MLP weight upload
const COMMANDS: usize=0x100;
const WEIGHTS: usize=0x10000;
const READY: u32=0;
const REQUEST: u32=1;
const DONE: u32=2;
const ERROR: u32=3;
const INIT: u32=4;
const INIT_DONE: u32=5;

#[inline]
fn pci_read(bus: u8, dev: u8, func: u8, reg: u8) -> u32 {
    let addr=0x8000_0000u32 | ((bus as u32)<<16) |
        ((dev as u32)<<11) | ((func as u32)<<8) | ((reg as u32)&0xfc);
    let result: u32;
    unsafe {
        asm!("out dx, eax",in("dx") 0xcf8u16,in("eax") addr,options(nostack,preserves_flags));
        asm!("in eax, dx",in("dx") 0xcfcu16,out("eax") result,options(nostack,preserves_flags));
    }
    result
}
fn locate() -> Option<usize> {
    for bus in 0..=255u16 { for dev in 0..32u8 { for function in 0..8u8 {
        let b=bus as u8;
        let id=pci_read(b,dev,function,0);
        if id as u16!=0x1af4 || (id>>16) as u16!=0x1110 { continue; }
        if pci_read(b,dev,function,4)&2==0 { continue; }
        let lo=pci_read(b,dev,function,0x18);
        if lo&1!=0 { continue; }
        let hi=match (lo>>1)&3 {
            2=>pci_read(b,dev,function,0x1c),0=>0,_=>continue
        };
        let physical=((hi as u64)<<32) | ((lo&!0xf) as u64);
        if physical==0 || physical>usize::MAX as u64-(REGION+CAPACITY) as u64 {continue;}
        let base=physical as usize+REGION;
        if unsafe {read_volatile(base as *const u32)}==MAGIC {
            serial_println!("[gpu-mlp] ivshmem {:02x}:{:02x}.{} BAR2={:#x}",b,dev,function,physical);
            return Some(base);
        }
    }}}
    None
}
#[inline]
fn put(base: usize, off: usize, value: u32) {
    unsafe {write_volatile((base+off) as *mut u32,value)};
    fence(Ordering::SeqCst);
}
#[inline]
fn get(base: usize, off: usize) -> u32 {
    let value=unsafe {read_volatile((base+off) as *const u32)};
    fence(Ordering::SeqCst);
    value
}
fn write_bytes(base: usize, off: usize, bytes: &[u8]) {
    let mut words=bytes.chunks_exact(4);
    for (i, chunk) in words.by_ref().enumerate() {
        unsafe {write_volatile((base+off+4*i) as *mut u32,
            u32::from_le_bytes([chunk[0],chunk[1],chunk[2],chunk[3]]))};
    }
    let written=bytes.len()-words.remainder().len();
    for (i,b) in words.remainder().iter().enumerate() {
        unsafe {write_volatile((base+off+written+i) as *mut u8,*b)};
    }
    fence(Ordering::SeqCst);
}
fn align64(n: usize) -> Option<usize> { n.checked_add(63).map(|n|n&!63) }
#[derive(Clone,Copy)]
struct Layout { wg:usize, wu:usize, wd:usize, gamma:usize, input:usize, output:usize }
fn layout(d:usize,f:usize)->Option<Layout> {
    if d!=2048 || f==0 || f>16384 || f%32!=0 {return None;}
    let gf=f.checked_mul(d/32)?.checked_mul(34)?;
    let df=d.checked_mul(f/32)?.checked_mul(34)?;
    let wg=WEIGHTS;
    let wu=align64(wg.checked_add(gf)?)?;
    let wd=align64(wu.checked_add(gf)?)?;
    let gamma=align64(wd.checked_add(df)?)?;
    let input=align64(gamma.checked_add(d*4)?)?;
    let output=align64(input.checked_add(d*4)?)?;
    if align64(output.checked_add(d * 4)?)? > CAPACITY { return None; }
    Some(Layout{wg,wu,wd,gamma,input,output})
}
fn raw_q8<'a>(mat: &'a QMat<'_>, rows:usize, cols:usize) -> Option<&'a [u8]> {
    let QMat::Q8(blocks)=mat else {return None;};
    let n=rows.checked_mul(cols/32)?;
    if cols%32!=0 || blocks.len()!=n || core::mem::size_of::<nr_tensor::q8::BlockQ8_0>()!=34 {
        return None;
    }
    Some(unsafe {core::slice::from_raw_parts(blocks.as_ptr() as *const u8,n*34)})
}
fn commands(dim:u32,ffn:u32,eps:f32,scale:f32)->Option<CommandList<8>> {
    let mut list=CommandList::<8>::new();
    list.push(GpuCommand::rms_norm(BufferId(3),BufferId(1),BufferId(2),dim,eps).ok()?).ok()?;
    list.push(GpuCommand::quantize_q8(BufferId(4),BufferId(3),dim).ok()?).ok()?;
    list.push(GpuCommand::q8_matvec(BufferId(6),BufferId(5),BufferId(4),ffn,dim).ok()?).ok()?;
    list.push(GpuCommand::q8_matvec(BufferId(8),BufferId(7),BufferId(4),ffn,dim).ok()?).ok()?;
    list.push(GpuCommand::swiglu(BufferId(6),BufferId(6),BufferId(8),ffn).ok()?).ok()?;
    list.push(GpuCommand::quantize_q8(BufferId(9),BufferId(6),ffn).ok()?).ok()?;
    list.push(GpuCommand::q8_matvec(BufferId(11),BufferId(10),BufferId(9),dim,ffn).ok()?).ok()?;
    list.push(GpuCommand::scaled_residual(BufferId(1),BufferId(1),BufferId(11),dim,scale).ok()?).ok()?;
    Some(list)
}
fn wait(base:usize, target:u32, sequence:u32, spins:usize)->bool {
    for _ in 0..spins {
        let status=get(base,STATUS);
        if status==ERROR {
            serial_println!("[gpu-mlp] executor failure code={}",get(base,ERROR_CODE));
            return false;
        }
        if status==target && get(base,COMPLETED_SEQ)==sequence {return true;}
        boot::stall(Duration::from_micros(100));
    }
    false
}

struct PciMlp { base:usize, shape:Layout, dim:usize, sequence:u32, calls:u32 }
impl MlpBackend for PciMlp {
    // On failure x is unchanged, so NightRun safely falls back to CPU.
    fn compute(&mut self, x: &mut [f32]) -> bool {
        if x.len()!=self.dim || get(self.base,STATUS)!=READY {return false;}
        for (i,&value) in x.iter().enumerate() {
            if !value.is_finite() {return false;}
            put(self.base,self.shape.input+4*i,value.to_bits());
        }
        self.sequence=self.sequence.wrapping_add(1);
        put(self.base,REQUEST_SEQ,self.sequence);
        put(self.base,STATUS,REQUEST);
        if !wait(self.base,DONE,self.sequence,1000) {
            serial_println!("[gpu-mlp] inference service timeout; disabling GPU layer");
            return false;
        }
        // Check the entire output before copying any values into the live state.
        for i in 0..self.dim {
            if !f32::from_bits(get(self.base,self.shape.output+i*4)).is_finite() {
                serial_println!("[gpu-mlp] non-finite GPU output; falling back to CPU");
                return false;
            }
        }
        for (i,val) in x.iter_mut().enumerate() {
            *val=f32::from_bits(get(self.base,self.shape.output+i*4));
        }
        self.calls=self.calls.wrapping_add(1);
        if self.calls<=3 || self.calls%64==0 {
            serial_println!("[gpu-mlp] real decode MLP #{} GPU graph={} us service={} us",
                self.calls,get(self.base,GPU_US),get(self.base,SERVICE_US));
        }
        put(self.base,STATUS,READY);
        true
    }

    fn on_parity_check(&mut self, ok: bool, max_error: f32) {
        serial_println!("[gpu-mlp] NightRun CPU parity {}: max_abs_error={}",
            if ok { "PASS" } else { "FAIL - CPU fallback" }, max_error);
    }
}

pub fn attach(infer: &mut InferCtx<'_>) {
    if !crate::gpu_resident::ready() {
        serial_println!("[gpu-mlp] resident model not ready; using CPU layer-0 MLP");
        return;
    }
    let Some(base)=locate() else {
        serial_println!("[gpu-mlp] GPU layer executor not found; normal CPU inference");
        return;
    };
    let d=infer.dims.dim;
    let f=infer.dims.ffn_dim;
    let Some(shape)=layout(d,f) else {
        serial_println!("[gpu-mlp] unsupported model shape; CPU fallback");
        return;
    };
    let layer=&infer.weights.layers[0];
    let Some(gate)=raw_q8(&layer.w_gate,f,d) else {
        serial_println!("[gpu-mlp] layer-0 gate is not Q8_0"); return;
    };
    let Some(up)=raw_q8(&layer.w_up,f,d) else {
        serial_println!("[gpu-mlp] layer-0 up is not Q8_0"); return;
    };
    let Some(down)=raw_q8(&layer.w_down,d,f) else {
        serial_println!("[gpu-mlp] layer-0 down is not Q8_0"); return;
    };
    let (eps,scale)=infer.mlp_params();
    let Some(cmds)=commands(d as u32,f as u32,eps,scale) else {
        serial_println!("[gpu-mlp] could not construct command buffer"); return;
    };
    if get(base,STATUS)!=READY {serial_println!("[gpu-mlp] executor not ready");return;}
    // Tensor registry already has the verified Q8_0 weights and FP32 gamma.
    // Preserve Q8_0 type/shape checks without transferring these bytes again.
    if gate.is_empty() || up.is_empty() || down.is_empty() || layer.ffn_norm.len()!=d {
        serial_println!("[gpu-mlp] resident tensor shape validation failed");
        return;
    }
    serial_println!("[gpu-mlp] using Stage 5A resident weight IDs 261,262,263,264 (no reupload)");
    for (i,c) in cmds.as_slice().iter().enumerate() {
        let bytes=unsafe {core::slice::from_raw_parts(c as *const _ as *const u8,48)};
        write_bytes(base,COMMANDS+i*48,bytes);
    }
    put(base,DIM,d as u32); put(base,FFN,f as u32);
    put(base,EPS_BITS,eps.to_bits()); put(base,SCALE_BITS,scale.to_bits());
    put(base,CMD_COUNT,8); put(base,RESIDENT_MODE,1); put(base,ERROR_CODE,0);
    put(base,REQUEST_SEQ,1); put(base,STATUS,INIT);
    if !wait(base,INIT_DONE,1,100_000) {
        serial_println!("[gpu-mlp] initialization failed/timed out; CPU fallback");
        return;
    }
    put(base,STATUS,READY);
    infer.set_mlp_backend(Box::leak(Box::new(PciMlp {base,shape,dim:d,sequence:1,calls:0})));
    serial_println!("[gpu-mlp] resident layer-0 decode MLP ACTIVE (CPU parity check enabled)");
}
