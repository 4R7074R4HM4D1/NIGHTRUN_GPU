//! Stage 9C: generic CUDA-driver-neutral launch descriptions authored by NightRun.
//! Modules and symbols are carried by UEFI; no NVIDIA pointers or CUDA types cross the ABI.
//!
//! The generic runtime validates buffer bounds and launches kernels, but does NOT
//! invent model operations, tensor layouts, sampling policies or token order.
use crate::GpuError;

pub const GENERIC_LAUNCH_ABI: u32 = 1;
pub const KIND_BUFFER: u32 = 1; // value=UEFI buffer handle; offset and span in bytes
pub const KIND_U32: u32 = 2;    // value=u32 immediate
pub const KIND_F32: u32 = 3;    // value=f32.to_bits()
pub const MAX_ARGS: usize = 8;
pub const SYMBOL_BYTES: usize = 48;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelArg {
    pub kind: u32,
    pub value: u32,
    pub span: u32,
    pub offset: u32,
}
impl KernelArg {
    pub const fn buffer(handle:u32,span:u32)->Self {Self{kind:KIND_BUFFER,value:handle,span,offset:0}}
    pub const fn u32(v:u32)->Self {Self{kind:KIND_U32,value:v,span:0,offset:0}}
    pub const fn f32(v:f32)->Self {Self{kind:KIND_F32,value:v.to_bits(),span:0,offset:0}}
    pub const fn empty()->Self {Self{kind:0,value:0,span:0,offset:0}}
}

/// 224-byte little-endian record (UEFI x86_64), eight explicit, typed arguments.
/// No host-selected grid, block, kernel symbol, or argument ordering.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelLaunch {
    pub module_id:u32,
    pub grid:[u32;3],
    pub block:[u32;3],
    pub shared_bytes:u32,
    pub arg_count:u32,
    pub reserved:u32,
    pub symbol:[u8;SYMBOL_BYTES],
    pub padding:[u32;2],
    pub args:[KernelArg;MAX_ARGS],
}
const _:()=assert!(core::mem::size_of::<KernelArg>()==16);
const _:()=assert!(core::mem::size_of::<KernelLaunch>()==224);
impl KernelLaunch {
    pub fn new(module_id:u32,name:&[u8],grid:[u32;3],block:[u32;3],
               args:&[KernelArg])->Result<Self,GpuError>{
        if !(1..=3).contains(&module_id) || name.is_empty() || name.len()>=SYMBOL_BYTES ||
           !name.iter().all(|c|c.is_ascii_alphanumeric() || *c==b'_') ||
           args.is_empty() || args.len()>MAX_ARGS ||
           grid.contains(&0) || block.contains(&0) ||
           (block[0] as u64)*(block[1] as u64)*(block[2] as u64)>1024 {
            return Err(GpuError::InvalidShape);
        }
        let mut desc=Self{module_id,grid,block,shared_bytes:0,arg_count:args.len() as u32,
              reserved:0,symbol:[0;SYMBOL_BYTES],padding:[0;2],args:[KernelArg::empty();MAX_ARGS]};
        desc.symbol[..name.len()].copy_from_slice(name);
        desc.args[..args.len()].copy_from_slice(args);
        Ok(desc)
    }
}

/// The output-head launch plan is assembled *in NightRun*, not in the executor.
/// Tensor IDs 2/3 are the already-resident final norm gamma / Q8 vocabulary matrix.
pub fn output_head_launches(dim:u32,vocab:u32,eps:f32)->Result<[KernelLaunch;3],GpuError>{
    if dim!=2048 || dim%32!=0 || vocab==0 || vocab>262144 || !eps.is_finite() || eps<=0.0 {
        return Err(GpuError::InvalidShape);
    }
    let weights=(vocab as u64)*(dim as u64/32)*34;
    if weights>u32::MAX as u64 {return Err(GpuError::InvalidShape);}
    let norm=KernelLaunch::new(1,b"nr_rmsnorm",[1,1,1],[256,1,1],&[
        KernelArg::buffer(2000,dim*4),KernelArg::buffer(2,dim*4),
        KernelArg::buffer(2017,dim*4),KernelArg::u32(dim),KernelArg::f32(eps)])?;
    let quant=KernelLaunch::new(1,b"nr_quantize_q8",[dim/32,1,1],[32,1,1],&[
        KernelArg::buffer(2017,dim*4),KernelArg::buffer(2018,dim/32*34),KernelArg::u32(dim)])?;
    let gemv=KernelLaunch::new(1,b"nr_q8_matvec",[(vocab+7)/8,1,1],[256,1,1],&[
        KernelArg::buffer(3,weights as u32),KernelArg::buffer(2018,dim/32*34),
        KernelArg::buffer(2019,vocab*4),KernelArg::u32(vocab),KernelArg::u32(dim)])?;
    Ok([norm,quant,gemv])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn head_is_uefi_authored_and_pointer_free() {
        let v=output_head_launches(2048,128256,1e-5).unwrap();
        assert_eq!(core::mem::size_of::<KernelLaunch>(),224);
        assert_eq!(&v[0].symbol[..10],b"nr_rmsnorm");
        assert_eq!(v[1].grid,[64,1,1]);
        assert_eq!(v[2].grid,[(128256+7)/8,1,1]);
        assert_eq!(v[2].args[0].value,3);
        assert_eq!(v[2].args[0].span,128256*64*34);
        assert_eq!(v[2].args[2].value,2019);
    }
    #[test]
    fn no_pointer_values_or_bad_launch(){
        assert!(KernelLaunch::new(1,b"bad symbol",[1,1,1],[32,1,1],&[KernelArg::u32(1)]).is_err());
        assert!(KernelLaunch::new(1,b"okay",[1,1,1],[1025,1,1],&[KernelArg::u32(1)]).is_err());
        assert!(output_head_launches(2048,0,1e-5).is_err());
    }
}
