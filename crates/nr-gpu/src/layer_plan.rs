//! NightRun-owned layer-0 decode command plan. UEFI constructs this list;
//! CUDA/QEMU only executes its operations, and never plans transformer layers.
use crate::{ABI_VERSION, BufferId, CommandList, GpuCommand, GpuError};

// Opaque device-buffer handles. IDs 256..264 are NightRun's persistent
// model tensor handles, 1000..1001 are resident KV memory allocations.
// 2000..2016 are per-layer transient buffers allocated by the backend.
pub const X:u32=2000; pub const NORM:u32=2001; pub const QNORM:u32=2002;
pub const Q:u32=2003; pub const K:u32=2004; pub const V:u32=2005;
pub const FREQUENCIES:u32=2006; pub const SCORES:u32=2007;
pub const ATTN_OUT:u32=2008; pub const ATTN_Q8:u32=2009;
pub const PROJECTED:u32=2010; pub const FFN_NORM:u32=2011;
pub const FFN_Q8:u32=2012; pub const GATE:u32=2013;
pub const UP:u32=2014; pub const GATE_Q8:u32=2015; pub const DOWN:u32=2016;
pub const KEYS:u32=1000; pub const VALUES:u32=1001;

// Operations 1..6 are defined in nr-gpu::Opcode, 7..10 extend it.
// Each command is the common 48-byte endian-stable GpuCommand ABI.
pub const ROPE:u32=7;
pub const KV_STORE:u32=8;
pub const GQA_SCORES:u32=9;
pub const GQA_VALUE:u32=10;
pub const COMMANDS:usize=21;

fn id(x:u32)->BufferId { BufferId(x) }
fn custom(op:u32,dst:u32,s0:u32,s1:u32,aux:u32,rows:u32,cols:u32,sbits:u32)->GpuCommand {
    GpuCommand{opcode:op,version:ABI_VERSION,flags:0,dst,src0:s0,src1:s1,aux,
        rows,cols,scalar0_bits:sbits,scalar1_bits:0,reserved:[0;2]}
}

pub fn layer0_decode(dim:u32,ffn:u32,kv:u32,heads:u32,kv_heads:u32,
                     hd:u32,eps:f32,residual:f32,scale:f32)
                     ->Result<CommandList<COMMANDS>,GpuError> {
    if dim!=2048 || ffn!=8192 || kv!=512 || heads!=32 || kv_heads!=8 || hd!=64 ||
        !eps.is_finite() || eps<=0. || !residual.is_finite() ||
        !scale.is_finite() || scale<=0. {
        return Err(GpuError::InvalidShape);
    }
    let mut p=CommandList::<COMMANDS>::new();
    // Attention pre-norm and shared activation quantization.
    p.push(GpuCommand::rms_norm(id(NORM),id(X),id(256),dim,eps)?)?;
    p.push(GpuCommand::quantize_q8(id(QNORM),id(NORM),dim)?)?;
    p.push(GpuCommand::q8_matvec(id(Q),id(257),id(QNORM),dim,dim)?)?;
    p.push(GpuCommand::q8_matvec(id(K),id(258),id(QNORM),kv,dim)?)?;
    p.push(GpuCommand::q8_matvec(id(V),id(259),id(QNORM),kv,dim)?)?;
    p.push(custom(ROPE,Q,FREQUENCIES,0,0,heads,hd,0))?;
    p.push(custom(ROPE,K,FREQUENCIES,0,0,kv_heads,hd,0))?;
    p.push(custom(KV_STORE,KEYS,K,V,VALUES,kv,0,0))?;
    p.push(custom(GQA_SCORES,SCORES,Q,KEYS,0,heads,hd,scale.to_bits()))?;
    p.push(custom(GQA_VALUE,ATTN_OUT,SCORES,VALUES,0,heads,hd,0))?;
    p.push(GpuCommand::quantize_q8(id(ATTN_Q8),id(ATTN_OUT),dim)?)?;
    p.push(GpuCommand::q8_matvec(id(PROJECTED),id(260),id(ATTN_Q8),dim,dim)?)?;
    p.push(GpuCommand::scaled_residual(id(X),id(X),id(PROJECTED),dim,residual)?)?;
    // FFN in the same command stream; every intermediate stays in VRAM.
    p.push(GpuCommand::rms_norm(id(FFN_NORM),id(X),id(261),dim,eps)?)?;
    p.push(GpuCommand::quantize_q8(id(FFN_Q8),id(FFN_NORM),dim)?)?;
    p.push(GpuCommand::q8_matvec(id(GATE),id(262),id(FFN_Q8),ffn,dim)?)?;
    p.push(GpuCommand::q8_matvec(id(UP),id(263),id(FFN_Q8),ffn,dim)?)?;
    p.push(GpuCommand::swiglu(id(GATE),id(GATE),id(UP),ffn)?)?;
    p.push(GpuCommand::quantize_q8(id(GATE_Q8),id(GATE),ffn)?)?;
    p.push(GpuCommand::q8_matvec(id(DOWN),id(264),id(GATE_Q8),dim,ffn)?)?;
    p.push(GpuCommand::scaled_residual(id(X),id(X),id(DOWN),dim,residual)?)?;
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn guest_owns_complete_schedule() {
        let p=layer0_decode(2048,8192,512,32,8,64,1e-5,1.0,0.125).unwrap();
        let ops: [u32;21] = core::array::from_fn(|i| p.as_slice()[i].opcode);
        assert_eq!(ops, [2,3,1,1,1,7,7,8,9,10,3,1,6,2,3,1,1,5,3,1,6]);
        assert_eq!(p.as_slice()[2].src0,257);
        assert_eq!(p.as_slice()[7].aux,1001);
        assert_eq!(p.as_slice().len(),21);
        assert_eq!(core::mem::size_of::<GpuCommand>(),48);
    }
}
