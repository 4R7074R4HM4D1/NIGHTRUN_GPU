//! NightRun UEFI-owned 16-layer decode plan (no CUDA, OS, or model logic in backend).
//! Every command is 48 bytes and includes an opaque layer index in its `flags`.
use crate::{CommandList,GpuError};
use crate::layer_plan::{layer0_decode,COMMANDS};

pub const LAYERS: usize = 16;
pub const ALL_COMMANDS: usize = LAYERS * COMMANDS;

/// Construct the complete transformer stack from NightRun model dimensions.
/// Intermediates use the SAME scratch handles between layers, keeping X resident.
/// Weight IDs are 256 + layer*16 + tensor_slot; no host chooses a layer order.
pub fn decoder_stack(dim:u32,ffn:u32,kv:u32,heads:u32,kv_heads:u32,hd:u32,
                     eps:f32,residual:f32,score_scale:f32)
                     ->Result<CommandList<ALL_COMMANDS>,GpuError> {
    let template=layer0_decode(dim,ffn,kv,heads,kv_heads,hd,eps,residual,score_scale)?;
    let mut all=CommandList::<ALL_COMMANDS>::new();
    for layer in 0..LAYERS {
        for entry in template.as_slice() {
            let mut cmd=*entry;
            cmd.flags=layer as u16;
            // Transform ONLY weight handles. Other IDs reference shared
            // scratch (2000+) or layer-strided KV allocations (1000,1001).
            for handle in [&mut cmd.dst,&mut cmd.src0,&mut cmd.src1,&mut cmd.aux] {
                if (256..=264).contains(handle) { *handle += layer as u32 * 16; }
            }
            all.push(cmd)?;
        }
    }
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_sixteen_layers_authored_by_guest() {
        let p=decoder_stack(2048,8192,512,32,8,64,1e-5,1.,0.125).unwrap();
        assert_eq!(p.as_slice().len(),336);
        assert_eq!(core::mem::size_of::<crate::GpuCommand>(),48);
        for (layer,chunk) in p.as_slice().chunks_exact(21).enumerate() {
            assert_eq!(chunk.len(),21);
            assert!(chunk.iter().all(|c|c.flags==layer as u16));
            assert_eq!(chunk[0].src1,256+(layer as u32)*16);
            assert_eq!(chunk[2].src0,257+(layer as u32)*16);
            assert_eq!(chunk[12].dst,2000); // persistent hidden state
            assert_eq!(chunk[7].dst,1000);  // layer-strided KV (driver interprets flags)
            assert_eq!(chunk[20].opcode,6);
        }
    }
}

// NIGHTRUN_STAGE8C_GPU_FINAL_V1: UEFI constructs its own 3-command final head.
// Input hidden is buffer 2000, output norm is persistent tensor 2, and
// classifier is resident tensor 3 (including tied-embedding alias).
pub fn output_head(dim:u32,vocab:u32,eps:f32)->Result<CommandList<3>,GpuError> {
    use crate::{BufferId,GpuCommand};
    if dim!=2048 || vocab==0 || vocab>262144 || !(eps>0.0 && eps.is_finite()) {
        return Err(GpuError::InvalidShape);
    }
    let mut commands=CommandList::<3>::new();
    commands.push(GpuCommand::rms_norm(BufferId(2017),BufferId(2000),BufferId(2),dim,eps)?)?;
    commands.push(GpuCommand::quantize_q8(BufferId(2018),BufferId(2017),dim)?)?;
    commands.push(GpuCommand::q8_matvec(BufferId(2019),BufferId(3),BufferId(2018),vocab,dim)?)?;
    Ok(commands)
}
#[cfg(test)]
mod output_head_tests {
    use super::*;
    #[test]
    fn final_head_is_guest_authored() {
        let p=output_head(2048,128256,1e-5).unwrap();
        assert_eq!(p.as_slice().len(),3);
        assert_eq!([p.as_slice()[0].opcode,p.as_slice()[1].opcode,p.as_slice()[2].opcode],[2,3,1]);
        assert_eq!(p.as_slice()[2].src0,3);
    }
}
