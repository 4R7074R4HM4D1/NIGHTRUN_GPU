//! Stable opaque tensor IDs and model metadata for persistent VRAM residency.
//! UEFI native and QEMU CUDA backends use these SAME IDs and descriptors.
//! No GPU address or CUDA type crosses the interface.
use crate::BufferId;

pub const EMBED_ID:BufferId=BufferId(1);
pub const OUTPUT_NORM_ID:BufferId=BufferId(2);
pub const OUTPUT_ID:BufferId=BufferId(3);
pub const KV_KEYS_ID:BufferId=BufferId(1000);
pub const KV_VALUES_ID:BufferId=BufferId(1001);
pub const SCRATCH_ID:BufferId=BufferId(1002);
pub const SLOT_ATTN_NORM:u32=0;
pub const SLOT_Q:u32=1;
pub const SLOT_K:u32=2;
pub const SLOT_V:u32=3;
pub const SLOT_O:u32=4;
pub const SLOT_FFN_NORM:u32=5;
pub const SLOT_GATE:u32=6;
pub const SLOT_UP:u32=7;
pub const SLOT_DOWN:u32=8;
pub const SLOT_Q_NORM:u32=9;
pub const SLOT_K_NORM:u32=10;

pub const fn layer_tensor_id(layer:u32,slot:u32)->Option<BufferId> {
    if layer>=32 || slot>=16 {None}
    else {Some(BufferId(256+layer*16+slot))}
}

/// Serialization is done explicitly, so no compiler padding is transmitted.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct ModelTensorDesc {
    pub id:u32,
    /// 0=FP32, 1=Q8_0, 2=Q4_K, 3=Q6_K
    pub dtype:u32,
    pub rows:u32,
    pub cols:u32,
    pub byte_len:u64,
}
impl ModelTensorDesc {
    pub const fn new(id:u32,dtype:u32,rows:u32,cols:u32,byte_len:u64)->Self {
        Self{id,dtype,rows,cols,byte_len}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn ids_do_not_collide() {
        assert_eq!(layer_tensor_id(0,SLOT_ATTN_NORM),Some(BufferId(256)));
        assert_eq!(layer_tensor_id(15,SLOT_DOWN),Some(BufferId(504)));
        assert_eq!(layer_tensor_id(32,0),None);
        assert_eq!(layer_tensor_id(0,16),None);
        assert!(layer_tensor_id(31,15).unwrap().0<KV_KEYS_ID.0);
    }
}
