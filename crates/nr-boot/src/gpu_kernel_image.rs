//! NIGHTRUN_STAGE9A_UEFI_OWNED_CUBIN_V1 - NightRun carries compiled sm_89 CUBIN images.
//! CUDA is used only by the temporary driver bridge, never by the UEFI app.
//! Build the three CUBIN files *before* compiling nr-boot.
//!
//! `include_bytes!` stores the exact GPU binaries in BOOTX64.EFI.

pub static MLP: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"),
    "/../../gpu-native/build/nr_layer_mlp_sm89.cubin"));
pub static ATTENTION: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"),
    "/../../gpu-native/build/nr_attention_sm89.cubin"));
pub static ROPE_KV: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"),
    "/../../gpu-native/build/nr_rope_kv_sm89.cubin"));
