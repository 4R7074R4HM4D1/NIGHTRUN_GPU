#![no_std]
//! NightRun-owned GPU command ABI. Does not depend on CUDA, QEMU, std or UEFI.
//!
//! Hardware backends translate GPU buffer handles and kernel commands to
//! their own allocation, DMA, and launch mechanisms. Tensor interpretation
//! and scheduling remain inside NightRun.

/// Q8_0 GGUF/NightRun block: little-endian f16 scale followed by 32 i8s.
pub mod runtime_launch; // NIGHTRUN_STAGE9C_GENERIC_GPU_LAUNCH_V1
pub const Q8_VALUES: u32 = 32;
pub const Q8_BLOCK_BYTES: u32 = 34;
pub const ABI_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuError {
    OutOfMemory,
    InvalidBuffer,
    InvalidShape,
    QueueFull,
    UnsupportedOperation,
    DeviceFailure,
    Timeout,
}

/// Backend-local opaque GPU allocation ID, NOT a guest physical address.
/// A native NVIDIA backend will map this to an allocation in GPU virtual memory.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferId(pub u32);

/// Source-stable kernel opcode: backend-specific launch parameters are not
/// permitted in this structure. Unsupported opcodes must return an error.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    Q8Matvec = 1,
    RmsNorm = 2,
    QuantizeQ8 = 3,
    ResidualAdd = 4,
    SwiGlu = 5,
    ScaledResidual = 6,
    Rope = 7,
    KvStore = 8,
    GqaScores = 9,
    GqaValue = 10,
    Q8EmbeddingRow = 11, // UEFI requests one row of resident Q8_0 tensor
    // All tensors stay on GPU; command order establishes dependencies.
}

/// Fixed-width, little-endian-on-the-wire command record (48 bytes).
/// Memory addresses NEVER cross this interface: only registered buffer IDs.
/// For Q8Matvec: dst=f32 output, src0=packed Q8 weights, src1=packed Q8 x,
/// rows/cols=weight matrix shape. Other fields must be zero.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuCommand {
    pub opcode: u32,
    pub version: u16,
    pub flags: u16,
    pub dst: u32,
    pub src0: u32,
    pub src1: u32,
    pub aux: u32,
    pub rows: u32,
    pub cols: u32,
    pub scalar0_bits: u32,
    pub scalar1_bits: u32,
    pub reserved: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<GpuCommand>() == 48);

/// The number of packed weight bytes required by a row-major Q8_0 matrix.
pub const fn q8_weight_bytes(rows: u32, cols: u32) -> Option<u64> {
    if rows == 0 || cols == 0 || cols % Q8_VALUES != 0 {
        return None;
    }
    let blocks = match (rows as u64).checked_mul((cols / Q8_VALUES) as u64) {
        Some(b) => b,
        None => return None,
    };
    blocks.checked_mul(Q8_BLOCK_BYTES as u64)
}

pub const fn q8_activation_bytes(cols: u32) -> Option<u64> {
    q8_weight_bytes(1, cols)
}

impl GpuCommand {
    /// dst=f32 normalized tensor, src0=f32 input, src1=f32 gamma,
    /// rows=number of elements, scalar0_bits=epsilon f32.
    pub fn rms_norm(dst: BufferId, input: BufferId, gamma: BufferId, len: u32, epsilon: f32) -> Result<Self, GpuError> {
        if len == 0 || len > 8192 || !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(GpuError::InvalidShape);
        }
        let mut cmd = Self::zero(Opcode::RmsNorm, dst, input, gamma);
        cmd.rows = len;
        cmd.scalar0_bits = epsilon.to_bits();
        Ok(cmd)
    }

    /// dst=packed Q8_0 vector, src0=f32 input, rows=number of elements.
    pub fn quantize_q8(dst: BufferId, input: BufferId, len: u32) -> Result<Self, GpuError> {
        if len == 0 || len % 32 != 0 { return Err(GpuError::InvalidShape); }
        let mut cmd = Self::zero(Opcode::QuantizeQ8, dst, input, BufferId(0));
        cmd.rows = len;
        Ok(cmd)
    }

    /// In-place dst[i] += src0[i], rows=number of f32 elements.
    pub fn residual_add(dst: BufferId, residual: BufferId, len: u32) -> Result<Self, GpuError> {
        if len == 0 { return Err(GpuError::InvalidShape); }
        let mut cmd = Self::zero(Opcode::ResidualAdd, dst, residual, BufferId(0));
        cmd.rows = len;
        Ok(cmd)
    }

    /// In-place SwiGLU: gate[i] = silu(gate[i]) * up[i].
    pub fn swiglu(dst: BufferId, gate: BufferId, up: BufferId, len: u32) -> Result<Self, GpuError> {
        if len == 0 { return Err(GpuError::InvalidShape); }
        let mut c = Self::zero(Opcode::SwiGlu, dst, gate, up);
        c.rows = len;
        Ok(c)
    }

    /// In-place x += scale * projection. src0=x, src1=projection.
    pub fn scaled_residual(dst: BufferId, x: BufferId, projection: BufferId,
                           len: u32, scale: f32) -> Result<Self, GpuError> {
        if len == 0 || !scale.is_finite() { return Err(GpuError::InvalidShape); }
        let mut c = Self::zero(Opcode::ScaledResidual, dst, x, projection);
        c.rows = len;
        c.scalar0_bits = scale.to_bits();
        Ok(c)
    }

    // NIGHTRUN_STAGE8D_GPU_EMBED_Q8_V1: lookup from resident Q8_0 embedding
    // tensor ID 1 into GPU hidden-state buffer 2000. UEFI owns token ID.
    pub fn q8_embedding_row(token:u32, vocab:u32, dim:u32, scale:f32)->Result<Self,GpuError> {
        if vocab==0 || vocab>262144 || token>=vocab || dim!=2048 ||
            !scale.is_finite() {return Err(GpuError::InvalidShape);}
        let mut cmd=Self::zero(Opcode::Q8EmbeddingRow,BufferId(2000),BufferId(1),BufferId(0));
        cmd.aux=token;
        cmd.rows=dim;
        cmd.cols=vocab;
        cmd.scalar0_bits=scale.to_bits();
        Ok(cmd)
    }

    fn zero(opcode: Opcode, dst: BufferId, src0: BufferId, src1: BufferId) -> Self {
        Self {
            opcode: opcode as u32,
            version: ABI_VERSION,
            flags: 0,
            dst: dst.0,
            src0: src0.0,
            src1: src1.0,
            aux: 0,
            rows: 0,
            cols: 0,
            scalar0_bits: 0,
            scalar1_bits: 0,
            reserved: [0; 2],
        }
    }

    pub fn q8_matvec(
        dst: BufferId,
        weights: BufferId,
        activation: BufferId,
        rows: u32,
        cols: u32,
    ) -> Result<Self, GpuError> {
        if q8_weight_bytes(rows, cols).is_none() {
            return Err(GpuError::InvalidShape);
        }
        Ok(Self {
            opcode: Opcode::Q8Matvec as u32,
            version: ABI_VERSION,
            flags: 0,
            dst: dst.0,
            src0: weights.0,
            src1: activation.0,
            aux: 0,
            rows,
            cols,
            scalar0_bits: 0,
            scalar1_bits: 0,
            reserved: [0; 2],
        })
    }
}

/// No heap allocation in command construction or submission; the entire
/// inference graph can eventually be queued then synchronized once.
pub struct CommandList<const N: usize> {
    commands: [GpuCommand; N],
    len: usize,
}
impl<const N: usize> CommandList<N> {
    pub const fn new() -> Self {
        Self {
            commands: [GpuCommand {
                opcode: 0, version: 0, flags: 0, dst: 0, src0: 0,
                src1: 0, aux: 0, rows: 0, cols: 0, scalar0_bits: 0,
                scalar1_bits: 0, reserved: [0; 2],
            }; N],
            len: 0,
        }
    }
    pub fn push(&mut self, command: GpuCommand) -> Result<(), GpuError> {
        if self.len == N {
            return Err(GpuError::QueueFull);
        }
        self.commands[self.len] = command;
        self.len += 1;
        Ok(())
    }
    pub fn as_slice(&self) -> &[GpuCommand] {
        &self.commands[..self.len]
    }
    pub fn clear(&mut self) {
        self.len = 0;
    }
}
impl<const N: usize> Default for CommandList<N> {
    fn default() -> Self { Self::new() }
}

/// A persistent execution device, not an LLM backend. Models, GPU commands,
/// attention/KV semantics and sampling remain controlled by NightRun.
///
/// All uploads are completed or ordered ahead of subsequent submitted work.
/// The backend must ensure coherent ordered execution until synchronize().
pub trait GpuBackend {
    fn allocate(&mut self, bytes: u64, alignment: u32) -> Result<BufferId, GpuError>;
    fn free(&mut self, buffer: BufferId) -> Result<(), GpuError>;
    fn upload(&mut self, dst: BufferId, offset: u64, data: &[u8]) -> Result<(), GpuError>;
    fn download(&mut self, src: BufferId, offset: u64, data: &mut [u8]) -> Result<(), GpuError>;
    fn submit(&mut self, commands: &[GpuCommand]) -> Result<(), GpuError>;
    fn synchronize(&mut self) -> Result<(), GpuError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layout_matches_gguf_q8() {
        assert_eq!(q8_weight_bytes(128256, 2048), Some(279_085_056));
        assert_eq!(q8_activation_bytes(2048), Some(2_176));
        assert_eq!(core::mem::size_of::<GpuCommand>(), 48);
    }
    #[test]
    fn invalid_shapes_rejected() {
        assert_eq!(q8_weight_bytes(1, 31), None);
        assert_eq!(q8_weight_bytes(0, 32), None);
        assert_eq!(GpuCommand::q8_matvec(BufferId(1), BufferId(2), BufferId(3), 5, 31), Err(GpuError::InvalidShape));
    }
    #[test]
    fn layer_chain_uses_same_abi_and_preserves_order() {
        let mut cmds = CommandList::<4>::new();
        cmds.push(GpuCommand::rms_norm(BufferId(3), BufferId(1), BufferId(2), 2048, 1.0e-5).unwrap()).unwrap();
        cmds.push(GpuCommand::quantize_q8(BufferId(4), BufferId(3), 2048).unwrap()).unwrap();
        cmds.push(GpuCommand::q8_matvec(BufferId(5), BufferId(6), BufferId(4), 2048, 2048).unwrap()).unwrap();
        cmds.push(GpuCommand::residual_add(BufferId(5), BufferId(1), 2048).unwrap()).unwrap();
        assert_eq!(cmds.as_slice()[0].opcode, 2);
        assert_eq!(cmds.as_slice()[1].opcode, 3);
        assert_eq!(cmds.as_slice()[2].opcode, 1);
        assert_eq!(cmds.as_slice()[3].opcode, 4);
        assert_eq!(core::mem::size_of::<GpuCommand>(), 48);
        assert_eq!(GpuCommand::quantize_q8(BufferId(1), BufferId(2), 2047), Err(GpuError::InvalidShape));
    }
    #[test]
    fn command_queue_is_fixed_and_ordered() {
        let mut q = CommandList::<2>::new();
        let c = GpuCommand::q8_matvec(BufferId(1), BufferId(2), BufferId(3), 16, 2048).unwrap();
        q.push(c).unwrap();
        q.push(c).unwrap();
        assert_eq!(q.push(c), Err(GpuError::QueueFull));
        assert_eq!(q.as_slice().len(), 2);
        assert_eq!(q.as_slice()[0].opcode, Opcode::Q8Matvec as u32);
        q.clear();
        assert_eq!(q.as_slice().len(), 0);
    }
}

/// Persistent GPU tensor registry shared with a future native UEFI backend.
pub mod resident;

pub mod layer_plan;

pub mod stack_plan;
