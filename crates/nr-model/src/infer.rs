//! Llama-family forward pass over .nrm weights: RMSNorm -> GQA attention
//! with rotary embeddings and an f16 KV cache -> SwiGLU MLP, quantized
//! matvecs throughout. Supports the Llama-3 recipe and the Qwen3 variant
//! (per-head Q/K RMSNorm, attention width != hidden width).
//!
//! Weight matrices are dtype-tagged (Q8_0 / Q4_K / Q6_K); dispatch happens
//! once per matvec call, never inside row loops. Activations are quantized
//! to the matching format (Q8_0 or Q8_K). Single token per step; all
//! buffers preallocated — generation never allocates.

use nr_tensor::kernels::{axpy_f16, dot_f16, matmul_q8, matvec_q8};
use nr_tensor::kquant::{self, BlockQ4K, BlockQ6K, BlockQ8K, QK_K};
use nr_tensor::q8::{self, BlockQ8_0, QK8_0};
use nr_tensor::{f32_to_f16, rope, vec as tvec};

// GPU_Q8_BYTE_PROBE_V1: Bitwise inspection of the same 34-byte packed blocks
// used by the CPU Q8 GEMV path. Read-only borrow, no allocation.
// GPU_GEMV_SCALAR_REFERENCE_V1: Q8_0 weights passed as read-only bytes.
fn gpu_q8_probe_weights<'a>(w:QMat<'a>)->&'a [u8] {
    match w {QMat::Q8(blocks)=>gpu_q8_probe_bytes(blocks),_=>&[]}
}
fn gpu_q8_probe_bytes(blocks:&[BlockQ8_0])->&[u8] {
    let bytes=blocks.len()*core::mem::size_of::<BlockQ8_0>();
    // SAFETY: BlockQ8_0 has an initialized repr(C) packed representation;
    // the byte view cannot outlive the borrowed blocks.
    unsafe {core::slice::from_raw_parts(blocks.as_ptr() as *const u8,bytes)}
}

use crate::format::{Arch, Model, ParseError, TensorDtype, TensorKind, TensorView};

/// A dtype-tagged weight matrix (rows x cols), viewed in place.
#[derive(Clone, Copy)]
pub enum QMat<'m> {
    Q8(&'m [BlockQ8_0]),
    Q4K(&'m [BlockQ4K]),
    Q6K(&'m [BlockQ6K]),
}

impl<'m> QMat<'m> {
    fn from_view(v: TensorView<'m>) -> Result<QMat<'m>, ParseError> {
        Ok(match v.dtype {
            TensorDtype::Q8_0 => QMat::Q8(v.q8()),
            TensorDtype::Q4K => QMat::Q4K(v.q4k()),
            TensorDtype::Q6K => QMat::Q6K(v.q6k()),
            TensorDtype::F32 => return Err(ParseError::BadTable),
        })
    }

    /// Dequantize row `r` (cols values) into `out`.
    pub fn dequant_row(&self, r: usize, cols: usize, out: &mut [f32]) {
        match self {
            QMat::Q8(blocks) => {
                let bpr = cols / QK8_0;
                q8::dequantize(&blocks[r * bpr..(r + 1) * bpr], out);
            }
            QMat::Q4K(blocks) => {
                let bpr = cols / QK_K;
                for (i, b) in blocks[r * bpr..(r + 1) * bpr].iter().enumerate() {
                    kquant::dequant_q4k(b, &mut out[i * QK_K..(i + 1) * QK_K]);
                }
            }
            QMat::Q6K(blocks) => {
                let bpr = cols / QK_K;
                for (i, b) in blocks[r * bpr..(r + 1) * bpr].iter().enumerate() {
                    kquant::dequant_q6k(b, &mut out[i * QK_K..(i + 1) * QK_K]);
                }
            }
        }
    }
}

/// Dimensions captured as plain usizes.
#[derive(Clone, Copy)]
pub struct Dims {
    pub dim: usize,
    /// n_heads * head_dim; equals `dim` for Llama, differs for Qwen3.
    pub att_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub kv_dim: usize,
    pub ffn_dim: usize,
    pub vocab: usize,
    pub ctx: usize,
}

pub struct LayerWeights<'m> {
    pub attn_norm: &'m [f32],
    /// Per-head RMSNorm weights (head_dim each); Qwen3 only.
    pub q_norm: Option<&'m [f32]>,
    pub k_norm: Option<&'m [f32]>,
    pub wq: QMat<'m>,
    pub wk: QMat<'m>,
    pub wv: QMat<'m>,
    pub wo: QMat<'m>,
    pub ffn_norm: &'m [f32],
    pub w_gate: QMat<'m>,
    pub w_up: QMat<'m>,
    pub w_down: QMat<'m>,
}

pub struct Weights<'m> {
    pub embed: QMat<'m>,
    pub layers: alloc::vec::Vec<LayerWeights<'m>>,
    pub out_norm: &'m [f32],
    /// Classifier; equals `embed` for tied models.
    pub output: QMat<'m>,
}

/// Prompt tokens processed per batched-prefill pass. Weight rows are
/// reused from cache across the batch, so prompt processing reads weights
/// ~batch-fold less often than token-at-a-time decode.
pub const MAX_BATCH: usize = 64;

/// NIGHTRUN_STAGE8A_UEFI_GPU_PREFILL_V1: set false and rebuild for a CPU-prefill A/B baseline.
pub const STAGE8A_GPU_PREFILL_ENABLED: bool = true;
/// Stage 8B A/B control: false restores Stage 8A sequential GPU prefill.
pub const STAGE8B_GPU_BATCH_ENABLED: bool = true;
/// NIGHTRUN_STAGE8D_GPU_EMBED_Q8_V1. Set false for CPU embedding A/B baseline.
pub const STAGE8D_GPU_EMBED_ENABLED: bool = true;

/// Scratch activation-quantization buffers; a matvec input is quantized
/// into the format its weight matrix needs.
struct Acts {
    q8: &'static mut [BlockQ8_0],
    q8k: &'static mut [BlockQ8K],
}

impl Acts {
    /// Quantize `x` and run `w.matvec` into y (rows = y.len(), cols = x.len()).
    fn matvec(&mut self, y: &mut [f32], w: QMat, x: &[f32]) {
        let cols = x.len();
        let rows = y.len();
        match w {
            QMat::Q8(blocks) => {
                let xq = &mut self.q8[..cols / QK8_0];
                q8::quantize(x, xq);
                matvec_q8(y, blocks, xq, rows, cols);
            }
            QMat::Q4K(blocks) => {
                let xk = &mut self.q8k[..cols / QK_K];
                kquant::quantize_q8k(x, xk);
                kquant::matvec_q4k(y, blocks, xk, rows, cols);
            }
            QMat::Q6K(blocks) => {
                let xk = &mut self.q8k[..cols / QK_K];
                kquant::quantize_q8k(x, xk);
                kquant::matvec_q6k(y, blocks, xk, rows, cols);
            }
        }
    }

    /// Batched: quantize `batch` activation rows (each `cols` long) and run
    /// `w` over all of them. `y` is `[batch][rows]`. Quantization per row
    /// uses the same functions as the decode path, so results are
    /// bit-identical to per-token matvec.
    fn matmul(&mut self, y: &mut [f32], w: QMat, xs: &[f32], batch: usize) {
        let cols = xs.len() / batch;
        let rows = y.len() / batch;
        match w {
            QMat::Q8(blocks) => {
                let bpr = cols / QK8_0;
                let xq = &mut self.q8[..batch * bpr];
                for b in 0..batch {
                    q8::quantize(
                        &xs[b * cols..(b + 1) * cols],
                        &mut xq[b * bpr..(b + 1) * bpr],
                    );
                }
                matmul_q8(y, blocks, xq, rows, cols, batch);
            }
            QMat::Q4K(blocks) => {
                let bpr = cols / QK_K;
                let xk = &mut self.q8k[..batch * bpr];
                for b in 0..batch {
                    kquant::quantize_q8k(
                        &xs[b * cols..(b + 1) * cols],
                        &mut xk[b * bpr..(b + 1) * bpr],
                    );
                }
                kquant::matmul_q4k(y, blocks, xk, rows, cols, batch);
            }
            QMat::Q6K(blocks) => {
                let bpr = cols / QK_K;
                let xk = &mut self.q8k[..batch * bpr];
                for b in 0..batch {
                    kquant::quantize_q8k(
                        &xs[b * cols..(b + 1) * cols],
                        &mut xk[b * bpr..(b + 1) * bpr],
                    );
                }
                kquant::matmul_q6k(y, blocks, xk, rows, cols, batch);
            }
        }
    }

    /// Batched matmuls sharing one input; each needed activation format is
    /// quantized exactly once.
    fn matmul_group(&mut self, jobs: &mut [(&mut [f32], QMat)], xs: &[f32], batch: usize) {
        let cols = xs.len() / batch;
        let mut q8_ready = false;
        let mut q8k_ready = false;
        for (y, w) in jobs.iter_mut() {
            let rows = y.len() / batch;
            match w {
                QMat::Q8(blocks) => {
                    let bpr = cols / QK8_0;
                    if !q8_ready {
                        for b in 0..batch {
                            q8::quantize(
                                &xs[b * cols..(b + 1) * cols],
                                &mut self.q8[b * bpr..(b + 1) * bpr],
                            );
                        }
                        q8_ready = true;
                    }
                    matmul_q8(y, blocks, &self.q8[..batch * bpr], rows, cols, batch);
                }
                QMat::Q4K(blocks) => {
                    let bpr = cols / QK_K;
                    if !q8k_ready {
                        for b in 0..batch {
                            kquant::quantize_q8k(
                                &xs[b * cols..(b + 1) * cols],
                                &mut self.q8k[b * bpr..(b + 1) * bpr],
                            );
                        }
                        q8k_ready = true;
                    }
                    kquant::matmul_q4k(y, blocks, &self.q8k[..batch * bpr], rows, cols, batch);
                }
                QMat::Q6K(blocks) => {
                    let bpr = cols / QK_K;
                    if !q8k_ready {
                        for b in 0..batch {
                            kquant::quantize_q8k(
                                &xs[b * cols..(b + 1) * cols],
                                &mut self.q8k[b * bpr..(b + 1) * bpr],
                            );
                        }
                        q8k_ready = true;
                    }
                    kquant::matmul_q6k(y, blocks, &self.q8k[..batch * bpr], rows, cols, batch);
                }
            }
        }
    }

    /// Run several matvecs that share one input vector, quantizing each
    /// needed format exactly once.
    fn matvec_group(&mut self, jobs: &mut [(&mut [f32], QMat)], x: &[f32]) {
        let cols = x.len();
        let mut q8_ready = false;
        let mut q8k_ready = false;
        for (y, w) in jobs.iter_mut() {
            match w {
                QMat::Q8(blocks) => {
                    if !q8_ready {
                        q8::quantize(x, &mut self.q8[..cols / QK8_0]);
                        q8_ready = true;
                    }
                    matvec_q8(y, blocks, &self.q8[..cols / QK8_0], y.len(), cols);
                }
                QMat::Q4K(blocks) => {
                    if !q8k_ready {
                        kquant::quantize_q8k(x, &mut self.q8k[..cols / QK_K]);
                        q8k_ready = true;
                    }
                    kquant::matvec_q4k(y, blocks, &self.q8k[..cols / QK_K], y.len(), cols);
                }
                QMat::Q6K(blocks) => {
                    if !q8k_ready {
                        kquant::quantize_q8k(x, &mut self.q8k[..cols / QK_K]);
                        q8k_ready = true;
                    }
                    kquant::matvec_q6k(y, blocks, &self.q8k[..cols / QK_K], y.len(), cols);
                }
            }
        }
    }
}

/// Optional UEFI-accessible inference backend. No heap allocation in compute().
/// Returns false to trigger the original CPU implementation.
// Read a serialized TSC on x86-64. The non-x86 value is a placeholder;
// CPU timing is used only in the x86-64 QEMU/KVM experiment.
#[inline]
fn cpu_profile_ticks() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: LFENCE/RDTSC are supported on the x86-64 QEMU target.
        unsafe {
            core::arch::x86_64::_mm_lfence();
            let t = core::arch::x86_64::_rdtsc();
            core::arch::x86_64::_mm_lfence();
            t
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    { 0 }
}

pub trait ClassifierBackend {
    fn compute(&mut self, input: &[f32], logits: &mut [f32]) -> bool;
}

/// Optional full first-layer MLP compute engine.  NightRun owns all layer
/// scheduling, weights, activation and parity validation; backend may not
/// modify x unless it completed successfully.
/// Optional GPU execution of live decode attention for one transformer layer.
/// CPU projection, cache mirror, and fallback remain NightRun-owned.
/// Execute an entire transformer layer using resident model allocations.
/// When compute() returns false, x and the current CPU KV cache must remain
/// valid for the normal NightRun CPU path.
/// All transformer scheduling stays in NightRun UEFI, NEVER the driver.
/// compute() must not alter x or CPU KV unless all GPU results validate.
pub trait StackBackend {
    fn compute(&mut self,x:&mut[f32],pos:usize,
               keys:&mut[u16],vals:&mut[u16])->bool;
    // NIGHTRUN_STAGE8A_UEFI_GPU_PREFILL_V1: observer hooks; UEFI controls prompt scheduling.
    // NIGHTRUN_STAGE8B_UEFI_16TOKEN_PREFILL_BATCH_V1: UEFI builds ALL embeddings
    // and owns ordered token positions. Backend only replays the command buffer.
    // Execute GPU embedding-row lookup followed by the UEFI-authored stack.
    // GPU result + KV rows commit together, or the CPU path is used instead.
    fn compute_token_q8(&mut self,_token:u32,_vocab:usize,_scale:f32,
        _x:&mut[f32],_pos:usize,_keys:&mut[u16],_vals:&mut[u16])->bool {false}
    // GPU embeddings for a UEFI-authorized contiguous prompt batch.
    fn compute_prefill_tokens_q8(&mut self,_tokens:&[u32],_vocab:usize,_scale:f32,
        _start:usize,_last:&mut[f32],_keys:&mut[u16],_vals:&mut[u16])->bool {false}
    fn compute_prefill_batch(&mut self,_embeddings:&[f32],_start:usize,_count:usize,
        _last:&mut[f32],_keys:&mut[u16],_vals:&mut[u16])->bool { false }
    // NIGHTRUN_STAGE8C_GPU_FINAL_V1: UEFI authored output-head commands.
    // Backend must commit logits only if the entire GPU request succeeded.
    fn compute_final(&mut self,_hidden:&[f32],_logits:&mut[f32])->bool { false }
    fn on_prefill_complete(&mut self,_start:usize,_tokens:usize) {}
    fn on_prefill_failure(&mut self,_failed_pos:usize,_gpu_tokens:usize) {}
    fn on_parity_check(&mut self,_ok:bool,_max_error:f32) {}
    fn trace_compare_layer(&mut self,_layer:usize,_cpu:&[f32]) {}
    // Stage 7 diagnostic: UEFI requests an isolated, exact-CPU-input replay.
    fn trace_begin_layer(&mut self,_layer:usize,_pos:usize,_cpu_input:&[f32],_keys:&[u16],_vals:&[u16])->bool {false}
    fn trace_compare_stage(&mut self,_layer:usize,_stage:usize,_cpu:&[f32]) {}
    fn trace_compare_cache(&mut self,_layer:usize,_stage:usize,_cpu:&[u16]) {}
    // GPU_DYNAMIC_Q8_ORIGIN_V1: CPU prequantization buffer supplied for comparisons only.
    fn trace_compare_q8(&mut self,_layer:usize,_stage:usize,
        _cpu:&[u8],_pre_quant:&[f32]) {}
    fn trace_compare_gemv(&mut self,_layer:usize,_stage:usize,
        _weights:&[u8],_activation:&[u8],_cpu:&[f32]) {}
}

pub trait LayerBackend {
    fn compute(&mut self, x: &mut [f32], pos: usize,
               keys: &mut [u16], vals: &mut [u16]) -> bool;
    fn on_parity_check(&mut self, _ok: bool, _max_error: f32) {}
}

pub trait AttentionBackend {
    fn compute(&mut self, query: &[f32], keys: &[u16], vals: &[u16],
               pos: usize, output: &mut [f32]) -> bool;
    fn on_parity_check(&mut self, _ok: bool, _max_error: f32) {}
}

pub trait MlpBackend {
    fn compute(&mut self, x: &mut [f32]) -> bool;
    fn on_parity_check(&mut self, _ok: bool, _max_error: f32) {}
}

pub struct InferCtx<'m> {
    pub dims: Dims,
    pub weights: Weights<'m>,
    norm_eps: f32,
    rope_style: rope::RopeStyle,
    // muP scalars (neutral 1.0 / precomputed for non-Granite models).
    embed_scale: f32,
    att_score_scale: f32,
    residual_scale: f32,
    logit_recip: f32,
    freqs: &'static mut [f32],
    // Scratch (all preallocated; generation never allocates).
    x: &'static mut [f32],
    xb: &'static mut [f32],
    xb2: &'static mut [f32],
    q: &'static mut [f32],
    attn_out: &'static mut [f32],
    k: &'static mut [f32],
    v: &'static mut [f32],
    att: &'static mut [f32],
    gate: &'static mut [f32],
    up: &'static mut [f32],
    logits: &'static mut [f32],
    acts: Acts,
    // Batched-prefill scratch, [MAX_BATCH][*]; reuses `att`/`logits`.
    bx: &'static mut [f32],
    bxb: &'static mut [f32],
    bq: &'static mut [f32],
    battn: &'static mut [f32],
    bk: &'static mut [f32],
    bv: &'static mut [f32],
    bgate: &'static mut [f32],
    bup: &'static mut [f32],
    /// f16 bits, [n_layers][ctx][kv_dim].
    key_cache: &'static mut [u16],
    val_cache: &'static mut [u16],
    pub pos: usize,
    classifier_backend: Option<&'static mut dyn ClassifierBackend>,
    stack_backend: Option<&'static mut dyn StackBackend>,
    stack_checks_remaining: usize,
    layer_backend: Option<&'static mut dyn LayerBackend>,
    layer_checks_remaining: usize,
    attention_backend: Option<&'static mut dyn AttentionBackend>,
    attention_checks_remaining: usize,
    mlp_backend: Option<&'static mut dyn MlpBackend>,
    mlp_checks_remaining: usize,
    classifier_ref: &'static mut [f32],
    classifier_checks: usize,
    cpu_classifier_calls: u64,
    cpu_classifier_ticks: u64,
    cpu_classifier_max_ticks: u64,
}

/// Allocation callback: (bytes, align) -> pointer valid for 'static.
pub type AllocFn<'f> = &'f mut dyn FnMut(usize, usize) -> *mut u8;

fn take<T>(alloc: AllocFn, n: usize) -> &'static mut [T] {
    let bytes = (n * core::mem::size_of::<T>()).max(1);
    let ptr = alloc(bytes, core::mem::align_of::<T>().max(64));
    assert!(!ptr.is_null(), "inference arena exhausted");
    // SAFETY: the callback contract hands us exclusive, zeroed, aligned
    // memory that lives for the session.
    unsafe { core::slice::from_raw_parts_mut(ptr as *mut T, n) }
}

fn dims_of(model: &Model, ctx: usize) -> Dims {
    let m = &model.meta;
    Dims {
        dim: m.dim as usize,
        att_dim: (m.n_heads * m.head_dim) as usize,
        n_layers: m.n_layers as usize,
        n_heads: m.n_heads as usize,
        n_kv_heads: m.n_kv_heads as usize,
        head_dim: m.head_dim as usize,
        kv_dim: (m.n_kv_heads * m.head_dim) as usize,
        ffn_dim: m.ffn_dim as usize,
        vocab: m.vocab as usize,
        ctx,
    }
}

impl<'m> InferCtx<'m> {
    /// Bytes of scratch + KV needed for a given context length (for
    /// sizing arenas before construction).
    pub fn required_bytes(model: &Model, ctx: usize) -> usize {
        let d = dims_of(model, ctx);
        let maxd = d.dim.max(d.att_dim).max(d.ffn_dim);
        let f32s = d.dim * 3
            + d.att_dim * 2
            + d.kv_dim * 2
            + d.n_heads * ctx
            + d.ffn_dim * 2
            + d.vocab * 2
            + d.head_dim / 2;
        let batch_f32s = MAX_BATCH * (d.dim * 2 + d.att_dim * 2 + d.kv_dim * 2 + d.ffn_dim * 2);
        let acts = MAX_BATCH
            * (maxd / QK8_0 * core::mem::size_of::<BlockQ8_0>()
                + maxd / QK_K * core::mem::size_of::<BlockQ8K>());
        let kv = 2 * d.n_layers * ctx * d.kv_dim * 2;
        (f32s + batch_f32s) * 4 + acts + kv + 64 * 40 // alignment slack
    }

    pub fn new(
        model: &'m Model<'m>,
        ctx: usize,
        alloc: AllocFn,
    ) -> Result<InferCtx<'m>, ParseError> {
        let m = &model.meta;
        let dims = dims_of(model, ctx);

        // Shape-checked tensor accessors.
        let f32_tensor =
            |kind: TensorKind, layer: usize, len: usize| -> Result<&'m [f32], ParseError> {
                let v = model.tensor(kind, layer)?;
                let s = v.f32();
                if s.len() != len {
                    return Err(ParseError::MissingTensor("bad f32 tensor shape"));
                }
                Ok(s)
            };
        let mat = |kind: TensorKind,
                   layer: usize,
                   rows: usize,
                   cols: usize|
         -> Result<QMat<'m>, ParseError> {
            let v = model.tensor(kind, layer)?;
            if v.rows != rows || v.cols != cols {
                return Err(ParseError::MissingTensor("bad matrix shape"));
            }
            QMat::from_view(v)
        };

        let embed = mat(TensorKind::TokEmbed, 0, dims.vocab, dims.dim)?;
        let is_qwen = m.arch == Arch::Qwen3;
        // Non-Qwen families must not carry qk-norm tensors.
        if !is_qwen && model.has_tensor(TensorKind::AttnQNorm) {
            return Err(ParseError::MissingTensor(
                "unexpected attn_q_norm for this arch",
            ));
        }
        let mut layers = alloc::vec::Vec::with_capacity(dims.n_layers);
        for l in 0..dims.n_layers {
            layers.push(LayerWeights {
                attn_norm: f32_tensor(TensorKind::AttnNorm, l, dims.dim)?,
                q_norm: if is_qwen {
                    Some(f32_tensor(TensorKind::AttnQNorm, l, dims.head_dim)?)
                } else {
                    None
                },
                k_norm: if is_qwen {
                    Some(f32_tensor(TensorKind::AttnKNorm, l, dims.head_dim)?)
                } else {
                    None
                },
                wq: mat(TensorKind::AttnQ, l, dims.att_dim, dims.dim)?,
                wk: mat(TensorKind::AttnK, l, dims.kv_dim, dims.dim)?,
                wv: mat(TensorKind::AttnV, l, dims.kv_dim, dims.dim)?,
                wo: mat(TensorKind::AttnO, l, dims.dim, dims.att_dim)?,
                ffn_norm: f32_tensor(TensorKind::FfnNorm, l, dims.dim)?,
                w_gate: mat(TensorKind::FfnGate, l, dims.ffn_dim, dims.dim)?,
                w_up: mat(TensorKind::FfnUp, l, dims.ffn_dim, dims.dim)?,
                w_down: mat(TensorKind::FfnDown, l, dims.dim, dims.ffn_dim)?,
            });
        }
        let out_norm = f32_tensor(TensorKind::OutputNorm, 0, dims.dim)?;
        let output = if m.flags & crate::format::FLAG_TIED_EMBEDDINGS != 0 {
            embed
        } else {
            mat(TensorKind::Output, 0, dims.vocab, dims.dim)?
        };
        let weights = Weights {
            embed,
            layers,
            out_norm,
            output,
        };

        // Rope frequencies: base theta spectrum, then either the GGUF's
        // precomputed frequency factors (divisors) or the header's Llama-3
        // scaling parameters.
        let freqs: &'static mut [f32] = take(alloc, dims.head_dim / 2);
        let scaling = (m.rope_factor > 0.0).then_some(rope::Llama3Scaling {
            factor: m.rope_factor,
            low_freq_factor: m.rope_low,
            high_freq_factor: m.rope_high,
            original_context: m.rope_orig_ctx,
        });
        if model.has_tensor(TensorKind::RopeFreqs) {
            rope::rope_freqs(freqs, dims.head_dim, m.rope_theta, None);
            let factors = model.tensor(TensorKind::RopeFreqs, 0)?.f32();
            for (f, &d) in freqs.iter_mut().zip(factors) {
                *f /= d;
            }
        } else {
            rope::rope_freqs(freqs, dims.head_dim, m.rope_theta, scaling);
        }

        let maxd = dims.dim.max(dims.att_dim).max(dims.ffn_dim);
        Ok(InferCtx {
            norm_eps: m.norm_eps,
            rope_style: match m.arch {
                Arch::Llama3 | Arch::Granite => rope::RopeStyle::Adjacent,
                Arch::Qwen3 => rope::RopeStyle::Neox,
            },
            embed_scale: m.embed_scale,
            att_score_scale: if m.attn_scale != 0.0 {
                m.attn_scale
            } else {
                1.0 / libm::sqrtf(m.head_dim as f32)
            },
            residual_scale: m.residual_scale,
            logit_recip: 1.0 / m.logit_scale,
            freqs,
            x: take(alloc, dims.dim),
            xb: take(alloc, dims.dim),
            xb2: take(alloc, dims.dim),
            q: take(alloc, dims.att_dim),
            attn_out: take(alloc, dims.att_dim),
            k: take(alloc, dims.kv_dim),
            v: take(alloc, dims.kv_dim),
            att: take(alloc, dims.n_heads * ctx),
            gate: take(alloc, dims.ffn_dim),
            up: take(alloc, dims.ffn_dim),
            logits: take(alloc, dims.vocab),
            acts: Acts {
                q8: take(alloc, MAX_BATCH * (maxd / QK8_0)),
                q8k: take(alloc, MAX_BATCH * (maxd / QK_K)),
            },
            bx: take(alloc, MAX_BATCH * dims.dim),
            bxb: take(alloc, MAX_BATCH * dims.dim.max(dims.att_dim)),
            bq: take(alloc, MAX_BATCH * dims.att_dim),
            battn: take(alloc, MAX_BATCH * dims.att_dim),
            bk: take(alloc, MAX_BATCH * dims.kv_dim),
            bv: take(alloc, MAX_BATCH * dims.kv_dim),
            bgate: take(alloc, MAX_BATCH * dims.ffn_dim),
            bup: take(alloc, MAX_BATCH * dims.ffn_dim),
            key_cache: take(alloc, dims.n_layers * ctx * dims.kv_dim),
            val_cache: take(alloc, dims.n_layers * ctx * dims.kv_dim),
            dims,
            weights,
            pos: 0,
            classifier_backend: None,
            stack_backend: None,
            stack_checks_remaining: 0,
            layer_backend: None,
            layer_checks_remaining: 0,
            attention_backend: None,
            attention_checks_remaining: 0,
            mlp_backend: None,
            mlp_checks_remaining: 0,
            classifier_ref: take(alloc, dims.vocab),
            classifier_checks: 0,
            cpu_classifier_calls: 0,
            cpu_classifier_ticks: 0,
            cpu_classifier_max_ticks: 0,
        })
    }

    /// Attach an optional GPU classifier after loading the model.
    /// CPU remains the fallback and is also the initial parity oracle.
    pub fn set_classifier_backend(&mut self, backend: &'static mut dyn ClassifierBackend) {
        self.classifier_backend = Some(backend);
        self.classifier_checks = 4;
    }

    /// Register an optional first-layer GPU MLP backend after model loading.
    pub fn set_stack_backend(&mut self, backend: &'static mut dyn StackBackend) {
        self.stack_backend=Some(backend);
        self.stack_checks_remaining=3;
    }

    pub fn set_layer_backend(&mut self, backend: &'static mut dyn LayerBackend) {
        self.layer_backend = Some(backend);
        self.layer_checks_remaining = 3;
    }

    pub fn layer0_rope_frequencies(&self) -> &[f32] { self.freqs }

    pub fn set_attention_backend(&mut self, backend: &'static mut dyn AttentionBackend) {
        self.attention_backend = Some(backend);
        self.attention_checks_remaining = 3;
    }

    pub fn attention_score_scale(&self) -> f32 { self.att_score_scale }
    // NIGHTRUN_STAGE8E_GPU_SAMPLER_V1: model-owned multiplier already applied to CPU logits.
    pub fn gpu_sampler_logit_scale(&self)->f32 {self.logit_recip}


    pub fn set_mlp_backend(&mut self, backend: &'static mut dyn MlpBackend) {
        self.mlp_backend = Some(backend);
        self.mlp_checks_remaining = 3;
    }

    /// Same coefficients used by NightRun's native Llama forward pass.
    pub fn mlp_params(&self) -> (f32, f32) {
        (self.norm_eps, self.residual_scale)
    }

    pub fn gpu_mlp_active(&self) -> bool {
        self.mlp_backend.is_some()
    }

    pub fn gpu_classifier_active(&self) -> bool {
        self.classifier_backend.is_some()
    }

    /// Reset CPU-only classifier counters before autoregressive decoding.
    pub fn reset_cpu_classifier_profile(&mut self) {
        self.cpu_classifier_calls = 0;
        self.cpu_classifier_ticks = 0;
        self.cpu_classifier_max_ticks = 0;
    }

    /// Returns (CPU calls, accumulated cycles, maximum call cycles).
    pub fn cpu_classifier_profile(&self) -> (u64, u64, u64) {
        (
            self.cpu_classifier_calls,
            self.cpu_classifier_ticks,
            self.cpu_classifier_max_ticks,
        )
    }

    fn classify(&mut self) {
        let gpu_ok = match self.classifier_backend.as_deref_mut() {
            Some(backend) => backend.compute(self.xb, self.logits),
            None => false,
        };
        if !gpu_ok {
            let t0 = cpu_profile_ticks();
            self.acts.matvec(self.logits, self.weights.output, self.xb);
            let dt = cpu_profile_ticks().wrapping_sub(t0);
            self.cpu_classifier_calls += 1;
            self.cpu_classifier_ticks = self.cpu_classifier_ticks.saturating_add(dt);
            self.cpu_classifier_max_ticks = self.cpu_classifier_max_ticks.max(dt);
            return;
        }
        // Compare real model logits for the first four calls. Because CUDA
        // accumulates dequantized FP32 values rather than NightRun's Q8 integer
        // dots, exact logit parity is not guaranteed. Fail closed if the
        // greedy decision differs or any logit is nonfinite.
        if self.classifier_checks != 0 {
            self.acts.matvec(self.classifier_ref, self.weights.output, self.xb);
            let mut gpu_best = 0usize;
            let mut cpu_best = 0usize;
            for i in 0..self.logits.len() {
                if !self.logits[i].is_finite() {
                    self.logits.copy_from_slice(self.classifier_ref);
                    self.classifier_backend = None;
                    return;
                }
                if self.logits[i] > self.logits[gpu_best] { gpu_best = i; }
                if self.classifier_ref[i] > self.classifier_ref[cpu_best] { cpu_best = i; }
            }
            if gpu_best != cpu_best {
                self.logits.copy_from_slice(self.classifier_ref);
                self.classifier_backend = None;
                return;
            }
            self.classifier_checks -= 1;
        }
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Logits from the most recent `forward` call.
    // NIGHTRUN_STAGE8C_GPU_FINAL_V1: one shared output operation for decode,
    // GPU prefill (batch/sequential), and the original CPU prefill fallback.
    // GPU takes the last hidden vector, resident FP32 gamma and resident Q8
    // output matrix. NightRun alone applies model logit scaling and sampling.
    fn finish_output(&mut self, last_bx_row: Option<usize>) {
        let dim=self.dims.dim;
        let gpu_ok = {
            let hidden: &[f32] = match last_bx_row {
                Some(row) => &self.bx[row*dim..(row+1)*dim],
                None => self.x,
            };
            match self.stack_backend.as_deref_mut() {
                Some(b) => b.compute_final(hidden,self.logits),
                None => false,
            }
        };
        if !gpu_ok {
            let hidden: &[f32] = match last_bx_row {
                Some(row) => &self.bx[row*dim..(row+1)*dim],
                None => self.x,
            };
            tvec::rmsnorm(self.xb,hidden,self.weights.out_norm,self.norm_eps);
            self.classify();
        }
        if self.logit_recip!=1.0 {
            tvec::scale_inplace(self.logits,self.logit_recip);
        }
    }

    pub fn logits(&self) -> &[f32] {
        self.logits
    }

    /// Remaining context slots.
    pub fn remaining(&self) -> usize {
        self.dims.ctx - self.pos
    }

    /// Run one token through the model at the current position, advancing
    /// it. Returns the logits over the vocabulary.
    pub fn forward(&mut self, token: u32) -> &[f32] {
        let d = self.dims;
        let pos = self.pos;
        assert!(pos < d.ctx, "context window exhausted");

        // NIGHTRUN_STAGE8D_GPU_EMBED_Q8_V1: GPU reads resident tensor ID 1.
        // CPU dequantization remains for non-Q8 models, parity and hard fallback.
        let gpu_embedding=STAGE8D_GPU_EMBED_ENABLED &&
            matches!(self.weights.embed,QMat::Q8(_)) && self.stack_backend.is_some();
        if !gpu_embedding {
            self.weights.embed.dequant_row(token as usize,d.dim,self.x);
            if self.embed_scale!=1.0 {tvec::scale_inplace(self.x,self.embed_scale);}
        }

        // Stage 7: UEFI schedules all 16 layers with one GPU request.
        // First three requests run both paths for full-stack parity. Once
        // verified, no CPU transformer layer runs on successful GPU decode.
        let mut stack_done = false;
        let mut stack_verify = false;
        if self.stack_backend.is_some() {
            stack_verify = self.stack_checks_remaining != 0;
            if stack_verify {
                if gpu_embedding {
                    self.weights.embed.dequant_row(token as usize,d.dim,&mut self.bx[..d.dim]);
                    if self.embed_scale!=1.0 {
                        tvec::scale_inplace(&mut self.bx[..d.dim],self.embed_scale);
                    }
                } else {self.bx[..d.dim].copy_from_slice(self.x);}
            }
            let gpu_ok=if gpu_embedding {
                self.stack_backend.as_deref_mut().unwrap().compute_token_q8(
                    token,d.vocab,self.embed_scale,self.x,pos,self.key_cache,self.val_cache)
            } else {
                self.stack_backend.as_deref_mut().unwrap().compute(
                    self.x,pos,self.key_cache,self.val_cache)
            };
            if gpu_ok {
                if stack_verify {
                    self.bxb[..d.dim].copy_from_slice(self.x);
                    self.x.copy_from_slice(&self.bx[..d.dim]);
                } else {stack_done=true;}
            } else {
                self.stack_backend=None;stack_verify=false;
                if gpu_embedding {
                    // GPU failure must not reuse the previous token's hidden state.
                    self.weights.embed.dequant_row(token as usize,d.dim,self.x);
                    if self.embed_scale!=1.0 {tvec::scale_inplace(self.x,self.embed_scale);}
                }
            }
        }
        if !stack_done {
        for l in 0..d.n_layers {
            let w = &self.weights.layers[l];
            // Isolated parity: replay THIS 21-command layer on exactly the
            // CPU's input activation, rather than an accumulated GPU input.
            // Requested by NightRun, not chosen by CUDA. First 3 tokens only.
            if stack_verify && l==4 {
                if let Some(b)=self.stack_backend.as_deref_mut() {
                    b.trace_begin_layer(l,pos,self.x,
                        &self.key_cache[l*d.ctx*d.kv_dim..(l+1)*d.ctx*d.kv_dim],
                        &self.val_cache[l*d.ctx*d.kv_dim..(l+1)*d.ctx*d.kv_dim]);
                }
            }

            // Stage 6B: complete GPU layer-0. One QEMU command per decoded
            // token; the GPU owns all sub-operations and the active KV cache.
            // First three calls run the original full CPU layer as a parity
            // oracle, then restore the GPU result only if it agrees.
            let mut gpu_full_layer_ok = false;
            let verify_full_layer = l == 0 && self.layer_checks_remaining != 0;
            if l == 0 && self.layer_backend.is_some() {
                if verify_full_layer {
                    self.bx[..d.dim].copy_from_slice(self.x);
                }
                let complete = self.layer_backend.as_deref_mut().unwrap().compute(
                    self.x, pos,
                    &mut self.key_cache[..d.ctx * d.kv_dim],
                    &mut self.val_cache[..d.ctx * d.kv_dim]);
                if complete {
                    gpu_full_layer_ok = true;
                    if !verify_full_layer {continue;}
                    // Copy GPU result to preallocated prefill scratch and
                    // run unmodified CPU layer from original input.
                    self.bxb[..d.dim].copy_from_slice(self.x);
                    self.x.copy_from_slice(&self.bx[..d.dim]);
                } else {
                    self.layer_backend = None;
                }
            }

            // Attention block.
            tvec::rmsnorm(self.xb, self.x, w.attn_norm, self.norm_eps);
            if stack_verify { if let Some(b)=self.stack_backend.as_deref_mut() {
                b.trace_compare_stage(l,0,self.xb);
            }}
            self.acts.matvec_group(
                &mut [(self.q, w.wq), (self.k, w.wk), (self.v, w.wv)],
                self.xb,
            );
            // GPU_Q8_BYTE_PROBE_V1: stage 1 attention-norm Q8 bytes.
            if stack_verify { if let Some(b)=self.stack_backend.as_deref_mut() {
                b.trace_compare_q8(l,1,gpu_q8_probe_bytes(&self.acts.q8[..d.dim/QK8_0]),self.xb);
            }}
            if stack_verify { if let Some(b)=self.stack_backend.as_deref_mut() {
                b.trace_compare_stage(l,2,self.q);
                b.trace_compare_stage(l,3,self.k);
                b.trace_compare_stage(l,4,self.v);
            }}

            // Qwen3: per-head RMSNorm on Q and K, before RoPE.
            if let (Some(qn), Some(kn)) = (w.q_norm, w.k_norm) {
                for head in self.q.chunks_exact_mut(d.head_dim) {
                    tvec::rmsnorm_inplace(head, qn, self.norm_eps);
                }
                for head in self.k.chunks_exact_mut(d.head_dim) {
                    tvec::rmsnorm_inplace(head, kn, self.norm_eps);
                }
            }

            rope::apply(self.q, d.head_dim, self.freqs, pos, self.rope_style);
            rope::apply(self.k, d.head_dim, self.freqs, pos, self.rope_style);
            if stack_verify { if let Some(b)=self.stack_backend.as_deref_mut() {
                b.trace_compare_stage(l,5,self.q);
                b.trace_compare_stage(l,6,self.k);
            }}

            // Append K/V to the cache as f16.
            let cache_row = (l * d.ctx + pos) * d.kv_dim;
            for i in 0..d.kv_dim {
                self.key_cache[cache_row + i] = f32_to_f16(self.k[i]);
                self.val_cache[cache_row + i] = f32_to_f16(self.v[i]);
            }
            if stack_verify { if let Some(b)=self.stack_backend.as_deref_mut() {
                b.trace_compare_cache(l,7,&self.key_cache[cache_row..cache_row+d.kv_dim]);
                b.trace_compare_cache(l,8,&self.val_cache[cache_row..cache_row+d.kv_dim]);
            }}

            // Stage 6A: layer-0 GQA runs on GPU-resident f16 KV when available.
            // The CPU still generates Q/K/V and mirrors K/V, so failure always
            // falls back to the original CPU implementation without losing context.
            let mut gpu_attention_ok = false;
            let verify_gpu_attention = l == 0 && self.attention_checks_remaining != 0;
            if l == 0 && self.attention_backend.is_some() {
                gpu_attention_ok = self.attention_backend.as_deref_mut().unwrap().compute(
                    self.q,
                    &self.key_cache[..d.ctx * d.kv_dim],
                    &self.val_cache[..d.ctx * d.kv_dim],
                    pos,
                    self.attn_out,
                );
                if !gpu_attention_ok { self.attention_backend = None; }
            }
            if !gpu_attention_ok || verify_gpu_attention {
                // Use a distinct preallocated CPU reference buffer so the
                // GPU output can be compared without aliasing mutable slices.
                let cpu_out: &mut [f32] = &mut self.battn[..d.att_dim];
                let gqa = d.n_heads / d.n_kv_heads;
                let scale = self.att_score_scale;
                for h in 0..d.n_heads {
                    let qh = &self.q[h * d.head_dim..(h + 1) * d.head_dim];
                    let kvh = (h / gqa) * d.head_dim;
                    let att = &mut self.att[h * d.ctx..h * d.ctx + pos + 1];
                    for (t, a) in att.iter_mut().enumerate() {
                        let krow = (l * d.ctx + t) * d.kv_dim + kvh;
                        *a = dot_f16(&self.key_cache[krow..krow + d.head_dim], qh) * scale;
                    }
                    tvec::softmax(att);
                    let out = &mut cpu_out[h * d.head_dim..(h + 1) * d.head_dim];
                    out.fill(0.0);
                    for (t, &a) in att.iter().enumerate() {
                        let vrow = (l * d.ctx + t) * d.kv_dim + kvh;
                        axpy_f16(out, a, &self.val_cache[vrow..vrow + d.head_dim]);
                    }
                }
                if gpu_attention_ok {
                    let mut max_error = 0.0f32;
                    let mut parity_ok = true;
                    for i in 0..d.att_dim {
                        let cpu = cpu_out[i];
                        let gpu = self.attn_out[i];
                        let error = (cpu - gpu).abs();
                        if error > max_error { max_error = error; }
                        if !gpu.is_finite() || error > 0.01 + 0.002 * cpu.abs() {
                            parity_ok = false;
                        }
                    }
                    self.attention_backend.as_deref_mut().unwrap()
                        .on_parity_check(parity_ok, max_error);
                    self.attention_checks_remaining -= 1;
                    if !parity_ok {
                        self.attn_out.copy_from_slice(&self.battn[..d.att_dim]);
                        self.attention_backend = None;
                    }
                } else {
                    self.attn_out.copy_from_slice(&self.battn[..d.att_dim]);
                }
            }

            if stack_verify { if let Some(b)=self.stack_backend.as_deref_mut() {
                b.trace_compare_stage(l,9,self.attn_out);
            }}
            self.acts.matvec(self.xb2, w.wo, self.attn_out);
            // GPU_Q8_BYTE_PROBE_V1: stage 10 attention-output Q8 bytes.
            if stack_verify { if let Some(b)=self.stack_backend.as_deref_mut() {
                b.trace_compare_q8(l,10,gpu_q8_probe_bytes(&self.acts.q8[..d.dim/QK8_0]),self.attn_out);
                b.trace_compare_stage(l,11,self.xb2);
                b.trace_compare_gemv(l,11,
                    gpu_q8_probe_weights(w.wo),
                    gpu_q8_probe_bytes(&self.acts.q8[..d.dim/QK8_0]),self.xb2);
            }}
            tvec::saxpy(self.x, self.residual_scale, self.xb2);
            if stack_verify { if let Some(b)=self.stack_backend.as_deref_mut() {
                b.trace_compare_stage(l,12,self.x);
            }}

            // First decode layer only: optional real GPU MLP. Prefill and all
            // remaining layers use the unchanged NightRun CPU implementation.
            // The first three GPU calls are checked against NightRun's own
            // Q8_0 CPU computation; mismatch disables the GPU backend.
            let mut gpu_mlp_ok = false;
            if l == 0 && self.mlp_backend.is_some() {
                let verify = self.mlp_checks_remaining != 0;
                if verify {
                    // xb has finished its attention role at this point.
                    self.xb.copy_from_slice(self.x);
                }
                let computed = self.mlp_backend.as_deref_mut().unwrap().compute(self.x);
                if computed {
                    if verify {
                        // Independently calculate the expected CPU MLP result
                        // from x_before (xb). Scratch remains preallocated.
                        tvec::rmsnorm(self.xb2, self.xb, w.ffn_norm, self.norm_eps);
                        self.acts.matvec_group(
                            &mut [(self.gate, w.w_gate), (self.up, w.w_up)],
                            self.xb2,
                        );
                        tvec::swiglu(self.gate, self.up);
                        self.acts.matvec(self.xb2, w.w_down, self.gate);
                        let mut parity_ok = true;
                        let mut max_error = 0.0f32;
                        for i in 0..d.dim {
                            let expected = self.xb[i] + self.residual_scale * self.xb2[i];
                            let actual = self.x[i];
                            let error = (expected - actual).abs();
                            if error > max_error { max_error = error; }
                            if !actual.is_finite() || error > 0.02 + 0.003 * expected.abs() {
                                parity_ok = false;
                            }
                        }
                        self.mlp_backend.as_deref_mut().unwrap()
                            .on_parity_check(parity_ok, max_error);
                        self.mlp_checks_remaining -= 1;
                        if !parity_ok {
                            self.x.copy_from_slice(self.xb);
                        }
                        gpu_mlp_ok = parity_ok;
                    } else {
                        gpu_mlp_ok = true;
                    }
                }
                if !gpu_mlp_ok {
                    // compute() leaves x unchanged on failure; parity mismatch
                    // explicitly restored x above. Continue with CPU MLP.
                    self.mlp_backend = None;
                }
            }
            if !gpu_mlp_ok {
                // Original CPU MLP, including the fallback path.
                tvec::rmsnorm(self.xb, self.x, w.ffn_norm, self.norm_eps);
                if stack_verify { if let Some(b)=self.stack_backend.as_deref_mut() {
                    b.trace_compare_stage(l,13,self.xb);
                }}
                self.acts
                    .matvec_group(&mut [(self.gate, w.w_gate), (self.up, w.w_up)], self.xb);
                // GPU_Q8_BYTE_PROBE_V1: stage 14 MLP input Q8 bytes.
                if stack_verify { if let Some(b)=self.stack_backend.as_deref_mut() {
                    b.trace_compare_q8(l,14,gpu_q8_probe_bytes(&self.acts.q8[..d.dim/QK8_0]),self.xb);
                    b.trace_compare_stage(l,15,self.gate);
                    b.trace_compare_gemv(l,15,
                        gpu_q8_probe_weights(w.w_gate),
                        gpu_q8_probe_bytes(&self.acts.q8[..d.dim/QK8_0]),self.gate);
                    b.trace_compare_stage(l,16,self.up);
                    b.trace_compare_gemv(l,16,
                        gpu_q8_probe_weights(w.w_up),
                        gpu_q8_probe_bytes(&self.acts.q8[..d.dim/QK8_0]),self.up);
                }}
                tvec::swiglu(self.gate, self.up);
                if stack_verify { if let Some(b)=self.stack_backend.as_deref_mut() {
                    b.trace_compare_stage(l,17,self.gate);
                }}
                self.acts.matvec(self.xb2, w.w_down, self.gate);
                // GPU_Q8_BYTE_PROBE_V1: stage 18 MLP output Q8 bytes.
                if stack_verify { if let Some(b)=self.stack_backend.as_deref_mut() {
                    b.trace_compare_q8(l,18,gpu_q8_probe_bytes(&self.acts.q8[..d.ffn_dim/QK8_0]),self.gate);
                    b.trace_compare_stage(l,19,self.xb2);
                        b.trace_compare_gemv(l,19,
                            gpu_q8_probe_weights(w.w_down),
                            gpu_q8_probe_bytes(&self.acts.q8[..d.ffn_dim/QK8_0]),self.xb2);
                }}
                tvec::saxpy(self.x, self.residual_scale, self.xb2);
                if stack_verify { if let Some(b)=self.stack_backend.as_deref_mut() {
                    b.trace_compare_stage(l,20,self.x);
                }}
            }
            if gpu_full_layer_ok && verify_full_layer {
                let mut max_err = 0.0f32;
                let mut parity_ok = true;
                for i in 0..d.dim {
                    let cpu = self.x[i];
                    let gpu = self.bxb[i];
                    let err = (cpu - gpu).abs();
                    if err > max_err { max_err = err; }
                    if !gpu.is_finite() || err > 0.025 + 0.005 * cpu.abs() {
                        parity_ok = false;
                    }
                }
                self.layer_backend.as_deref_mut().unwrap()
                    .on_parity_check(parity_ok, max_err);
                self.layer_checks_remaining -= 1;
                if parity_ok {
                    self.x.copy_from_slice(&self.bxb[..d.dim]);
                } else {
                    // CPU reference is already in x and CPU KV is correct.
                    self.layer_backend = None;
                }
            }
            // Diagnostic-only: compare GPU and CPU after this exact layer.
            if stack_verify {
                if let Some(backend)=self.stack_backend.as_deref_mut() {
                    backend.trace_compare_layer(l, self.x);
                }
            }
        }

        if stack_verify {
            let mut max_err=0.0f32;
            let mut parity_ok=true;
            for i in 0..d.dim {
                let cpu=self.x[i];let gpu=self.bxb[i];
                let err=(cpu-gpu).abs();
                if err>max_err {max_err=err;}
                // Conservative relative+absolute full-model criterion.
                if !gpu.is_finite() || err>0.05+0.01*cpu.abs() {parity_ok=false;}
            }
            self.stack_backend.as_deref_mut().unwrap()
                .on_parity_check(parity_ok,max_err);
            self.stack_checks_remaining-=1;
            if parity_ok {self.x.copy_from_slice(&self.bxb[..d.dim]);}
            else {self.stack_backend=None;}
        }
        }

        // Final norm + classifier.
        self.finish_output(None);

        self.pos += 1;
        self.logits
    }

    /// Batched prefill of up to MAX_BATCH prompt tokens. Numerically
    /// bit-identical to calling `forward` per token (same kernels, same
    /// order); returns the logits of the last token.
    #[allow(clippy::needless_range_loop)] // j writes two caches at one offset
    pub fn prefill_chunk(&mut self, tokens: &[u32]) -> &[f32] {
        let d = self.dims;
        let b = tokens.len();
        assert!((1..=MAX_BATCH).contains(&b));
        assert!(self.pos + b <= d.ctx, "context window exhausted");

        // NIGHTRUN_STAGE8B_UEFI_16TOKEN_PREFILL_BATCH_V1:
        // UEFI alone constructs ordered embedding inputs and requested positions.
        // CUDA is a low-level replay/queue executor, never an inference engine.
        if STAGE8A_GPU_PREFILL_ENABLED && STAGE8B_GPU_BATCH_ENABLED &&
           self.stack_backend.is_some() {
            const GPU_PREFILL_BATCH:usize=16;
            if b>GPU_PREFILL_BATCH {
                for sub in tokens.chunks(GPU_PREFILL_BATCH) {
                    self.prefill_chunk(sub);
                }
                return self.logits;
            }
            let start=self.pos;
            let gpu_embed=STAGE8D_GPU_EMBED_ENABLED &&
                matches!(self.weights.embed,QMat::Q8(_));
            let ok=if gpu_embed {
                self.stack_backend.as_deref_mut().unwrap().compute_prefill_tokens_q8(
                    tokens,d.vocab,self.embed_scale,start,self.x,self.key_cache,self.val_cache)
            } else {
                let mut embeddings=alloc::vec![0.0f32;b*d.dim];
                for (i,&token) in tokens.iter().enumerate() {
                    let dst=&mut embeddings[i*d.dim..(i+1)*d.dim];
                    self.weights.embed.dequant_row(token as usize,d.dim,dst);
                    if self.embed_scale!=1.0 {tvec::scale_inplace(dst,self.embed_scale);}
                }
                self.stack_backend.as_deref_mut().unwrap().compute_prefill_batch(
                    &embeddings,start,b,self.x,self.key_cache,self.val_cache)
            };
            if ok {
                self.pos+=b;
                self.stack_backend.as_deref_mut().unwrap().on_prefill_complete(start,b);
                self.finish_output(None);
                return self.logits;
            }
            // Whole-batch transaction: no CPU KV entries committed on failure.
            // The host may have partially altered GPU VRAM, so disable that
            // backend and replay this entire batch using the original CPU path.
            self.stack_backend.as_deref_mut().unwrap().on_prefill_failure(start,0);
            self.stack_backend=None;
            return self.prefill_chunk(tokens);
        }

        // NIGHTRUN_STAGE8A_CHUNK_ENTRYPOINT_FIX_V1:
        // app.rs calls prefill_chunk() directly; this must use the GPU stack.
        // NightRun UEFI owns the token loop, 336-command dispatch and KV policy.
        if STAGE8A_GPU_PREFILL_ENABLED && self.stack_backend.is_some() {
            let start = self.pos;
            for (index, &token) in tokens.iter().enumerate() {
                let gpu_embed=STAGE8D_GPU_EMBED_ENABLED &&
                    matches!(self.weights.embed,QMat::Q8(_));
                if !gpu_embed {
                    self.weights.embed.dequant_row(token as usize,d.dim,self.x);
                    if self.embed_scale!=1.0 {tvec::scale_inplace(self.x,self.embed_scale);}
                }
                let ok=if gpu_embed {
                    self.stack_backend.as_deref_mut().unwrap().compute_token_q8(
                        token,d.vocab,self.embed_scale,self.x,self.pos,
                        self.key_cache,self.val_cache)
                } else {
                    self.stack_backend.as_deref_mut().unwrap().compute(
                        self.x,self.pos,self.key_cache,self.val_cache)
                };
                if !ok {
                    self.stack_backend.as_deref_mut().unwrap()
                        .on_prefill_failure(self.pos, index);
                    self.stack_backend = None;
                    // Successful prefix rows remain in the CPU KV mirror.
                    // Resume at the failed token, without skip or duplication.
                    return self.prefill_chunk(&tokens[index..]);
                }
                self.pos += 1;
            }
            self.stack_backend.as_deref_mut().unwrap()
                .on_prefill_complete(start, b);
            // Original prefill_chunk() classifies the last token of each
            // chunk. Preserve exactly that externally observable behavior.
            self.finish_output(None);
            return self.logits;
        }

        let base = self.pos;

        // Embeddings (+ muP embed scale) per row.
        for (i, &t) in tokens.iter().enumerate() {
            let row = &mut self.bx[i * d.dim..(i + 1) * d.dim];
            self.weights.embed.dequant_row(t as usize, d.dim, row);
            if self.embed_scale != 1.0 {
                tvec::scale_inplace(row, self.embed_scale);
            }
        }

        for l in 0..d.n_layers {
            let w = &self.weights.layers[l];

            // Attention block: batched norm + QKV.
            for i in 0..b {
                tvec::rmsnorm(
                    &mut self.bxb[i * d.dim..(i + 1) * d.dim],
                    &self.bx[i * d.dim..(i + 1) * d.dim],
                    w.attn_norm,
                    self.norm_eps,
                );
            }
            self.acts.matmul_group(
                &mut [
                    (&mut self.bq[..b * d.att_dim], w.wq),
                    (&mut self.bk[..b * d.kv_dim], w.wk),
                    (&mut self.bv[..b * d.kv_dim], w.wv),
                ],
                &self.bxb[..b * d.dim],
                b,
            );

            // Per token: qk-norm, rope at its own position, KV append.
            for i in 0..b {
                let pos = base + i;
                let q = &mut self.bq[i * d.att_dim..(i + 1) * d.att_dim];
                let k = &mut self.bk[i * d.kv_dim..(i + 1) * d.kv_dim];
                if let (Some(qn), Some(kn)) = (w.q_norm, w.k_norm) {
                    for head in q.chunks_exact_mut(d.head_dim) {
                        tvec::rmsnorm_inplace(head, qn, self.norm_eps);
                    }
                    for head in k.chunks_exact_mut(d.head_dim) {
                        tvec::rmsnorm_inplace(head, kn, self.norm_eps);
                    }
                }
                rope::apply(q, d.head_dim, self.freqs, pos, self.rope_style);
                rope::apply(k, d.head_dim, self.freqs, pos, self.rope_style);
                let cache_row = (l * d.ctx + pos) * d.kv_dim;
                for j in 0..d.kv_dim {
                    self.key_cache[cache_row + j] = f32_to_f16(k[j]);
                    self.val_cache[cache_row + j] = f32_to_f16(self.bv[i * d.kv_dim + j]);
                }
            }

            // Causal attention per token: token i sees positions 0..=base+i
            // only (later batch tokens' cache rows are beyond its window).
            let gqa = d.n_heads / d.n_kv_heads;
            let scale = self.att_score_scale;
            for i in 0..b {
                let pos = base + i;
                for h in 0..d.n_heads {
                    let qh = &self.bq
                        [i * d.att_dim + h * d.head_dim..i * d.att_dim + (h + 1) * d.head_dim];
                    let kvh = (h / gqa) * d.head_dim;
                    let att = &mut self.att[h * d.ctx..h * d.ctx + pos + 1];
                    for (t, a) in att.iter_mut().enumerate() {
                        let krow = (l * d.ctx + t) * d.kv_dim + kvh;
                        *a = dot_f16(&self.key_cache[krow..krow + d.head_dim], qh) * scale;
                    }
                    tvec::softmax(att);
                    let out = &mut self.battn
                        [i * d.att_dim + h * d.head_dim..i * d.att_dim + (h + 1) * d.head_dim];
                    out.fill(0.0);
                    for (t, &a) in att.iter().enumerate() {
                        let vrow = (l * d.ctx + t) * d.kv_dim + kvh;
                        axpy_f16(out, a, &self.val_cache[vrow..vrow + d.head_dim]);
                    }
                }
            }

            // Batched output projection + residual.
            self.acts.matmul(
                &mut self.bxb[..b * d.dim],
                w.wo,
                &self.battn[..b * d.att_dim],
                b,
            );
            for i in 0..b {
                tvec::saxpy(
                    &mut self.bx[i * d.dim..(i + 1) * d.dim],
                    self.residual_scale,
                    &self.bxb[i * d.dim..(i + 1) * d.dim],
                );
            }

            // MLP block: batched norm, gate/up, swiglu rows, down, residual.
            for i in 0..b {
                tvec::rmsnorm(
                    &mut self.bxb[i * d.dim..(i + 1) * d.dim],
                    &self.bx[i * d.dim..(i + 1) * d.dim],
                    w.ffn_norm,
                    self.norm_eps,
                );
            }
            self.acts.matmul_group(
                &mut [
                    (&mut self.bgate[..b * d.ffn_dim], w.w_gate),
                    (&mut self.bup[..b * d.ffn_dim], w.w_up),
                ],
                &self.bxb[..b * d.dim],
                b,
            );
            for i in 0..b {
                tvec::swiglu(
                    &mut self.bgate[i * d.ffn_dim..(i + 1) * d.ffn_dim],
                    &self.bup[i * d.ffn_dim..(i + 1) * d.ffn_dim],
                );
            }
            self.acts.matmul(
                &mut self.bxb[..b * d.dim],
                w.w_down,
                &self.bgate[..b * d.ffn_dim],
                b,
            );
            for i in 0..b {
                tvec::saxpy(
                    &mut self.bx[i * d.dim..(i + 1) * d.dim],
                    self.residual_scale,
                    &self.bxb[i * d.dim..(i + 1) * d.dim],
                );
            }
        }

        // Only the last token's logits are needed.
        self.finish_output(Some(b-1));

        self.pos += b;
        self.logits
    }

    /// Stage 8A: UEFI drives the existing 336-command GPU stack for each
    /// prompt token, maintaining GPU-resident history AND the CPU FP16 KV
    /// mirror. CPU does embeddings and a *single* final norm/classifier.
    /// This first integration is sequential, not batched GPU prefill.
    pub fn prefill(&mut self, tokens: &[u32]) -> &[f32] {
        assert!(!tokens.is_empty());
        let d = self.dims;
        assert!(self.pos + tokens.len() <= d.ctx, "context window exhausted");
        if STAGE8A_GPU_PREFILL_ENABLED && self.stack_backend.is_some() {
            let start = self.pos;
            for (index, &token) in tokens.iter().enumerate() {
                // Embedding and position ownership remain in NightRun UEFI.
                let gpu_embed=STAGE8D_GPU_EMBED_ENABLED &&
                    matches!(self.weights.embed,QMat::Q8(_));
                if !gpu_embed {
                    self.weights.embed.dequant_row(token as usize,d.dim,self.x);
                    if self.embed_scale!=1.0 {tvec::scale_inplace(self.x,self.embed_scale);}
                }
                let ok=if gpu_embed {
                    self.stack_backend.as_deref_mut().unwrap().compute_token_q8(
                        token,d.vocab,self.embed_scale,self.x,self.pos,
                        self.key_cache,self.val_cache)
                } else {
                    self.stack_backend.as_deref_mut().unwrap().compute(
                        self.x,self.pos,self.key_cache,self.val_cache)
                };
                if !ok {
                    // Failed GPU commands have not committed any CPU KV row.
                    // Prefix tokens remain valid in our mirrored CPU cache.
                    self.stack_backend.as_deref_mut().unwrap()
                        .on_prefill_failure(self.pos, index);
                    self.stack_backend = None;
                    // Resume from the *failed* token: no skip or duplicate.
                    for chunk in tokens[index..].chunks(MAX_BATCH) {
                        self.prefill_chunk(chunk);
                    }
                    return self.logits;
                }
                self.pos += 1;
            }
            self.stack_backend.as_deref_mut().unwrap()
                .on_prefill_complete(start, tokens.len());
            // Only the last prompt token needs logits. The existing
            // NightRun classifier backend and sampler remain unchanged.
            self.finish_output(None);
            return self.logits;
        }
        // Original batched CPU prefill remains the baseline and fallback.
        for chunk in tokens.chunks(MAX_BATCH) {
            self.prefill_chunk(chunk);
        }
        self.logits
    }
}
