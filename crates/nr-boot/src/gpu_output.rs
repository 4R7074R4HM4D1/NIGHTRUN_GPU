//! UEFI/QEMU ivshmem backend for packed-Q8_0 NightRun output classifier.
//! Research prototype: this is a virtual PCI adapter to a WSL CUDA service,
//! NOT a native NVIDIA driver or physical GPU passthrough.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::arch::asm;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};
use core::time::Duration;

use nr_model::infer::{ClassifierBackend, InferCtx, QMat};
use nr_tensor::q8::{self, BlockQ8_0, QK8_0};
use uefi::boot;

const PCI_VENDOR: u16 = 0x1af4;
const PCI_DEVICE: u16 = 0x1110;
const BAR_BYTES: usize = 8 * 1024 * 1024 * 1024;

// Protocol v2: header includes Q8_0 format code; all other offsets retained.
const STATUS: usize = 0x00;
const REQUEST_SEQ: usize = 0x04;
const COMPLETED_SEQ: usize = 0x08;
const ERROR_CODE: usize = 0x0c;
const ROWS: usize = 0x10;
const COLS: usize = 0x14;
const HOST_US: usize = 0x18;
const WEIGHT_FORMAT: usize = 0x1c;
const Q8_BLOCK_BYTES: usize = 34;

const MATRIX: usize = 0x1000;
const VECTOR: usize = 0x5000_0000;
const OUTPUT: usize = 0x5002_0000;
const ST_IDLE: u32 = 0;
const ST_LOAD: u32 = 1;
const ST_LOADED: u32 = 2;
const ST_GEMV: u32 = 3;
const ST_DONE: u32 = 4;
const ST_ERROR: u32 = 5;

#[inline]
fn pci_read(bus: u8, dev: u8, func: u8, reg: u8) -> u32 {
    let addr = 0x8000_0000u32 | ((bus as u32) << 16)
        | ((dev as u32) << 11) | ((func as u32) << 8)
        | ((reg as u32) & 0xfc);
    let value: u32;
    // SAFETY: UEFI x86_64 CPL0 PCI configuration I/O.
    unsafe {
        asm!("out dx, eax", in("dx") 0xcf8u16, in("eax") addr,
             options(nostack, preserves_flags));
        asm!("in eax, dx", in("dx") 0xcfcu16, out("eax") value,
             options(nostack, preserves_flags));
    }
    value
}

fn find_ivshmem() -> Option<u64> {
    for bus in 0..=255u16 {
        for dev in 0..32u8 {
            for func in 0..8u8 {
                let b = bus as u8;
                let id = pci_read(b, dev, func, 0x00);
                if id as u16 != PCI_VENDOR || (id >> 16) as u16 != PCI_DEVICE {
                    continue;
                }
                let command = pci_read(b, dev, func, 0x04);
                if command & 0x2 == 0 { return None; }
                let low = pci_read(b, dev, func, 0x18);
                if low & 1 != 0 { return None; }
                let kind = (low >> 1) & 3;
                let high = match kind {
                    0 => 0u32,
                    2 => pci_read(b, dev, func, 0x1c),
                    _ => return None,
                };
                let base = ((high as u64) << 32) | ((low as u64) & !0xf);
                if base == 0 || base > usize::MAX as u64 - BAR_BYTES as u64 {
                    return None;
                }
                serial_println!("[gpu-output] ivshmem {:02x}:{:02x}.{} BAR2={:#x}", b, dev, func, base);
                return Some(base);
            }
        }
    }
    None
}

#[inline]
unsafe fn read_u32(base: u64, offset: usize) -> u32 {
    fence(Ordering::SeqCst);
    let v = unsafe { read_volatile((base as usize + offset) as *const u32) };
    fence(Ordering::SeqCst);
    v
}

#[inline]
unsafe fn write_u32(base: u64, offset: usize, v: u32) {
    fence(Ordering::SeqCst);
    unsafe { write_volatile((base as usize + offset) as *mut u32, v); }
    fence(Ordering::SeqCst);
}

/// Copy packed quantized bytes using volatile 32-bit stores. BAR offsets
/// are 4-byte aligned. This is a UEFI PCI-memory experiment, not a general
/// DMA implementation. Preserve byte-exact GGUF Q8_0 packing.
unsafe fn copy_bytes_to_bar(base: u64, offset: usize, data: &[u8]) {
    let addr = base as usize + offset;
    let mut groups = data.chunks_exact(4);
    for (i, chunk) in groups.by_ref().enumerate() {
        let value = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        unsafe { write_volatile((addr + i * 4) as *mut u32, value); }
    }
    let tail = groups.remainder();
    for (i, &byte) in tail.iter().enumerate() {
        unsafe { write_volatile((addr + data.len() - tail.len() + i) as *mut u8, byte); }
    }
    fence(Ordering::SeqCst);
}

fn wait(base: u64, status: u32, seq: u32, timeout_ms: usize) -> bool {
    // Poll every 50 us. Timeout is approximately timeout_ms, apart from
    // firmware scheduling jitter. No allocations in the generation loop.
    for _ in 0..timeout_ms.saturating_mul(20) {
        let s = unsafe { read_u32(base, STATUS) };
        if s == ST_ERROR {
            serial_println!("[gpu-output] CUDA service error {}", unsafe { read_u32(base, ERROR_CODE) });
            return false;
        }
        if s == status && unsafe { read_u32(base, COMPLETED_SEQ) } == seq {
            return true;
        }
        boot::stall(Duration::from_micros(50));
    }
    serial_println!("[gpu-output] timed out waiting for seq {}", seq);
    false
}

struct OutputBridge {
    base: u64,
    rows: usize,
    cols: usize,
    seq: u32,
    enabled: bool,
    calls: u32,
    // Permanent scratch; decoding performs no heap allocations.
    xq: Vec<BlockQ8_0>,
}

impl ClassifierBackend for OutputBridge {
    fn compute(&mut self, x: &[f32], logits: &mut [f32]) -> bool {
        if !self.enabled || x.len() != self.cols || logits.len() != self.rows {
            return false;
        }
        // Quantize using NightRun's native Q8_0 algorithm, then transfer
        // packed [f16 scale, 32 signed i8] blocks without dequantizing.
        q8::quantize(x, &mut self.xq);
        let byte_len = self.xq.len() * Q8_BLOCK_BYTES;
        // SAFETY: BlockQ8_0 is repr(C, packed) and exactly 34 bytes.
        let packed = unsafe {
            core::slice::from_raw_parts(self.xq.as_ptr() as *const u8, byte_len)
        };
        unsafe { copy_bytes_to_bar(self.base, VECTOR, packed); }
        self.seq = self.seq.wrapping_add(1);
        if self.seq == 0 { self.seq = 1; }
        unsafe {
            write_u32(self.base, REQUEST_SEQ, self.seq);
            write_u32(self.base, STATUS, ST_GEMV);
        }
        if !wait(self.base, ST_DONE, self.seq, 3000) {
            self.enabled = false;
            serial_println!("[gpu-output] switching permanently to CPU classifier");
            return false;
        }
        for (i, out) in logits.iter_mut().enumerate() {
            // SAFETY: BAR2, OUTPUT range, pre-validated during setup.
            let bits = unsafe { read_volatile((self.base as usize + OUTPUT + i * 4) as *const u32) };
            *out = f32::from_bits(bits);
        }
        fence(Ordering::SeqCst);
        self.calls += 1;
        if self.calls <= 3 || self.calls % 64 == 0 {
            serial_println!("[gpu-output] classifier #{} CUDA service={} us", self.calls,
                            unsafe { read_u32(self.base, HOST_US) });
        }
        true
    }
}

/// Called once, after `InferCtx::new` and before entering the chat loop.
/// Only the Q8_0 classifier is supported by this first prototype.
pub fn try_attach(infer: &mut InferCtx<'static>) -> bool {
    let (rows, cols) = (infer.dims.vocab, infer.dims.dim);
    let blocks = match infer.weights.output {
        QMat::Q8(blocks) => blocks,
        _ => {
            serial_println!("[gpu-output] model classifier is not Q8_0: CPU fallback");
            return false;
        }
    };
    if cols == 0 || rows == 0 || cols % QK8_0 != 0
        || core::mem::size_of::<BlockQ8_0>() != Q8_BLOCK_BYTES
    {
        return false;
    }
    let Some(block_count) = rows.checked_mul(cols / QK8_0) else {
        return false;
    };
    if blocks.len() != block_count { return false; }
    let Some(matrix_bytes) = block_count.checked_mul(Q8_BLOCK_BYTES) else {
        return false;
    };
    let vector_bytes = (cols / QK8_0) * Q8_BLOCK_BYTES;
    if MATRIX + matrix_bytes > VECTOR || VECTOR + vector_bytes > OUTPUT
        || OUTPUT + rows * 4 > BAR_BYTES
    {
        serial_println!("[gpu-output] classifier too large for fixed bridge layout");
        return false;
    }
    let Some(base) = find_ivshmem() else {
        serial_println!("[gpu-output] no ivshmem: CPU-only");
        return false;
    };
    serial_println!("[gpu-output] uploading packed Q8_0 classifier {}x{} ({} MiB)",
                    rows, cols, matrix_bytes / (1024 * 1024));

    // Copy the quantized model bytes exactly once. No FP32 staging matrix.
    unsafe {
        write_u32(base, STATUS, ST_IDLE);
        write_u32(base, REQUEST_SEQ, 1);
        write_u32(base, COMPLETED_SEQ, 0);
        write_u32(base, ERROR_CODE, 0);
        write_u32(base, ROWS, rows as u32);
        write_u32(base, COLS, cols as u32);
        write_u32(base, WEIGHT_FORMAT, 8); // Protocol v2: raw packed Q8_0
    }
    // Reuse the model's Q8_0 data directly. No FP32 expansion in UEFI RAM.
    // SAFETY: BlockQ8_0 has a packed, 34-byte stable representation.
    let packed = unsafe {
        core::slice::from_raw_parts(blocks.as_ptr() as *const u8, matrix_bytes)
    };
    serial_println!("[gpu-output] writing {} packed bytes through BAR2", matrix_bytes);
    unsafe { copy_bytes_to_bar(base, MATRIX, packed); }
    serial_println!("[gpu-output] packed weights copied to BAR2");
    fence(Ordering::SeqCst);
    unsafe { write_u32(base, STATUS, ST_LOAD); }
    if !wait(base, ST_LOADED, 1, 120_000) {
        serial_println!("[gpu-output] weight upload failed: CPU-only");
        return false;
    }
    let backend = Box::leak(Box::new(OutputBridge {
        base, rows, cols, seq: 1, enabled: true, calls: 0,
        xq: vec![BlockQ8_0 { d: 0, qs: [0; QK8_0] }; cols / QK8_0],
    }));
    infer.set_classifier_backend(backend);
    serial_println!("[gpu-output] real model classifier is now GPU-backed");
    true
}
