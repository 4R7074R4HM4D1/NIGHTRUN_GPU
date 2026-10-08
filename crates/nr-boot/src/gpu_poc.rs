//! Stage 3: QEMU ivshmem proof of NightRun-owned GPU command execution.
//! The host is a CUDA kernel executor, NOT an LLM inference runtime.
//! The real Q8_0 weights come from the model loaded by NightRun.
//! Uses a reserved region of the EXISTING 8 GiB ivshmem PCI BAR.
use core::arch::asm;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};
use core::time::Duration;
use nr_gpu::{BufferId, CommandList, GpuCommand};
use nr_model::infer::{InferCtx, QMat};
use uefi::boot;

const VENDOR: u16 = 0x1af4;
const DEVICE: u16 = 0x1110;
const REGION: usize = 0x6000_0000; // 1.5 GiB, away from legacy Q8 classifier
const REGION_LIMIT: usize = 0x0100_0000; // 16 MiB
const MAGIC: u32 = 0x3347_504e; // "NPG3" in little-endian bytes, protocol v3
const OFF_MAGIC: usize = 0x00;
const OFF_STATUS: usize = 0x04;
const OFF_REQUEST_SEQ: usize = 0x08;
const OFF_COMPLETED_SEQ: usize = 0x0c;
const OFF_ERROR: usize = 0x10;
const OFF_CMD_COUNT: usize = 0x14;
const OFF_DIM: usize = 0x18;
const OFF_SERVICE_US: usize = 0x1c;
const OFF_GPU_US: usize = 0x20;
const OFF_COMMANDS: usize = 0x100;
const OFF_WEIGHTS: usize = 0x10000;
const OFF_INPUT: usize = 0x500000;
const OFF_GAMMA: usize = 0x502000;
const OFF_OUTPUT: usize = 0x504000;
const OFF_NORMALIZED: usize = 0x506000;
const OFF_QUANTIZED: usize = 0x508000;
const READY: u32 = 0;
const REQUEST: u32 = 1;
const DONE: u32 = 2;
const ERROR: u32 = 3;
const DIM: usize = 2048;
const BLOCKS: usize = DIM / 32;
const ROW_BYTES: usize = BLOCKS * 34;
const WEIGHT_BYTES: usize = DIM * ROW_BYTES;

#[inline]
fn pci_read(bus: u8, dev: u8, func: u8, reg: u8) -> u32 {
    let address = 0x8000_0000u32 | ((bus as u32) << 16)
        | ((dev as u32) << 11) | ((func as u32) << 8)
        | (reg as u32 & 0xfc);
    let result: u32;
    unsafe {
        asm!("out dx, eax", in("dx") 0xcf8u16, in("eax") address,
             options(nostack, preserves_flags));
        asm!("in eax, dx", in("dx") 0xcfcu16, out("eax") result,
             options(nostack, preserves_flags));
    }
    result
}

fn locate_bridge() -> Option<usize> {
    for bus in 0..=255u16 {
        for dev in 0..32u8 {
            for function in 0..8u8 {
                let b = bus as u8;
                let id = pci_read(b, dev, function, 0);
                if id as u16 != VENDOR || (id >> 16) as u16 != DEVICE { continue; }
                if pci_read(b, dev, function, 4) & 2 == 0 { continue; }
                let lo = pci_read(b, dev, function, 0x18);
                if lo & 1 != 0 { continue; }
                let kind = (lo >> 1) & 3;
                let hi = if kind == 2 { pci_read(b, dev, function, 0x1c) }
                         else if kind == 0 { 0 } else { continue; };
                let address = ((hi as u64) << 32) | ((lo & !0xf) as u64);
                if address == 0 || address > usize::MAX as u64 - (REGION + REGION_LIMIT) as u64 {
                    continue;
                }
                let base = address as usize;
                // A service-created signature prevents touching unrelated devices.
                if unsafe { read_volatile((base + REGION + OFF_MAGIC) as *const u32) } == MAGIC {
                    serial_println!("[gpu-poc] PCI {:02x}:{:02x}.{} BAR2={:#x}", b, dev, function, base);
                    return Some(base + REGION);
                }
            }
        }
    }
    None
}

#[inline]
unsafe fn read_u32(base: usize, offset: usize) -> u32 {
    let v = unsafe { read_volatile((base + offset) as *const u32) };
    fence(Ordering::SeqCst);
    v
}
#[inline]
unsafe fn write_u32(base: usize, offset: usize, value: u32) {
    unsafe { write_volatile((base + offset) as *mut u32, value); }
    fence(Ordering::SeqCst);
}
fn write_bytes(base: usize, offset: usize, data: &[u8]) {
    let mut parts = data.chunks_exact(4);
    for (i, word) in parts.by_ref().enumerate() {
        unsafe {
            write_volatile((base + offset + 4*i) as *mut u32,
                u32::from_le_bytes([word[0], word[1], word[2], word[3]]));
        }
    }
    let used = data.len() - parts.remainder().len();
    for (i, byte) in parts.remainder().iter().enumerate() {
        unsafe { write_volatile((base + offset + used + i) as *mut u8, *byte); }
    }
    fence(Ordering::SeqCst);
}
fn input_value(i: usize) -> f32 {
    ((i.wrapping_mul(37) % 1001) as i32 - 500) as f32 * 0.0007
}
fn gamma_value(i: usize) -> f32 { 1.0 + (i.wrapping_mul(29) % 127) as f32 * 0.0001 }

fn half_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits >> 10) & 31) as u32;
    let frac = ((bits & 1023) as u32) << 13;
    if exp == 0 {
        if frac == 0 { return f32::from_bits(sign); }
        let mut mant = (bits & 1023) as u32;
        let mut exponent = 113u32;
        while mant & 1024 == 0 { mant <<= 1; exponent -= 1; }
        return f32::from_bits(sign | (exponent << 23) | ((mant & 1023) << 13));
    }
    if exp == 31 { return f32::from_bits(sign | 0x7f80_0000 | frac); }
    f32::from_bits(sign | ((exp + 112) << 23) | frac)
}

fn check_result(base: usize, weights: &[u8]) -> bool {
    let rows = [0usize, 1, 8, 64, 128, 512, 1024, 2047];
    let mut max_error = 0.0f32;
    for &row in &rows {
        let mut expected = input_value(row);
        let start = row * ROW_BYTES;
        for block in 0..BLOCKS {
            let off = start + block * 34;
            let wh = u16::from_le_bytes([weights[off], weights[off + 1]]);
            let qoff = OFF_QUANTIZED + block*34;
            let xh = (unsafe { read_u32(base, qoff & !3) } >> ((qoff & 2)*8)) as u16;
            let mut dot = 0i32;
            for j in 0..32 {
                let w = weights[off + 2 + j] as i8 as i32;
                let q = unsafe { read_volatile((base + qoff + 2 + j) as *const i8) } as i32;
                dot += w*q;
            }
            expected += (dot as f32) * half_f32(wh) * half_f32(xh);
        }
        let actual = f32::from_bits(unsafe { read_u32(base, OFF_OUTPUT + row*4) });
        let error = (actual - expected).abs();
        if !actual.is_finite() || error > 0.003 + 0.0002*expected.abs() {
            serial_println!("[gpu-poc] FAIL row={} expected={} actual={} err={}", row, expected, actual, error);
            return false;
        }
        if error > max_error { max_error = error; }
    }
    serial_println!("[gpu-poc] PASS: 4 GPU operations, 8 CPU-checked rows, maxerr={}", max_error);
    true
}

/// Run once during model initialization. This demonstration does not replace
/// NightRun's normal inference and never modifies the context or model.
pub fn run(infer: &InferCtx<'_>) {
    let Some(base) = locate_bridge() else {
        serial_println!("[gpu-poc] executor not detected; continuing normal inference");
        return;
    };
    if infer.dims.dim != DIM { serial_println!("[gpu-poc] unexpected model dim"); return; }
    let blocks = match &infer.weights.output {
        QMat::Q8(q) => q,
        _ => { serial_println!("[gpu-poc] output classifier not Q8_0"); return; }
    };
    if core::mem::size_of::<nr_tensor::q8::BlockQ8_0>() != 34 || blocks.len() < DIM*BLOCKS {
        serial_println!("[gpu-poc] packed classifier length mismatch");
        return;
    }
    let packed = unsafe { core::slice::from_raw_parts(blocks.as_ptr() as *const u8, WEIGHT_BYTES) };
    let mut cmds = CommandList::<4>::new();
    let construct = (|| {
        cmds.push(GpuCommand::rms_norm(BufferId(3), BufferId(1), BufferId(2), DIM as u32, 1e-5)?)?;
        cmds.push(GpuCommand::quantize_q8(BufferId(4), BufferId(3), DIM as u32)?)?;
        cmds.push(GpuCommand::q8_matvec(BufferId(6), BufferId(5), BufferId(4), DIM as u32, DIM as u32)?)?;
        cmds.push(GpuCommand::residual_add(BufferId(6), BufferId(1), DIM as u32)?)?;
        Ok::<(), nr_gpu::GpuError>(())
    })();
    if construct.is_err() { serial_println!("[gpu-poc] cannot construct commands"); return; }
    serial_println!("[gpu-poc] uploading {} bytes of real classifier weights through PCI BAR", WEIGHT_BYTES);
    write_bytes(base, OFF_WEIGHTS, packed);
    for i in 0..DIM {
        unsafe {
            write_volatile((base + OFF_INPUT + i*4) as *mut u32, input_value(i).to_bits());
            write_volatile((base + OFF_GAMMA + i*4) as *mut u32, gamma_value(i).to_bits());
        }
    }
    // Serialize the hardware-independent 48-byte GpuCommand records verbatim.
    for (i, cmd) in cmds.as_slice().iter().enumerate() {
        let bytes = unsafe { core::slice::from_raw_parts(cmd as *const _ as *const u8, 48) };
        write_bytes(base, OFF_COMMANDS + i*48, bytes);
    }
    fence(Ordering::SeqCst);
    unsafe {
        write_u32(base, OFF_DIM, DIM as u32);
        write_u32(base, OFF_CMD_COUNT, 4);
        write_u32(base, OFF_ERROR, 0);
        write_u32(base, OFF_REQUEST_SEQ, 1);
        write_u32(base, OFF_STATUS, REQUEST);
    }
    serial_println!("[gpu-poc] submitted RMSNorm -> QuantizeQ8 -> Q8Matvec -> ResidualAdd");
    let mut finished = false;
    for _ in 0..150_000 {
        let state = unsafe { read_u32(base, OFF_STATUS) };
        if state == ERROR {
            serial_println!("[gpu-poc] executor error={}", unsafe { read_u32(base, OFF_ERROR) });
            break;
        }
        if state == DONE && unsafe { read_u32(base, OFF_COMPLETED_SEQ) } == 1 {
            finished = true;
            break;
        }
        boot::stall(Duration::from_micros(100));
    }
    if !finished {
        serial_println!("[gpu-poc] did not complete; normal inference unaffected");
        return;
    }
    serial_println!("[gpu-poc] host service={} us GPU graph={} us",
        unsafe { read_u32(base, OFF_SERVICE_US) }, unsafe { read_u32(base, OFF_GPU_US) });
    let _ = check_result(base, packed);
    unsafe { write_u32(base, OFF_STATUS, READY); }
}
