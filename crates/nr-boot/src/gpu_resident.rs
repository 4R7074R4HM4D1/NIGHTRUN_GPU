//! Stage 5A: persistent GPU allocations for all model inference weights.
//!
//! The UEFI side owns model tensor IDs, metadata, order, and Q8_0/Q4K/Q6K
//! bytes. CUDA is ONLY the temporary QEMU transport/execution backend.
//! This is intentionally not yet a GPU forward-pass implementation.
use core::arch::asm;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering, AtomicBool};
use core::time::Duration;
use nr_gpu::resident::{layer_tensor_id, ModelTensorDesc, SLOT_ATTN_NORM, SLOT_Q,
    SLOT_K, SLOT_V, SLOT_O, SLOT_FFN_NORM, SLOT_GATE, SLOT_UP, SLOT_DOWN,
    SLOT_Q_NORM, SLOT_K_NORM, EMBED_ID, OUTPUT_NORM_ID, OUTPUT_ID,
    KV_KEYS_ID, KV_VALUES_ID, SCRATCH_ID};
use nr_model::infer::{InferCtx, QMat};
use nr_model::crc32::Crc32;
use uefi::boot;

// Distinct from Q8 classifier (base), Stage 3 (+0x60000000),
// and Stage 4 (+0x80000000).
const REGION: usize = 0xA000_0000; // +2.5 GiB in the existing 8 GiB BAR2
const REGION_CAP: usize = 0x0200_0000; // 32 MiB maximum mapped region
const DATA_OFF: usize = 0x1000;
const CHUNK: usize = 4 * 1024 * 1024;
const MAGIC: u32 = 0x3547_504e; // 'NPG5'
const STATUS: usize = 4;
const SEQ: usize = 8;
const DONE_SEQ: usize = 12;
const ERROR: usize = 16;
const OP: usize = 20;
const ID: usize = 24;
const DTYPE: usize = 28;
const ROWS: usize = 32;
const COLS: usize = 36;
const BYTES_LO: usize = 40;
const BYTES_HI: usize = 44;
const OFFSET_LO: usize = 48;
const OFFSET_HI: usize = 52;
const CHUNK_LEN: usize = 56;
const CRC: usize = 60;
const STATS_TENSORS: usize = 64;
const STATS_BYTES_LO: usize = 68;
const STATS_BYTES_HI: usize = 72;
const STATS_KV_LO: usize = 76;
const STATS_KV_HI: usize = 80;
const READY: u32 = 0;
const REQUEST: u32 = 1;
const DONE: u32 = 2;
const FAILED: u32 = 3;
const CREATE: u32 = 1;
const WRITE: u32 = 2;
const ALIAS: u32 = 3;
const ALLOC: u32 = 4;
const FINISH: u32 = 5;

#[inline]
fn pci_read(bus: u8, dev: u8, func: u8, reg: u8) -> u32 {
    let addr = 0x8000_0000u32 | ((bus as u32) << 16) |
        ((dev as u32) << 11) | ((func as u32) << 8) | ((reg as u32) & 0xfc);
    let result: u32;
    unsafe {
        asm!("out dx, eax", in("dx") 0xcf8u16, in("eax") addr,
             options(nostack, preserves_flags));
        asm!("in eax, dx", in("dx") 0xcfcu16, out("eax") result,
             options(nostack, preserves_flags));
    }
    result
}
fn locate() -> Option<usize> {
    for bus in 0..=255u16 { for dev in 0..32u8 { for fun in 0..8u8 {
        let b = bus as u8;
        if pci_read(b, dev, fun, 0) != 0x1110_1af4 { continue; }
        if pci_read(b, dev, fun, 4) & 2 == 0 { continue; }
        let lo = pci_read(b, dev, fun, 0x18);
        if lo & 1 != 0 { continue; }
        let hi = match (lo >> 1) & 3 { 2 => pci_read(b, dev, fun, 0x1c), 0 => 0, _ => continue };
        let phys = ((hi as u64) << 32) | (lo & !0xf) as u64;
        if phys == 0 || phys > usize::MAX as u64 - (REGION + REGION_CAP) as u64 { continue; }
        let base = phys as usize + REGION;
        if unsafe { read_volatile(base as *const u32) } == MAGIC {
            serial_println!("[gpu-resident] PCI {:02x}:{:02x}.{} BAR2={:#x}",b,dev,fun,phys);
            return Some(base);
        }
    }}}
    None
}
#[inline]
fn put(base: usize, off: usize, v: u32) {
    unsafe { write_volatile((base+off) as *mut u32, v); }
}
#[inline]
fn get(base: usize, off: usize) -> u32 {
    unsafe { read_volatile((base+off) as *const u32) }
}
#[inline]
fn put64(base:usize, lo:usize, hi:usize, v:u64) {
    put(base,lo,v as u32); put(base,hi,(v >> 32) as u32);
}
#[inline]
fn get64(base:usize, lo:usize, hi:usize)->u64 {
    get(base,lo) as u64 | ((get(base,hi) as u64)<<32)
}
fn stage_bytes(base:usize, bytes:&[u8]) {
    // No per-word fence: transfer 4 MiB of WB shared BAR data, then publish
    // request once with a release fence. The service does not read until then.
    let mut words=bytes.chunks_exact(4);
    for (i,w) in words.by_ref().enumerate() {
        let v=u32::from_le_bytes([w[0],w[1],w[2],w[3]]);
        unsafe { write_volatile((base+DATA_OFF+4*i) as *mut u32,v); }
    }
    let done=bytes.len()-words.remainder().len();
    for (i,&b) in words.remainder().iter().enumerate() {
        unsafe { write_volatile((base+DATA_OFF+done+i) as *mut u8,b); }
    }
    fence(Ordering::Release);
}
fn chunk_crc(data:&[u8])->u32 {
    let mut h=Crc32::new(); h.update(data); h.finish()
}

struct Link { base:usize, seq:u32, tensor_count:u32, bytes:u64 }
impl Link {
    fn issue(&mut self, op:u32, id:u32, dtype:u32, rows:u32, cols:u32,
             size:u64, off:u64, len:u32, crc:u32) -> bool {
        let before = get(self.base,STATUS);
        if before != READY {
            serial_println!("[gpu-resident] NOT_READY before op={} id={} status={} local_seq={} shared_seq={} done_seq={} err={}",
                op, id, before, self.seq, get(self.base,SEQ),
                get(self.base,DONE_SEQ), get(self.base,ERROR));
            return false;
        }
        self.seq=self.seq.wrapping_add(1);
        put(self.base,OP,op); put(self.base,ID,id); put(self.base,DTYPE,dtype);
        put(self.base,ROWS,rows); put(self.base,COLS,cols);
        put64(self.base,BYTES_LO,BYTES_HI,size);
        put64(self.base,OFFSET_LO,OFFSET_HI,off);
        put(self.base,CHUNK_LEN,len); put(self.base,CRC,crc);
        put(self.base,SEQ,self.seq);
        fence(Ordering::Release);
        put(self.base,STATUS,REQUEST);
        fence(Ordering::SeqCst);
        // ~5s per operation; 4MiB copies and checksum verification normally
        // take far less. Timeouts never affect normal NightRun inference.
        for _ in 0..50_000 {
            let status=get(self.base,STATUS);
            fence(Ordering::Acquire);
            if status==FAILED {
                serial_println!("[gpu-resident] service failure op={} id={} error={}",op,id,get(self.base,ERROR));
                return false;
            }
            if status==DONE && get(self.base,DONE_SEQ)==self.seq {
                put(self.base,STATUS,READY);
                fence(Ordering::SeqCst);
                if self.seq <= 3 {
                    serial_println!("[gpu-resident] ACK op={} id={} seq={} status_after_ack={}",
                        op, id, self.seq, get(self.base,STATUS));
                }
                return true;
            }
            boot::stall(Duration::from_micros(100));
        }
        serial_println!("[gpu-resident] timeout op={} id={}; CPU inference preserved",op,id);
        false
    }
    fn upload(&mut self, info:ModelTensorDesc, bytes:&[u8])->bool {
        if bytes.len() as u64 != info.byte_len {
            serial_println!("[gpu-resident] BAD_LENGTH id={} bytes={} expected={}",
                info.id, bytes.len(), info.byte_len);
            return false;
        }
        if self.seq <= 2 {
            serial_println!("[gpu-resident] START tensor={} bytes={} status={} local_seq={}",
                info.id, info.byte_len, get(self.base,STATUS), self.seq);
        }
        if !self.issue(CREATE,info.id,info.dtype,info.rows,info.cols,info.byte_len,0,0,0) {
            return false;
        }
        for (part,chunk) in bytes.chunks(CHUNK).enumerate() {
            if self.seq == 1 {
                serial_println!("[gpu-resident] staging first chunk id={} len={} status={}",
                    info.id, chunk.len(), get(self.base,STATUS));
            }
            stage_bytes(self.base,chunk);
            if !self.issue(WRITE,info.id,info.dtype,info.rows,info.cols,
                            info.byte_len,(part*CHUNK) as u64,chunk.len() as u32,chunk_crc(chunk)) {
                return false;
            }
        }
        self.tensor_count+=1;
        self.bytes+=info.byte_len;
        true
    }
    fn alias(&mut self, id:u32, source:u32)->bool {
        self.issue(ALIAS,id,0,0,0,0,source as u64,0,0)
    }
    fn allocate(&mut self,id:u32, bytes:u64)->bool {
        self.issue(ALLOC,id,0,0,0,bytes,0,0,0)
    }
}

fn raw_mat<'a>(m:&'a QMat<'_>)->(u32,&'a [u8]) {
    match m {
        QMat::Q8(blocks) => (1,unsafe {core::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,blocks.len() * core::mem::size_of::<nr_tensor::q8::BlockQ8_0>())}),
        QMat::Q4K(blocks) => (2,unsafe {core::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,blocks.len() * core::mem::size_of::<nr_tensor::kquant::BlockQ4K>())}),
        QMat::Q6K(blocks) => (3,unsafe {core::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,blocks.len() * core::mem::size_of::<nr_tensor::kquant::BlockQ6K>())}),
    }
}
fn f32_bytes(values:&[f32])->&[u8] {
    unsafe {core::slice::from_raw_parts(values.as_ptr() as *const u8, values.len()*4)}
}
fn send_f32(link:&mut Link,id:u32,values:&[f32])->bool {
    let Some(cols)=u32::try_from(values.len()).ok() else {return false;};
    link.upload(ModelTensorDesc::new(id,0,1,cols,values.len() as u64*4),f32_bytes(values))
}
fn send_qmat(link:&mut Link,id:u32,mat:&QMat<'_>,rows:usize,cols:usize)->bool {
    let Some(rows)=u32::try_from(rows).ok() else {return false;};
    let Some(cols)=u32::try_from(cols).ok() else {return false;};
    let (dtype,bytes)=raw_mat(mat);
    link.upload(ModelTensorDesc::new(id,dtype,rows,cols,bytes.len() as u64),bytes)
}

static MODEL_RESIDENT_READY: AtomicBool = AtomicBool::new(false);

/// True only after all GPU tensor uploads and host-side verification succeeded.
pub fn ready() -> bool { MODEL_RESIDENT_READY.load(Ordering::Acquire) }

pub fn upload(infer:&InferCtx<'_>) {
    MODEL_RESIDENT_READY.store(false, Ordering::Release);
    let Some(base)=locate() else {
        serial_println!("[gpu-resident] service absent; existing inference unchanged");
        return;
    };
    if get(base,STATUS)!=READY {
        serial_println!("[gpu-resident] executor is not ready; skipping GPU residency");
        return;
    }
    let d=&infer.dims;
    if d.n_layers>32 || d.dim==0 || d.kv_dim==0 || d.ctx==0 {
        serial_println!("[gpu-resident] unsupported model dimensions; CPU fallback");
        return;
    }
    let mut link=Link{base,seq:0,tensor_count:0,bytes:0};
    serial_println!("[gpu-resident] uploading ALL inference weights ({} layers) into persistent VRAM",d.n_layers);
    // Stable tensor IDs: 1=embedding, 2=output norm, 3=classifier.
    // 256+16*l+slot are individual layer tensors.
    let mut ok=send_qmat(&mut link,EMBED_ID.0,&infer.weights.embed,d.vocab,d.dim)
        && send_f32(&mut link,OUTPUT_NORM_ID.0,infer.weights.out_norm);
    if !ok {serial_println!("[gpu-resident] upload failed during global tensors");return;}
    // A tied output is aliased in VRAM, not uploaded twice.
    let (ed,eb)=raw_mat(&infer.weights.embed);
    let (od,ob)=raw_mat(&infer.weights.output);
    if ed==od && eb.as_ptr()==ob.as_ptr() && eb.len()==ob.len() {
        ok=link.alias(OUTPUT_ID.0,EMBED_ID.0);
    } else {
        ok=send_qmat(&mut link,OUTPUT_ID.0,&infer.weights.output,d.vocab,d.dim);
    }
    if !ok {serial_println!("[gpu-resident] output tensor upload failed");return;}
    for (i,w) in infer.weights.layers.iter().enumerate() {
        let id=|slot:u32| layer_tensor_id(i as u32,slot).unwrap().0;
        ok=send_f32(&mut link,id(SLOT_ATTN_NORM),w.attn_norm)
            && send_qmat(&mut link,id(SLOT_Q),&w.wq,d.att_dim,d.dim)
            && send_qmat(&mut link,id(SLOT_K),&w.wk,d.kv_dim,d.dim)
            && send_qmat(&mut link,id(SLOT_V),&w.wv,d.kv_dim,d.dim)
            && send_qmat(&mut link,id(SLOT_O),&w.wo,d.dim,d.att_dim)
            && send_f32(&mut link,id(SLOT_FFN_NORM),w.ffn_norm)
            && send_qmat(&mut link,id(SLOT_GATE),&w.w_gate,d.ffn_dim,d.dim)
            && send_qmat(&mut link,id(SLOT_UP),&w.w_up,d.ffn_dim,d.dim)
            && send_qmat(&mut link,id(SLOT_DOWN),&w.w_down,d.dim,d.ffn_dim);
        if ok { if let Some(q)=w.q_norm {ok=send_f32(&mut link,id(SLOT_Q_NORM),q);} }
        if ok { if let Some(k)=w.k_norm {ok=send_f32(&mut link,id(SLOT_K_NORM),k);} }
        if !ok {serial_println!("[gpu-resident] layer {} upload failed; CPU inference unchanged",i);return;}
        if i%4==3 || i+1==d.n_layers {
            serial_println!("[gpu-resident] uploaded {}/{} layers; {} MiB resident",
                i+1,d.n_layers,link.bytes/1048576);
        }
    }
    let Some(kv_bytes)=(d.n_layers as u64)
        .checked_mul(d.ctx as u64)
        .and_then(|n|n.checked_mul(d.kv_dim as u64))
        .and_then(|n|n.checked_mul(2)) else {return;};
    if !link.allocate(KV_KEYS_ID.0,kv_bytes) || !link.allocate(KV_VALUES_ID.0,kv_bytes) ||
       !link.allocate(SCRATCH_ID.0,32*1024*1024) {
        serial_println!("[gpu-resident] KV/workspace allocation failed");return;
    }
    if !link.issue(FINISH,0,0,0,0,0,0,0,0) {return;}
    let host_tensors=get(base,STATS_TENSORS);
    let host_bytes=get64(base,STATS_BYTES_LO,STATS_BYTES_HI);
    let host_kv=get64(base,STATS_KV_LO,STATS_KV_HI);
    if host_tensors!=link.tensor_count || host_bytes!=link.bytes {
        serial_println!("[gpu-resident] statistics mismatch: guest {} tensors {} bytes vs host {} tensors {} bytes",
            link.tensor_count,link.bytes,host_tensors,host_bytes);
        return;
    }
    serial_println!("[gpu-resident] PASS: {} tensors, {} MiB packed model weights resident in RTX VRAM",
        host_tensors,host_bytes/1048576);
    serial_println!("[gpu-resident] GPU KV cache reserved={} MiB (keys+values), scratch=32 MiB",
        host_kv/1048576);
    MODEL_RESIDENT_READY.store(true, Ordering::Release);
    serial_println!("[gpu-resident] model residency READY; resident MLP may attach");
}
