#!/usr/bin/env python3
"""Standalone NightRun-format Q8_0 x Q8_0 classifier benchmark for Triton.

The model's row-major Q8_0 blocks contain a little-endian f16 scale followed
by 32 signed int8 samples (34 bytes/block). This script tests the *actual*
block arithmetic and memory format, but generates synthetic weights. It does
not yet offload NightRun's loaded model or its UEFI inference pipeline.

Examples:
  python benchmark_q8_triton.py --rows 4096 --repeats 30
  python benchmark_q8_triton.py --rows 128256 --repeats 30 --full-verify
  python benchmark_q8_triton.py --rows 128256 --weights-raw /path/to/q8_classifier.bin
"""
import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl


BLOCK_ITEMS = 32
BYTES_PER_BLOCK = 34
Q8_DTYPE = np.dtype([("scale", "<f2"), ("qs", "i1", (BLOCK_ITEMS,))], align=False)
assert Q8_DTYPE.itemsize == BYTES_PER_BLOCK


@triton.jit
def q8_q8_classifier_kernel(
    W, X, Y,
    ROWS: tl.constexpr,
    BLOCKS_PER_ROW: tl.constexpr,
    B: tl.constexpr,
    BR: tl.constexpr,
):
    # Program owns BR output rows; B rounds blocks/row to a power of two.
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    b = tl.arange(0, B)
    k = tl.arange(0, 32)
    valid = (r[:, None] < ROWS) & (b[None, :] < BLOCKS_PER_ROW)

    # Q8_0 scales are f16, stored as LITTLE-ENDIAN two-byte bit patterns.
    wb = (r[:, None] * BLOCKS_PER_ROW + b[None, :]) * 34
    sw_lo = tl.load(W + wb, mask=valid, other=0).to(tl.uint16)
    sw_hi = tl.load(W + wb + 1, mask=valid, other=0).to(tl.uint16)
    sw_bits = sw_lo | (sw_hi << 8)
    sw = sw_bits.to(tl.float16, bitcast=True).to(tl.float32)

    xb = b * 34
    sx_lo = tl.load(X + xb, mask=(b < BLOCKS_PER_ROW), other=0).to(tl.uint16)
    sx_hi = tl.load(X + xb + 1, mask=(b < BLOCKS_PER_ROW), other=0).to(tl.uint16)
    sx_bits = sx_lo | (sx_hi << 8)
    sx = sx_bits.to(tl.float16, bitcast=True).to(tl.float32)

    # Signed i8 weights and activations. Expand [row, block, 32].
    w_ptrs = W + wb[:, :, None] + 2 + k[None, None, :]
    qw = tl.load(w_ptrs, mask=valid[:, :, None], other=0)
    qw = qw.to(tl.int8, bitcast=True).to(tl.int32)

    x_ptrs = X + xb[:, None] + 2 + k[None, :]
    qx = tl.load(x_ptrs, mask=(b[:, None] < BLOCKS_PER_ROW), other=0)
    qx = qx.to(tl.int8, bitcast=True).to(tl.int32)

    # NightRun: per-block signed int32 dot, multiply by f16 block scales,
    # then accumulate the floating-point block contributions.
    dot32 = tl.sum(qw * qx[None, :, :], axis=2)
    contribution = dot32.to(tl.float32) * sw * sx[None, :]
    logits = tl.sum(contribution, axis=1)
    tl.store(Y + r, logits, mask=(r < ROWS))


def synthetic_q8_weights(rows: int, cols: int, seed: int) -> np.ndarray:
    """Generate valid, row-major packed Q8_0 weights in manageable chunks."""
    bpr = cols // BLOCK_ITEMS
    raw = np.empty(rows * bpr * BYTES_PER_BLOCK, dtype=np.uint8)
    packed = raw.view(Q8_DTYPE).reshape(rows, bpr)
    rng = np.random.default_rng(seed)
    chunk = 256
    for start in range(0, rows, chunk):
        stop = min(rows, start + chunk)
        shape = (stop - start, bpr)
        packed["scale"][start:stop] = rng.uniform(0.001, 0.03, size=shape).astype(np.float16)
        packed["qs"][start:stop] = rng.integers(
            -127, 128, size=(*shape, 32), dtype=np.int8
        )
    return raw


def quantize_activation_q8(x: np.ndarray) -> np.ndarray:
    """Match NightRun's f16-rounded amax/127 and round-half-away-from-zero."""
    bpr = x.size // BLOCK_ITEMS
    xb = x.reshape(bpr, BLOCK_ITEMS)
    amax = np.max(np.abs(xb), axis=1)
    scale = (amax / np.float32(127.0)).astype(np.float16)
    scale_f32 = scale.astype(np.float32)
    inverse = np.divide(
        np.float32(1.0), scale_f32,
        out=np.zeros_like(scale_f32), where=(scale_f32 > 0)
    )
    scaled = xb * inverse[:, None]
    rounded = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    q = np.clip(rounded, -127, 127).astype(np.int8)
    packed = np.empty(bpr, dtype=Q8_DTYPE)
    packed["scale"] = scale
    packed["qs"] = q
    return packed.view(np.uint8).copy()


def cpu_reference(raw: np.ndarray, x_raw: np.ndarray, rows: int, cols: int,
                  selected_rows: np.ndarray) -> np.ndarray:
    """NightRun-style block dot reference, vectorized only over requested rows."""
    bpr = cols // 32
    w = raw.view(Q8_DTYPE).reshape(rows, bpr)
    x = x_raw.view(Q8_DTYPE)
    xq = x["qs"].astype(np.int32)
    sx = x["scale"].astype(np.float32)
    answers = np.empty(selected_rows.size, dtype=np.float32)
    for i, row in enumerate(selected_rows):
        wr = w[int(row)]
        qprod = wr["qs"].astype(np.int32) * xq
        dots = qprod.sum(axis=1, dtype=np.int32).astype(np.float32)
        ws = wr["scale"].astype(np.float32)
        answers[i] = np.sum((dots * ws) * sx, dtype=np.float32)
    return answers


def cuda_event_and_wall(op, repeats: int, warmup: int):
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()

    # Individual requests approximate autoregressive per-token behavior.
    gpu_ms, wall_ms = [], []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter_ns()
        start.record()
        op()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter_ns() - t0) / 1e6)
        gpu_ms.append(start.elapsed_time(end))

    # Continuous dispatch illustrates hardware steady state with no idle gap.
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    batch_start = torch.cuda.Event(enable_timing=True)
    batch_end = torch.cuda.Event(enable_timing=True)
    batch_start.record()
    for _ in range(repeats):
        op()
    batch_end.record()
    batch_end.synchronize()
    continuous_gpu = batch_start.elapsed_time(batch_end) / repeats
    continuous_wall = (time.perf_counter_ns() - t0) / 1e6 / repeats
    return {
        "per_request_gpu_ms": statistics.median(gpu_ms),
        "per_request_wall_ms": statistics.median(wall_ms),
        "continuous_gpu_ms": continuous_gpu,
        "continuous_wall_ms": continuous_wall,
        "per_request_p95_wall_ms": float(np.percentile(wall_ms, 95)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=128256)
    p.add_argument("--cols", type=int, default=2048)
    p.add_argument("--block-rows", type=int, choices=[1, 2, 4, 8, 16], default=4)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--check-rows", type=int, default=32)
    p.add_argument("--full-verify", action="store_true", help="Check every logit against CPU blocks (slow)")
    p.add_argument("--weights-raw", type=Path, help="Optional raw, packed Q8_0 classifier tensor")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.rows <= 0 or args.cols <= 0 or args.cols % 32:
        p.error("rows and cols must be positive, and cols divisible by 32")
    if args.repeats < 1 or args.warmup < 0:
        p.error("repeats must be >= 1 and warmup must be >= 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    bpr = args.cols // 32
    total_bytes = args.rows * bpr * 34
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton: {triton.__version__}")
    print(f"Q8_0 matrix: {args.rows} x {args.cols}; {total_bytes / 2**20:.1f} MiB")
    print(f"Q8_0 activation: {args.cols} values; {bpr*34} bytes")
    if args.weights_raw:
        if args.weights_raw.stat().st_size != total_bytes:
            raise ValueError(f"Raw classifier must be exactly {total_bytes} bytes")
        print(f"Reading raw Q8_0 weights: {args.weights_raw}", flush=True)
        raw = np.fromfile(args.weights_raw, dtype=np.uint8)
    else:
        print("Generating synthetic Q8_0 blocks...", flush=True)
        raw = synthetic_q8_weights(args.rows, args.cols, args.seed)

    rng = np.random.default_rng(args.seed + 1)
    x_float = rng.standard_normal(args.cols).astype(np.float32)
    x_raw = quantize_activation_q8(x_float)
    print("Transferring quantized weights to GPU...", flush=True)
    t0 = time.perf_counter_ns()
    W_gpu = torch.from_numpy(raw).to("cuda")
    X_gpu = torch.from_numpy(x_raw).to("cuda")
    Y_gpu = torch.empty(args.rows, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    print(f"One-time GPU upload: {(time.perf_counter_ns()-t0)/1e6:.1f} ms", flush=True)

    num_blocks = triton.next_power_of_2(bpr)
    grid = (triton.cdiv(args.rows, args.block_rows),)

    def launch():
        q8_q8_classifier_kernel[grid](
            W_gpu, X_gpu, Y_gpu,
            ROWS=args.rows,
            BLOCKS_PER_ROW=bpr,
            B=num_blocks,
            BR=args.block_rows,
            num_warps=4,
            enable_fp_fusion=False,
        )

    print("Compiling + running the Triton kernel...", flush=True)
    launch()
    torch.cuda.synchronize()

    y = Y_gpu.cpu().numpy()
    if args.full_verify:
        checked = np.arange(args.rows, dtype=np.int64)
    else:
        count = min(args.check_rows, args.rows)
        random_rows = rng.choice(args.rows, size=count, replace=False)
        checked = np.unique(np.concatenate(([0, args.rows - 1, int(np.argmax(y))], random_rows)))
    expected = cpu_reference(raw, x_raw, args.rows, args.cols, checked)
    got = y[checked]
    max_diff = np.max(np.abs(expected - got))
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=2e-2)
    print(f"Correctness PASS: {checked.size} rows, max_abs_error={max_diff:.6g}")
    print(f"GPU top-1 row: {np.argmax(y)}, logit={float(np.max(y)):.5f}")

    print("Benchmarking warm per-request and continuous dispatch...", flush=True)
    timings = cuda_event_and_wall(launch, args.repeats, args.warmup)
    for k, v in timings.items():
        print(f"{k}: {v:.4f} ms")
    print("CPU classifier baseline (your previous NightRun run): 4.9830 ms")
    if timings["per_request_wall_ms"] > 0:
        print(f"Illustrative kernel-only speedup vs that baseline: "
              f"{4.983 / timings['per_request_wall_ms']:.2f}x")
    print("NOTE: This excludes UEFI PCI transport, model integration, and token generation.")


if __name__ == "__main__":
    main()
