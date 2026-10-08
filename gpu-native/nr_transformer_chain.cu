// NightRun-owned GPU primitives: first reusable transformer execution slice.
// All kernels contain device code only. Compiled offline for sm_89.
// Q8_0 layout: [FP16 scale (2)][signed int8 q[32]] = 34 bytes.
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

// Keep the exact Stage 1 matvec kernel and ABI symbol in the same CUBIN.
#include "nr_q8_matvec.cu"

extern "C" __global__
void nr_rmsnorm(const float* __restrict__ x,
                const float* __restrict__ gamma,
                float* __restrict__ out,
                unsigned int count,
                float eps) {
    // Single-token vector, up to 8192 elements; one CTA owns the vector.
    __shared__ float warp_sums[8];
    __shared__ float inv_rms;
    const unsigned int tid = threadIdx.x;
    float acc = 0.f;
    for (unsigned int i=tid; i<count; i+=blockDim.x) {
        float v = x[i];
        acc = fmaf(v, v, acc);
    }
    for (int d=16; d>0; d>>=1) acc += __shfl_down_sync(0xffffffffu, acc, d);
    if ((tid & 31u)==0) warp_sums[tid>>5] = acc;
    __syncthreads();
    if (tid < 32) {
        float s = (tid < 8) ? warp_sums[tid] : 0.f;
        for (int d=16; d>0; d>>=1) s += __shfl_down_sync(0xffffffffu, s, d);
        if (tid==0) inv_rms = rsqrtf(s / float(count) + eps);
    }
    __syncthreads();
    const float inv = inv_rms;
    for (unsigned int i=tid; i<count; i+=blockDim.x) out[i] = x[i] * inv * gamma[i];
}

extern "C" __global__
void nr_quantize_q8(const float* __restrict__ x,
                    unsigned char* __restrict__ packed,
                    unsigned int count) {
    // Each warp quantizes one 32-element Q8_0 block.
    const unsigned int block = blockIdx.x;
    const unsigned int lane = threadIdx.x;  // launched with blockDim=32
    const unsigned int idx = block * 32u + lane;
    if (idx >= count) return;
    float val = x[idx];
    float max_abs = fabsf(val);
    for (int d=16; d>0; d>>=1) max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, d));
    max_abs = __shfl_sync(0xffffffffu, max_abs, 0);
    // NightRun CPU Q8_0 parity: quantize using the f16-rounded scale.
    // nr-tensor::q8::quantize stores the half scale first, converts it
    // back to f32, then computes inv = 1.0 / d for each 32-value block.
    const __half d_half = __float2half_rn(max_abs / 127.f);
    const float d = __half2float(d_half);
    const unsigned long long off = static_cast<unsigned long long>(block) * 34ull;
    if (lane == 0) {
        *reinterpret_cast<uint16_t*>(packed + off) = __half_as_ushort(d_half);
    }
    const float inv = d > 0.f ? (1.f / d) : 0.f;
    const float xq = val * inv;
    // Rust rounds half-away-from-zero by adding +/-0.5 before i32 cast
    // (truncation toward zero). nearbyintf would use ties-to-even instead.
    const float rounded = xq >= 0.f ? xq + 0.5f : xq - 0.5f;
    int q = static_cast<int>(rounded);
    q = max(-127, min(127, q));
    packed[off + 2ull + lane] = static_cast<unsigned char>(static_cast<int8_t>(q));
}

extern "C" __global__
void nr_residual_add(float* __restrict__ dst,
                     const float* __restrict__ residual,
                     unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] += residual[i];
}
