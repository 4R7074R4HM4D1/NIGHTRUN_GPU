// NightRun-owned, standalone Q8_0 GPU binary for the Ada RTX 4080 (sm_89).
// This is GPU *device code*. It uses neither PyTorch, Triton nor llama.cpp.
// Compile as a CUBIN to separate execution from CUDA runtime calls.
// Block layout: [2-byte little-endian f16 scale][32 signed int8 values].
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__(256)
void nr_q8_matvec(const unsigned char* __restrict__ weights,
                  const unsigned char* __restrict__ activation,
                  float* __restrict__ output,
                  unsigned int rows,
                  unsigned int cols) {
    // One warp computes one output row. One lane handles one byte of each
    // block; integer products are warp-reduced before scaling by the two f16s.
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= rows) return; // all lanes within the warp take same branch

    const unsigned int nblocks = cols >> 5;
    const unsigned long long row_base =
        (unsigned long long)warp * (unsigned long long)nblocks * 34ull;
    float sum = 0.f;
    for (unsigned int b = 0; b < nblocks; ++b) {
        const unsigned long long woff = row_base + (unsigned long long)b * 34ull;
        const unsigned long long xoff = (unsigned long long)b * 34ull;
        const int w = (int)(int8_t)weights[woff + 2ull + lane];
        const int x = (int)(int8_t)activation[xoff + 2ull + lane];
        int dot = w * x;
        #pragma unroll
        for (int delta = 16; delta >= 1; delta >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, delta);
        }
        if (lane == 0u) {
            // Both scale locations are 2-byte aligned.
            const uint16_t wd = *reinterpret_cast<const uint16_t*>(weights + woff);
            const uint16_t xd = *reinterpret_cast<const uint16_t*>(activation + xoff);
            const float sw = __half2float(__ushort_as_half(wd));
            const float sx = __half2float(__ushort_as_half(xd));
            // Mirrors NightRun's scalar per-block sum (numerical parity
            // requires tolerance because SIMD/other GPU algorithms regroup).
            sum += (float)dot * sw * sx;
        }
    }
    if (lane == 0u) output[warp] = sum;
}
