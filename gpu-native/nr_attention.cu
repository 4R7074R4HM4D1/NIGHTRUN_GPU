// NightRun-owned decode-time GQA attention kernels for GPU-resident f16 KV.
// Offline compiled for sm_89; no CUDA runtime or libc functions at kernel launch.
#include <cuda_fp16.h>
#include <math.h>

extern "C" __global__
void nr_gqa_scores(const float* q, const __half* keys, float* scores,
                   unsigned pos, unsigned ctx, unsigned heads,
                   unsigned kv_heads, unsigned head_dim, unsigned kv_dim,
                   float scale) {
    const unsigned h = blockIdx.x;
    const unsigned t = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= heads || t > pos) return;
    const unsigned kvh = h / (heads / kv_heads);
    const __half* k = keys + size_t(t)*kv_dim + size_t(kvh)*head_dim;
    const float* x = q + size_t(h)*head_dim;
    float dot = 0.0f;
    for (unsigned i=0;i<head_dim;++i) dot += x[i]*__half2float(k[i]);
    scores[size_t(h)*ctx+t] = dot*scale;
}

extern "C" __global__
void nr_gqa_value(const float* scores, const __half* values, float* result,
                  unsigned pos, unsigned ctx, unsigned heads,
                  unsigned kv_heads, unsigned head_dim, unsigned kv_dim) {
    const unsigned h=blockIdx.x;
    const unsigned i=threadIdx.x;
    if (h>=heads) return;
    const unsigned kvh=h/(heads/kv_heads);
    const float* s=scores+size_t(h)*ctx;
    __shared__ float mx, denom;
    if(i==0){
        float m=-INFINITY;
        for(unsigned t=0;t<=pos;++t) m=fmaxf(m,s[t]);
        float d=0.0f;
        for(unsigned t=0;t<=pos;++t) d+=expf(s[t]-m);
        mx=m;denom=d;
    }
    __syncthreads();
    if(i<head_dim){
        float z=0.0f;
        for(unsigned t=0;t<=pos;++t){
            float p=expf(s[t]-mx)/denom;
            z += p*__half2float(values[size_t(t)*kv_dim+size_t(kvh)*head_dim+i]);
        }
        result[size_t(h)*head_dim+i]=z;
    }
}
