// NightRun GPU layer-0 RoPE and KV-cache write: CUDA CUBIN, sm_89.
// Llama-3 adjacent pair rotation. No OS/CUDA-runtime dependencies in device code.
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>
extern "C" __global__ void nr_rope_adjacent(float* x, const float* freqs,
                                               unsigned pos, unsigned heads, unsigned head_dim) {
    const unsigned j=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned half=head_dim/2;
    if(j>=heads*half) return;
    const unsigned head=j/half, i=j%half;
    const float angle=float(pos)*freqs[i];
    float s,c; sincosf(angle,&s,&c);
    const unsigned p=head*head_dim+2*i;
    const float a=x[p],b=x[p+1];
    x[p]=a*c-b*s;
    x[p+1]=a*s+b*c;
}
extern "C" __global__ void nr_kv_store(const float* k,const float* v,
                                         __half* kc,__half* vc,unsigned pos,unsigned kv_dim) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=kv_dim)return;
    kc[size_t(pos)*kv_dim+i]=__float2half_rn(k[i]);
    vc[size_t(pos)*kv_dim+i]=__float2half_rn(v[i]);
}
