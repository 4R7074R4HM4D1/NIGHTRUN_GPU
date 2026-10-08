// NightRun Stage 4: real layer-0 SwiGLU MLP primitives.
// Device-only CUBIN, usable by a future native command-submission backend.
// Kernel entrypoints and Q8_0 format are NightRun owned.
#include "nr_transformer_chain.cu"

extern "C" __global__
void nr_swiglu(float* gate, const float* up, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float v = gate[i];
        // Silu(v) * up, as in NightRun's SwiGLU MLP.
        gate[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

extern "C" __global__
void nr_scaled_residual(float* x, const float* projected,
                        unsigned int count, float scale) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = fmaf(projected[i], scale, x[i]);
}

// NIGHTRUN_STAGE8D_GPU_EMBED_Q8_V1: no runtime table upload; weights are
// persistent tensor ID 1. Token IDs are supplied in UEFI GPU commands.
extern "C" __global__
void nr_q8_embedding_row(const unsigned char* __restrict__ table,
                         float* __restrict__ out,
                         unsigned int token,unsigned int dim,float scale) {
    const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=dim)return;
    const unsigned long long row=(unsigned long long)token*(dim/32u)*34ull;
    const unsigned long long off=row+(i/32u)*34ull;
    const unsigned short bits=(unsigned short)table[off] |
                              ((unsigned short)table[off+1ull]<<8);
    const __half h=__ushort_as_half(bits);
    const int8_t q=(int8_t)table[off+2ull+(i%32u)];
    // Same order as NightRun's Q8 dequantize then muP embed scaling.
    out[i]=(__half2float(h)*(float)q)*scale;
}

// NIGHTRUN_STAGE8E_GPU_SAMPLER_V1.
// UEFI selects temperature, top-p and RNG. These are numerical primitives.
// Stage A: independently rank top 64 of every 512 logits (no host sorting).
extern "C" __global__
void nr_sample_top64_partial(const float* logits,unsigned vocab,
                             float* partial,unsigned* ids) {
    if(threadIdx.x!=0)return;
    const unsigned begin=blockIdx.x*512u;
    const unsigned base=blockIdx.x*64u;
    float best[64];unsigned best_id[64];
    for(unsigned i=0;i<64;i++){best[i]=-INFINITY;best_id[i]=0xffffffffu;}
    const unsigned end=(begin+512u<vocab)?begin+512u:vocab;
    for(unsigned j=begin;j<end;++j){
        const float v=logits[j];
        if(!isfinite(v))continue; // UEFI has independently checked all logits.
        if(v<best[63] || (v==best[63] && j>=best_id[63]))continue;
        unsigned at=0;
        while(at<64 && (v<best[at] || (v==best[at] && j>=best_id[at])))++at;
        if(at<64){
            for(unsigned i=63;i>at;--i){best[i]=best[i-1];best_id[i]=best_id[i-1];}
            best[at]=v;best_id[at]=j;
        }
    }
    for(unsigned i=0;i<64;i++){partial[base+i]=best[i];ids[base+i]=best_id[i];}
}

// Stage B: merge candidate fragments, temperature softmax, nucleus cutoff.
// Single-thread merge is deliberately simple: coverage before optimization.
// Output is 520 bytes: [count,mass,64 IDs,64 unnormalized probabilities].
extern "C" __global__
void nr_sample_top64_nucleus(const float* partial,const unsigned* part_ids,
                              unsigned count,float logit_scale,float temperature,
                              float top_p,unsigned* out) {
    if(blockIdx.x!=0 || threadIdx.x!=0)return;
    float best[64];unsigned ids[64];
    for(unsigned i=0;i<64;++i){best[i]=-INFINITY;ids[i]=0xffffffffu;}
    for(unsigned j=0;j<count*64u;++j){
        const unsigned id=part_ids[j];
        if(id==0xffffffffu)continue;
        const float v=partial[j];
        if(v<best[63] || (v==best[63] && id>=ids[63]))continue;
        unsigned at=0;
        while(at<64 && (v<best[at] || (v==best[at] && id>=ids[at])))++at;
        if(at<64){
            for(unsigned i=63;i>at;--i){best[i]=best[i-1];ids[i]=ids[i-1];}
            best[at]=v;ids[at]=id;
        }
    }
    if(ids[63]==0xffffffffu || !isfinite(best[0])){out[0]=0;return;}
    const float max_scaled=best[0]*logit_scale;
    float probs[64];float sum=0.f;
    for(unsigned i=0;i<64;++i){
        const float scaled=best[i]*logit_scale;
        const float p=expf((scaled-max_scaled)/temperature);
        probs[i]=p;sum+=p;
    }
    const float cutoff=top_p*sum;
    float cumulative=0.f;unsigned n=64;
    for(unsigned i=0;i<64;++i){
        cumulative+=probs[i];
        if(cumulative>=cutoff){n=i+1;break;}
    }
    out[0]=n;out[1]=__float_as_uint(cumulative);
    for(unsigned i=0;i<64;++i){
        out[2+i]=ids[i];out[66+i]=__float_as_uint(probs[i]);
    }
}
