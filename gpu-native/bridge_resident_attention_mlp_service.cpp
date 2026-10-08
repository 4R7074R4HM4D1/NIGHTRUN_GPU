// NightRun Stage 5A GPU-resident model tensor allocator / upload service.
// QEMU transport only. No tokenizer, GGUF/.nrm parser, inference scheduling,
// or model execution. All tensor IDs, formats, shapes are NightRun-controlled.
#include <cuda.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <limits>
#include <cmath>

#define CU_OK(call) do { CUresult _cuda_rc=(call); if(_cuda_rc!=CUDA_SUCCESS) { \
    const char *name=nullptr, *msg=nullptr; \
    cuGetErrorName(_cuda_rc,&name); cuGetErrorString(_cuda_rc,&msg); \
    throw std::runtime_error(std::string(#call)+": "+(name?name:"error")+" "+(msg?msg:"")); \
} } while(0)

namespace p {
constexpr size_t region=0xA0000000ULL;
constexpr size_t capacity=0x02000000ULL;
constexpr size_t data=0x1000;
constexpr size_t max_chunk=4*1024*1024;
constexpr uint32_t magic=0x3547504e;
constexpr size_t status=4, seq=8, done_seq=12, error=16, op=20, id=24,
    dtype=28, rows=32, cols=36, bytes_lo=40, bytes_hi=44,
    off_lo=48, off_hi=52, chunk_len=56, crc=60,
    stats_tensors=64,stats_bytes_lo=68,stats_bytes_hi=72,
    stats_kv_lo=76,stats_kv_hi=80;
constexpr uint32_t ready=0, request=1, done=2, failed=3;
constexpr uint32_t create=1, write=2, alias=3, alloc=4, finish=5;
constexpr uint32_t kv_keys=1000,kv_vals=1001,scratch=1002;
}
static uint64_t u64(uint32_t lo,uint32_t hi) {
    return uint64_t(lo)|(uint64_t(hi)<<32);
}
static uint32_t crc32(const uint8_t* bytes,size_t n) {
    uint32_t c=0xffffffffU;
    for(size_t i=0;i<n;++i) {
        c^=bytes[i];
        for(int j=0;j<8;++j) c=(c>>1) ^ (0xedb88320U & -(c&1u));
    }
    return ~c;
}
struct Shared {
    int fd{-1}; uint8_t *mapping{nullptr},*at{nullptr};size_t length{0};
    explicit Shared(const char* path) {
        fd=open(path,O_RDWR);if(fd<0)throw std::runtime_error("cannot open ivshmem backing file");
        struct stat st{};
        if(fstat(fd,&st) || st.st_size<0 || uint64_t(st.st_size)<p::region+p::capacity)
            throw std::runtime_error("shared file too small (need existing 8GiB mapping)");
        length=size_t(st.st_size);
        void* addr=mmap(nullptr,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
        if(addr==MAP_FAILED) throw std::runtime_error("mmap failed");
        mapping=static_cast<uint8_t*>(addr);at=mapping+p::region;
    }
    ~Shared(){if(mapping)munmap(mapping,length);if(fd>=0)close(fd);}
    uint32_t get(size_t off)const {
        return __atomic_load_n(reinterpret_cast<uint32_t*>(at+off),__ATOMIC_ACQUIRE);
    }
    void put(size_t off,uint32_t x){
        __atomic_store_n(reinterpret_cast<uint32_t*>(at+off),x,__ATOMIC_RELEASE);
    }
    uint64_t get64(size_t l,size_t h) const {return u64(get(l),get(h));}
    void put64(size_t l,size_t h,uint64_t v){put(l,uint32_t(v));put(h,uint32_t(v>>32));}
};
struct Tensor {
    CUdeviceptr ptr{0}; uint64_t bytes{0};uint32_t dtype{0},rows{0},cols{0};
    bool owned{true};
    uint64_t uploaded{0};
};
struct Executor {
    CUcontext context{};
    std::unordered_map<uint32_t,Tensor> tensors;
    std::vector<uint8_t> verify;
    uint64_t uploaded_bytes=0;
    uint64_t kv_bytes=0;
    uint32_t weight_tensors=0;
    bool finished=false;
    Executor() : verify(p::max_chunk) {
        CU_OK(cuInit(0));CUdevice dev{};CU_OK(cuDeviceGet(&dev,0));
        char name[128]{}; CU_OK(cuDeviceGetName(name,sizeof(name),dev));
        std::cout<<"[gpu-resident-host] GPU: "<<name<<std::endl;
        CU_OK(cuCtxCreate(&context,0,dev));
    }
    ~Executor(){
        if(context){
            for(const auto& [id,t]:tensors) if(t.owned && t.ptr) cuMemFree(t.ptr);
            cuCtxDestroy(context);
        }
    }
    static void validate_meta(uint32_t dtype,uint32_t rows,uint32_t cols,uint64_t bytes) {
        if(rows==0 || cols==0 || bytes==0 || bytes>(uint64_t(4)<<30))
            throw std::runtime_error("invalid tensor dimensions or size");
        uint64_t elements=uint64_t(rows)*cols;
        uint64_t expected=0;
        switch(dtype){
          case 0: expected=elements*4;break;
          case 1: if(elements%32)break;expected=elements/32*34;break;
          case 2: if(elements%256)break;expected=elements/256*144;break;
          case 3: if(elements%256)break;expected=elements/256*210;break;
          default: break;
        }
        if(!expected || bytes!=expected)throw std::runtime_error("tensor dtype/shape/length mismatch");
    }
    void command(Shared& s) {
        if(finished)throw std::runtime_error("upload session already finalized");
        const auto op=s.get(p::op),id=s.get(p::id),dtype=s.get(p::dtype);
        const auto rows=s.get(p::rows),cols=s.get(p::cols);
        const uint64_t bytes=s.get64(p::bytes_lo,p::bytes_hi);
        const uint64_t off=s.get64(p::off_lo,p::off_hi);
        const uint32_t len=s.get(p::chunk_len);
        if(op==p::create){
            validate_meta(dtype,rows,cols,bytes);
            if(id==0 || id>=1000 || tensors.count(id) || tensors.size()>=550)
                throw std::runtime_error("tensor ID invalid or already allocated");
            Tensor t{};t.bytes=bytes;t.dtype=dtype;t.rows=rows;t.cols=cols;
            CU_OK(cuMemAlloc(&t.ptr,size_t(bytes)));
            tensors.emplace(id,t);
        } else if(op==p::write) {
            auto it=tensors.find(id);
            if(it==tensors.end() || !it->second.owned || it->second.dtype!=dtype ||
               it->second.rows!=rows || it->second.cols!=cols || it->second.bytes!=bytes)
                throw std::runtime_error("upload missing or mismatched tensor");
            auto &t=it->second;
            if(len==0 || len>p::max_chunk || off!=t.uploaded ||
                off>bytes || uint64_t(len)>bytes-off || p::data+len>p::capacity)
                throw std::runtime_error("out-of-order or out-of-bounds chunk");
            const uint32_t wanted=s.get(p::crc);
            if(crc32(s.at+p::data,len)!=wanted)
                throw std::runtime_error("shared-memory chunk CRC mismatch");
            CU_OK(cuMemcpyHtoD(t.ptr+off,s.at+p::data,len));
            // Verify the GPU VRAM bytes, not merely the host-side upload buffer.
            CU_OK(cuMemcpyDtoH(verify.data(),t.ptr+off,len));
            if(crc32(verify.data(),len)!=wanted)
                throw std::runtime_error("GPU VRAM readback CRC mismatch");
            t.uploaded+=len;
            if(t.uploaded==t.bytes){
                uploaded_bytes+=t.bytes;++weight_tensors;
            }
        } else if(op==p::alias){
            if(id==0 || id>=1000 || tensors.count(id) || off==id)
                throw std::runtime_error("invalid alias ID");
            auto it=tensors.find(uint32_t(off));
            if(it==tensors.end() || it->second.uploaded!=it->second.bytes)
                throw std::runtime_error("alias source missing or not complete");
            Tensor t=it->second;t.owned=false;
            tensors.emplace(id,t);
        } else if(op==p::alloc){
            if((id!=p::kv_keys && id!=p::kv_vals && id!=p::scratch) ||
               tensors.count(id) || bytes==0 || bytes>uint64_t(2)*1024*1024*1024)
                throw std::runtime_error("invalid persistent runtime allocation");
            Tensor t{};t.bytes=bytes;
            CU_OK(cuMemAlloc(&t.ptr,size_t(bytes)));
            CU_OK(cuMemsetD8(t.ptr,0,size_t(bytes)));
            tensors.emplace(id,t);
            if(id==p::kv_keys || id==p::kv_vals)kv_bytes+=bytes;
        } else if(op==p::finish) {
            if(!tensors.count(p::kv_keys) || !tensors.count(p::kv_vals) || !tensors.count(p::scratch))
                throw std::runtime_error("runtime buffers missing");
            if(!tensors.count(1) || !tensors.count(2) || !tensors.count(3))
                throw std::runtime_error("global model tensors missing");
            for(const auto& [key,t]:tensors)
                if(key<1000 && t.uploaded!=t.bytes)
                    throw std::runtime_error("model tensor upload incomplete");
            // Registry is now immutable and held in CUDA context until exit.
            finished=true;
            s.put(p::stats_tensors,weight_tensors);
            s.put64(p::stats_bytes_lo,p::stats_bytes_hi,uploaded_bytes);
            s.put64(p::stats_kv_lo,p::stats_kv_hi,kv_bytes);
            std::cout<<"[gpu-resident-host] PASS: "<<weight_tensors
                     <<" model tensors, "<<(uploaded_bytes/1048576.0)
                     <<" MiB verified in VRAM, KV "<<(kv_bytes/1048576.0)
                     <<" MiB, scratch 32 MiB"<<std::endl;
            std::cout<<"[gpu-resident-host] allocations persistent; waiting (do not terminate)"<<std::endl;
        } else throw std::runtime_error("unknown NightRun residency operation");
    }
};

// Stage 5B: live Stage 4 MLP shares the CUDA context and the exact Stage 5A
// CUdeviceptr allocations. There is NO second host-side weight upload.
namespace mlp {
constexpr size_t region=0x80000000ULL;
constexpr size_t capacity=0x08000000ULL;
constexpr uint32_t magic=0x3447504e;
constexpr size_t status=4,seq=8,done_seq=12,error=16,dim=20,ffn=24;
constexpr size_t service_us=28,gpu_us=32,eps_bits=36,scale_bits=40,count=44;
constexpr size_t resident_mode=48,commands=0x100,weights=0x10000;
constexpr uint32_t ready=0,request=1,done=2,failed=3,init=4,init_done=5;
constexpr size_t align64(size_t n){return (n+63)&~size_t(63);}
struct Layout{size_t wg,wu,wd,gamma,input,output,limit;};
Layout layout(unsigned d,unsigned f){
    if(d!=2048 || f==0 || f>16384 || f%32)
        throw std::runtime_error("unsupported MLP dimensions");
    const auto gf=size_t(f)*(d/32)*34,df=size_t(d)*(f/32)*34;
    const auto wg=weights,wu=align64(wg+gf),wd=align64(wu+gf);
    const auto ga=align64(wd+df),in=align64(ga+size_t(d)*4);
    const auto out=align64(in+size_t(d)*4),end=align64(out+size_t(d)*4);
    if(end>capacity)throw std::runtime_error("MLP bridge layout too large");
    return{wg,wu,wd,ga,in,out,end};
}
}
struct Cmd {
    uint32_t op;uint16_t ver,flags;
    uint32_t dst,src0,src1,aux,rows,cols,s0,s1,res[2];
};
static_assert(sizeof(Cmd)==48,"Stage 4 GPU command ABI mismatch");
struct MlpLink {
    Shared& s;
    uint8_t* at;
    explicit MlpLink(Shared& sh):s(sh),at(sh.mapping+mlp::region) {}
    uint32_t get(size_t off)const{
        return __atomic_load_n(reinterpret_cast<uint32_t*>(at+off),__ATOMIC_ACQUIRE);
    }
    void put(size_t off,uint32_t v){
        __atomic_store_n(reinterpret_cast<uint32_t*>(at+off),v,__ATOMIC_RELEASE);
    }
};
static float float_bits(uint32_t bits){float x;std::memcpy(&x,&bits,4);return x;}
static bool valid_mlp_commands(const MlpLink& m,unsigned d,unsigned f,float eps,float scale){
    if(m.get(mlp::count)!=8 || m.get(mlp::resident_mode)!=1)return false;
    const uint32_t ops[]={2,3,1,1,5,3,1,6};
    const uint32_t dst[]={3,4,6,8,6,9,11,1};
    const uint32_t s0[]={1,3,5,7,6,6,10,1};
    const uint32_t s1[]={2,0,4,4,8,0,9,11};
    const uint32_t rows[]={d,d,f,f,f,f,d,d};
    const uint32_t cols[]={0,0,d,d,0,0,f,0};
    Cmd cmds[8];std::memcpy(cmds,m.at+mlp::commands,sizeof(cmds));
    for(unsigned i=0;i<8;++i){
        const auto& c=cmds[i];
        if(c.op!=ops[i] || c.ver!=1 || c.flags!=0 || c.dst!=dst[i] ||
           c.src0!=s0[i] || c.src1!=s1[i] || c.aux!=0 ||
           c.rows!=rows[i] || c.cols!=cols[i] || c.s1!=0 ||
           c.res[0]!=0 || c.res[1]!=0)return false;
        if(i!=0 && i!=7 && c.s0!=0)return false;
        if(i==0 && c.s0!=m.get(mlp::eps_bits))return false;
        if(i==7 && c.s0!=m.get(mlp::scale_bits))return false;
    }
    return eps>0 && std::isfinite(eps) && std::isfinite(scale);
}
struct MlpExec {
    Executor& residency;
    CUmodule module{}; CUfunction norm{},quant{},gemv{},swiglu{},add{};
    CUstream stream{}; CUgraph graph{}; CUgraphExec graph_exec{};
    CUevent start{},stop{};
    CUdeviceptr x{},normalized{},qnorm{},gate{},up{},qgate{},down{};
    CUdeviceptr gamma{},wg{},wu{},wd{}; // borrowed, never allocated/freed here
    unsigned dim{},ffn{};float eps{},scale{};bool initialized=false;
    explicit MlpExec(Executor& r,const char* cubin):residency(r) {
        CU_OK(cuModuleLoad(&module,cubin));
        CU_OK(cuModuleGetFunction(&norm,module,"nr_rmsnorm"));
        CU_OK(cuModuleGetFunction(&quant,module,"nr_quantize_q8"));
        CU_OK(cuModuleGetFunction(&gemv,module,"nr_q8_matvec"));
        CU_OK(cuModuleGetFunction(&swiglu,module,"nr_swiglu"));
        CU_OK(cuModuleGetFunction(&add,module,"nr_scaled_residual"));
        CU_OK(cuStreamCreate(&stream,CU_STREAM_DEFAULT));
        CU_OK(cuEventCreate(&start,CU_EVENT_DEFAULT));
        CU_OK(cuEventCreate(&stop,CU_EVENT_DEFAULT));
        std::cout<<"[gpu-stage5b-host] shared CUDA context; offline sm_89 CUBIN loaded"<<std::endl;
    }
    // No freeing borrowed Stage 5A model weights; the owning registry manages them.
    const Tensor& tensor(uint32_t id,uint32_t dtype,uint32_t rows,uint32_t cols) const {
        const auto it=residency.tensors.find(id);
        if(it==residency.tensors.end())throw std::runtime_error("resident tensor ID missing: "+std::to_string(id));
        const Tensor& t=it->second;
        if(t.dtype!=dtype || t.rows!=rows || t.cols!=cols || t.uploaded!=t.bytes)
            throw std::runtime_error("resident tensor metadata mismatch: "+std::to_string(id));
        return t;
    }
    void initialize(MlpLink& m) {
        if(initialized)throw std::runtime_error("duplicate MLP initialization");
        if(!residency.finished)throw std::runtime_error("model residency not yet finalized");
        dim=m.get(mlp::dim);ffn=m.get(mlp::ffn);
        eps=float_bits(m.get(mlp::eps_bits));scale=float_bits(m.get(mlp::scale_bits));
        (void)mlp::layout(dim,ffn);
        if(!valid_mlp_commands(m,dim,ffn,eps,scale))
            throw std::runtime_error("invalid resident MLP command list or mode flag");
        // Same fixed ID ABI declared by nr_gpu::resident.
        gamma=tensor(261,0,1,dim).ptr;       // 256+SLOT_FFN_NORM
        wg=tensor(262,1,ffn,dim).ptr;        // 256+SLOT_GATE
        wu=tensor(263,1,ffn,dim).ptr;        // 256+SLOT_UP
        wd=tensor(264,1,dim,ffn).ptr;        // 256+SLOT_DOWN
        CU_OK(cuMemAlloc(&x,size_t(dim)*4));
        CU_OK(cuMemAlloc(&normalized,size_t(dim)*4));
        CU_OK(cuMemAlloc(&qnorm,size_t(dim/32)*34));
        CU_OK(cuMemAlloc(&gate,size_t(ffn)*4));
        CU_OK(cuMemAlloc(&up,size_t(ffn)*4));
        CU_OK(cuMemAlloc(&qgate,size_t(ffn/32)*34));
        CU_OK(cuMemAlloc(&down,size_t(dim)*4));
        constexpr unsigned threads=256;
        void* n[]={&x,&gamma,&normalized,&dim,&eps};
        void* q1[]={&normalized,&qnorm,&dim};
        void* g[]={&wg,&qnorm,&gate,&ffn,&dim};
        void* u[]={&wu,&qnorm,&up,&ffn,&dim};
        void* s[]={&gate,&up,&ffn};
        void* q2[]={&gate,&qgate,&ffn};
        void* d[]={&wd,&qgate,&down,&dim,&ffn};
        void* r[]={&x,&down,&dim,&scale};
        CU_OK(cuStreamBeginCapture(stream,CU_STREAM_CAPTURE_MODE_GLOBAL));
        CU_OK(cuLaunchKernel(norm,1,1,1,threads,1,1,0,stream,n,nullptr));
        CU_OK(cuLaunchKernel(quant,dim/32,1,1,32,1,1,0,stream,q1,nullptr));
        CU_OK(cuLaunchKernel(gemv,(ffn+7)/8,1,1,threads,1,1,0,stream,g,nullptr));
        CU_OK(cuLaunchKernel(gemv,(ffn+7)/8,1,1,threads,1,1,0,stream,u,nullptr));
        CU_OK(cuLaunchKernel(swiglu,(ffn+threads-1)/threads,1,1,threads,1,1,0,stream,s,nullptr));
        CU_OK(cuLaunchKernel(quant,ffn/32,1,1,32,1,1,0,stream,q2,nullptr));
        CU_OK(cuLaunchKernel(gemv,(dim+7)/8,1,1,threads,1,1,0,stream,d,nullptr));
        CU_OK(cuLaunchKernel(add,(dim+threads-1)/threads,1,1,threads,1,1,0,stream,r,nullptr));
        CU_OK(cuStreamEndCapture(stream,&graph));
        CU_OK(cuGraphInstantiate(&graph_exec,graph,0));
        initialized=true;
        std::cout<<"[gpu-stage5b-host] layer-0 MLP borrowing resident tensor IDs 261,262,263,264; "
                 "NO weight HtoD; dim="<<dim<<" ffn="<<ffn<<std::endl;
    }
    uint32_t execute(MlpLink& m) {
        if(!initialized)throw std::runtime_error("MLP requested before resident initialize");
        const auto o=mlp::layout(dim,ffn);
        CU_OK(cuMemcpyHtoD(x,m.at+o.input,size_t(dim)*4));
        CU_OK(cuEventRecord(start,stream));
        CU_OK(cuGraphLaunch(graph_exec,stream));
        CU_OK(cuEventRecord(stop,stream));
        CU_OK(cuStreamSynchronize(stream));
        CU_OK(cuMemcpyDtoH(m.at+o.output,x,size_t(dim)*4));
        float ms=0;CU_OK(cuEventElapsedTime(&ms,start,stop));
        return uint32_t(ms*1000+0.5f);
    }
};

// Stage 6A: real layer-0 GQA attention, using the SAME resident KV buffers.
// NightRun computes Q/K/V + RoPE and owns a CPU KV mirror for safe fallback.
namespace attn {
constexpr size_t region=0xC0000000ULL,capacity=0x00400000ULL;
constexpr uint32_t magic=0x3641504e;
constexpr size_t status=4,seq=8,done_seq=12,error=16,op=20,pos=24,ctx=28,
    heads=32,kv_heads=36,head_dim=40,kv_dim=44,service_us=48,gpu_us=52,
    sync_start=56,sync_rows=60,scale_bits=64;
constexpr size_t query=0x1000,key=0x5000,val=0x6000,output=0x8000,data=0x10000;
constexpr uint32_t ready=0,request=1,done=2,failed=3,init=1,sync=2,run=3;
}
struct AttentionLink {
    uint8_t* at;
    explicit AttentionLink(Shared& sh):at(sh.mapping+attn::region){}
    uint32_t get(size_t off)const {
        return __atomic_load_n(reinterpret_cast<uint32_t*>(at+off),__ATOMIC_ACQUIRE);
    }
    void put(size_t off,uint32_t v) {
        __atomic_store_n(reinterpret_cast<uint32_t*>(at+off),v,__ATOMIC_RELEASE);
    }
};
struct AttentionExec {
    Executor& r;
    CUmodule mod{};CUfunction score{},value{};
    CUstream stream{};CUevent start{},stop{};
    CUdeviceptr q{},scores{},out{},keys{},vals{};
    unsigned ctx{},heads{},kv_heads{},hd{},kv{},next_pos{};
    float scale{};
    bool initialized=false;
    unsigned count=0;
    AttentionExec(Executor& residency,const char* cubin):r(residency){
        CU_OK(cuModuleLoad(&mod,cubin));
        CU_OK(cuModuleGetFunction(&score,mod,"nr_gqa_scores"));
        CU_OK(cuModuleGetFunction(&value,mod,"nr_gqa_value"));
        CU_OK(cuStreamCreate(&stream,CU_STREAM_DEFAULT));
        CU_OK(cuEventCreate(&start,CU_EVENT_DEFAULT));
        CU_OK(cuEventCreate(&stop,CU_EVENT_DEFAULT));
        std::cout<<"[gpu-stage6-host] offline GQA attention CUBIN loaded"<<std::endl;
    }
    ~AttentionExec(){
        if(q)cuMemFree(q);
        if(scores)cuMemFree(scores);
        if(out)cuMemFree(out);
        if(stop)cuEventDestroy(stop);
        if(start)cuEventDestroy(start);
        if(stream)cuStreamDestroy(stream);
        if(mod)cuModuleUnload(mod);
    }
    void initialize(AttentionLink& l) {
        if(initialized||!r.finished)throw std::runtime_error("attention initialized without completed residency");
        ctx=l.get(attn::ctx);heads=l.get(attn::heads);kv_heads=l.get(attn::kv_heads);
        hd=l.get(attn::head_dim);kv=l.get(attn::kv_dim);
        scale=float_bits(l.get(attn::scale_bits));
        if(ctx==0 || ctx>16384 || heads!=32 || kv_heads!=8 || hd!=64 || kv!=512 ||
           !(scale>0 && std::isfinite(scale)))throw std::runtime_error("unsupported attention dimensions");
        auto k=r.tensors.find(p::kv_keys),v=r.tensors.find(p::kv_vals);
        if(k==r.tensors.end()||v==r.tensors.end())throw std::runtime_error("resident KV buffers missing");
        if(k->second.bytes<uint64_t(ctx)*kv*2 || v->second.bytes<uint64_t(ctx)*kv*2)
            throw std::runtime_error("resident KV buffers too small");
        keys=k->second.ptr;vals=v->second.ptr;
        CU_OK(cuMemAlloc(&q,size_t(heads)*hd*4));
        CU_OK(cuMemAlloc(&scores,size_t(heads)*ctx*4));
        CU_OK(cuMemAlloc(&out,size_t(heads)*hd*4));
        next_pos=0; initialized=true;
        std::cout<<"[gpu-stage6-host] layer-0 attention using GPU-resident KV IDs 1000/1001"
                 <<" (no separate KV allocation)"<<std::endl;
    }
    void sync_history(AttentionLink& l){
        if(!initialized)throw std::runtime_error("attention not initialized");
        const unsigned begin=l.get(attn::sync_start),rows=l.get(attn::sync_rows);
        if(begin==0)next_pos=0; // Restart from CPU cache after prefill or context reset.
        if(begin!=next_pos || uint64_t(begin)+rows>ctx ||
           uint64_t(rows)*kv*4>attn::capacity-attn::data)
            throw std::runtime_error("KV history chunk range/order invalid");
        const size_t bytes=size_t(rows)*kv*2;
        if(bytes){
            CU_OK(cuMemcpyHtoD(keys+size_t(begin)*kv*2,l.at+attn::data,bytes));
            CU_OK(cuMemcpyHtoD(vals+size_t(begin)*kv*2,l.at+attn::data+bytes,bytes));
        }
        next_pos=begin+rows;
    }
    uint32_t execute(AttentionLink& l){
        if(!initialized)throw std::runtime_error("attention not initialized");
        unsigned pos=l.get(attn::pos);
        if(pos>=ctx || pos!=next_pos)throw std::runtime_error("attention KV position out of sequence");
        CU_OK(cuMemcpyHtoD(q,l.at+attn::query,size_t(heads)*hd*4));
        CU_OK(cuMemcpyHtoD(keys+size_t(pos)*kv*2,l.at+attn::key,size_t(kv)*2));
        CU_OK(cuMemcpyHtoD(vals+size_t(pos)*kv*2,l.at+attn::val,size_t(kv)*2));
        const unsigned rows=(pos+1+127)/128;
        void* s[]={&q,&keys,&scores,&pos,&ctx,&heads,&kv_heads,&hd,&kv,&scale};
        void* v[]={&scores,&vals,&out,&pos,&ctx,&heads,&kv_heads,&hd,&kv};
        CU_OK(cuEventRecord(start,stream));
        CU_OK(cuLaunchKernel(score,heads,rows,1,128,1,1,0,stream,s,nullptr));
        CU_OK(cuLaunchKernel(value,heads,1,1,64,1,1,0,stream,v,nullptr));
        CU_OK(cuEventRecord(stop,stream));
        CU_OK(cuStreamSynchronize(stream));
        CU_OK(cuMemcpyDtoH(l.at+attn::output,out,size_t(heads)*hd*4));
        float ms=0;CU_OK(cuEventElapsedTime(&ms,start,stop));
        next_pos=pos+1;
        ++count;
        if(count<=3 || count%64==0){
            std::cout<<"[gpu-stage6-host] real layer-0 GQA #"<<count<<" pos="<<pos
                     <<" GPU_us="<<uint32_t(ms*1000+0.5f)<<std::endl;
        }
        return uint32_t(ms*1000+0.5f);
    }
};

int main(int argc,char**argv) {
    if(argc!=4){std::cerr<<"Usage: "<<argv[0]
        <<" <gpu_bridge_8g.bin> <nr_layer_mlp_sm89.cubin> <nr_attention_sm89.cubin>\n";return 2;}
    try {
        Shared shm(argv[1]);
        Executor registry;               // one CUDA context, owns all model tensors
        MlpExec mlp_engine(registry,argv[2]);
        MlpLink mlp_link(shm);
        AttentionExec attn_engine(registry,argv[3]);
        AttentionLink attn_link(shm);
        shm.put(0,p::magic);shm.put(p::error,0);shm.put(p::done_seq,0);
        shm.put(p::status,p::ready);
        mlp_link.put(0,mlp::magic);mlp_link.put(mlp::error,0);
        mlp_link.put(mlp::done_seq,0);mlp_link.put(mlp::status,mlp::ready);
        attn_link.put(0,attn::magic);attn_link.put(attn::error,0);
        attn_link.put(attn::done_seq,0);attn_link.put(attn::status,attn::ready);
        std::cout<<"[gpu-stage5b-host] residency at +0xa0000000, resident MLP at +0x80000000"<<std::endl;
        unsigned count=0;
        for(;;) {
            if(shm.get(p::status)==p::request) {
                const uint32_t seq=shm.get(p::seq);
                try {
                    registry.command(shm);
                    shm.put(p::done_seq,seq);shm.put(p::status,p::done);
                } catch(const std::exception& e) {
                    std::cerr<<"[gpu-stage5b-host] residency error seq="<<seq<<": "<<e.what()<<std::endl;
                    shm.put(p::error,1);shm.put(p::done_seq,seq);shm.put(p::status,p::failed);
                }
                continue;
            }
            const uint32_t state=mlp_link.get(mlp::status);
            if(state==mlp::init || state==mlp::request) {
                const uint32_t seq=mlp_link.get(mlp::seq);
                const auto t0=std::chrono::steady_clock::now();
                try {
                    if(state==mlp::init) {
                        mlp_engine.initialize(mlp_link);
                        mlp_link.put(mlp::done_seq,seq);
                        mlp_link.put(mlp::status,mlp::init_done);
                    } else {
                        if(mlp_link.get(mlp::dim)!=mlp_engine.dim ||
                            mlp_link.get(mlp::ffn)!=mlp_engine.ffn)
                            throw std::runtime_error("unexpected MLP dimensions");
                        const uint32_t graph_us=mlp_engine.execute(mlp_link);
                        const auto t1=std::chrono::steady_clock::now();
                        const uint32_t service_us=uint32_t(std::chrono::duration_cast<
                            std::chrono::microseconds>(t1-t0).count());
                        mlp_link.put(mlp::gpu_us,graph_us);
                        mlp_link.put(mlp::service_us,service_us);
                        mlp_link.put(mlp::done_seq,seq);
                        mlp_link.put(mlp::status,mlp::done);
                        ++count;
                        if(count<=3 || count%64==0)
                            std::cout<<"[gpu-stage5b-host] resident layer-0 MLP #"<<count
                                     <<" service_us="<<service_us<<" graph_us="<<graph_us<<std::endl;
                    }
                } catch(const std::exception& e) {
                    std::cerr<<"[gpu-stage5b-host] MLP error: "<<e.what()<<std::endl;
                    mlp_link.put(mlp::error,state==mlp::init?1:2);
                    mlp_link.put(mlp::status,mlp::failed);
                }
                continue;
            }
            if(attn_link.get(attn::status)==attn::request){
                const uint32_t seq=attn_link.get(attn::seq);
                const uint32_t op=attn_link.get(attn::op);
                const auto t0=std::chrono::steady_clock::now();
                try{
                    uint32_t gpu_us=0;
                    if(op==attn::init)attn_engine.initialize(attn_link);
                    else if(op==attn::sync)attn_engine.sync_history(attn_link);
                    else if(op==attn::run)gpu_us=attn_engine.execute(attn_link);
                    else throw std::runtime_error("invalid attention operation");
                    const auto t1=std::chrono::steady_clock::now();
                    const uint32_t service_us=uint32_t(std::chrono::duration_cast<
                        std::chrono::microseconds>(t1-t0).count());
                    attn_link.put(attn::gpu_us,gpu_us);
                    attn_link.put(attn::service_us,service_us);
                    attn_link.put(attn::done_seq,seq);
                    attn_link.put(attn::status,attn::done);
                } catch(const std::exception& e){
                    std::cerr<<"[gpu-stage6-host] ATTENTION ERROR seq="<<seq<<": "<<e.what()<<std::endl;
                    attn_link.put(attn::error,1);
                    attn_link.put(attn::done_seq,seq);
                    attn_link.put(attn::status,attn::failed);
                }
                continue;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    } catch(const std::exception& e){
        std::cerr<<"[gpu-stage5b-host] FATAL: "<<e.what()<<std::endl;return 1;
    }
}
