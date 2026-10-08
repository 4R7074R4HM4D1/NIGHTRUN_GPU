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
    CUmodule module{}; CUfunction norm{},quant{},gemv{},swiglu{},add{},embed{},sample_partial{},sample_finish{};
    CUstream stream{}; CUgraph graph{}; CUgraphExec graph_exec{};
    CUevent start{},stop{};
    CUdeviceptr x{},normalized{},qnorm{},gate{},up{},qgate{},down{};
    CUdeviceptr gamma{},wg{},wu{},wd{}; // borrowed, never allocated/freed here
    unsigned dim{},ffn{};float eps{},scale{};bool initialized=false;
    // NIGHTRUN_STAGE9A_UEFI_OWNED_CUBIN_V1: Both legacy file mode and UEFI image mode use same kernel bindings.
    void bind_module(CUmodule loaded) {
        if(module)throw std::runtime_error("MLP CUBIN already loaded");
        module=loaded;
        CU_OK(cuModuleGetFunction(&norm,module,"nr_rmsnorm"));
        CU_OK(cuModuleGetFunction(&quant,module,"nr_quantize_q8"));
        CU_OK(cuModuleGetFunction(&gemv,module,"nr_q8_matvec"));
        CU_OK(cuModuleGetFunction(&swiglu,module,"nr_swiglu"));
        CU_OK(cuModuleGetFunction(&add,module,"nr_scaled_residual"));
        CU_OK(cuModuleGetFunction(&embed,module,"nr_q8_embedding_row"));
        CU_OK(cuModuleGetFunction(&sample_partial,module,"nr_sample_top64_partial"));
        CU_OK(cuModuleGetFunction(&sample_finish,module,"nr_sample_top64_nucleus"));
    }
    void load_bytes(const uint8_t* bytes) {
        if(module)throw std::runtime_error("MLP CUBIN already loaded");
        CUmodule loaded{};CU_OK(cuModuleLoadData(&loaded,bytes));bind_module(loaded);
    }
    explicit MlpExec(Executor& r,const char* cubin):residency(r) {
        if(cubin){CUmodule loaded{};CU_OK(cuModuleLoad(&loaded,cubin));bind_module(loaded);}
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
    // NIGHTRUN_STAGE9A_UEFI_OWNED_CUBIN_V1: CUDA loads the exact ELF supplied by UEFI.
    void bind_module(CUmodule loaded){
        if(mod)throw std::runtime_error("GQA CUBIN already loaded");
        mod=loaded;
        CU_OK(cuModuleGetFunction(&score,mod,"nr_gqa_scores"));
        CU_OK(cuModuleGetFunction(&value,mod,"nr_gqa_value"));
    }
    void load_bytes(const uint8_t* bytes){
        if(mod)throw std::runtime_error("GQA CUBIN already loaded");
        CUmodule loaded{};CU_OK(cuModuleLoadData(&loaded,bytes));bind_module(loaded);
    }
    AttentionExec(Executor& residency,const char* cubin):r(residency){
        if(cubin){CUmodule loaded{};CU_OK(cuModuleLoad(&loaded,cubin));bind_module(loaded);}
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


// Stage 6B — Full layer-0 GPU inference with actual NightRun weights, one
// PCI command per token. This service dispatches only GPU commands and tensor
// allocations; all token sampling/scheduling remain in UEFI.
namespace layer {
constexpr size_t region=0xE0000000ULL, capacity=0x01000000ULL;
constexpr uint32_t magic=0x364c504e;
constexpr size_t status=4,seq=8,ack=12,error=16,op=20,pos=24,ctx=28,
  dim=32,ffn=36,hd=40,heads=44,kv_heads=48,kv_dim=52,eps=56,
  residual=60,attn_scale=64,service_us=68,gpu_us=72,sync_start=76,sync_rows=80,cmd_count=84,cmd_abi=88,commands=0x9000;
constexpr size_t freq=0x1000,input=0x2000,out_k=0x5000,out_v=0x5800,data=0x10000;
constexpr size_t sync_layer=96,out_v_stack=0xD000,data_stack=0x20000;
constexpr size_t diag_layer=100,diag_valid=104,diag_base=0x600000,diag_stride=0x10000;
constexpr uint32_t diag=4,snapshot_kv=5;
// Stage 8B: guest-authored bounded prefill request. Existing opcodes preserved.
constexpr uint32_t prefill_batch=6;
// NIGHTRUN_STAGE8C_GPU_FINAL_V1: extra UEFI command frame.
constexpr uint32_t final_head=7;
// NIGHTRUN_STAGE8D_GPU_EMBED_Q8_V1: CUDA only dispatches UEFI authored descriptors.
constexpr uint32_t run_token_embed=8,run_batch_embed=9;
// NIGHTRUN_STAGE8E_GPU_SAMPLER_V1: UEFI controls parameters and RNG.
constexpr uint32_t sample=10;
// NIGHTRUN_STAGE9A_UEFI_OWNED_CUBIN_V1 - module image upload from UEFI, not from host filesystem.
constexpr uint32_t load_cubin=11;
constexpr size_t module_id=132,module_bytes=136,module_crc=140;
constexpr size_t module_data=0x800000,module_max=0x400000;
constexpr size_t sample_temp=116,sample_top_p=120,sample_scale=124,
                 sample_expected=128,sample_out=0x390000;
constexpr size_t embed_cmds=0x380000;
constexpr size_t final_vocab=112,final_logits=0x240000,final_commands=0x360000;
constexpr unsigned max_vocab=262144;
constexpr size_t batch_count=108,batch_input=0x100000,
    batch_keys=0x180000,batch_vals=0x1c0000;
constexpr unsigned max_batch=16;
// Diagnostic buffer: 16 layers * 2048 FP32 values, inside 16 MiB PCI region.
constexpr size_t trace=0x400000;
constexpr uint32_t ready=0,request=1,done=2,failed=3,init=1,sync=2,run=3;
}
// NIGHTRUN_STAGE9C_GENERIC_GPU_LAUNCH_V1: GPU runtime ABI (no model opcodes).
// Each descriptor is a bounded symbol + grid/block + typed arguments. UEFI
// decides the operation order. The bridge resolves handles and calls CUDA.
namespace nr_launch {
constexpr size_t abi=0x3a02a0,cap=0x3a02a4,records=0x3a0000,bytes=224,head_count=3;
constexpr size_t diag_ordinal=0x3a02a8;
constexpr uint32_t abi_version=1,cap_magic=0x314c524e;
constexpr uint32_t buf=1,u32=2,f32=3;
}

struct LayerLink {
    uint8_t* at;
    explicit LayerLink(Shared& sh):at(sh.mapping+layer::region){}
    uint32_t get(size_t o)const{return __atomic_load_n(reinterpret_cast<const uint32_t*>(at+o),__ATOMIC_ACQUIRE);}
    void put(size_t o,uint32_t v){__atomic_store_n(reinterpret_cast<uint32_t*>(at+o),v,__ATOMIC_RELEASE);}
};
struct FullLayerExec {
    Executor& registry; MlpExec& m; AttentionExec& a;
    CUmodule mod{}; CUfunction rope{},kvwrite{};
    CUstream stream{}; CUevent start{},stop{};
    CUdeviceptr x{},normalized{},qnorm{},q{},k{},v{},frequencies{},scores{},attout{};
    CUdeviceptr prefill_embeddings{};
    CUdeviceptr final_norm{},final_q8{},final_logits_dev{},sample_parts{},sample_ids{},sample_output_dev{};
    unsigned final_logits_capacity=0;
    unsigned generic_final_calls=0;
    uint32_t last_final_seq=0;
    CUdeviceptr qa{},projected{},ffn_norm{},ffn_q{},gate{},up{},qgate{},down{};
    CUdeviceptr keys{},vals{},attngamma{},wq{},wk{},wv{},wo{},ffngamma{},wg{},wu{},wd{};
    unsigned dim{},ffn{},ctx{},hd{},heads{},kv_heads{},kv_dim{},next_pos{},calls{},n_layers{1};
    unsigned next_pos_by_layer[16]{};
    float eps{},residual{},attn_scale{};
    bool initialized=false;
    std::vector<Cmd> plan; // submitted by NightRun UEFI, NOT inferred by the CUDA service
    // NIGHTRUN_STAGE9A_UEFI_OWNED_CUBIN_V1: GPU firmware is carried inside BOOTX64.EFI.
    void bind_module(CUmodule loaded){
        if(mod)throw std::runtime_error("RoPE CUBIN already loaded");
        mod=loaded;
        CU_OK(cuModuleGetFunction(&rope,mod,"nr_rope_adjacent"));
        CU_OK(cuModuleGetFunction(&kvwrite,mod,"nr_kv_store"));
    }
    void load_bytes(const uint8_t* bytes){
        if(mod)throw std::runtime_error("RoPE CUBIN already loaded");
        CUmodule loaded{};CU_OK(cuModuleLoadData(&loaded,bytes));bind_module(loaded);
    }
    FullLayerExec(Executor& r,MlpExec& mlp,AttentionExec& attn,const char* cubin)
        :registry(r),m(mlp),a(attn) {
        if(cubin){CUmodule loaded{};CU_OK(cuModuleLoad(&loaded,cubin));bind_module(loaded);}
        CU_OK(cuStreamCreate(&stream,CU_STREAM_DEFAULT));
        CU_OK(cuEventCreate(&start,CU_EVENT_DEFAULT));
        CU_OK(cuEventCreate(&stop,CU_EVENT_DEFAULT));
        std::cout<<"[gpu-stage7-host] offline NightRun compute kernels ready; low-level CUDA driver only"<<std::endl;
    }
    ~FullLayerExec(){
        for(CUdeviceptr ptr:{x,normalized,qnorm,q,k,v,frequencies,scores,attout,qa,
                             projected,ffn_norm,ffn_q,gate,up,qgate,down})
            if(ptr)cuMemFree(ptr);
        if(prefill_embeddings)cuMemFree(prefill_embeddings);
        if(final_norm)cuMemFree(final_norm);
        if(final_q8)cuMemFree(final_q8);
        if(final_logits_dev)cuMemFree(final_logits_dev);
        if(sample_parts)cuMemFree(sample_parts);
        if(sample_ids)cuMemFree(sample_ids);
        if(sample_output_dev)cuMemFree(sample_output_dev);
        if(start)cuEventDestroy(start);
        if(stop)cuEventDestroy(stop);
        if(stream)cuStreamDestroy(stream);
        if(mod)cuModuleUnload(mod);
    }
    void initialize(LayerLink& l){
        if(initialized || !registry.finished)throw std::runtime_error("full layer init before residency");
        dim=l.get(layer::dim);ffn=l.get(layer::ffn);ctx=l.get(layer::ctx);
        hd=l.get(layer::hd);heads=l.get(layer::heads);kv_heads=l.get(layer::kv_heads);
        kv_dim=l.get(layer::kv_dim);
        eps=float_bits(l.get(layer::eps));residual=float_bits(l.get(layer::residual));
        attn_scale=float_bits(l.get(layer::attn_scale));
        if(dim!=2048 || ffn!=8192 || ctx<1 || ctx>16384 || hd!=64 || heads!=32 ||
           kv_heads!=8 || kv_dim!=512 || !(eps>0 && std::isfinite(eps)) ||
           !(attn_scale>0 && std::isfinite(attn_scale)) || !std::isfinite(residual))
            throw std::runtime_error("full-layer dimensions or scales unsupported");
        // Reuse only the packed Q8_0 and FP32 weight tensors Stage 5A uploaded.
        attngamma=m.tensor(256,0,1,dim).ptr;
        wq=m.tensor(257,1,dim,dim).ptr; wk=m.tensor(258,1,kv_dim,dim).ptr;
        wv=m.tensor(259,1,kv_dim,dim).ptr;wo=m.tensor(260,1,dim,dim).ptr;
        ffngamma=m.tensor(261,0,1,dim).ptr;
        wg=m.tensor(262,1,ffn,dim).ptr;wu=m.tensor(263,1,ffn,dim).ptr;
        wd=m.tensor(264,1,dim,ffn).ptr;
        const auto kk=registry.tensors.find(p::kv_keys),vv=registry.tensors.find(p::kv_vals);
        if(kk==registry.tensors.end() || vv==registry.tensors.end() ||
           kk->second.bytes<uint64_t(ctx)*kv_dim*2 ||
           vv->second.bytes<uint64_t(ctx)*kv_dim*2)
            throw std::runtime_error("GPU resident KV memory missing/undersized");
        keys=kk->second.ptr;vals=vv->second.ptr;
        const auto allocate=[&](CUdeviceptr& dst,size_t len){CU_OK(cuMemAlloc(&dst,len));};
        allocate(prefill_embeddings,size_t(layer::max_batch)*dim*4);
        allocate(final_norm,size_t(dim)*4);
        allocate(final_q8,size_t(dim/32)*34);
        // 262144/512 partitions x 64 entries, reused for every token.
        allocate(sample_parts,512u*64u*4u);
        allocate(sample_ids,512u*64u*4u);
        allocate(sample_output_dev,520u);
        allocate(x,size_t(dim)*4);allocate(normalized,size_t(dim)*4);
        allocate(qnorm,size_t(dim/32)*34);
        allocate(q,size_t(dim)*4);allocate(k,size_t(kv_dim)*4);allocate(v,size_t(kv_dim)*4);
        allocate(frequencies,size_t(hd/2)*4);allocate(scores,size_t(heads)*ctx*4);
        allocate(attout,size_t(dim)*4);allocate(qa,size_t(dim/32)*34);
        allocate(projected,size_t(dim)*4);allocate(ffn_norm,size_t(dim)*4);
        allocate(ffn_q,size_t(dim/32)*34);allocate(gate,size_t(ffn)*4);
        allocate(up,size_t(ffn)*4);allocate(qgate,size_t(ffn/32)*34);
        allocate(down,size_t(dim)*4);
        CU_OK(cuMemcpyHtoD(frequencies,l.at+layer::freq,size_t(hd/2)*4));
        // UEFI's fixed-width, pointer-free command ABI is the source of truth.
        const uint32_t n=l.get(layer::cmd_count);
        if(l.get(layer::cmd_abi)!=1 || (n!=21 && n!=336) ||
           layer::commands+size_t(n)*sizeof(Cmd)>layer::capacity)
            throw std::runtime_error("unsupported NightRun UEFI command list");
        plan.resize(n);
        std::memcpy(plan.data(), l.at+layer::commands, size_t(n)*sizeof(Cmd));
        n_layers=n/21; // observed from UEFI command count; never guessed from model
        for(size_t i=0;i<plan.size();i++) {
            const auto& c=plan[i];
            if(c.ver!=1 || c.flags!=i/21 || c.res[0] || c.res[1] || c.s1 ||
               c.op<1 || c.op>10)
                throw std::runtime_error("invalid UEFI-ordered layer-tagged command descriptor");
        }
        // Fail closed if any layer's resident weights are missing/mismatched.
        // GPU driver validates allocations; NightRun already decided the ordering.
        for(unsigned li=0;li<n_layers;li++) {
            const uint32_t b=256+li*16;
            m.tensor(b+0,0,1,dim);m.tensor(b+1,1,dim,dim);
            m.tensor(b+2,1,kv_dim,dim);m.tensor(b+3,1,kv_dim,dim);
            m.tensor(b+4,1,dim,dim);m.tensor(b+5,0,1,dim);
            m.tensor(b+6,1,ffn,dim);m.tensor(b+7,1,ffn,dim);
            m.tensor(b+8,1,dim,ffn);
        }
        next_pos=0;for(unsigned i=0;i<16;i++)next_pos_by_layer[i]=0;
        calls=0;initialized=true;
        std::cout<<"[gpu-stage7-host] received "<<plan.size()
                 <<" ordered NightRun UEFI GPU commands; driver-dispatch only"<<std::endl;
        std::cout<<"[gpu-stage7-host] attached "<<n_layers
                 <<" resident GPU layers, UEFI-authored commands="<<plan.size()
                 <<"; GPU KV handles 1000/1001; NO weight HtoD"<<std::endl;
    }
    void sync_history(LayerLink& l){
        if(!initialized)throw std::runtime_error("full layer not initialized");
        const unsigned begin=l.get(layer::sync_start),rows=l.get(layer::sync_rows);
        const unsigned li=n_layers==1?0:l.get(layer::sync_layer);
        if(li>=n_layers)throw std::runtime_error("KV sync layer outside UEFI model");
        if(begin==0)next_pos_by_layer[li]=0;
        const size_t data_off=n_layers==1?layer::data:layer::data_stack;
        if(begin!=next_pos_by_layer[li] || uint64_t(begin)+rows>ctx ||
           uint64_t(rows)*kv_dim*4>layer::capacity-data_off)
            throw std::runtime_error("GPU KV history order or size invalid");
        const size_t bytes=size_t(rows)*kv_dim*2;
        const size_t off=(size_t(li)*ctx+begin)*kv_dim*2;
        if(bytes){
            CU_OK(cuMemcpyHtoD(keys+off,l.at+data_off,bytes));
            CU_OK(cuMemcpyHtoD(vals+off,l.at+data_off+bytes,bytes));
        }
        next_pos_by_layer[li]=begin+rows;
    }
    // GPU_KV_HISTORY_PROBE_V1: read-only PCI-visible snapshot of resident KV.
    // The guest chooses layer/range and performs all CPU-vs-GPU comparisons.
    void snapshot_kv_history(LayerLink& l) {
        if (!initialized) throw std::runtime_error("KV probe before GPU layer initialization");
        const unsigned li=l.get(layer::sync_layer);
        const unsigned begin=l.get(layer::sync_start);
        const unsigned rows=l.get(layer::sync_rows);
        if (li>=n_layers || rows==0 || rows>256 || begin>ctx || rows>ctx-begin)
            throw std::runtime_error("GPU KV snapshot layer/range invalid");
        const size_t nbytes=size_t(rows)*kv_dim*2;
        if (layer::data_stack+2*nbytes>layer::diag_base)
            throw std::runtime_error("GPU KV snapshot overlaps diagnostic shared memory");
        const size_t off=(size_t(li)*ctx+begin)*kv_dim*2;
        // RUN kernels use this non-default stream. Ensure their output is visible.
        CU_OK(cuStreamSynchronize(stream));
        CU_OK(cuMemcpyDtoH(l.at+layer::data_stack,keys+off,nbytes));
        CU_OK(cuMemcpyDtoH(l.at+layer::data_stack+nbytes,vals+off,nbytes));
    }
    // CUDA is ONLY a hardware executor. No transformer-layer control flow,
    // sampling, model parsing or token scheduling lives in this function.
    struct DeviceBuf { CUdeviceptr ptr; size_t bytes; };
    DeviceBuf buffer(uint32_t id) const {
        const auto it=registry.tensors.find(id);
        if(it!=registry.tensors.end())
            return {it->second.ptr,size_t(it->second.bytes)};
        switch(id) {
          case 2000:return{x,size_t(dim)*4};
          case 2001:return{normalized,size_t(dim)*4};
          case 2002:return{qnorm,size_t(dim/32)*34};
          case 2003:return{q,size_t(dim)*4};
          case 2004:return{k,size_t(kv_dim)*4};
          case 2005:return{v,size_t(kv_dim)*4};
          case 2006:return{frequencies,size_t(hd/2)*4};
          case 2007:return{scores,size_t(heads)*ctx*4};
          case 2008:return{attout,size_t(dim)*4};
          case 2009:return{qa,size_t(dim/32)*34};
          case 2010:return{projected,size_t(dim)*4};
          case 2011:return{ffn_norm,size_t(dim)*4};
          case 2012:return{ffn_q,size_t(dim/32)*34};
          case 2013:return{gate,size_t(ffn)*4};
          case 2014:return{up,size_t(ffn)*4};
          case 2015:return{qgate,size_t(ffn/32)*34};
          case 2016:return{down,size_t(dim)*4};
          case 2017:return{final_norm,size_t(dim)*4};
          case 2018:return{final_q8,size_t(dim/32)*34};
          case 2019:return{final_logits_dev,size_t(final_logits_capacity)*4};
          default:throw std::runtime_error("GPU command references unknown handle");
        }
    }
    CUdeviceptr checked(uint32_t id,size_t needed) const {
        const DeviceBuf b=buffer(id);
        if(!b.ptr || b.bytes<needed)
            throw std::runtime_error("NightRun GPU command has undersized buffer");
        return b.ptr;
    }
    void dispatch(const Cmd& c,unsigned position){
        constexpr unsigned block=256;
        unsigned n=c.rows, cols=c.cols;
        if(n==0 || n>(c.op==1?layer::max_vocab:16384))
            throw std::runtime_error("invalid GPU command length");
        if(c.flags>=n_layers)throw std::runtime_error("invalid UEFI layer tag");
        const size_t layer_stride=size_t(ctx)*kv_dim*2;
        const size_t kv_layer_offset=size_t(c.flags)*layer_stride;
        switch(c.op){
          case 1: { // Quantized GEMV
            if(!cols || cols%32 || cols>16384)throw std::runtime_error("bad Q8 shape");
            CUdeviceptr w=checked(c.src0,size_t(n)*(cols/32)*34);
            CUdeviceptr in=checked(c.src1,size_t(cols/32)*34);
            CUdeviceptr out=checked(c.dst,size_t(n)*4);
            void* args[]={&w,&in,&out,&n,&cols};
            CU_OK(cuLaunchKernel(m.gemv,(n+7)/8,1,1,block,1,1,0,stream,args,nullptr));
            break;
          }
          case 2: { // RMSNorm
            if(n>8192)throw std::runtime_error("bad norm length");
            CUdeviceptr in=checked(c.src0,size_t(n)*4);
            CUdeviceptr gamma=checked(c.src1,size_t(n)*4);
            CUdeviceptr out=checked(c.dst,size_t(n)*4);
            float eps=float_bits(c.s0);
            if(!(eps>0) || !std::isfinite(eps))throw std::runtime_error("bad norm epsilon");
            void* args[]={&in,&gamma,&out,&n,&eps};
            CU_OK(cuLaunchKernel(m.norm,1,1,1,block,1,1,0,stream,args,nullptr));
            break;
          }
          case 3: { // FP32 to Q8_0
            if(n%32)throw std::runtime_error("invalid quant length");
            CUdeviceptr in=checked(c.src0,size_t(n)*4);
            CUdeviceptr out=checked(c.dst,size_t(n/32)*34);
            void* args[]={&in,&out,&n};
            CU_OK(cuLaunchKernel(m.quant,n/32,1,1,32,1,1,0,stream,args,nullptr));
            break;
          }
          case 5: { // SwiGLU in place on gate
            if(c.src0!=c.dst)throw std::runtime_error("SwiGLU must be in-place");
            CUdeviceptr gate=checked(c.dst,size_t(n)*4);
            CUdeviceptr up=checked(c.src1,size_t(n)*4);
            void* args[]={&gate,&up,&n};
            CU_OK(cuLaunchKernel(m.swiglu,(n+block-1)/block,1,1,block,1,1,0,stream,args,nullptr));
            break;
          }
          case 6: { // scaled residual in-place
            if(c.src0!=c.dst)throw std::runtime_error("residual must be in-place");
            CUdeviceptr in=checked(c.dst,size_t(n)*4);
            CUdeviceptr proj=checked(c.src1,size_t(n)*4);
            float scale=float_bits(c.s0);
            if(!std::isfinite(scale))throw std::runtime_error("bad residual scale");
            void* args[]={&in,&proj,&n,&scale};
            CU_OK(cuLaunchKernel(m.add,(n+block-1)/block,1,1,block,1,1,0,stream,args,nullptr));
            break;
          }
          case 7: { // RoPE in-place for query or key
            if(cols!=hd || (n!=heads && n!=kv_heads) || n*cols>dim)
                throw std::runtime_error("unsupported RoPE shape");
            CUdeviceptr vec=checked(c.dst,size_t(n)*cols*4);
            CUdeviceptr freq=checked(c.src0,size_t(hd/2)*4);
            void* args[]={&vec,&freq,&position,&n,&cols};
            CU_OK(cuLaunchKernel(rope,(n*cols/2+255)/256,1,1,256,1,1,0,stream,args,nullptr));
            break;
          }
          case 8: { // K/V write to GPU-resident cache
            if(n!=kv_dim || c.dst!=1000 || c.aux!=1001)
                throw std::runtime_error("invalid KV cache target");
            CUdeviceptr kk=checked(c.src0,size_t(n)*4);
            CUdeviceptr vv=checked(c.src1,size_t(n)*4);
            CUdeviceptr dstk=checked(c.dst,size_t(n_layers)*layer_stride)+kv_layer_offset;
            CUdeviceptr dstv=checked(c.aux,size_t(n_layers)*layer_stride)+kv_layer_offset;
            void* args[]={&kk,&vv,&dstk,&dstv,&position,&n};
            CU_OK(cuLaunchKernel(kvwrite,(n+255)/256,1,1,256,1,1,0,stream,args,nullptr));
            break;
          }
          case 9: { // GQA score calculation
            if(n!=heads || cols!=hd)throw std::runtime_error("bad GQA shape");
            CUdeviceptr query=checked(c.src0,size_t(dim)*4);
            CUdeviceptr kk=checked(c.src1,size_t(n_layers)*layer_stride)+kv_layer_offset;
            CUdeviceptr out=checked(c.dst,size_t(heads)*ctx*4);
            float scale=float_bits(c.s0);
            if(!std::isfinite(scale) || scale<=0)throw std::runtime_error("bad GQA scale");
            void* args[]={&query,&kk,&out,&position,&ctx,&heads,&kv_heads,&hd,&kv_dim,&scale};
            CU_OK(cuLaunchKernel(a.score,heads,(position+1+127)/128,1,128,1,1,0,stream,args,nullptr));
            break;
          }
          case 10: { // weighted value reduction
            if(n!=heads || cols!=hd)throw std::runtime_error("bad GQA value shape");
            CUdeviceptr sc=checked(c.src0,size_t(heads)*ctx*4);
            CUdeviceptr vv=checked(c.src1,size_t(n_layers)*layer_stride)+kv_layer_offset;
            CUdeviceptr out=checked(c.dst,size_t(dim)*4);
            void* args[]={&sc,&vv,&out,&position,&ctx,&heads,&kv_heads,&hd,&kv_dim};
            CU_OK(cuLaunchKernel(a.value,heads,1,1,hd,1,1,0,stream,args,nullptr));
            break;
          }
          default:throw std::runtime_error("unsupported NightRun GPU opcode");
        }
    }
    void diagnose(LayerLink& l) {
        const uint32_t ordinal=l.get(nr_launch::diag_ordinal);
        if(!initialized || calls==0 || ordinal<1 || ordinal>3 || ordinal>calls)
            throw std::runtime_error("isolated diagnostic outside parity window");
        const unsigned li=l.get(layer::diag_layer),position=l.get(layer::pos);
        if(li>=n_layers || position>=ctx || next_pos_by_layer[li]!=position+1 ||
           plan.size()!=size_t(n_layers)*21)
            throw std::runtime_error("invalid UEFI diagnostic layer or position");
        // Save the exact GPU layer KV prefix, substitute UEFI CPU history,
        // then restore all original rows (including current) after replay.
        const size_t base=size_t(li)*ctx*kv_dim*2;
        const size_t row_off=base+size_t(position)*kv_dim*2;
        const size_t count=size_t(position+1)*kv_dim;
        const size_t history_bytes=size_t(position)*kv_dim*2;
        if(history_bytes*2>layer::diag_base-layer::data_stack)
            throw std::runtime_error("CPU KV history exceeds isolated diagnostic region");
        std::vector<uint16_t> old_k(count),old_v(count);
        CU_OK(cuStreamSynchronize(stream));
        CU_OK(cuMemcpyDtoH(old_k.data(),keys+base,count*2));
        CU_OK(cuMemcpyDtoH(old_v.data(),vals+base,count*2));
        auto restore=[&](){
            CU_OK(cuStreamSynchronize(stream));
            CU_OK(cuMemcpyHtoD(keys+base,old_k.data(),count*2));
            CU_OK(cuMemcpyHtoD(vals+base,old_v.data(),count*2));
        };
        l.put(layer::diag_valid,0);
        try {
            // Replay against CPU supplied historical KV, not the drifting
            // primary GPU KV. This is a diagnostic-only temporary overwrite.
            if(history_bytes) {
                CU_OK(cuMemcpyHtoD(keys+base,l.at+layer::data_stack,history_bytes));
                CU_OK(cuMemcpyHtoD(vals+base,l.at+layer::data_stack+history_bytes,history_bytes));
            }
            CU_OK(cuMemcpyHtoD(x,l.at+layer::input,size_t(dim)*4));
            for(unsigned j=0;j<21;++j) {
                const Cmd& c=plan[size_t(li)*21+j];
                dispatch(c,position);  // UEFI authored operation + operands
                // Temporary driver only copies selected operation outputs.
                const size_t slot=layer::diag_base+size_t(j)*layer::diag_stride;
                if(j==7) {
                    CU_OK(cuStreamSynchronize(stream));
                    CU_OK(cuMemcpyDtoH(l.at+slot,keys+row_off,kv_dim*2));
                    CU_OK(cuMemcpyDtoH(l.at+layer::diag_base+size_t(8)*layer::diag_stride,
                                       vals+row_off,kv_dim*2));
                } else if(c.op==3 && (j==1 || j==10 || j==14 || j==18)) {
                    // GPU_Q8_BYTE_PROBE_V1: snapshot packed Q8_0 activation.
                    // No interpretation, only DMA/copy of UEFI-selected output.
                    const size_t bytes=(size_t(c.rows)/32u)*34u;
                    if(bytes>layer::diag_stride)
                        throw std::runtime_error("Q8 diagnostic exceeds slot");
                    CU_OK(cuStreamSynchronize(stream));
                    const CUdeviceptr src=checked(c.dst,bytes);
                    CU_OK(cuMemcpyDtoH(l.at+slot,src,bytes));
                } else if(c.op==1 || c.op==2 || c.op==5 || c.op==6 ||
                          c.op==7 || c.op==10) {
                    const size_t n=c.op==7?size_t(c.rows)*c.cols:
                                   c.op==10?size_t(dim):size_t(c.rows);
                    if(n*4>layer::diag_stride)
                        throw std::runtime_error("diagnostic snapshot larger than slot");
                    CU_OK(cuStreamSynchronize(stream));
                    const CUdeviceptr src=checked(c.dst,n*4);
                    CU_OK(cuMemcpyDtoH(l.at+slot,src,n*4));
                }
            }
        } catch(...) {restore();throw;}
        restore();
        l.put(layer::diag_valid,1);
        std::cout<<"[gpu-opdiag-host] isolated UEFI layer="<<li<<" pos="<<position
                 <<" snapshots READY; original GPU KV history restored"<<std::endl;
    }
    // NIGHTRUN_STAGE8B_UEFI_16TOKEN_PREFILL_BATCH_V1:
    // Low-level replay of UEFI-supplied token embeddings and command list.
    // No model loading, token decisions, or layer scheduling are authored here.
    // NIGHTRUN_STAGE8C_GPU_FINAL_V1. The 3 GPU operations and buffer IDs
    // are sent by NightRun UEFI; this is only a low-level dispatch loop.
    // Validate and execute one UEFI-authored Q8 embedding command.
    // This is a primitive launch, not a model-inference or token decision.
    void launch_embedding_cmd(const Cmd& c) {
        if(c.op!=11 || c.ver!=1 || c.flags!=0 || c.dst!=2000 || c.src0!=1 ||
           c.src1!=0 || c.rows!=dim || c.cols<1 || c.cols>layer::max_vocab ||
           c.aux>=c.cols || c.s1!=0 || c.res[0]!=0 || c.res[1]!=0)
            throw std::runtime_error("invalid UEFI embedding command");
        float scale=float_bits(c.s0);
        if(!std::isfinite(scale))throw std::runtime_error("invalid embedding scale");
        const auto& t=m.tensor(1,1,c.cols,dim);
        CUdeviceptr table=t.ptr;
        CUdeviceptr output=checked(c.dst,size_t(dim)*4);
        unsigned token=c.aux;
        unsigned n=dim;
        void* args[]={&table,&output,&token,&n,&scale};
        CU_OK(cuLaunchKernel(m.embed,(n+255)/256,1,1,256,1,1,0,stream,args,nullptr));
    }
// NIGHTRUN_STAGE9C_GENERIC_GPU_LAUNCH_V1. The CUDA bridge does not
// select kernel symbols, launch geometry, or argument order.
void dispatch_generic(LayerLink& l,unsigned index) {
    if(index>=nr_launch::head_count)throw std::runtime_error("generic launch index invalid");
    const size_t at=nr_launch::records+size_t(index)*nr_launch::bytes;
    if(at+nr_launch::bytes>layer::trace)throw std::runtime_error("generic launch outside PCI range");
    const uint32_t module_id=l.get(at);
    CUmodule gpu_module = module_id==1 ? m.module : module_id==2 ? a.mod :
                           module_id==3 ? mod : nullptr;
    if(!gpu_module)throw std::runtime_error("UEFI kernel module not loaded");
    uint32_t grid[3],block[3];
    for(unsigned i=0;i<3;i++){grid[i]=l.get(at+4+4*i);block[i]=l.get(at+16+4*i);}
    const unsigned shared=l.get(at+28), argc=l.get(at+32);
    if(!grid[0] || !grid[1] || !grid[2] || grid[0]>65535 ||
       grid[1]>65535 || grid[2]>65535 || !block[0] || !block[1] || !block[2] ||
       uint64_t(block[0])*block[1]*block[2]>1024 || shared>49152 ||
       argc==0 || argc>8 || l.get(at+36) || l.get(at+88) || l.get(at+92))
       throw std::runtime_error("invalid guest-authored kernel launch shape");
    char symbol[49]{};
    std::memcpy(symbol,l.at+at+40,48);
    const size_t name_len=strnlen(symbol,48);
    if(!name_len || name_len==48)throw std::runtime_error("unterminated kernel symbol");
    for(size_t i=0;i<name_len;++i) {
        const char c=symbol[i];
        if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||
             (c>='0'&&c<='9')||c=='_'))
            throw std::runtime_error("invalid kernel symbol character");
    }
    for(size_t i=name_len+1;i<48;++i)
        if(symbol[i])throw std::runtime_error("kernel symbol padding invalid");
    CUfunction func{};
    CU_OK(cuModuleGetFunction(&func,gpu_module,symbol));
    CUdeviceptr ptrs[8]{};uint32_t immediates[8]{};void* params[8]{};
    for(unsigned i=0;i<8;i++) {
        const size_t p=at+96+size_t(i)*16;
        const uint32_t kind=l.get(p),value=l.get(p+4),span=l.get(p+8),offset=l.get(p+12);
        if(i>=argc) {
            if(kind||value||span||offset)throw std::runtime_error("nonzero unused kernel arg");
            continue;
        }
        if(kind==nr_launch::buf) {
            const DeviceBuf b=buffer(value);
            if(!b.ptr || !span || offset>b.bytes || span>b.bytes-offset)
                throw std::runtime_error("GPU launch buffer outside allocated bounds");
            ptrs[i]=b.ptr+offset;params[i]=&ptrs[i];
        } else if(kind==nr_launch::u32 || kind==nr_launch::f32) {
            if(span || offset)throw std::runtime_error("immediate kernel argument malformed");
            immediates[i]=value;params[i]=&immediates[i];
        } else throw std::runtime_error("unknown UEFI GPU argument type");
    }
    CU_OK(cuLaunchKernel(func,grid[0],grid[1],grid[2],
                        block[0],block[1],block[2],shared,stream,params,nullptr));
}

    uint32_t execute_final(LayerLink& l) {
        if(!initialized)throw std::runtime_error("output request before GPU stack init");
        const unsigned vocab=l.get(layer::final_vocab);
        if(vocab==0 || vocab>layer::max_vocab ||
           layer::final_logits+size_t(vocab)*4>layer::trace)
            throw std::runtime_error("bad final vocabulary/shared-buffer size");
        m.tensor(2,0,1,dim);
        m.tensor(3,1,vocab,dim);
        if(final_logits_capacity!=vocab) {
            if(final_logits_dev)CU_OK(cuMemFree(final_logits_dev));
            final_logits_dev=0;
            CU_OK(cuMemAlloc(&final_logits_dev,size_t(vocab)*4));
            final_logits_capacity=vocab;
        }
        Cmd commands[3];
        std::memcpy(commands,l.at+layer::final_commands,sizeof(commands));
        const unsigned op[3]={2,3,1},dst[3]={2017,2018,2019},
            src0[3]={2000,2017,3},src1[3]={2,0,2018},
            rows[3]={dim,dim,vocab},cols[3]={0,0,dim};
        for(unsigned i=0;i<3;++i) {
            const auto& c=commands[i];
            if(c.ver!=1 || c.flags!=0 || c.op!=op[i] || c.dst!=dst[i] ||
               c.src0!=src0[i] || c.src1!=src1[i] || c.aux ||
               c.rows!=rows[i] || c.cols!=cols[i] ||
               c.res[0] || c.res[1] || c.s1 ||
               c.s0!=(i==0?l.get(layer::eps):0))
                throw std::runtime_error("invalid guest-authored final head commands");
        }
        // Copy exact UEFI accepted hidden vector: prior CPU parity diagnostics
        // may have used the same temporary GPU scratch during a replay.
        CU_OK(cuMemcpyHtoDAsync(x,l.at+layer::input,size_t(dim)*4,stream));
        CU_OK(cuEventRecord(start,stream));
        const auto runtime_abi=l.get(nr_launch::abi);
        if(runtime_abi==nr_launch::abi_version) {
            for(unsigned i=0;i<nr_launch::head_count;++i)dispatch_generic(l,i);
            ++generic_final_calls;
            if(generic_final_calls<=3 || generic_final_calls%64==0)
                std::cout<<"[gpu-runtime-host] generic output #"<<generic_final_calls
                         <<" launches=3 (UEFI symbols/grids/args)"<<std::endl;
        } else if(runtime_abi==0) {
            for(const auto& c:commands)dispatch(c,0); // backward compatible
        } else throw std::runtime_error("unknown UEFI runtime launch ABI");
        CU_OK(cuEventRecord(stop,stream));
        CU_OK(cuStreamSynchronize(stream));
        CU_OK(cuMemcpyDtoH(l.at+layer::final_logits,final_logits_dev,size_t(vocab)*4));
        last_final_seq=l.get(layer::seq);
        float ms=0.f;CU_OK(cuEventElapsedTime(&ms,start,stop));
        return uint32_t(ms*1000.f+0.5f);
    }
    // Dispatches only UEFI-requested sampling arithmetic. Never draws a token.
    uint32_t execute_sampler(LayerLink& l) {
        if(!initialized || !final_logits_dev || !last_final_seq ||
           last_final_seq!=l.get(layer::sample_expected))
            throw std::runtime_error("GPU sampling request has no matching output head");
        unsigned vocab=l.get(layer::final_vocab);
        float temperature=float_bits(l.get(layer::sample_temp));
        float top_p=float_bits(l.get(layer::sample_top_p));
        float scale=float_bits(l.get(layer::sample_scale));
        if(vocab<64 || vocab>layer::max_vocab || final_logits_capacity!=vocab ||
           !(temperature>0.f) || !std::isfinite(temperature) ||
           !(top_p>0.f && top_p<=1.f) || !std::isfinite(scale) || scale<=0.f ||
           layer::sample_out+520>layer::trace)
            throw std::runtime_error("invalid UEFI sampling parameters");
        last_final_seq=0; // one consumption per output head, including errors.
        unsigned partitions=(vocab+511u)/512u;
        CUdeviceptr logits=final_logits_dev,parts=sample_parts,ids=sample_ids;
        CUdeviceptr result=sample_output_dev;
        void* partial_args[]={&logits,&vocab,&parts,&ids};
        void* final_args[]={&parts,&ids,&partitions,&scale,&temperature,&top_p,&result};
        CU_OK(cuEventRecord(start,stream));
        CU_OK(cuLaunchKernel(m.sample_partial,partitions,1,1,1,1,1,0,stream,partial_args,nullptr));
        CU_OK(cuLaunchKernel(m.sample_finish,1,1,1,1,1,1,0,stream,final_args,nullptr));
        CU_OK(cuEventRecord(stop,stream));
        CU_OK(cuStreamSynchronize(stream));
        CU_OK(cuMemcpyDtoH(l.at+layer::sample_out,sample_output_dev,520u));
        float ms=0.f;CU_OK(cuEventElapsedTime(&ms,start,stop));
        return uint32_t(ms*1000.f+0.5f);
    }
    uint32_t execute_prefill_batch(LayerLink& l,bool gpu_embeds=false) {
        if(!initialized || n_layers!=16 || plan.size()!=336)
            throw std::runtime_error("Stage8B needs initialized 16-layer UEFI command plan");
        const unsigned first=l.get(layer::pos), count=l.get(layer::batch_count);
        if(count<1 || count>layer::max_batch || first>=ctx || count>ctx-first)
            throw std::runtime_error("invalid UEFI GPU prefill batch range");
        for(unsigned li=0;li<n_layers;++li)
            if(next_pos_by_layer[li]!=first)
                throw std::runtime_error("GPU prefill batch KV position out of sequence");
        // Transfer ALL UEFI-computed embeddings in one PCI -> GPU operation.
        if(!gpu_embeds) {
            CU_OK(cuMemcpyHtoD(prefill_embeddings,l.at+layer::batch_input,
                               size_t(count)*dim*4));
        }
        CU_OK(cuEventRecord(start,stream));
        for(unsigned token=0;token<count;++token) {
            const unsigned pos=first+token;
            // Each 336-command replay and its position is explicitly requested
            // by UEFI's batch descriptor. Driver just submits opaque commands.
            if(gpu_embeds) {
                Cmd c{};
                std::memcpy(&c,l.at+layer::embed_cmds+size_t(token)*sizeof(Cmd),sizeof(c));
                launch_embedding_cmd(c);
            } else {
                CU_OK(cuMemcpyDtoDAsync(x,prefill_embeddings+size_t(token)*dim*4,
                                         size_t(dim)*4,stream));
            }
            for(const Cmd& cmd:plan) dispatch(cmd,pos);
        }
        CU_OK(cuEventRecord(stop,stream));
        CU_OK(cuStreamSynchronize(stream));
        // Mirror ALL accepted K/V rows to UEFI in 32 contiguous transfers,
        // rather than a separate transfer per layer per token.
        CU_OK(cuMemcpyDtoH(l.at+layer::input,x,size_t(dim)*4));
        for(unsigned li=0;li<n_layers;++li) {
            const size_t src=(size_t(li)*ctx+first)*kv_dim*2;
            const size_t dst=size_t(li)*count*kv_dim*2;
            const size_t bytes=size_t(count)*kv_dim*2;
            CU_OK(cuMemcpyDtoH(l.at+layer::batch_keys+dst,keys+src,bytes));
            CU_OK(cuMemcpyDtoH(l.at+layer::batch_vals+dst,vals+src,bytes));
        }
        float ms=0; CU_OK(cuEventElapsedTime(&ms,start,stop));
        for(unsigned li=0;li<n_layers;++li)next_pos_by_layer[li]=first+count;
        next_pos=first+count;calls+=count;
        const unsigned gpu_us=uint32_t(ms*1000+0.5f);
        std::cout<<"[gpu-stage8b-host] replay=UEFI_BATCH count="<<count
                 <<" positions="<<first<<".."<<(first+count-1)
                 <<" uefi_commands_per_token="<<plan.size()
                 <<" GPU_us="<<gpu_us<<std::endl;
        return gpu_us;
    }
    uint32_t execute(LayerLink& l,bool gpu_embed=false){
        if(!initialized)throw std::runtime_error("full layer not initialized");
        unsigned pos=l.get(layer::pos);
        if(pos>=ctx)throw std::runtime_error("GPU stack position exceeds context");
        for(unsigned li=0;li<n_layers;li++)
            if(next_pos_by_layer[li]!=pos)
                throw std::runtime_error("layer-specific GPU KV sequence mismatch");
        if(!gpu_embed) CU_OK(cuMemcpyHtoD(x,l.at+layer::input,size_t(dim)*4));
        CU_OK(cuEventRecord(start,stream));
        if(gpu_embed) {
            Cmd c{};
            std::memcpy(&c,l.at+layer::embed_cmds,sizeof(c));
            launch_embedding_cmd(c);
        }
        // The order and operands come solely from the NightRun UEFI command list.
        for(size_t i=0;i<plan.size();++i) {
            dispatch(plan[i],pos);
            // Diagnostic only. On the first 3 token requests, capture the
            // hidden state at each guest-authored 21-command layer boundary.
            // This does NOT perform inference scheduling on the host.
            if(calls<3 && (i+1)%21==0) {
                CU_OK(cuStreamSynchronize(stream));
                const size_t layer_index=i/21;
                CU_OK(cuMemcpyDtoH(l.at+layer::trace+layer_index*size_t(dim)*4,
                                   x,size_t(dim)*4));
            }
        }
        CU_OK(cuEventRecord(stop,stream));
        CU_OK(cuStreamSynchronize(stream));
        // Return complete hidden state and only THIS token's K/V (CPU mirror).
        CU_OK(cuMemcpyDtoH(l.at+layer::input,x,size_t(dim)*4));
        // Each GPU layer's new K/V is mirrored to UEFI for CPU fallback.
        const size_t kout=n_layers==1?layer::out_k:layer::out_k;
        const size_t vout=n_layers==1?layer::out_v:layer::out_v_stack;
        for(unsigned li=0;li<n_layers;li++) {
            const size_t off=(size_t(li)*ctx+pos)*kv_dim*2;
            CU_OK(cuMemcpyDtoH(l.at+kout+size_t(li)*kv_dim*2,
                               keys+off,size_t(kv_dim)*2));
            CU_OK(cuMemcpyDtoH(l.at+vout+size_t(li)*kv_dim*2,
                               vals+off,size_t(kv_dim)*2));
        }
        float ms=0;CU_OK(cuEventElapsedTime(&ms,start,stop));
        for(unsigned li=0;li<n_layers;li++)next_pos_by_layer[li]=pos+1;
        next_pos=pos+1;++calls;
        if(calls<=3 || calls%64==0)
            std::cout<<"[gpu-stage7-host] "<<n_layers<<" GPU layers #"<<calls
                     <<" pos="<<pos<<" GPU_us="<<uint32_t(ms*1000+0.5f)<<std::endl;
        return uint32_t(ms*1000+0.5f);
    }
};

int main(int argc,char**argv) {
    // NIGHTRUN_STAGE9A_UEFI_OWNED_CUBIN_V1: New runtime takes only ivshmem file; 5-arg form preserved for rollback.
    if(argc!=2 && argc!=5){std::cerr<<"Usage: "<<argv[0]
        <<" <gpu_bridge_8g.bin> [legacy three CUBIN filenames]\n";return 2;}
    try {
        Shared shm(argv[1]);
        Executor registry;               // one CUDA context, owns all model tensors
        MlpExec mlp_engine(registry,argc==5?argv[2]:nullptr);
        MlpLink mlp_link(shm);
        AttentionExec attn_engine(registry,argc==5?argv[3]:nullptr);
        AttentionLink attn_link(shm);
        FullLayerExec layer_engine(registry,mlp_engine,attn_engine,argc==5?argv[4]:nullptr);
        LayerLink layer_link(shm);
        shm.put(0,p::magic);shm.put(p::error,0);shm.put(p::done_seq,0);
        shm.put(p::status,p::ready);
        mlp_link.put(0,mlp::magic);mlp_link.put(mlp::error,0);
        mlp_link.put(mlp::done_seq,0);mlp_link.put(mlp::status,mlp::ready);
        attn_link.put(0,attn::magic);attn_link.put(attn::error,0);
        attn_link.put(attn::done_seq,0);attn_link.put(attn::status,attn::ready);
        layer_link.put(0,layer::magic);layer_link.put(layer::error,0);
        layer_link.put(nr_launch::cap,nr_launch::cap_magic); // generic runtime supported
        layer_link.put(layer::ack,0);layer_link.put(layer::status,layer::ready);
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
            if(layer_link.get(layer::status)==layer::request){
                const uint32_t seq=layer_link.get(layer::seq);
                const uint32_t op=layer_link.get(layer::op);
                const auto t0=std::chrono::steady_clock::now();
                try{
                    uint32_t gpu_us=0;
                    if(op==layer::load_cubin){
                        const uint32_t id=layer_link.get(layer::module_id);
                        const uint32_t bytes=layer_link.get(layer::module_bytes);
                        const uint32_t want=layer_link.get(layer::module_crc);
                        if(id<1 || id>3 || bytes<64 || bytes>layer::module_max ||
                           layer::module_data+size_t(bytes)>layer::capacity)
                            throw std::runtime_error("invalid UEFI CUBIN header");
                        const uint8_t* image=layer_link.at+layer::module_data;
                        if(std::memcmp(image,"\x7f" "ELF",4)!=0 || crc32(image,bytes)!=want)
                            throw std::runtime_error("UEFI CUBIN ELF signature or CRC invalid");
                        if(id==1)mlp_engine.load_bytes(image);
                        else if(id==2)attn_engine.load_bytes(image);
                        else layer_engine.load_bytes(image);
                        std::cout<<"[gpu-kernels-host] UEFI module="<<id<<" bytes="<<bytes
                                 <<" CRC verified; cuModuleLoadData OK"<<std::endl;
                    }
                    else if(op==layer::init){
                        if(!mlp_engine.module || !attn_engine.mod || !layer_engine.mod)
                            throw std::runtime_error("UEFI kernel modules not loaded");
                        layer_engine.initialize(layer_link);
                    }
                    else if(op==layer::sync)layer_engine.sync_history(layer_link);
                    else if(op==layer::run)gpu_us=layer_engine.execute(layer_link);
                    else if(op==layer::prefill_batch)gpu_us=layer_engine.execute_prefill_batch(layer_link);
                    else if(op==layer::final_head)gpu_us=layer_engine.execute_final(layer_link);
                    else if(op==layer::run_token_embed)gpu_us=layer_engine.execute(layer_link,true);
                    else if(op==layer::run_batch_embed)gpu_us=layer_engine.execute_prefill_batch(layer_link,true);
                    else if(op==layer::sample)gpu_us=layer_engine.execute_sampler(layer_link);
                    else if(op==layer::diag)layer_engine.diagnose(layer_link);
                    else if(op==layer::snapshot_kv)layer_engine.snapshot_kv_history(layer_link);
                    else throw std::runtime_error("invalid full-layer operation");
                    const auto t1=std::chrono::steady_clock::now();
                    const uint32_t service_us=uint32_t(std::chrono::duration_cast<
                        std::chrono::microseconds>(t1-t0).count());
                    layer_link.put(layer::gpu_us,gpu_us);
                    layer_link.put(layer::service_us,service_us);
                    layer_link.put(layer::ack,seq);
                    layer_link.put(layer::status,layer::done);
                }catch(const std::exception& e){
                    std::cerr<<"[gpu-stage7-host] layer-stack error seq="<<seq<<": "
                             <<e.what()<<std::endl;
                    layer_link.put(layer::error,1);
                    layer_link.put(layer::ack,seq);
                    layer_link.put(layer::status,layer::failed);
                }
                continue;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    } catch(const std::exception& e){
        std::cerr<<"[gpu-stage5b-host] FATAL: "<<e.what()<<std::endl;return 1;
    }
}
