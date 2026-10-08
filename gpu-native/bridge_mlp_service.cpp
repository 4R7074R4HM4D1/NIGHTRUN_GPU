// NightRun Stage 4: execute the actual first-layer SwiGLU MLP requested by UEFI.
// NOT a tokenizer, model runner, or llama.cpp backend. All model scheduling is
// done by NightRun. CUDA Driver API is only the development launch backend.
#include <cuda.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <memory>

#define CU_CHECK(expr) do { CUresult rc=(expr); if(rc!=CUDA_SUCCESS) { \
 const char *nm=nullptr,*msg=nullptr; cuGetErrorName(rc,&nm); cuGetErrorString(rc,&msg); \
 throw std::runtime_error(std::string(#expr)+": "+(nm?nm:"unknown")+" "+(msg?msg:"")); \
 }} while(0)

namespace p {
constexpr size_t region=0x80000000ULL; // 2 GiB, independent of Stage 3 region
constexpr size_t region_capacity=0x08000000ULL; // 128 MiB
constexpr uint32_t magic=0x3447504e; // NPG4
constexpr size_t status=4, seq=8, done_seq=12, error=16, dim=20, ffn=24;
constexpr size_t service_us=28, gpu_us=32, eps_bits=36, scale_bits=40, count=44;
constexpr size_t commands=0x100, weights=0x10000;
constexpr uint32_t READY=0, REQUEST=1, DONE=2, ERROR=3, INIT=4, INIT_DONE=5;
constexpr size_t align64(size_t n) { return (n+63)&~size_t(63); }
struct Layout { size_t wg,wu,wd,gamma,input,output,limit; };
Layout layout(unsigned d,unsigned f) {
    if(d!=2048 || f==0 || f>16384 || f%32) throw std::runtime_error("unsupported layer shape");
    const size_t gf=size_t(f)*(d/32)*34;
    const size_t df=size_t(d)*(f/32)*34;
    const size_t wg=weights, wu=align64(wg+gf), wd=align64(wu+gf);
    const size_t ga=align64(wd+df), in=align64(ga+size_t(d)*4);
    const size_t out=align64(in+size_t(d)*4), end=align64(out+size_t(d)*4);
    if(end>region_capacity) throw std::runtime_error("layer exceeds bridge capacity");
    return {wg,wu,wd,ga,in,out,end};
}
}

struct Cmd {
    uint32_t op; uint16_t ver,flags;
    uint32_t dst,src0,src1,aux,rows,cols,s0,s1,res[2];
};
static_assert(sizeof(Cmd)==48,"command ABI must be exactly 48 bytes");

struct Shared {
    int fd{-1}; uint8_t* mapping{nullptr}; uint8_t* at{nullptr}; size_t bytes{};
    explicit Shared(const char* filename) {
        fd=open(filename,O_RDWR); if(fd<0) throw std::runtime_error("could not open bridge file");
        struct stat st{};
        if(fstat(fd,&st)!=0 || st.st_size<0 || static_cast<uint64_t>(st.st_size)<p::region+p::region_capacity)
            throw std::runtime_error("bridge file must cover Stage 4 region (use existing 8 GiB file)");
        bytes=static_cast<size_t>(st.st_size);
        void* addr=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
        if(addr==MAP_FAILED) throw std::runtime_error("mmap failed");
        mapping=static_cast<uint8_t*>(addr); at=mapping+p::region;
    }
    ~Shared() { if(mapping) munmap(mapping,bytes); if(fd>=0) close(fd); }
    uint32_t get(size_t offset) const {
        return __atomic_load_n(reinterpret_cast<uint32_t*>(at+offset), __ATOMIC_ACQUIRE);
    }
    void put(size_t offset,uint32_t val) {
        __atomic_store_n(reinterpret_cast<uint32_t*>(at+offset),val,__ATOMIC_RELEASE);
    }
};

static float as_float(uint32_t bits) { float x; std::memcpy(&x,&bits,sizeof(x)); return x; }

static bool valid_commands(const Shared& m,unsigned dim,unsigned ffn,float eps,float scale) {
    if(m.get(p::count)!=8) return false;
    const uint32_t ops[]={2,3,1,1,5,3,1,6};
    const uint32_t dst[]={3,4,6,8,6,9,11,1};
    const uint32_t s0[]={1,3,5,7,6,6,10,1};
    const uint32_t s1[]={2,0,4,4,8,0,9,11};
    const uint32_t rows[]={dim,dim,ffn,ffn,ffn,ffn,dim,dim};
    const uint32_t cols[]={0,0,dim,dim,0,0,ffn,0};
    Cmd commands[8]; std::memcpy(commands,m.at+p::commands,sizeof(commands));
    for(unsigned i=0;i<8;++i) {
        const auto& c=commands[i];
        if(c.op!=ops[i] || c.ver!=1 || c.flags!=0 || c.dst!=dst[i] ||
            c.src0!=s0[i] || c.src1!=s1[i] || c.aux!=0 ||
            c.rows!=rows[i] || c.cols!=cols[i] || c.s1!=0 || c.res[0]!=0 || c.res[1]!=0) return false;
        if(i!=0 && i!=7 && c.s0!=0) return false;
        if(i==0 && c.s0!=m.get(p::eps_bits)) return false;
        if(i==7 && c.s0!=m.get(p::scale_bits)) return false;
    }
    return eps>0 && std::isfinite(eps) && std::isfinite(scale);
}

struct Device {
    CUcontext ctx{}; CUmodule module{};
    CUfunction norm{},quant{},gemv{},swiglu{},add{};
    CUstream stream{}; CUgraph graph{}; CUgraphExec executable{};
    CUevent start{},stop{};
    CUdeviceptr x{},gamma{},normalized{},qnorm{},wg{},wu{},wd{},gate{},up{},qgate{},down{};
    unsigned dim{},ffn{}; float eps{},scale{};
    bool initialized=false;
    explicit Device(const char* cubin) {
        CU_CHECK(cuInit(0)); CUdevice dev{}; CU_CHECK(cuDeviceGet(&dev,0));
        char name[128]{}; CU_CHECK(cuDeviceGetName(name,128,dev));
        std::cout<<"[gpu-mlp-host] GPU: "<<name<<std::endl;
        CU_CHECK(cuCtxCreate(&ctx,0,dev)); CU_CHECK(cuModuleLoad(&module,cubin));
        CU_CHECK(cuModuleGetFunction(&norm,module,"nr_rmsnorm"));
        CU_CHECK(cuModuleGetFunction(&quant,module,"nr_quantize_q8"));
        CU_CHECK(cuModuleGetFunction(&gemv,module,"nr_q8_matvec"));
        CU_CHECK(cuModuleGetFunction(&swiglu,module,"nr_swiglu"));
        CU_CHECK(cuModuleGetFunction(&add,module,"nr_scaled_residual"));
        CU_CHECK(cuStreamCreate(&stream,CU_STREAM_DEFAULT));
        CU_CHECK(cuEventCreate(&start,CU_EVENT_DEFAULT));
        CU_CHECK(cuEventCreate(&stop,CU_EVENT_DEFAULT));
        std::cout<<"[gpu-mlp-host] offline NightRun sm_89 CUBIN loaded; awaiting real layer-0 weights"<<std::endl;
    }
    void initialize(Shared& m) {
        if(initialized) throw std::runtime_error("already initialized; restart service for new model");
        dim=m.get(p::dim); ffn=m.get(p::ffn);
        eps=as_float(m.get(p::eps_bits)); scale=as_float(m.get(p::scale_bits));
        const auto offsets=p::layout(dim,ffn);
        if(!valid_commands(m,dim,ffn,eps,scale)) throw std::runtime_error("invalid NightRun GPU MLP command list");
        const size_t gf=size_t(ffn)*(dim/32)*34, df=size_t(dim)*(ffn/32)*34;
        CU_CHECK(cuMemAlloc(&x,size_t(dim)*4));
        CU_CHECK(cuMemAlloc(&gamma,size_t(dim)*4));
        CU_CHECK(cuMemAlloc(&normalized,size_t(dim)*4));
        CU_CHECK(cuMemAlloc(&qnorm,size_t(dim/32)*34));
        CU_CHECK(cuMemAlloc(&wg,gf)); CU_CHECK(cuMemAlloc(&wu,gf)); CU_CHECK(cuMemAlloc(&wd,df));
        CU_CHECK(cuMemAlloc(&gate,size_t(ffn)*4)); CU_CHECK(cuMemAlloc(&up,size_t(ffn)*4));
        CU_CHECK(cuMemAlloc(&qgate,size_t(ffn/32)*34)); CU_CHECK(cuMemAlloc(&down,size_t(dim)*4));
        CU_CHECK(cuMemcpyHtoD(wg,m.at+offsets.wg,gf));
        CU_CHECK(cuMemcpyHtoD(wu,m.at+offsets.wu,gf));
        CU_CHECK(cuMemcpyHtoD(wd,m.at+offsets.wd,df));
        CU_CHECK(cuMemcpyHtoD(gamma,m.at+offsets.gamma,size_t(dim)*4));
        unsigned threads=256;
        void* n[]={&x,&gamma,&normalized,&dim,&eps};
        void* q1[]={&normalized,&qnorm,&dim};
        void* g[]={&wg,&qnorm,&gate,&ffn,&dim};
        void* u[]={&wu,&qnorm,&up,&ffn,&dim};
        void* s[]={&gate,&up,&ffn};
        void* q2[]={&gate,&qgate,&ffn};
        void* d[]={&wd,&qgate,&down,&dim,&ffn};
        void* r[]={&x,&down,&dim,&scale};
        CU_CHECK(cuStreamBeginCapture(stream,CU_STREAM_CAPTURE_MODE_GLOBAL));
        CU_CHECK(cuLaunchKernel(norm,1,1,1,threads,1,1,0,stream,n,nullptr));
        CU_CHECK(cuLaunchKernel(quant,dim/32,1,1,32,1,1,0,stream,q1,nullptr));
        CU_CHECK(cuLaunchKernel(gemv,(ffn+7)/8,1,1,threads,1,1,0,stream,g,nullptr));
        CU_CHECK(cuLaunchKernel(gemv,(ffn+7)/8,1,1,threads,1,1,0,stream,u,nullptr));
        CU_CHECK(cuLaunchKernel(swiglu,(ffn+threads-1)/threads,1,1,threads,1,1,0,stream,s,nullptr));
        CU_CHECK(cuLaunchKernel(quant,ffn/32,1,1,32,1,1,0,stream,q2,nullptr));
        CU_CHECK(cuLaunchKernel(gemv,(dim+7)/8,1,1,threads,1,1,0,stream,d,nullptr));
        CU_CHECK(cuLaunchKernel(add,(dim+threads-1)/threads,1,1,threads,1,1,0,stream,r,nullptr));
        CU_CHECK(cuStreamEndCapture(stream,&graph));
        CU_CHECK(cuGraphInstantiate(&executable,graph,0));
        initialized=true;
        std::cout<<"[gpu-mlp-host] real layer-0 Q8_0 weights cached: "
                 <<(2*gf+df)/1048576.0<<" MiB; dim="<<dim<<" ffn="<<ffn<<std::endl;
    }
    uint32_t execute(Shared& m) {
        if(!initialized) throw std::runtime_error("execution before weights uploaded");
        const auto o=p::layout(dim,ffn);
        CU_CHECK(cuMemcpyHtoD(x,m.at+o.input,size_t(dim)*4));
        CU_CHECK(cuEventRecord(start,stream));
        CU_CHECK(cuGraphLaunch(executable,stream));
        CU_CHECK(cuEventRecord(stop,stream));
        CU_CHECK(cuStreamSynchronize(stream));
        CU_CHECK(cuMemcpyDtoH(m.at+o.output,x,size_t(dim)*4));
        float ms=0; CU_CHECK(cuEventElapsedTime(&ms,start,stop));
        return static_cast<uint32_t>(ms*1000+0.5f);
    }
};

int main(int argc,char** argv) {
    if(argc!=3) { std::cerr<<"Usage: "<<argv[0]<<" <gpu_bridge_8g.bin> <nr_layer_mlp_sm89.cubin>\n"; return 2; }
    try {
        Shared mem(argv[1]); Device gpu(argv[2]);
        mem.put(0,p::magic); mem.put(p::done_seq,0); mem.put(p::error,0); mem.put(p::status,p::READY);
        std::cout<<"[gpu-mlp-host] waiting at PCI shared offset +0x"<<std::hex<<p::region<<std::dec<<std::endl;
        unsigned completed=0;
        for(;;) {
            const uint32_t state=mem.get(p::status);
            if(state!=p::INIT && state!=p::REQUEST) {
                std::this_thread::sleep_for(std::chrono::microseconds(50)); continue;
            }
            const auto time0=std::chrono::steady_clock::now();
            const uint32_t seq=mem.get(p::seq);
            try {
                if(state==p::INIT) {
                    gpu.initialize(mem);
                    mem.put(p::done_seq,seq); mem.put(p::status,p::INIT_DONE);
                    continue;
                }
                if(!gpu.initialized || mem.get(p::dim)!=gpu.dim || mem.get(p::ffn)!=gpu.ffn)
                    throw std::runtime_error("uninitialized service or unexpected model shape");
                const uint32_t gpu_us=gpu.execute(mem);
                const auto time1=std::chrono::steady_clock::now();
                uint32_t service_us=static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::microseconds>(time1-time0).count());
                mem.put(p::gpu_us,gpu_us); mem.put(p::service_us,service_us);
                mem.put(p::done_seq,seq); mem.put(p::status,p::DONE);
                ++completed;
                if(completed<=3 || completed%64==0)
                    std::cout<<"[gpu-mlp-host] actual layer-0 MLP #"<<completed
                             <<" seq="<<seq<<" service_us="<<service_us<<" graph_us="<<gpu_us<<std::endl;
            } catch(const std::exception& e) {
                std::cerr<<"[gpu-mlp-host] FAILED: "<<e.what()<<std::endl;
                mem.put(p::error,state==p::INIT?1:2); mem.put(p::status,p::ERROR);
            }
        }
    } catch(const std::exception& e) {
        std::cerr<<"[gpu-mlp-host] FATAL: "<<e.what()<<std::endl; return 1;
    }
}
