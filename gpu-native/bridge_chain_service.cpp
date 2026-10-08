// NightRun Stage 3 QEMU proof of concept: consume NightRun-owned GpuCommand list
// from ivshmem and execute the Stage 2 offline sm_89 CUBIN using CUDA Driver API.
// This is NOT llama.cpp, a transformer service, or a native UEFI GPU driver.
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
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#define CU_OK(expr) do { \
 CUresult _r=(expr); if (_r!=CUDA_SUCCESS) { const char *_n=nullptr,*_s=nullptr; \
 cuGetErrorName(_r,&_n); cuGetErrorString(_r,&_s); \
 throw std::runtime_error(std::string(#expr)+": "+(_n?_n:"?")+" "+(_s?_s:"")); \
 }} while(0)

namespace proto {
constexpr size_t region=0x60000000ull;
constexpr size_t region_min=0x510000;
constexpr uint32_t magic=0x3347504eu;
constexpr size_t status=4, request_seq=8, completed_seq=12, error=16;
constexpr size_t cmd_count=20, dim=24, service_us=28, gpu_us=32, commands=0x100;
constexpr size_t weights=0x10000, input=0x500000, gamma=0x502000;
constexpr size_t output=0x504000, normalized=0x506000, quantized=0x508000;
constexpr uint32_t READY=0, REQUEST=1, DONE=2, ERROR=3;
constexpr unsigned D=2048;
constexpr size_t weight_bytes=size_t(D)*(D/32)*34;
constexpr size_t qbytes=(D/32)*34;
}

struct Command {
    uint32_t opcode; uint16_t version, flags;
    uint32_t dst, src0, src1, aux, rows, cols, scalar0_bits, scalar1_bits, reserved[2];
};
static_assert(sizeof(Command)==48, "GPU command wire layout mismatch");

struct SharedFile {
    int fd{-1}; uint8_t* map{nullptr}; size_t bytes{0};
    explicit SharedFile(const std::string& path) {
        fd=open(path.c_str(), O_RDWR);
        if (fd<0) throw std::runtime_error("cannot open shared memory: " + path);
        struct stat st{};
        if (fstat(fd,&st)!=0 || st.st_size < 0 || uint64_t(st.st_size)<proto::region+proto::region_min)
            throw std::runtime_error("shared memory file too small; expected existing 8 GiB bridge");
        bytes=static_cast<size_t>(st.st_size);
        void* p=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
        if (p==MAP_FAILED) throw std::runtime_error("mmap failed");
        map=static_cast<uint8_t*>(p)+proto::region;
    }
    ~SharedFile() { if(map) munmap(map-proto::region,bytes); if(fd>=0) close(fd); }
    uint32_t get(size_t off) const {
        return __atomic_load_n(reinterpret_cast<uint32_t*>(map+off),__ATOMIC_ACQUIRE);
    }
    void put(size_t off,uint32_t value) {
        __atomic_store_n(reinterpret_cast<uint32_t*>(map+off),value,__ATOMIC_RELEASE);
    }
};

struct Gpu {
    CUcontext ctx{}; CUmodule mod{}; CUfunction norm{},quant{},matvec{},add{};
    CUdeviceptr dw{},dx{},dg{},dn{},dq{},dy{};
    CUstream stream{}; CUgraph graph{}; CUgraphExec exec{}; CUevent start{},stop{};
    unsigned dim=proto::D; float eps=1e-5f;
    explicit Gpu(const std::string& cubin) {
        CU_OK(cuInit(0)); CUdevice dev{}; CU_OK(cuDeviceGet(&dev,0));
        char name[128]{}; CU_OK(cuDeviceGetName(name,sizeof(name),dev));
        std::cout << "[gpu-poc-host] GPU: " << name << '\n';
        CU_OK(cuCtxCreate(&ctx,0,dev)); CU_OK(cuModuleLoad(&mod,cubin.c_str()));
        CU_OK(cuModuleGetFunction(&norm,mod,"nr_rmsnorm"));
        CU_OK(cuModuleGetFunction(&quant,mod,"nr_quantize_q8"));
        CU_OK(cuModuleGetFunction(&matvec,mod,"nr_q8_matvec"));
        CU_OK(cuModuleGetFunction(&add,mod,"nr_residual_add"));
        CU_OK(cuMemAlloc(&dw,proto::weight_bytes));
        CU_OK(cuMemAlloc(&dx,dim*sizeof(float)));
        CU_OK(cuMemAlloc(&dg,dim*sizeof(float)));
        CU_OK(cuMemAlloc(&dn,dim*sizeof(float)));
        CU_OK(cuMemAlloc(&dq,proto::qbytes));
        CU_OK(cuMemAlloc(&dy,dim*sizeof(float)));
        CU_OK(cuStreamCreate(&stream,CU_STREAM_DEFAULT));
        void* an[]={&dx,&dg,&dn,&dim,&eps};
        void* aq[]={&dn,&dq,&dim};
        void* am[]={&dw,&dq,&dy,&dim,&dim};
        void* aa[]={&dy,&dx,&dim};
        CU_OK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
        CU_OK(cuLaunchKernel(norm,1,1,1,256,1,1,0,stream,an,nullptr));
        CU_OK(cuLaunchKernel(quant,dim/32,1,1,32,1,1,0,stream,aq,nullptr));
        CU_OK(cuLaunchKernel(matvec,(dim+7)/8,1,1,256,1,1,0,stream,am,nullptr));
        CU_OK(cuLaunchKernel(add,(dim+255)/256,1,1,256,1,1,0,stream,aa,nullptr));
        CU_OK(cuStreamEndCapture(stream,&graph));
        CU_OK(cuGraphInstantiate(&exec,graph,0));
        CU_OK(cuEventCreate(&start,CU_EVENT_DEFAULT));
        CU_OK(cuEventCreate(&stop,CU_EVENT_DEFAULT));
        std::cout << "[gpu-poc-host] offline sm_89 CUBIN loaded; CUDA graph ready\n";
    }
    uint32_t execute(SharedFile& mem) {
        CU_OK(cuMemcpyHtoD(dw,mem.map+proto::weights,proto::weight_bytes));
        CU_OK(cuMemcpyHtoD(dx,mem.map+proto::input,dim*sizeof(float)));
        CU_OK(cuMemcpyHtoD(dg,mem.map+proto::gamma,dim*sizeof(float)));
        CU_OK(cuEventRecord(start,stream));
        CU_OK(cuGraphLaunch(exec,stream));
        CU_OK(cuEventRecord(stop,stream));
        CU_OK(cuEventSynchronize(stop));
        float gpu_ms=0.f; CU_OK(cuEventElapsedTime(&gpu_ms,start,stop));
        CU_OK(cuMemcpyDtoH(mem.map+proto::output,dy,dim*sizeof(float)));
        CU_OK(cuMemcpyDtoH(mem.map+proto::normalized,dn,dim*sizeof(float)));
        CU_OK(cuMemcpyDtoH(mem.map+proto::quantized,dq,proto::qbytes));
        CU_OK(cuStreamSynchronize(stream));
        return static_cast<uint32_t>(gpu_ms*1000.f+0.5f);
    }
};

static bool valid_commands(SharedFile& m) {
    if (m.get(proto::cmd_count)!=4 || m.get(proto::dim)!=proto::D) return false;
    Command c[4]; std::memcpy(c,m.map+proto::commands,sizeof(c));
    const uint32_t opcode[]={2,3,1,4};
    const uint32_t dst[]={3,4,6,6},src0[]={1,3,5,1},src1[]={2,0,4,0};
    for(int i=0;i<4;++i) {
        if(c[i].opcode!=opcode[i] || c[i].version!=1 || c[i].flags!=0 ||
           c[i].dst!=dst[i] || c[i].src0!=src0[i] || c[i].src1!=src1[i] ||
           c[i].aux!=0 || c[i].rows!=proto::D ||
           c[i].scalar1_bits!=0 || c[i].reserved[0]!=0 || c[i].reserved[1]!=0) return false;
    }
    if (c[0].cols!=0 || c[1].cols!=0 || c[2].cols!=proto::D || c[3].cols!=0) return false;
    float eps=0.f; std::memcpy(&eps,&c[0].scalar0_bits,sizeof(eps));
    if(std::fabs(eps-1e-5f)>1e-10f || c[1].scalar0_bits!=0 || c[2].scalar0_bits!=0 || c[3].scalar0_bits!=0)
        return false;
    return true;
}

int main(int argc,char** argv) {
    if(argc!=3) {
        std::cerr << "Usage: " << argv[0] << " <existing gpu_bridge_8g.bin> <nr_transformer_chain_sm89.cubin>\n";
        return 2;
    }
    try {
        SharedFile mem(argv[1]); Gpu gpu(argv[2]);
        // Reset ONLY the reserved Stage 3 header, not the classifier protocol.
        mem.put(0,proto::magic);
        mem.put(proto::completed_seq,0);
        mem.put(proto::error,0);
        mem.put(proto::status,proto::READY);
        std::cout << "[gpu-poc-host] waiting for NightRun QEMU command list at +0x"
                  << std::hex << proto::region << std::dec << std::endl;
        unsigned requests=0;
        for(;;) {
            if(mem.get(proto::status)!=proto::REQUEST) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
            }
            const auto t0=std::chrono::steady_clock::now();
            const uint32_t seq=mem.get(proto::request_seq);
            if(!valid_commands(mem)) {
                mem.put(proto::error,1); mem.put(proto::status,proto::ERROR);
                std::cerr << "[gpu-poc-host] invalid NightRun command list\n";
                continue;
            }
            try {
                uint32_t gpu_us=gpu.execute(mem);
                const auto t1=std::chrono::steady_clock::now();
                uint32_t total_us=static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count());
                mem.put(proto::gpu_us,gpu_us); mem.put(proto::service_us,total_us);
                mem.put(proto::completed_seq,seq);
                mem.put(proto::status,proto::DONE);
                std::cout << "[gpu-poc-host] completed #" << ++requests << " seq=" << seq
                          << " gpu_graph_us=" << gpu_us << " service_us=" << total_us << std::endl;
            } catch(const std::exception& ex) {
                std::cerr << "[gpu-poc-host] CUDA execution failed: " << ex.what() << '\n';
                mem.put(proto::error,2); mem.put(proto::status,proto::ERROR);
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "[gpu-poc-host] FATAL: " << ex.what() << '\n'; return 1;
    }
}
