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

#define CU_OK(call) do { CUresult r=(call); if(r!=CUDA_SUCCESS) { \
    const char *name=nullptr, *msg=nullptr; \
    cuGetErrorName(r,&name); cuGetErrorString(r,&msg); \
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
int main(int argc,char**argv){
    if(argc!=2){std::cerr<<"Usage: "<<argv[0]<<" <gpu_bridge_8g.bin>\n";return 2;}
    try {
        Shared s(argv[1]);Executor gpu;
        s.put(0,p::magic);s.put(p::error,0);s.put(p::done_seq,0);
        s.put(p::status,p::ready);
        std::cout<<"[gpu-resident-host] waiting at ivshmem +0x"<<std::hex
                 <<p::region<<std::dec<<", 4 MiB per upload request"<<std::endl;
        for(;;){
            if(s.get(p::status)!=p::request){
                std::this_thread::sleep_for(std::chrono::microseconds(50));continue;
            }
            const uint32_t seq=s.get(p::seq);
            try {
                gpu.command(s);
                s.put(p::done_seq,seq);
                s.put(p::status,p::done);
            } catch(const std::exception& e) {
                std::cerr<<"[gpu-resident-host] ERROR seq="<<seq<<": "<<e.what()<<std::endl;
                s.put(p::error,1);s.put(p::done_seq,seq);s.put(p::status,p::failed);
                break;
            }
        }
    } catch(const std::exception& e){
        std::cerr<<"[gpu-resident-host] fatal: "<<e.what()<<std::endl;return 1;
    }
    return 0;
}
