// NightRun Stage 2: one CUDA Graph launches a sequence of NightRun-owned kernels
// with GPU-resident intermediates. NO llama.cpp, Triton or PyTorch.
// CPU scalar calculations are used strictly as validation oracle.
#include <cuda.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CU_OK(expr) do { \
    CUresult _r = (expr); \
    if (_r != CUDA_SUCCESS) { \
        const char* _name=nullptr; const char* _desc=nullptr; \
        cuGetErrorName(_r, &_name); cuGetErrorString(_r, &_desc); \
        throw std::runtime_error(std::string(#expr) + ": " + \
            (_name ? _name : "unknown") + " " + (_desc ? _desc : "")); \
    } \
} while (0)

static uint16_t u16(const uint8_t* p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
static float f16_to_f32(uint16_t h) {
    const float sign = (h & 0x8000) ? -1.f : 1.f;
    unsigned exponent = (h>>10) & 31u, mant = h & 1023u;
    if (!exponent) return sign * std::ldexp(float(mant), -24);
    if (exponent==31) return mant ? NAN : sign*INFINITY;
    return sign*std::ldexp(float(mant+1024), int(exponent)-25);
}
static float reference_row(const uint8_t* w, const uint8_t* x, unsigned dim) {
    float sum = 0;
    for (unsigned b=0; b<dim/32; ++b) {
        const uint8_t* wb=w+size_t(b)*34;
        const uint8_t* xb=x+size_t(b)*34;
        int dot=0;
        for (unsigned j=0; j<32; ++j)
            dot += int(int8_t(wb[2+j])) * int(int8_t(xb[2+j]));
        sum += float(dot) * f16_to_f32(u16(wb)) * f16_to_f32(u16(xb));
    }
    return sum;
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <chain.cubin> [--weights classifier_real_q8.bin] [--dim 2048] [--repeats 100]\n";
            return 2;
        }
        std::string cubin=argv[1], weight_path;
        unsigned dim=2048, repeats=100;
        for (int i=2; i<argc; i+=2) {
            if (i+1>=argc) throw std::runtime_error("argument requires value: " + std::string(argv[i]));
            std::string name=argv[i], val=argv[i+1];
            if (name=="--weights") weight_path=val;
            else if (name=="--dim") dim=unsigned(std::stoul(val));
            else if (name=="--repeats") repeats=unsigned(std::stoul(val));
            else throw std::runtime_error("unknown argument: " + name);
        }
        if (dim==0 || dim>8192 || (dim%32)!=0 || repeats==0 || repeats>100000)
            throw std::runtime_error("dim must be a nonzero multiple of 32 <=8192; repeats >0");
        const size_t weight_bytes = size_t(dim)*(dim/32)*34;
        const size_t activation_bytes = size_t(dim/32)*34;
        const float eps=1.0e-5f;
        std::vector<uint8_t> w(weight_bytes);
        if (!weight_path.empty()) {
            std::ifstream f(weight_path, std::ios::binary);
            if (!f) throw std::runtime_error("cannot open weights file: " + weight_path);
            f.read(reinterpret_cast<char*>(w.data()), std::streamsize(w.size()));
            if (f.gcount()!=std::streamsize(w.size()))
                throw std::runtime_error("weight file smaller than prefix required for square matrix");
            std::cout << "Using " << weight_bytes << " bytes (first " << dim << " rows) from " << weight_path << '\n';
        } else {
            for (size_t b=0; b<size_t(dim)*(dim/32); ++b) {
                size_t off=b*34;
                w[off]=0; w[off+1]=0x14; // f16 2^-10
                for (unsigned j=0; j<32; ++j)
                    w[off+2+j]=uint8_t(int8_t(int((b*17+j*23)%255)-127));
            }
            std::cout << "Using deterministic synthetic Q8_0 square matrix\n";
        }
        std::vector<float> x(dim), gamma(dim), norm(dim), result(dim);
        for (unsigned i=0; i<dim; ++i) {
            x[i]=0.25f*std::sin(float(i)*0.013f)+0.33f*std::cos(float(i)*0.035f);
            gamma[i]=1.f+0.01f*std::sin(float(i)*0.031f);
        }
        const unsigned bytes_vec=dim*sizeof(float);
        CU_OK(cuInit(0));
        CUdevice dev; CU_OK(cuDeviceGet(&dev, 0));
        char name[128]={0}; CU_OK(cuDeviceGetName(name, 128, dev));
        std::cout << "GPU: " << name << '\n';
        CUcontext ctx; CU_OK(cuCtxCreate(&ctx, 0, dev));
        CUmodule mod; CU_OK(cuModuleLoad(&mod, cubin.c_str()));
        CUfunction f_norm, f_quant, f_matvec, f_add;
        CU_OK(cuModuleGetFunction(&f_norm, mod, "nr_rmsnorm"));
        CU_OK(cuModuleGetFunction(&f_quant, mod, "nr_quantize_q8"));
        CU_OK(cuModuleGetFunction(&f_matvec, mod, "nr_q8_matvec"));
        CU_OK(cuModuleGetFunction(&f_add, mod, "nr_residual_add"));
        CUdeviceptr dw, dx, dg, dn, dq, dy;
        CU_OK(cuMemAlloc(&dw, weight_bytes));
        CU_OK(cuMemAlloc(&dx, bytes_vec));
        CU_OK(cuMemAlloc(&dg, bytes_vec));
        CU_OK(cuMemAlloc(&dn, bytes_vec));
        CU_OK(cuMemAlloc(&dq, activation_bytes));
        CU_OK(cuMemAlloc(&dy, bytes_vec));
        CU_OK(cuMemcpyHtoD(dw, w.data(), weight_bytes));
        CU_OK(cuMemcpyHtoD(dx, x.data(), bytes_vec));
        CU_OK(cuMemcpyHtoD(dg, gamma.data(), bytes_vec));
        CUstream stream; CU_OK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
        // Command sequence: RMSNorm -> QuantizeQ8 -> Q8 matvec -> residual add.
        // All intermediates are device pointers; no transfer/sync between steps.
        void* args_norm[] = {&dx, &dg, &dn, &dim, (void*)&eps};
        void* args_quant[] = {&dn, &dq, &dim};
        void* args_matvec[] = {&dw, &dq, &dy, &dim, &dim};
        void* args_add[] = {&dy, &dx, &dim};
        auto enqueue_four = [&] {
            CU_OK(cuLaunchKernel(f_norm, 1,1,1, 256,1,1, 0,stream, args_norm, nullptr));
            CU_OK(cuLaunchKernel(f_quant, dim/32,1,1, 32,1,1, 0,stream, args_quant, nullptr));
            CU_OK(cuLaunchKernel(f_matvec, (dim+7)/8,1,1, 256,1,1, 0,stream, args_matvec, nullptr));
            CU_OK(cuLaunchKernel(f_add, (dim+255)/256,1,1, 256,1,1, 0,stream, args_add, nullptr));
        };
        enqueue_four();
        CU_OK(cuStreamSynchronize(stream));
        std::vector<uint8_t> q(activation_bytes);
        CU_OK(cuMemcpyDtoH(norm.data(), dn, bytes_vec));
        CU_OK(cuMemcpyDtoH(q.data(), dq, activation_bytes));
        CU_OK(cuMemcpyDtoH(result.data(), dy, bytes_vec));
        // Verify normalization separately, then dot product with actual GPU
        // quantized bytes. This avoids mistaking a quantization rounding tie for
        // a matvec bug while still catching quantization layout failures.
        double sumsq=0;
        for (float v:x) sumsq+=double(v)*double(v);
        const float inv=1.f/std::sqrt(float(sumsq/dim)+eps);
        float max_norm_err=0, max_output_err=0;
        for (unsigned i=0; i<dim; ++i) {
            float ref=x[i]*inv*gamma[i];
            max_norm_err=std::max(max_norm_err, std::abs(norm[i]-ref));
        }
        unsigned checked=std::min(128u,dim);
        for (unsigned k=0; k<checked; ++k) {
            unsigned r=unsigned((uint64_t(k)*dim)/checked);
            float expected=reference_row(w.data()+size_t(r)*(dim/32)*34, q.data(), dim) + x[r];
            float err=std::abs(result[r]-expected);
            max_output_err=std::max(max_output_err,err);
            if (!std::isfinite(result[r]) || err>0.003f+0.0002f*std::abs(expected)) {
                std::cerr << "FAIL row " << r << " expected " << expected
                          << " actual " << result[r] << " error " << err << '\n';
                return 1;
            }
        }
        if (max_norm_err>0.0003f) throw std::runtime_error("RMSNorm mismatch");
        // Check normalized block extrema and packed q values independently.
        float max_scale_err=0;
        unsigned max_q_diff=0;
        for (unsigned block=0; block<dim/32; ++block) {
            float ma=0;
            for (unsigned j=0; j<32; ++j) ma=std::max(ma,std::abs(norm[block*32+j]));
            const float expected_d=ma/127.f;
            const float actual_d=f16_to_f32(u16(q.data()+size_t(block)*34));
            max_scale_err=std::max(max_scale_err,std::abs(expected_d-actual_d));
            for (unsigned j=0; j<32; ++j) {
                float v=norm[block*32+j];
                int expected_q=expected_d>0 ? int(std::nearbyint(v/expected_d)) : 0;
                expected_q=std::max(-127,std::min(127,expected_q));
                int actual_q=int(int8_t(q[size_t(block)*34+2+j]));
                max_q_diff=std::max(max_q_diff,unsigned(std::abs(expected_q-actual_q)));
            }
        }
        if (max_scale_err>1e-5f || max_q_diff>1)
            throw std::runtime_error("packed Q8 quantization differs significantly from scalar reference");
        std::cout << "CORRECTNESS PASS: RMSNorm + Q8 quant + Q8 matvec + residual; "
                  << checked << " reference rows\n"
                  << "max_norm_error=" << max_norm_err
                  << " max_scale_error=" << max_scale_err
                  << " max_q_step_difference=" << max_q_diff
                  << " max_output_error=" << max_output_err << '\n';
        // Record the same launches as a CUDA Graph. GPU driver API is only
        // this development executor; the command sequence belongs to NightRun.
        CUgraph graph;
        CUgraphExec executable;
        CU_OK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
        enqueue_four();
        CU_OK(cuStreamEndCapture(stream, &graph));
        CU_OK(cuGraphInstantiate(&executable, graph, 0));
        for (unsigned i=0; i<10; ++i) CU_OK(cuGraphLaunch(executable, stream));
        CU_OK(cuStreamSynchronize(stream));
        CUevent start,stop;
        CU_OK(cuEventCreate(&start,CU_EVENT_DEFAULT));
        CU_OK(cuEventCreate(&stop,CU_EVENT_DEFAULT));
        double gpu_ms_sum=0,wall_ms_sum=0;
        for (unsigned i=0; i<repeats; ++i) {
            auto t0=std::chrono::steady_clock::now();
            CU_OK(cuEventRecord(start,stream));
            CU_OK(cuGraphLaunch(executable,stream));
            CU_OK(cuEventRecord(stop,stream));
            CU_OK(cuEventSynchronize(stop));
            float elapsed=0;
            CU_OK(cuEventElapsedTime(&elapsed,start,stop));
            auto t1=std::chrono::steady_clock::now();
            gpu_ms_sum+=elapsed;
            wall_ms_sum+=std::chrono::duration<double,std::milli>(t1-t0).count();
        }
        std::cout << std::fixed << std::setprecision(4)
                  << "dim=" << dim << " weights_MiB=" << double(weight_bytes)/1048576.0 << '\n'
                  << "gpu_graph_exec_avg_ms: " << gpu_ms_sum/repeats << '\n'
                  << "cuda_graph_launch_sync_wall_avg_ms: " << wall_ms_sum/repeats << '\n'
                  << "GPU buffers resident; no intermediate host round-trips; one graph replay per chain.\n"
                  << "NOTE: Validated transformer slice, not yet a full model or native UEFI launch.\n";
        CU_OK(cuEventDestroy(start)); CU_OK(cuEventDestroy(stop));
        CU_OK(cuGraphExecDestroy(executable)); CU_OK(cuGraphDestroy(graph));
        CU_OK(cuStreamDestroy(stream));
        CU_OK(cuMemFree(dy)); CU_OK(cuMemFree(dq)); CU_OK(cuMemFree(dn));
        CU_OK(cuMemFree(dg)); CU_OK(cuMemFree(dx)); CU_OK(cuMemFree(dw));
        CU_OK(cuModuleUnload(mod)); CU_OK(cuCtxDestroy(ctx));
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "ERROR: " << exc.what() << '\n';
        return 1;
    }
}
