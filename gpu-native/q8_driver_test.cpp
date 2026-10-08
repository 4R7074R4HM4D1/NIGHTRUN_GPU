// NightRun CUDA Driver API test for an offline compiled CUBIN.
// This is ONLY a development/test launcher, not the final UEFI GPU driver.
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

#define CU_OK(expression) do {                                                  \
    CUresult r = (expression);                                                   \
    if (r != CUDA_SUCCESS) {                                                      \
        const char* name = nullptr;                                              \
        const char* detail = nullptr;                                            \
        cuGetErrorName(r, &name);                                                  \
        cuGetErrorString(r, &detail);                                              \
        throw std::runtime_error(std::string(#expression) + ": " +                \
           (name ? name : "unknown") + " " + (detail ? detail : ""));             \
    }                                                                            \
} while (0)

static uint16_t load_u16_le(const uint8_t* p) {
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}

static float f16_to_f32(uint16_t h) {
    const int sign = (h & 0x8000) ? -1 : 1;
    const unsigned exp = (h >> 10) & 31;
    const unsigned mant = h & 1023;
    if (exp == 0) return sign * std::ldexp(float(mant), -24);
    if (exp == 31) return mant == 0 ? sign * INFINITY : NAN;
    return sign * std::ldexp(float(mant + 1024), int(exp) - 25);
}

// Mirrors nr_tensor::kernels::dot_q8_scalar over packed Q8_0 data.
static float reference_row(const uint8_t* w, const uint8_t* x, unsigned cols) {
    float sum = 0.0f;
    for (unsigned b = 0; b < cols / 32; ++b) {
        const uint8_t* wb = w + size_t(b) * 34;
        const uint8_t* xb = x + size_t(b) * 34;
        int dot = 0;
        for (unsigned k = 0; k < 32; ++k) {
            dot += int(int8_t(wb[2 + k])) * int(int8_t(xb[2 + k]));
        }
        sum += float(dot) * f16_to_f32(load_u16_le(wb)) *
               f16_to_f32(load_u16_le(xb));
    }
    return sum;
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0]
                << " <kernel.cubin> [--weights packed_q8.bin]"
                << " [--rows 128256] [--cols 2048] [--repeats 50]\n";
            return 2;
        }
        std::string cubin = argv[1], weights_path;
        unsigned rows = 4096, cols = 2048, repeats = 50;
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (i + 1 == argc) throw std::runtime_error("missing value for " + arg);
            std::string val = argv[++i];
            if (arg == "--weights") weights_path = val;
            else if (arg == "--rows") rows = static_cast<unsigned>(std::stoul(val));
            else if (arg == "--cols") cols = static_cast<unsigned>(std::stoul(val));
            else if (arg == "--repeats") repeats = static_cast<unsigned>(std::stoul(val));
            else throw std::runtime_error("unknown argument " + arg);
        }
        if (rows == 0 || cols == 0 || cols % 32 || repeats == 0 || rows > 2000000 || cols > 65536)
            throw std::runtime_error("invalid rows/cols/repeats");
        size_t nblocks = cols / 32;
        size_t weight_bytes = size_t(rows) * nblocks * 34;
        size_t xbytes = nblocks * 34;
        std::vector<uint8_t> weights(weight_bytes);
        if (!weights_path.empty()) {
            std::ifstream f(weights_path, std::ios::binary | std::ios::ate);
            if (!f) throw std::runtime_error("can't open " + weights_path);
            auto len = f.tellg();
            if (len < 0 || static_cast<uint64_t>(len) != weight_bytes)
                throw std::runtime_error("packed weight file size mismatch");
            f.seekg(0);
            f.read(reinterpret_cast<char*>(weights.data()), static_cast<std::streamsize>(weight_bytes));
            if (!f) throw std::runtime_error("weight file read failed");
        } else {
            // Synthetic input with exact f16 = 2^-10 scale and signed int8s.
            for (size_t b = 0; b < size_t(rows) * nblocks; ++b) {
                size_t off = b * 34;
                weights[off] = 0x00;
                weights[off + 1] = 0x14; // 0x1400 = 2^-10
                for (unsigned j = 0; j < 32; ++j)
                    weights[off + 2 + j] = uint8_t(int8_t(int((b * 17 + j * 23) % 255) - 127));
            }
        }
        std::vector<uint8_t> x(xbytes);
        for (size_t b = 0; b < nblocks; ++b) {
            x[b * 34] = 0x00;
            x[b * 34 + 1] = 0x24; // 0x2400 = 2^-6
            for (unsigned j = 0; j < 32; ++j)
                x[b * 34 + 2 + j] = uint8_t(int8_t(int((b * 19 + j * 11) % 255) - 127));
        }
        std::vector<float> output(rows);

        CU_OK(cuInit(0));
        CUdevice dev;
        CU_OK(cuDeviceGet(&dev, 0));
        char dev_name[256];
        CU_OK(cuDeviceGetName(dev_name, sizeof(dev_name), dev));
        int major = 0, minor = 0;
        CU_OK(cuDeviceComputeCapability(&major, &minor, dev));
        std::cout << "GPU: " << dev_name << " sm_" << major << minor << '\n';
        CUcontext context;
        CU_OK(cuCtxCreate(&context, 0, dev));
        CUmodule module;
        CU_OK(cuModuleLoad(&module, cubin.c_str()));
        CUfunction fn;
        CU_OK(cuModuleGetFunction(&fn, module, "nr_q8_matvec"));
        CUdeviceptr dweights = 0, dx = 0, dy = 0;
        CU_OK(cuMemAlloc(&dweights, weight_bytes));
        CU_OK(cuMemAlloc(&dx, xbytes));
        CU_OK(cuMemAlloc(&dy, size_t(rows) * sizeof(float)));
        CU_OK(cuMemcpyHtoD(dweights, weights.data(), weight_bytes));
        CU_OK(cuMemcpyHtoD(dx, x.data(), xbytes));
        const unsigned warps_per_block = 8;
        unsigned grid = (rows + warps_per_block - 1) / warps_per_block;
        void* args[] = { &dweights, &dx, &dy, &rows, &cols };
        auto launch = [&] {
            CU_OK(cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
        };
        launch();
        CU_OK(cuCtxSynchronize());
        CU_OK(cuMemcpyDtoH(output.data(), dy, size_t(rows) * sizeof(float)));
        // Check dispersed samples, not just the first rows.
        unsigned checked = std::min<unsigned>(rows, 128);
        float max_abs = 0.0f;
        for (unsigned i = 0; i < checked; ++i) {
            unsigned r = (uint64_t(i) * rows) / checked;
            float ref = reference_row(weights.data() + size_t(r) * nblocks * 34, x.data(), cols);
            float got = output[r];
            float delta = std::abs(got - ref);
            max_abs = std::max(max_abs, delta);
            // Floating point summation differs between GPUs/CPUs and FMA paths.
            if (!std::isfinite(got) || delta > 0.002f + 1.0e-4f * std::abs(ref)) {
                std::cerr << "CORRECTNESS FAIL row=" << r << " cpu=" << ref
                          << " gpu=" << got << " abs=" << delta << '\n';
                return 1;
            }
        }
        std::cout << "Correctness PASS: " << checked << " rows, max_abs_error="
                  << max_abs << '\n';
        CUevent start, stop;
        CU_OK(cuEventCreate(&start, CU_EVENT_DEFAULT));
        CU_OK(cuEventCreate(&stop, CU_EVENT_DEFAULT));
        double gpu_total_ms = 0.0, wall_total_ms = 0.0;
        for (unsigned i = 0; i < repeats; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            CU_OK(cuEventRecord(start, nullptr));
            launch();
            CU_OK(cuEventRecord(stop, nullptr));
            CU_OK(cuEventSynchronize(stop));
            float gpu_ms = 0;
            CU_OK(cuEventElapsedTime(&gpu_ms, start, stop));
            auto t1 = std::chrono::steady_clock::now();
            gpu_total_ms += gpu_ms;
            wall_total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "rows=" << rows << " cols=" << cols
                  << " packed_weights_MiB=" << double(weight_bytes) / 1048576.0 << '\n';
        std::cout << "gpu_kernel_avg_ms: " << gpu_total_ms / repeats << '\n';
        std::cout << "cuda_launch_sync_wall_avg_ms: " << wall_total_ms / repeats << '\n';
        std::cout << "NOTE: CUDA Driver API launcher, not UEFI driver or PCI-bridge timing.\n";
        CU_OK(cuEventDestroy(start));
        CU_OK(cuEventDestroy(stop));
        CU_OK(cuMemFree(dweights));
        CU_OK(cuMemFree(dx));
        CU_OK(cuMemFree(dy));
        CU_OK(cuModuleUnload(module));
        CU_OK(cuCtxDestroy(context));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
