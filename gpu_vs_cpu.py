
import time
import statistics
import torch

CPU_THREADS = 8
WARMUP = 5
REPEATS = 30

SHAPES = [
    (2048, 2048),
    (4096, 4096),
    (8192, 4096),
]

torch.set_num_threads(CPU_THREADS)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

device = torch.device("cuda")

print("GPU:", torch.cuda.get_device_name(0))
print("CPU threads:", torch.get_num_threads())
print("Data type: FP32")
print()

def benchmark(fn, repeats=REPEATS):
    for _ in range(WARMUP):
        fn()

    torch.cuda.synchronize()
    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(
            (time.perf_counter() - start) * 1000
        )

    return statistics.median(times)

print(
    f"{'Matrix':<14} "
    f"{'CPU ms':>10} "
    f"{'GPU ms':>10} "
    f"{'GPU+I/O ms':>12} "
    f"{'Speedup':>10}"
)
print("-" * 64)

for rows, cols in SHAPES:
    torch.manual_seed(42)

    # CPU data
    W_cpu = torch.randn(rows, cols)
    x_cpu = torch.randn(cols)

    # Cache weights and vector in GPU memory
    W_gpu = W_cpu.to(device)
    x_gpu = x_cpu.to(device)

    # CPU computation
    def cpu_run():
        return torch.mv(W_cpu, x_cpu)

    # GPU computation: weights already resident
    def gpu_run():
        return torch.mv(W_gpu, x_gpu)

    # GPU with activation upload and output download
    def gpu_io_run():
        x = x_cpu.to(device)
        y = torch.mv(W_gpu, x)
        return y.cpu()

    cpu_time = benchmark(cpu_run)
    gpu_time = benchmark(gpu_run)
    gpu_io_time = benchmark(gpu_io_run)

    # Correctness verification
    cpu_result = cpu_run()
    gpu_result = gpu_run().cpu()

    torch.testing.assert_close(
        cpu_result,
        gpu_result,
        rtol=1e-3,
        atol=1e-2
    )

    speedup = cpu_time / gpu_time

    print(
        f"{rows}x{cols:<8} "
        f"{cpu_time:>10.3f} "
        f"{gpu_time:>10.3f} "
        f"{gpu_io_time:>12.3f} "
        f"{speedup:>9.2f}x"
    )

print()
print("CPU and GPU results verified.")
print("GPU memory:", torch.cuda.memory_allocated() // 1048576, "MiB")
