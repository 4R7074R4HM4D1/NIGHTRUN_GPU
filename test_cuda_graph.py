import time
import statistics
import torch

N = 4096
REPEATS = 50

assert torch.cuda.is_available()

W = torch.randn(N, N, device="cuda")
x = torch.randn(N, device="cuda")
y = torch.empty(N, device="cuda")

stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())

with torch.cuda.stream(stream):
    for _ in range(20):
        torch.mv(W, x, out=y)
stream.synchronize()

graph = torch.cuda.CUDAGraph()

with torch.cuda.graph(graph, stream=stream):
    torch.mv(W, x, out=y)

with torch.cuda.stream(stream):
    for _ in range(20):
        graph.replay()
stream.synchronize()

def bench(name, operation, delay=0):
    event_times = []
    wall_times = []

    for _ in range(REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        t0 = time.perf_counter_ns()

        with torch.cuda.stream(stream):
            start.record()
            operation()
            end.record()

        end.synchronize()

        wall_times.append(
            (time.perf_counter_ns() - t0) / 1e6
        )
        event_times.append(start.elapsed_time(end))

        if delay:
            time.sleep(delay)

    print(
        f"{name:<26} "
        f"CUDA={statistics.median(event_times):.3f} ms "
        f"Wall={statistics.median(wall_times):.3f} ms"
    )

print("GPU:", torch.cuda.get_device_name(0))
print()

bench(
    "Normal GEMV",
    lambda: torch.mv(W, x, out=y)
)

bench(
    "CUDA Graph",
    lambda: graph.replay()
)

bench(
    "Normal GEMV + 5ms idle",
    lambda: torch.mv(W, x, out=y),
    delay=0.005
)

bench(
    "CUDA Graph + 5ms idle",
    lambda: graph.replay(),
    delay=0.005
)
