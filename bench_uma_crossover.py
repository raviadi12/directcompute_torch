"""Find the GPU vs CPU matmul crossover point for UMA (iGPU) systems.

On UMA, CPU↔GPU data transfer is a fast memcpy within shared DRAM,
so we need to find at what matrix size GPU becomes faster than CPU (NumPy/MKL).
"""
import numpy as np
import time
import ctypes
from nn_engine import lib, Tensor, _run_mm, _IS_IGPU

def bench_gpu_mm(M, K, N, A_np, B_np, warmup=5, iters=20):
    result = np.empty((M, N), dtype=np.float32)
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    a = Tensor(A_np, track=False)
    b = Tensor(B_np, track=False)
    for _ in range(warmup):
        out = lib.CreateBuffer(None, M * N)
        _run_mm(a.gpu_buf, b.gpu_buf, out, M, K, N, 0)
        lib.ReadBuffer(out, result_ptr)
        lib.ReleaseBuffer(out)
    times = []
    for _ in range(iters):
        out = lib.CreateBuffer(None, M * N)
        t0 = time.perf_counter()
        _run_mm(a.gpu_buf, b.gpu_buf, out, M, K, N, 0)
        lib.ReadBuffer(out, result_ptr)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        lib.ReleaseBuffer(out)
    lib.ReleaseBuffer(a.gpu_buf); a.gpu_buf = None
    lib.ReleaseBuffer(b.gpu_buf); b.gpu_buf = None
    return np.median(times)

def bench_gpu_mm_noreadback(M, K, N, A_np, B_np, warmup=5, iters=20):
    """GPU matmul without reading result back (stays on GPU)."""
    a = Tensor(A_np, track=False)
    b = Tensor(B_np, track=False)
    for _ in range(warmup):
        out = lib.CreateBuffer(None, M * N)
        _run_mm(a.gpu_buf, b.gpu_buf, out, M, K, N, 0)
        lib.FlushGPU()
        lib.ReleaseBuffer(out)
    times = []
    for _ in range(iters):
        out = lib.CreateBuffer(None, M * N)
        t0 = time.perf_counter()
        _run_mm(a.gpu_buf, b.gpu_buf, out, M, K, N, 0)
        lib.FlushGPU()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        lib.ReleaseBuffer(out)
    lib.ReleaseBuffer(a.gpu_buf); a.gpu_buf = None
    lib.ReleaseBuffer(b.gpu_buf); b.gpu_buf = None
    return np.median(times)

def bench_cpu_mm(M, K, N, A_np, B_np, warmup=5, iters=20):
    for _ in range(warmup):
        _ = A_np @ B_np
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        C = A_np @ B_np
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times)

def bench_hybrid_mm(M, K, N, A_np, B_np, warmup=5, iters=20):
    """Simulate hybrid: read A,B from GPU, CPU matmul, upload result to GPU."""
    a = Tensor(A_np, track=False)
    b = Tensor(B_np, track=False)
    a_cpu = np.empty((M, K), dtype=np.float32)
    b_cpu = np.empty((K, N), dtype=np.float32)
    a_ptr = a_cpu.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b_cpu.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    for _ in range(warmup):
        lib.ReadBuffer(a.gpu_buf, a_ptr)
        lib.ReadBuffer(b.gpu_buf, b_ptr)
        C = a_cpu @ b_cpu
        out = lib.CreateBuffer(C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), M * N)
        lib.ReleaseBuffer(out)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        lib.ReadBuffer(a.gpu_buf, a_ptr)
        lib.ReadBuffer(b.gpu_buf, b_ptr)
        C = a_cpu @ b_cpu
        out = lib.CreateBuffer(C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), M * N)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        lib.ReleaseBuffer(out)

    lib.ReleaseBuffer(a.gpu_buf); a.gpu_buf = None
    lib.ReleaseBuffer(b.gpu_buf); b.gpu_buf = None
    return np.median(times)

print(f"UMA Crossover Benchmark ({'iGPU' if _IS_IGPU else 'dGPU'})")
print(f"{'Size':>20s}  {'GPU(ms)':>8s}  {'GPU-only':>8s}  {'CPU(ms)':>8s}  {'Hybrid':>8s}  {'Winner':>8s}  {'Speedup':>8s}")
print(f"{'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

# Test sizes: from tiny to large, including NN-relevant sizes
sizes = [
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (128, 84, 10),       # LeNet fc2
    (128, 120, 84),      # LeNet fc1
    (128, 400, 120),     # LeNet fc0
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
]

for M, K, N in sizes:
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    gpu_ms = bench_gpu_mm(M, K, N, A, B)
    gpu_only = bench_gpu_mm_noreadback(M, K, N, A, B)
    cpu_ms = bench_cpu_mm(M, K, N, A, B)
    hyb_ms = bench_hybrid_mm(M, K, N, A, B)

    best = min(gpu_ms, cpu_ms, hyb_ms, gpu_only)
    if best == gpu_ms: winner = "GPU"
    elif best == gpu_only: winner = "GPU-nrb"
    elif best == cpu_ms: winner = "CPU"
    else: winner = "Hybrid"
    
    flops = 2 * M * K * N
    label = f"{M}x{K}x{N}"
    print(f"{label:>20s}  {gpu_ms:>8.3f}  {gpu_only:>8.3f}  {cpu_ms:>8.3f}  {hyb_ms:>8.3f}  {winner:>8s}  {gpu_ms/best:>7.2f}x")
