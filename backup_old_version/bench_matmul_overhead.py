"""Matmul benchmark: isolate dispatch overhead from GPU compute."""
import numpy as np, time, ctypes
from nn_engine import lib, Tensor, _run_mm, release_all_buffers

sizes = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
]

for M, K, N in sizes:
    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(K, N).astype(np.float32)
    a = Tensor(A_np, track=False)
    b = Tensor(B_np, track=False)
    result = np.empty((M, N), dtype=np.float32)
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Warmup (with full sync)
    for _ in range(3):
        out = lib.CreateBuffer(None, M * N)
        _run_mm(a.gpu_buf, b.gpu_buf, out, M, K, N, 0)
        lib.ReadBuffer(out, result_ptr)
        lib.ReleaseBuffer(out)

    # Test 1: Dispatch only (Python ctypes overhead, no GPU work measured)
    out = lib.CreateBuffer(None, M * N)
    lib.ReadBuffer(out, result_ptr)  # drain queue
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        _run_mm(a.gpu_buf, b.gpu_buf, out, M, K, N, 0)
    t_dispatch = (time.perf_counter() - t0) * 1000 / iters
    lib.ReadBuffer(out, result_ptr)  # drain
    lib.ReleaseBuffer(out)

    # Test 2: Dispatch + full sync (true GPU compute time)
    # Each iteration: dispatch, then ReadBuffer forces GPU to finish
    times_sync = []
    for _ in range(10):
        out = lib.CreateBuffer(None, M * N)
        t0 = time.perf_counter()
        _run_mm(a.gpu_buf, b.gpu_buf, out, M, K, N, 0)
        lib.ReadBuffer(out, result_ptr)  # TRUE sync — blocks until GPU done
        t1 = time.perf_counter()
        times_sync.append((t1 - t0) * 1000)
        lib.ReleaseBuffer(out)

    # Test 3: Batch N dispatches + 1 sync (amortized overhead)
    batch_n = 10
    out = lib.CreateBuffer(None, M * N)
    lib.ReadBuffer(out, result_ptr)  # drain
    t0 = time.perf_counter()
    for _ in range(batch_n):
        _run_mm(a.gpu_buf, b.gpu_buf, out, M, K, N, 0)
    lib.ReadBuffer(out, result_ptr)  # sync all
    t_batch = (time.perf_counter() - t0) * 1000 / batch_n
    lib.ReleaseBuffer(out)

    median_sync = np.median(times_sync)
    flops = 2.0 * M * K * N
    gflops_sync = flops / (median_sync / 1000) / 1e9
    gflops_batch = flops / (t_batch / 1000) / 1e9

    print(f"\n{M}x{K}x{N}:")
    print(f"  Python dispatch overhead:      {t_dispatch:.3f} ms")
    print(f"  Dispatch + ReadBuffer (1:1):   {median_sync:.3f} ms  ({gflops_sync:.1f} GFLOPS)")
    print(f"  Batched {batch_n} dispatches + sync:  {t_batch:.3f} ms/op  ({gflops_batch:.1f} GFLOPS)")
    print(f"  Sync overhead per op:          {median_sync - t_dispatch:.3f} ms")

    lib.ReleaseBuffer(a.gpu_buf); a.gpu_buf = None
    lib.ReleaseBuffer(b.gpu_buf); b.gpu_buf = None
