"""Raw matmul benchmark: DirectCompute GPU vs PyTorch CPU + correctness verification."""
import numpy as np
import time
import torch

# -- DirectCompute setup --
from nn_engine import lib, Tensor, release_all_buffers, _run_mm
import ctypes

def bench_directcompute(M, K, N, A_np, B_np, warmup=3, iters=10):
    """Benchmark DirectCompute matmul. Returns (median_ms, result_array)."""
    result = np.empty((M, N), dtype=np.float32)
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    a = Tensor(A_np, track=False)
    b = Tensor(B_np, track=False)

    # Warmup
    for _ in range(warmup):
        out = lib.CreateBuffer(None, M * N)
        _run_mm(a.gpu_buf, b.gpu_buf, out, M, K, N, 0)
        lib.ReadBuffer(out, result_ptr)
        lib.ReleaseBuffer(out)

    # Timed runs
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
    return np.median(times), result.copy()

def bench_pytorch(M, K, N, A_np, B_np, warmup=3, iters=10):
    """Benchmark PyTorch CPU matmul. Returns (median_ms, result_array)."""
    A = torch.from_numpy(A_np)
    B = torch.from_numpy(B_np)
    for _ in range(warmup):
        _ = torch.mm(A, B)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        C = torch.mm(A, B)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times), C.numpy()

def bench_numpy(M, K, N, A_np, B_np, warmup=3, iters=10):
    """Benchmark NumPy matmul. Returns (median_ms, result_array)."""
    for _ in range(warmup):
        _ = A_np @ B_np
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        C = A_np @ B_np
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times), C

def verify(dc_result, ref_result, label):
    """Compare DirectCompute result against double-precision reference.
    Uses numpy allclose-style tolerance: |dc - ref| <= atol + rtol * |ref|
    Returns (max_abs_error, mean_abs_error, fraction_close, pass/fail)."""
    abs_err = np.abs(dc_result - ref_result)
    max_abs = float(np.max(abs_err))
    mean_abs = float(np.mean(abs_err))
    
    # allclose check: |a - b| <= atol + rtol * |b|
    # FP32 eps = 1.19e-7, for K accumulations error ~ sqrt(K) * eps * |value|
    # Use generous tolerances for large matmuls
    atol = 1e-4   # absolute tolerance for near-zero values
    rtol = 1e-3   # relative tolerance (0.1%)
    close_mask = abs_err <= (atol + rtol * np.abs(ref_result))
    frac_close = float(np.mean(close_mask))
    
    passed = frac_close > 0.999  # 99.9% of elements must be within tolerance
    return max_abs, mean_abs, frac_close, passed

# -- Run benchmarks --
sizes = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    # AlexNet-like shapes
    (64, 363, 193600),     # Conv1 forward
    (192, 1600, 46656),    # Conv2 forward  
    (384, 1728, 10816),    # Conv3 forward
]

# Header
print(f"{'Size':>25s}  {'DC ms':>8s}  {'Torch ms':>9s}  {'NumPy ms':>9s}  {'DC/Torch':>9s}  {'MaxAbsErr':>10s}  {'MeanAbsErr':>11s}  {'%Close':>7s}  {'Verify':>6s}")
print(f"{'-'*25}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*11}  {'-'*7}  {'-'*6}")

all_pass = True

for M, K, N in sizes:
    # Use SAME input data for all backends
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(K, N).astype(np.float32)
    
    # Double-precision reference (ground truth)
    ref_f64 = (A_np.astype(np.float64) @ B_np.astype(np.float64)).astype(np.float32)

    dc_ms, dc_res = bench_directcompute(M, K, N, A_np, B_np)
    pt_ms, pt_res = bench_pytorch(M, K, N, A_np, B_np)
    np_ms, np_res = bench_numpy(M, K, N, A_np, B_np)

    # Verify DirectCompute against double-precision reference
    max_abs, mean_abs, frac_close, passed = verify(dc_res, ref_f64, f"{M}x{K}x{N}")
    all_pass = all_pass and passed

    ratio = dc_ms / pt_ms
    winner = "DC" if dc_ms < pt_ms else "CPU"
    tag = "PASS" if passed else "FAIL"
    label = f"{M}x{K}x{N}"
    
    print(f"{label:>25s}  {dc_ms:>8.2f}  {pt_ms:>9.2f}  {np_ms:>9.2f}  {ratio:>8.2f}x  {max_abs:>10.6f}  {mean_abs:>11.7f}  {frac_close*100:>6.2f}%  {tag:>6s}  [{winner}]")

print(f"\n{'='*110}")
if all_pass:
    print("  ALL SIZES PASSED (>99.9% elements within atol=1e-4, rtol=1e-3)")
else:
    print("  WARNING: SOME SIZES FAILED (>0.1% elements outside tolerance)")
print(f"  Tolerance: |DC - ref| <= 0.0001 + 0.001 * |ref|  (generous for FP32 accumulation)")
print(f"{'='*110}")
