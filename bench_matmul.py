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

HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available — TorchGPU columns will show N/A")

def bench_pytorch_gpu(M, K, N, A_np, B_np, warmup=3, iters=10):
    """Benchmark PyTorch CUDA matmul. Returns (median_ms, result_array) or (NaN, None)."""
    if not HAS_CUDA:
        return float('nan'), None
    A = torch.from_numpy(A_np).cuda()
    B = torch.from_numpy(B_np).cuda()
    for _ in range(warmup):
        _ = torch.mm(A, B)
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        C = torch.mm(A, B)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    result = C.cpu().numpy()
    del A, B, C
    return np.median(times), result

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
    # LeNet-like shapes (batch=32)
    (32, 256, 120),        # Linear1 forward
    (32, 120, 84),         # Linear2 forward
    (6, 25, 18432),        # Conv1 forward (batch folded into N)
    (16, 150, 2048),       # Conv2 forward (batch folded into N)
    # AlexNet-like shapes (batch=64)
    (64, 363, 200704),     # Conv1: 64x(3*11*11) @ (363)x(64*56*56)
    (192, 1600, 46656),    # Conv2: 192x(64*5*5) @ (1600)x(64*27*27)
    (384, 1728, 10816),    # Conv3: 384x(192*3*3) @ (1728)x(64*13*13)
    (256, 3456, 10816),    # Conv4: 256x(384*3*3) @ (3456)x(64*13*13)
    (64, 9216, 512),       # Linear1: 64x9216 @ 9216x512
    (64, 512, 512),        # Linear2: 64x512 @ 512x512
]

# Header
print(f"{'Size':>25s}  {'DC ms':>8s}  {'Torch ms':>9s}  {'TorchGPU':>9s}  {'NumPy ms':>9s}  {'DC/Torch':>9s}  {'DC/GPU':>7s}  {'MaxAbsErr':>10s}  {'%Close':>7s}  {'Verify':>6s}")
print(f"{'-'*25}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*10}  {'-'*7}  {'-'*6}")

all_pass = True

for M, K, N in sizes:
    label = f"{M}x{K}x{N}"

    # Use SAME input data for all backends
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(K, N).astype(np.float32)
    
    # Double-precision reference (ground truth)
    ref_f64 = (A_np.astype(np.float64) @ B_np.astype(np.float64)).astype(np.float32)

    dc_ms, dc_res = bench_directcompute(M, K, N, A_np, B_np)
    pt_ms, pt_res = bench_pytorch(M, K, N, A_np, B_np)
    gpu_ms, gpu_res = bench_pytorch_gpu(M, K, N, A_np, B_np)
    np_ms, np_res = bench_numpy(M, K, N, A_np, B_np)

    # Verify DirectCompute against double-precision reference
    max_abs, mean_abs, frac_close, passed = verify(dc_res, ref_f64, f"{M}x{K}x{N}")
    all_pass = all_pass and passed

    ratio = dc_ms / pt_ms
    gpu_ratio_str = f"{dc_ms / gpu_ms:>6.2f}x" if not np.isnan(gpu_ms) else "    N/A"
    gpu_ms_str = f"{gpu_ms:>9.2f}" if not np.isnan(gpu_ms) else "      N/A"
    winner = "DC" if dc_ms < pt_ms else "CPU"
    tag = "PASS" if passed else "FAIL"
    label = f"{M}x{K}x{N}"
    
    print(f"{label:>25s}  {dc_ms:>8.2f}  {pt_ms:>9.2f}  {gpu_ms_str}  {np_ms:>9.2f}  {ratio:>8.2f}x  {gpu_ratio_str}  {max_abs:>10.6f}  {frac_close*100:>6.2f}%  {tag:>6s}  [{winner}]")

print(f"\n{'='*110}")
if all_pass:
    print("  ALL SIZES PASSED (>99.9% elements within atol=1e-4, rtol=1e-3)")
else:
    print("  WARNING: SOME SIZES FAILED (>0.1% elements outside tolerance)")
print(f"  Tolerance: |DC - ref| <= 0.0001 + 0.001 * |ref|  (generous for FP32 accumulation)")
print(f"{'='*110}")

# ── Batched vs Folded matmul comparison ──
# In a real NN, "batch matmul" can mean two things:
#   1. FOLDED (what we do): reshape batch into M or N dimension, run ONE big GEMM
#      e.g. conv forward: filters[outC, inC*kH*kW] @ im2col[inC*kH*kW, batch*outH*outW]
#      e.g. linear forward: input[batch, in_features] @ weight[in_features, out_features]
#   2. TRUE BATCHED: dispatch B independent small GEMMs (one per sample)
#
# Folded is faster because one large GEMM saturates the GPU better than many small ones.

print(f"\n{'='*120}")
print("  BATCHED MATMUL COMPARISON: DC Folded vs DC Looped vs PyTorch (CPU)")
print(f"{'='*120}")
print(f"  Shows how our folded approach compares to PyTorch's CPU matmul on the same shapes.\n")

batch_tests = [
    # --- LeNet (batch=32) ---
    ("LeNet Linear1  (32x256 @ 256x120)",   32, 256, 120,  1, 256, 120),
    ("LeNet Linear2  (32x120 @ 120x84)",    32, 120, 84,   1, 120, 84),
    ("LeNet Conv1    (6x25 @ 25x18432)",    6, 25, 32*24*24, 6, 25, 24*24),
    ("LeNet Conv2    (16x150 @ 150x2048)",  16, 150, 32*8*8,  16, 150, 8*8),
    # --- AlexNet (batch=64) ---
    ("AlexNet Conv1  (64x363 @ 363x200704)",  64, 363, 64*56*56,  64, 363, 56*56),
    ("AlexNet Conv2  (192x1600 @ 1600x46656)",192, 1600, 64*27*27, 192, 1600, 27*27),
    ("AlexNet Conv3  (384x1728 @ 1728x10816)",384, 1728, 64*13*13, 384, 1728, 13*13),
    ("AlexNet Lin1   (64x9216 @ 9216x512)",   64, 9216, 512,   1, 9216, 512),
    ("AlexNet Lin2   (64x512 @ 512x512)",     64, 512, 512,    1, 512, 512),
]

hdr = f"  {'Description':<45s}  {'DC Fold':>8s}  {'DC Loop':>8s}  {'Torch':>8s}  {'TorchGPU':>8s}  {'Fold/Torch':>11s}  {'Fold/GPU':>9s}"
print(hdr)
print(f"  {'-'*45}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*11}  {'-'*9}")

WARMUP, ITERS = 2, 5

for desc, M_f, K_f, N_f, M_s, K_s, N_s in batch_tests:
    # Derive batch count from folded vs per-sample dimensions
    if N_f != N_s:
        BATCH = N_f // N_s  # conv: batch folded into N
    elif M_f != M_s:
        BATCH = M_f // M_s  # linear: batch folded into M
    else:
        BATCH = 1

    # Skip looped DC for huge conv shapes (64 individual dispatches too slow)
    skip_loop = (M_s * N_s * BATCH > 200_000_000)

    np.random.seed(42)
    A_big = np.random.randn(M_f, K_f).astype(np.float32)
    B_big = np.random.randn(K_f, N_f).astype(np.float32)

    # --- DC folded (1 GEMM) — SyncGPU for accurate timing ---
    a_t = Tensor(A_big, track=False)
    b_t = Tensor(B_big, track=False)
    for _ in range(WARMUP):
        out = lib.CreateBuffer(None, M_f * N_f)
        _run_mm(a_t.gpu_buf, b_t.gpu_buf, out, M_f, K_f, N_f, 0)
        lib.SyncGPU()
        lib.ReleaseBuffer(out)
    fold_times = []
    for _ in range(ITERS):
        out = lib.CreateBuffer(None, M_f * N_f)
        t0 = time.perf_counter()
        _run_mm(a_t.gpu_buf, b_t.gpu_buf, out, M_f, K_f, N_f, 0)
        lib.SyncGPU()
        t1 = time.perf_counter()
        fold_times.append((t1 - t0) * 1000)
        lib.ReleaseBuffer(out)
    lib.ReleaseBuffer(a_t.gpu_buf); a_t.gpu_buf = None
    lib.ReleaseBuffer(b_t.gpu_buf); b_t.gpu_buf = None

    # --- DC looped (B small GEMMs) ---
    if skip_loop:
        loop_ms = float('nan')
    else:
        A_small = np.random.randn(M_s, K_s).astype(np.float32)
        B_small = np.random.randn(K_s, N_s).astype(np.float32)
        a_s = Tensor(A_small, track=False)
        b_s = Tensor(B_small, track=False)
        for _ in range(WARMUP):
            for _ in range(BATCH):
                out = lib.CreateBuffer(None, M_s * N_s)
                _run_mm(a_s.gpu_buf, b_s.gpu_buf, out, M_s, K_s, N_s, 0)
                lib.ReleaseBuffer(out)
            lib.SyncGPU()
        loop_times = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            for _ in range(BATCH):
                out = lib.CreateBuffer(None, M_s * N_s)
                _run_mm(a_s.gpu_buf, b_s.gpu_buf, out, M_s, K_s, N_s, 0)
                lib.ReleaseBuffer(out)
            lib.SyncGPU()
            t1 = time.perf_counter()
            loop_times.append((t1 - t0) * 1000)
        lib.ReleaseBuffer(a_s.gpu_buf); a_s.gpu_buf = None
        lib.ReleaseBuffer(b_s.gpu_buf); b_s.gpu_buf = None
        loop_ms = np.median(loop_times)

    # --- PyTorch CPU folded (same big shape) ---
    A_t = torch.from_numpy(A_big)
    B_t = torch.from_numpy(B_big)
    for _ in range(WARMUP):
        _ = torch.mm(A_t, B_t)
    torch_times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        _ = torch.mm(A_t, B_t)
        t1 = time.perf_counter()
        torch_times.append((t1 - t0) * 1000)

    # --- PyTorch GPU folded (same big shape) ---
    if HAS_CUDA:
        A_g = torch.from_numpy(A_big).cuda()
        B_g = torch.from_numpy(B_big).cuda()
        for _ in range(WARMUP):
            _ = torch.mm(A_g, B_g)
            torch.cuda.synchronize()
        tgpu_times = []
        for _ in range(ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = torch.mm(A_g, B_g)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            tgpu_times.append((t1 - t0) * 1000)
        del A_g, B_g
        tgpu_ms = np.median(tgpu_times)
    else:
        tgpu_ms = float('nan')

    fold_ms = np.median(fold_times)
    torch_ms = np.median(torch_times)
    fold_ratio = fold_ms / torch_ms if torch_ms > 0 else float('inf')
    gpu_ratio_str = f"{fold_ms / tgpu_ms:>8.1f}x" if not np.isnan(tgpu_ms) else "      N/A"
    tgpu_str = f"{tgpu_ms:>7.3f}" if not np.isnan(tgpu_ms) else "     N/A"

    if np.isnan(loop_ms):
        loop_str = "skip"
    else:
        loop_str = f"{loop_ms:>7.3f}"

    print(f"  {desc:<45s}  {fold_ms:>7.3f}  {loop_str:>8s}  {torch_ms:>7.3f}  {tgpu_str:>8s}  {fold_ratio:>10.1f}x  {gpu_ratio_str:>9s}")

print(f"\n  Note: DC times use SyncGPU (D3D11 query fence) for accurate GPU timing.")
print(f"  TorchGPU times use torch.cuda.synchronize() for accurate CUDA timing.")
print(f"  DC Fold = our engine's actual approach (1 big GEMM with batch folded into dims).")
print(f"  DC Loop = naive per-sample dispatch (B tiny GEMMs).")
print(f"  Fold/Torch >1x = Torch CPU wins. <1x = DC GPU wins.")
print(f"  Fold/GPU >1x = CUDA wins. <1x = DC wins.")
print(f"{'='*120}")
