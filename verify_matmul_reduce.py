"""Verify matmul_reduce produces same results as normal matmul."""
import numpy as np
import nn_engine as nn
import ctypes

def check_close(a, b, name, atol=1e-3):
    diff = np.max(np.abs(a - b))
    status = "PASSED" if diff < atol else "FAILED"
    print(f"  {status}: {name} | Max diff: {diff:.6e}")
    return diff < atol

np.random.seed(42)
print("=" * 60)
print("  Verifying matmul_reduce correctness")
print("=" * 60)

# Test 1: Conv1 dFilter dimensions (M=6, K=18432, N=25, flags=2)
print("\nTest 1: Conv1 dFilter — M=6, K=18432, N=25, flags=2 (B^T)")
A = np.random.randn(6, 18432).astype(np.float32) * 0.01
B = np.random.randn(25, 18432).astype(np.float32) * 0.01  # B^T stored as (N, K)

# NumPy reference: C = A × B^T
C_ref = A @ B.T

# GPU compute
a_gpu = nn.Tensor(A)
b_gpu = nn.Tensor(B)
out_handle = nn.lib.CreateBuffer(None, 6 * 25)
cb = (ctypes.c_uint * 4)(6, 18432, 25, 2)
srvs = (ctypes.c_void_p * 2)(a_gpu.gpu_buf, b_gpu.gpu_buf)
uavs = (ctypes.c_void_p * 1)(out_handle)
threads = (ctypes.c_uint * 3)(25, 6, 1)  # matmul_reduce dispatch
nn.lib.RunShader(b"matmul_reduce", srvs, 2, uavs, 1, threads, cb, 16)
C_gpu = np.zeros((6, 25), dtype=np.float32)
nn.lib.ReadBuffer(out_handle, C_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
nn.lib.ReleaseBuffer(out_handle)

ok1 = check_close(C_gpu, C_ref, "Conv1 dFilter (6×25, K=18432)")

# Test 2: Conv2 dFilter dimensions (M=16, K=2048, N=150, flags=2)
print("\nTest 2: Conv2 dFilter — M=16, K=2048, N=150, flags=2 (B^T)")
A2 = np.random.randn(16, 2048).astype(np.float32) * 0.01
B2 = np.random.randn(150, 2048).astype(np.float32) * 0.01

C_ref2 = A2 @ B2.T

a2_gpu = nn.Tensor(A2)
b2_gpu = nn.Tensor(B2)
out2_handle = nn.lib.CreateBuffer(None, 16 * 150)
cb2 = (ctypes.c_uint * 4)(16, 2048, 150, 2)
srvs2 = (ctypes.c_void_p * 2)(a2_gpu.gpu_buf, b2_gpu.gpu_buf)
uavs2 = (ctypes.c_void_p * 1)(out2_handle)
threads2 = (ctypes.c_uint * 3)(150, 16, 1)
nn.lib.RunShader(b"matmul_reduce", srvs2, 2, uavs2, 1, threads2, cb2, 16)
C_gpu2 = np.zeros((16, 150), dtype=np.float32)
nn.lib.ReadBuffer(out2_handle, C_gpu2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
nn.lib.ReleaseBuffer(out2_handle)

ok2 = check_close(C_gpu2, C_ref2, "Conv2 dFilter (16×150, K=2048)")

# Test 3: Normal matmul (no transpose) — small M, N
print("\nTest 3: No transpose — M=8, K=4096, N=32, flags=0")
A3 = np.random.randn(8, 4096).astype(np.float32) * 0.01
B3 = np.random.randn(4096, 32).astype(np.float32) * 0.01

C_ref3 = A3 @ B3

a3_gpu = nn.Tensor(A3)
b3_gpu = nn.Tensor(B3)
out3_handle = nn.lib.CreateBuffer(None, 8 * 32)
cb3 = (ctypes.c_uint * 4)(8, 4096, 32, 0)
srvs3 = (ctypes.c_void_p * 2)(a3_gpu.gpu_buf, b3_gpu.gpu_buf)
uavs3 = (ctypes.c_void_p * 1)(out3_handle)
threads3 = (ctypes.c_uint * 3)(32, 8, 1)
nn.lib.RunShader(b"matmul_reduce", srvs3, 2, uavs3, 1, threads3, cb3, 16)
C_gpu3 = np.zeros((8, 32), dtype=np.float32)
nn.lib.ReadBuffer(out3_handle, C_gpu3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
nn.lib.ReleaseBuffer(out3_handle)

ok3 = check_close(C_gpu3, C_ref3, "No transpose (8×32, K=4096)")

# Test 4: A transpose (flags=1)
print("\nTest 4: A transpose — M=10, K=512, N=20, flags=1")
A4_T = np.random.randn(512, 10).astype(np.float32) * 0.01  # stored transposed
B4 = np.random.randn(512, 20).astype(np.float32) * 0.01

C_ref4 = A4_T.T @ B4  # (10, 512) × (512, 20) = (10, 20)

a4_gpu = nn.Tensor(A4_T)
b4_gpu = nn.Tensor(B4)
out4_handle = nn.lib.CreateBuffer(None, 10 * 20)
cb4 = (ctypes.c_uint * 4)(10, 512, 20, 1)
srvs4 = (ctypes.c_void_p * 2)(a4_gpu.gpu_buf, b4_gpu.gpu_buf)
uavs4 = (ctypes.c_void_p * 1)(out4_handle)
threads4 = (ctypes.c_uint * 3)(20, 10, 1)
nn.lib.RunShader(b"matmul_reduce", srvs4, 2, uavs4, 1, threads4, cb4, 16)
C_gpu4 = np.zeros((10, 20), dtype=np.float32)
nn.lib.ReadBuffer(out4_handle, C_gpu4.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
nn.lib.ReleaseBuffer(out4_handle)

ok4 = check_close(C_gpu4, C_ref4, "A^T × B (10×20, K=512)")

nn.release_all_buffers()

print(f"\n{'='*60}")
all_ok = all([ok1, ok2, ok3, ok4])
print(f"  ALL TESTS {'PASSED' if all_ok else 'FAILED'}!")
print(f"{'='*60}")
