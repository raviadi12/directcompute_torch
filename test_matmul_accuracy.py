"""Test matmul accuracy across different dispatch paths.
matmul_universal (TS=16): M<64 or N<64
matmul_coarsened (TS=64): M>=64 and N>=64
"""
import numpy as np
import nn_engine as nn

def test_matmul(M, K, N, label):
    rng = np.random.RandomState(42)
    A = rng.randn(M, K).astype(np.float32) * 0.1
    B = rng.randn(K, N).astype(np.float32) * 0.1
    
    expected = A @ B
    
    a_t = nn.Tensor(A, requires_grad=False, track=False)
    b_t = nn.Tensor(B, requires_grad=False, track=False)
    
    # Use the raw matmul via engine
    import ctypes
    out_handle = nn.lib.MatMulAlloc(a_t.gpu_buf, b_t.gpu_buf, M, K, N, 0)
    out_t = nn.Tensor.from_gpu(out_handle, (M, N))
    result = out_t.sync()
    
    diff = np.abs(expected - result)
    print(f"{label:40s} M={M:>5} K={K:>5} N={N:>5} | "
          f"max={diff.max():.8f} mean={diff.mean():.8f} "
          f"rel_max={diff.max() / (np.abs(expected).max() + 1e-10):.6f}")
    
    nn.lib.ReleaseBuffer(out_handle)
    nn.lib.ReleaseBuffer(a_t.gpu_buf)
    nn.lib.ReleaseBuffer(b_t.gpu_buf)
    return diff.max()

print("=" * 120)
print("Matmul Accuracy Test")
print("=" * 120)

# Cases that use matmul_universal (M<64 or N<64)
print("\n--- matmul_universal (M<64 or N<64) ---")
test_matmul(4, 27, 32768, "base4 enc1 conv1")
test_matmul(16, 27, 32768, "base16 enc1 conv1")
test_matmul(16, 144, 32768, "base16 enc1 conv2")
test_matmul(32, 144, 8192, "base16 enc2 conv1")
test_matmul(32, 288, 8192, "base16 enc2 conv2")

# Cases that use matmul_coarsened (M>=64 AND N>=64)
print("\n--- matmul_coarsened (M>=64 and N>=64) ---")
test_matmul(64, 288, 2048, "base16 enc3 conv1")
test_matmul(64, 576, 2048, "base16 enc3 conv2")
test_matmul(128, 576, 512, "base16 bot conv1")
test_matmul(128, 1152, 512, "base16 bot conv2")

# Decoder with concat: up3 has 128+64=192 input channels
test_matmul(64, 1728, 2048, "base16 up3 conv1 (192*9=1728)")
test_matmul(64, 576, 2048, "base16 up3 conv2")
test_matmul(32, 864, 8192, "base16 up2 conv1 (96*9=864)")
test_matmul(32, 288, 8192, "base16 up2 conv2")
test_matmul(16, 432, 32768, "base16 up1 conv1 (48*9=432)")
test_matmul(16, 144, 32768, "base16 up1 conv2")

# Edge cases near boundary
print("\n--- Boundary cases ---")
test_matmul(63, 288, 2048, "M=63 (universal)")
test_matmul(64, 288, 63, "N=63 (universal)")
test_matmul(64, 288, 64, "M=64,N=64 (coarsened)")

# Larger K test
print("\n--- Large K ---")
test_matmul(64, 2048, 2048, "large K=2048")
test_matmul(128, 4096, 512, "very large K=4096")
