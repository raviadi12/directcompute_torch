import numpy as np
import nn_engine as nn

def check_close(a, b, name, rtol=1e-3, atol=1e-4):
    diff = np.abs(a - b)
    max_diff = np.max(diff)
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        print(f"FAILED: {name} | Max diff: {max_diff}")
    else:
        print(f"PASSED: {name} | Max diff: {max_diff}")

# 1. Leaky ReLU
np.random.seed(42)
x_np = np.random.randn(2, 3, 4, 4).astype(np.float32)
alpha = 0.1
y_np = np.where(x_np > 0, x_np, x_np * alpha)
dy_np = np.random.randn(2, 3, 4, 4).astype(np.float32)
dx_np = dy_np * np.where(x_np > 0, 1.0, alpha)

x = nn.Tensor(x_np, requires_grad=True)
y = nn.leaky_relu(x, alpha)
out = y.sync()
check_close(out, y_np, "LeakyReLU Forward")

y.backward(nn.Tensor(dy_np))
dx = x.grad.sync()
check_close(dx, dx_np, "LeakyReLU Backward")

# 2. UpSample (Nearest Neighbor)
x_np = np.arange(1*2*2*2).reshape(1, 2, 2, 2).astype(np.float32)
scale = 2
y_np = x_np.repeat(scale, axis=2).repeat(scale, axis=3)
dy_np = np.ones((1, 2, 4, 4), dtype=np.float32)
dx_np = np.ones((1, 2, 2, 2), dtype=np.float32) * (scale * scale)

x = nn.Tensor(x_np, requires_grad=True)
y = nn.upsample2d(x, scale, scale)
out = y.sync()
check_close(out, y_np, "UpSample2D Forward")

y.backward(nn.Tensor(dy_np))
dx = x.grad.sync()
check_close(dx, dx_np, "UpSample2D Backward")

# 3. Concat
A_np = np.random.randn(2, 3, 4, 4).astype(np.float32)
B_np = np.random.randn(2, 5, 4, 4).astype(np.float32)
y_np = np.concatenate([A_np, B_np], axis=1)

A = nn.Tensor(A_np, requires_grad=True)
B = nn.Tensor(B_np, requires_grad=True)
y = nn.concat([A, B], axis=1)
out = y.sync()
check_close(out, y_np, "Concat Forward")

dy_np = np.copy(y_np) # gradient is same as output for simple check
y.backward(nn.Tensor(dy_np))
dA = A.grad.sync()
dB = B.grad.sync()
check_close(dA, dy_np[:, :3, :, :], "Concat Backward A")
check_close(dB, dy_np[:, 3:, :, :], "Concat Backward B")

# 4. Sigmoid
print("Checking Sigmoid...")
sig_in = np.random.randn(2, 3, 4, 4).astype(np.float32)
sig_out_np = 1.0 / (1.0 + np.exp(-sig_in))
dsig_np = sig_out_np * (1.0 - sig_out_np)

sig_tensor = nn.Tensor(sig_in, requires_grad=True)
sig_out_gpu = nn.sigmoid(sig_tensor)
check_close(sig_out_gpu.sync(), sig_out_np, "Sigmoid Forward")

sig_out_gpu.backward(nn.Tensor(np.ones_like(sig_out_np)))
check_close(sig_tensor.grad.sync(), dsig_np, "Sigmoid Backward")

nn.release_all_buffers()
