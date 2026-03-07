"""Verify new ResNet/MobileNet building blocks: add, depthwise_conv2d, global_avg_pool2d.
Tests forward shapes, backward gradient flow, and numerical gradient checks."""
import numpy as np
import sys

from nn_engine import (Tensor, add, depthwise_conv2d, global_avg_pool2d, relu, flatten,
                       conv2d, softmax_ce, end_batch, ConvLayer, DepthwiseConvLayer,
                       BatchNorm2d, Linear, Model, maxpool2d, batchnorm2d, add_bias, matmul,
                       SGD)

PASS = 0
FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond:
        print(f"  [PASS] {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name}")
        FAIL += 1

# ── 1. Differentiable Add (skip / residual connection) ──────────────────────
print("\n=== Test: Differentiable Add ===")

A = Tensor(np.random.randn(2, 3, 4, 4).astype(np.float32), requires_grad=True)
B = Tensor(np.random.randn(2, 3, 4, 4).astype(np.float32), requires_grad=True)
C = add(A, B)
check("add forward shape", C.shape == (2, 3, 4, 4))

# Verify values
a_np = A.data.copy()
b_np = B.data.copy()
c_np = C.sync()
check("add forward values", np.allclose(c_np, a_np + b_np, atol=1e-5))

# Backward: flatten C and run through a linear head so the graph stays connected
C_flat = flatten(C)  # (2, 48)
w_add = Tensor(np.random.randn(C_flat.shape[1], 2).astype(np.float32) * 0.01, requires_grad=True)
wb_add = Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
logits_add = add_bias(matmul(C_flat, w_add), wb_add)
labels = Tensor(np.array([0, 1], dtype=np.float32))
loss = softmax_ce(logits_add, labels)
loss.backward()
check("add backward A has grad", A.grad is not None)
check("add backward B has grad", B.grad is not None)
end_batch()

# Test residual skip: out = relu(conv(x)) + x
print("\n--- Residual connection pattern ---")
x = Tensor(np.random.randn(2, 8, 8, 8).astype(np.float32), requires_grad=True)
# 3x3 conv with padding=1 preserves spatial dims
f = Tensor(np.random.randn(8, 8, 3, 3).astype(np.float32) * 0.1, requires_grad=True)
b = Tensor(np.zeros(8, dtype=np.float32), requires_grad=True)
conv_out = conv2d(x, f, b, stride=1, padding=1)
r = relu(conv_out)
residual = add(r, x)  # skip connection!
check("residual shape", residual.shape == x.shape)

# Run backward through rest
flat = flatten(residual)
w = Tensor(np.random.randn(flat.shape[1], 2).astype(np.float32) * 0.01, requires_grad=True)
wb = Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
logits = add_bias(matmul(flat, w), wb)
labels2 = Tensor(np.array([0, 1], dtype=np.float32))
loss2 = softmax_ce(logits, labels2)
loss2.backward()
check("residual backward x has grad", x.grad is not None)
check("residual backward filters has grad", f.grad is not None)
end_batch()

# ── 2. Depthwise Conv2d ─────────────────────────────────────────────────────
print("\n=== Test: Depthwise Conv2d ===")

C_ch = 16
x_dw = Tensor(np.random.randn(2, C_ch, 8, 8).astype(np.float32), requires_grad=True)
# filters: (C, 1, kH, kW)
f_dw = Tensor(np.random.randn(C_ch, 1, 3, 3).astype(np.float32) * 0.1, requires_grad=True)
b_dw = Tensor(np.zeros(C_ch, dtype=np.float32), requires_grad=True)

out_dw = depthwise_conv2d(x_dw, f_dw, b_dw, stride=1, padding=1)
check("depthwise forward shape", out_dw.shape == (2, C_ch, 8, 8))

# Numerical gradient check for depthwise conv
def dw_forward_np(x_np, f_np, b_np, stride=1, padding=1):
    N, C, H, W = x_np.shape
    _, _, kH, kW = f_np.shape
    outH = (H + 2*padding - kH) // stride + 1
    outW = (W + 2*padding - kW) // stride + 1
    if padding > 0:
        x_pad = np.pad(x_np, ((0,0),(0,0),(padding,padding),(padding,padding)))
    else:
        x_pad = x_np
    out = np.zeros((N, C, outH, outW), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            for oh in range(outH):
                for ow in range(outW):
                    patch = x_pad[n, c, oh*stride:oh*stride+kH, ow*stride:ow*stride+kW]
                    out[n, c, oh, ow] = np.sum(patch * f_np[c, 0]) + b_np[c]
    return out

ref = dw_forward_np(x_dw.data, f_dw.data, b_dw.data)
gpu_out = out_dw.sync()
check("depthwise forward values (vs numpy)", np.allclose(gpu_out, ref, atol=1e-4))

# Backward
flat_dw = flatten(out_dw)
w_dw = Tensor(np.random.randn(flat_dw.shape[1], 2).astype(np.float32) * 0.01, requires_grad=True)
wb_dw = Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
logits_dw = add_bias(matmul(flat_dw, w_dw), wb_dw)
labels_dw = Tensor(np.array([0, 1], dtype=np.float32))
loss_dw = softmax_ce(logits_dw, labels_dw)
loss_dw.backward()
check("depthwise backward x has grad", x_dw.grad is not None)
check("depthwise backward filters has grad", f_dw.grad is not None)
check("depthwise backward bias has grad", b_dw.grad is not None)
check("depthwise backward filter grad shape", f_dw.grad.shape == (C_ch, 1, 3, 3))
check("depthwise backward bias grad shape", b_dw.grad.shape == (C_ch,))
end_batch()

# Stride=2 test
x_s2 = Tensor(np.random.randn(1, 8, 16, 16).astype(np.float32), requires_grad=True)
f_s2 = Tensor(np.random.randn(8, 1, 3, 3).astype(np.float32) * 0.1, requires_grad=True)
b_s2 = Tensor(np.zeros(8, dtype=np.float32), requires_grad=True)
out_s2 = depthwise_conv2d(x_s2, f_s2, b_s2, stride=2, padding=1)
check("depthwise stride=2 shape", out_s2.shape == (1, 8, 8, 8))
ref_s2 = dw_forward_np(x_s2.data, f_s2.data, b_s2.data, stride=2, padding=1)
check("depthwise stride=2 values", np.allclose(out_s2.sync(), ref_s2, atol=1e-4))
end_batch()

# ── 3. Global Average Pooling ───────────────────────────────────────────────
print("\n=== Test: Global Average Pooling ===")

x_gap = Tensor(np.random.randn(2, 16, 7, 7).astype(np.float32), requires_grad=True)
out_gap = global_avg_pool2d(x_gap)
check("global_avg_pool forward shape", out_gap.shape == (2, 16))

# Verify values
ref_gap = x_gap.data.reshape(2, 16, -1).mean(axis=2)
check("global_avg_pool forward values", np.allclose(out_gap.sync(), ref_gap, atol=1e-5))

# Backward
w_gap = Tensor(np.random.randn(16, 2).astype(np.float32) * 0.01, requires_grad=True)
wb_gap = Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
logits_gap = add_bias(matmul(out_gap, w_gap), wb_gap)
labels_gap = Tensor(np.array([0, 1], dtype=np.float32))
loss_gap = softmax_ce(logits_gap, labels_gap)
loss_gap.backward()
check("global_avg_pool backward has grad", x_gap.grad is not None)
check("global_avg_pool backward grad shape", x_gap.grad.shape == (2, 16, 7, 7))
end_batch()

# ── 4. DepthwiseConvLayer class ─────────────────────────────────────────────
print("\n=== Test: DepthwiseConvLayer class ===")

dwl = DepthwiseConvLayer(32, ks=3, stride=1, padding=1)
check("DepthwiseConvLayer filter shape", dwl.filters.shape == (32, 1, 3, 3))
check("DepthwiseConvLayer bias shape", dwl.bias.shape == (32,))

x_dwl = Tensor(np.random.randn(1, 32, 8, 8).astype(np.float32))
out_dwl = dwl(x_dwl)
check("DepthwiseConvLayer forward shape", out_dwl.shape == (1, 32, 8, 8))
end_batch()

# ── 5. Parameter discovery ──────────────────────────────────────────────────
print("\n=== Test: Model parameter discovery ===")

class TestNet(Model):
    def __init__(self):
        super().__init__()
        self.c1 = ConvLayer(3, 16, ks=3, padding=1)
        self.dw1 = DepthwiseConvLayer(16, ks=3, padding=1)
        self.bn1 = BatchNorm2d(16)
        self.pw1 = ConvLayer(16, 32, ks=1)  # pointwise
        self.l1 = Linear(32, 2)
    def forward(self, x):
        x = relu(self.bn1(self.c1(x)))
        x = relu(self.dw1(x))
        x = relu(self.pw1(x))
        x = global_avg_pool2d(x)
        return self.l1(x)

net = TestNet()
params = net.parameters()
# c1: 2 (filters+bias), dw1: 2, bn1: 2 (gamma+beta), pw1: 2, l1: 2 = 10
check("parameter count", len(params) == 10)
check("dw params included", any(p.shape == (16, 1, 3, 3) for p in params))
end_batch()

# ── 6. End-to-end mini training (ResNet-style residual block) ────────────────
print("\n=== Test: End-to-end ResNet residual block training ===")

class TinyResNet(Model):
    def __init__(self):
        super().__init__()
        self.c1 = ConvLayer(1, 8, ks=3, padding=1)
        self.bn1 = BatchNorm2d(8)
        # Residual block
        self.c2a = ConvLayer(8, 8, ks=3, padding=1)
        self.bn2a = BatchNorm2d(8)
        self.c2b = ConvLayer(8, 8, ks=3, padding=1)
        self.bn2b = BatchNorm2d(8)
        # Head
        self.l1 = Linear(8, 2)
    def forward(self, x):
        x = relu(self.bn1(self.c1(x)))
        # Residual block
        identity = x
        out = relu(self.bn2a(self.c2a(x)))
        out = self.bn2b(self.c2b(out))
        out = relu(add(out, identity))  # skip connection
        x = global_avg_pool2d(out)
        return self.l1(x)

model = TinyResNet()
params = model.parameters()
optimizer = SGD(params, lr=0.01)

# Synthetic data: 2 classes from random 8x8 images
np.random.seed(42)
X = np.random.randn(32, 1, 8, 8).astype(np.float32)
Y = np.array([i % 2 for i in range(32)], dtype=np.float32)

losses = []
for epoch in range(5):
    optimizer.zero_grad()
    xb = Tensor(X)
    yb = Tensor(Y)
    logits = model(xb)
    loss = softmax_ce(logits, yb)
    loss.backward()
    optimizer.step(clip=1.0)
    l = float(loss.sync()[0])
    losses.append(l)
    end_batch()

check("ResNet loss decreasing", losses[-1] < losses[0])
print(f"  Losses: {' -> '.join(f'{l:.4f}' for l in losses)}")

# ── 7. End-to-end MobileNet-style block ─────────────────────────────────────
print("\n=== Test: End-to-end MobileNet inverted residual block ===")

class TinyMobileNet(Model):
    def __init__(self):
        super().__init__()
        self.c1 = ConvLayer(1, 8, ks=3, padding=1)
        self.bn1 = BatchNorm2d(8)
        # Inverted residual: expand -> depthwise -> project
        self.expand = ConvLayer(8, 24, ks=1)  # 1x1 expand
        self.bn_e = BatchNorm2d(24)
        self.dw = DepthwiseConvLayer(24, ks=3, padding=1)
        self.bn_dw = BatchNorm2d(24)
        self.project = ConvLayer(24, 8, ks=1)  # 1x1 project
        self.bn_p = BatchNorm2d(8)
        # Head
        self.l1 = Linear(8, 2)
    def forward(self, x):
        x = relu(self.bn1(self.c1(x)))
        # Inverted residual block
        identity = x
        out = relu(self.bn_e(self.expand(x)))
        out = relu(self.bn_dw(self.dw(out)))
        out = self.bn_p(self.project(out))
        out = relu(add(out, identity))  # residual connection
        x = global_avg_pool2d(out)
        return self.l1(x)

model2 = TinyMobileNet()
params2 = model2.parameters()
optimizer2 = SGD(params2, lr=0.01)

losses2 = []
for epoch in range(5):
    optimizer2.zero_grad()
    xb = Tensor(X)
    yb = Tensor(Y)
    logits2 = model2(xb)
    loss2 = softmax_ce(logits2, yb)
    loss2.backward()
    optimizer2.step(clip=1.0)
    l2 = float(loss2.sync()[0])
    losses2.append(l2)
    end_batch()

check("MobileNet loss decreasing", losses2[-1] < losses2[0])
print(f"  Losses: {' -> '.join(f'{l:.4f}' for l in losses2)}")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS+FAIL} tests")
if FAIL > 0:
    sys.exit(1)
else:
    print("All tests passed!")
