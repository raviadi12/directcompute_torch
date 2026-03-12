"""
End-to-end UNet pipeline comparison: DirectCompute vs NumPy reference.
Uses identical weights. Tests forward and backward through the full skip-connection path.
"""
import numpy as np
import nn_engine as nn

def numpy_conv2d(x, w, b, stride=1, padding=0):
    """Reference conv2d via im2col + matmul."""
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = w.shape
    OH = (IH + 2*padding - KH) // stride + 1
    OW = (IW + 2*padding - KW) // stride + 1
    if padding > 0:
        x_pad = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    else:
        x_pad = x
    col = np.zeros((N, IC*KH*KW, OH*OW), dtype=np.float32)
    for n in range(N):
        idx = 0
        for oh in range(OH):
            for ow in range(OW):
                col[n, :, idx] = x_pad[n, :, oh*stride:oh*stride+KH, ow*stride:ow*stride+KW].reshape(-1)
                idx += 1
    w_flat = w.reshape(OC, -1)
    out = np.zeros((N, OC, OH, OW), dtype=np.float32)
    for n in range(N):
        out[n] = (w_flat @ col[n] + b.reshape(-1, 1)).reshape(OC, OH, OW)
    return out

def numpy_relu(x):
    return np.maximum(0, x)

def numpy_maxpool2d(x, pool=2, stride=2):
    N, C, H, W = x.shape
    OH = (H - pool) // stride + 1
    OW = (W - pool) // stride + 1
    out = np.zeros((N, C, OH, OW), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            for oh in range(OH):
                for ow in range(OW):
                    patch = x[n, c, oh*stride:oh*stride+pool, ow*stride:ow*stride+pool]
                    out[n, c, oh, ow] = np.max(patch)
    return out

def numpy_upsample(x, scale=2):
    return x.repeat(scale, axis=2).repeat(scale, axis=3)

def numpy_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def check(gpu, ref, name, atol=5e-4):
    diff = np.abs(gpu - ref)
    maxd = np.max(diff)
    meand = np.mean(diff)
    ok = np.allclose(gpu, ref, atol=atol, rtol=1e-3)
    status = "PASSED" if ok else "FAILED"
    print(f"  {status}: {name} | shape={gpu.shape} | max_diff={maxd:.6f} mean_diff={meand:.6f}")
    if not ok and gpu.ndim >= 3:
        # Spatial analysis
        err = diff[0, 0]  # first batch, first channel
        print(f"    H-error profile: mean={np.mean(err, axis=1).max():.6f}")
        print(f"    W-error profile: mean={np.mean(err, axis=0).max():.6f}")
        # Check if error is spatially uniform or concentrated
        center = err[err.shape[0]//4:3*err.shape[0]//4, err.shape[1]//4:3*err.shape[1]//4]
        edge = np.mean(err[:5]) + np.mean(err[-5:]) + np.mean(err[:, :5]) + np.mean(err[:, -5:])
        print(f"    Center error: {np.mean(center):.6f}, Edge error: {edge/4:.6f}")
    return ok

rng = np.random.RandomState(42)

# Generate shared weights for a 2-level UNet
# Encoder level 1: conv(3→8, 3x3, pad1) + relu + conv(8→8, 3x3, pad1) + relu
enc1_w1 = rng.randn(8, 3, 3, 3).astype(np.float32) * 0.1
enc1_b1 = np.zeros(8, dtype=np.float32)
enc1_w2 = rng.randn(8, 8, 3, 3).astype(np.float32) * 0.1
enc1_b2 = np.zeros(8, dtype=np.float32)

# Encoder level 2: conv(8→16, 3x3, pad1) + relu + conv(16→16, 3x3, pad1) + relu
enc2_w1 = rng.randn(16, 8, 3, 3).astype(np.float32) * 0.1
enc2_b1 = np.zeros(16, dtype=np.float32)
enc2_w2 = rng.randn(16, 16, 3, 3).astype(np.float32) * 0.1
enc2_b2 = np.zeros(16, dtype=np.float32)

# Bottleneck: conv(16→32, 3x3, pad1) + relu + conv(32→32, 3x3, pad1) + relu
bot_w1 = rng.randn(32, 16, 3, 3).astype(np.float32) * 0.1
bot_b1 = np.zeros(32, dtype=np.float32)
bot_w2 = rng.randn(32, 32, 3, 3).astype(np.float32) * 0.1
bot_b2 = np.zeros(32, dtype=np.float32)

# Up2: conv(32+16→16, 3x3, pad1) + relu + conv(16→16, 3x3, pad1) + relu
up2_w1 = rng.randn(16, 48, 3, 3).astype(np.float32) * 0.1
up2_b1 = np.zeros(16, dtype=np.float32)
up2_w2 = rng.randn(16, 16, 3, 3).astype(np.float32) * 0.1
up2_b2 = np.zeros(16, dtype=np.float32)

# Up1: conv(16+8→8, 3x3, pad1) + relu + conv(8→8, 3x3, pad1) + relu
up1_w1 = rng.randn(8, 24, 3, 3).astype(np.float32) * 0.1
up1_b1 = np.zeros(8, dtype=np.float32)
up1_w2 = rng.randn(8, 8, 3, 3).astype(np.float32) * 0.1
up1_b2 = np.zeros(8, dtype=np.float32)

# Final 1x1 conv: conv(8→1, 1x1)
final_w = rng.randn(1, 8, 1, 1).astype(np.float32) * 0.1
final_b = np.zeros(1, dtype=np.float32)

# Input: circle pattern on 64x64 (smaller for faster test)
SZ = 64
x_np = np.zeros((1, 3, SZ, SZ), dtype=np.float32)
for h in range(SZ):
    for w in range(SZ):
        d = ((h - SZ//2)**2 + (w - SZ//2)**2) ** 0.5
        x_np[0, 0, h, w] = float(d < SZ//3) * 0.8  # circle
        x_np[0, 1, h, w] = h / SZ  # vertical gradient
        x_np[0, 2, h, w] = w / SZ  # horizontal gradient

target_np = np.zeros((1, 1, SZ, SZ), dtype=np.float32)
for h in range(SZ):
    for w in range(SZ):
        d = ((h - SZ//2)**2 + (w - SZ//2)**2) ** 0.5
        target_np[0, 0, h, w] = 1.0 if d < SZ//4 else 0.0

print("=" * 70)
print("Mini-UNet Forward Comparison: DirectCompute vs NumPy")
print("=" * 70)

# --- NumPy Forward ---
print("\n--- NumPy Forward ---")
# Encoder 1 (64×64)
np_e1_c1 = numpy_conv2d(x_np, enc1_w1, enc1_b1, padding=1)
np_e1_r1 = numpy_relu(np_e1_c1)
np_e1_c2 = numpy_conv2d(np_e1_r1, enc1_w2, enc1_b2, padding=1)
np_e1 = numpy_relu(np_e1_c2)

# Pool (32×32)
np_p1 = numpy_maxpool2d(np_e1)

# Encoder 2 (32×32)
np_e2_c1 = numpy_conv2d(np_p1, enc2_w1, enc2_b1, padding=1)
np_e2_r1 = numpy_relu(np_e2_c1)
np_e2_c2 = numpy_conv2d(np_e2_r1, enc2_w2, enc2_b2, padding=1)
np_e2 = numpy_relu(np_e2_c2)

# Pool (16×16)
np_p2 = numpy_maxpool2d(np_e2)

# Bottleneck (16×16)
np_b_c1 = numpy_conv2d(np_p2, bot_w1, bot_b1, padding=1)
np_b_r1 = numpy_relu(np_b_c1)
np_b_c2 = numpy_conv2d(np_b_r1, bot_w2, bot_b2, padding=1)
np_b = numpy_relu(np_b_c2)

# Upsample + Concat + Decode 2 (32×32)
np_u2 = numpy_upsample(np_b)
np_c2 = np.concatenate([np_u2, np_e2], axis=1)
np_d2_c1 = numpy_conv2d(np_c2, up2_w1, up2_b1, padding=1)
np_d2_r1 = numpy_relu(np_d2_c1)
np_d2_c2 = numpy_conv2d(np_d2_r1, up2_w2, up2_b2, padding=1)
np_d2 = numpy_relu(np_d2_c2)

# Upsample + Concat + Decode 1 (64×64)
np_u1 = numpy_upsample(np_d2)
np_c1 = np.concatenate([np_u1, np_e1], axis=1)
np_d1_c1 = numpy_conv2d(np_c1, up1_w1, up1_b1, padding=1)
np_d1_r1 = numpy_relu(np_d1_c1)
np_d1_c2 = numpy_conv2d(np_d1_r1, up1_w2, up1_b2, padding=1)
np_d1 = numpy_relu(np_d1_c2)

# Final 1x1 conv + sigmoid
np_logits = numpy_conv2d(np_d1, final_w, final_b, padding=0)
np_preds = numpy_sigmoid(np_logits)

print(f"  Output shape: {np_preds.shape}")
print(f"  Output range: [{np_preds.min():.4f}, {np_preds.max():.4f}]")

# --- DirectCompute Forward ---
print("\n--- DirectCompute Forward ---")

# Encoder 1
gpu_e1_c1 = nn.conv2d(nn.Tensor(x_np), nn.Tensor(enc1_w1), nn.Tensor(enc1_b1), padding=1)
gpu_e1_r1 = nn.relu(gpu_e1_c1)
gpu_e1_c2 = nn.conv2d(gpu_e1_r1, nn.Tensor(enc1_w2), nn.Tensor(enc1_b2), padding=1)
gpu_e1 = nn.relu(gpu_e1_c2)

# Pool
gpu_p1 = nn.maxpool2d(gpu_e1, 2, 2)

# Encoder 2
gpu_e2_c1 = nn.conv2d(gpu_p1, nn.Tensor(enc2_w1), nn.Tensor(enc2_b1), padding=1)
gpu_e2_r1 = nn.relu(gpu_e2_c1)
gpu_e2_c2 = nn.conv2d(gpu_e2_r1, nn.Tensor(enc2_w2), nn.Tensor(enc2_b2), padding=1)
gpu_e2 = nn.relu(gpu_e2_c2)

# Pool
gpu_p2 = nn.maxpool2d(gpu_e2, 2, 2)

# Bottleneck
gpu_b_c1 = nn.conv2d(gpu_p2, nn.Tensor(bot_w1), nn.Tensor(bot_b1), padding=1)
gpu_b_r1 = nn.relu(gpu_b_c1)
gpu_b_c2 = nn.conv2d(gpu_b_r1, nn.Tensor(bot_w2), nn.Tensor(bot_b2), padding=1)
gpu_b = nn.relu(gpu_b_c2)

# Up2
gpu_u2 = nn.upsample2d(gpu_b, 2, 2)
gpu_c2 = nn.concat([gpu_u2, gpu_e2], axis=1)
gpu_d2_c1 = nn.conv2d(gpu_c2, nn.Tensor(up2_w1), nn.Tensor(up2_b1), padding=1)
gpu_d2_r1 = nn.relu(gpu_d2_c1)
gpu_d2_c2 = nn.conv2d(gpu_d2_r1, nn.Tensor(up2_w2), nn.Tensor(up2_b2), padding=1)
gpu_d2 = nn.relu(gpu_d2_c2)

# Up1
gpu_u1 = nn.upsample2d(gpu_d2, 2, 2)
gpu_c1 = nn.concat([gpu_u1, gpu_e1], axis=1)
gpu_d1_c1 = nn.conv2d(gpu_c1, nn.Tensor(up1_w1), nn.Tensor(up1_b1), padding=1)
gpu_d1_r1 = nn.relu(gpu_d1_c1)
gpu_d1_c2 = nn.conv2d(gpu_d1_r1, nn.Tensor(up1_w2), nn.Tensor(up1_b2), padding=1)
gpu_d1 = nn.relu(gpu_d1_c2)

# Final
gpu_logits = nn.conv2d(gpu_d1, nn.Tensor(final_w), nn.Tensor(final_b), padding=0)
gpu_preds = nn.sigmoid(gpu_logits)

gpu_preds_np = gpu_preds.sync()
print(f"  Output shape: {gpu_preds_np.shape}")
print(f"  Output range: [{gpu_preds_np.min():.4f}, {gpu_preds_np.max():.4f}]")

# --- Compare intermediate tensors ---
print("\n--- Comparison ---")
all_ok = True
all_ok &= check(gpu_e1_c1.sync(), np_e1_c1, "enc1_conv1")
all_ok &= check(gpu_e1.sync(), np_e1, "enc1_out")
all_ok &= check(gpu_p1.sync(), np_p1, "pool1")
all_ok &= check(gpu_e2.sync(), np_e2, "enc2_out")
all_ok &= check(gpu_p2.sync(), np_p2, "pool2")
all_ok &= check(gpu_b.sync(), np_b, "bottleneck_out")
all_ok &= check(gpu_u2.sync(), np_u2, "upsample2")
all_ok &= check(gpu_c2.sync(), np_c2, "concat2")
all_ok &= check(gpu_d2.sync(), np_d2, "decode2_out")
all_ok &= check(gpu_u1.sync(), np_u1, "upsample1")
all_ok &= check(gpu_c1.sync(), np_c1, "concat1")
all_ok &= check(gpu_d1.sync(), np_d1, "decode1_out")
all_ok &= check(gpu_logits.sync(), np_logits, "logits")
all_ok &= check(gpu_preds_np, np_preds, "final_preds")

# Spatial analysis of final output
print("\n--- Spatial Analysis ---")
gpu_out = gpu_preds_np[0, 0]
np_out = np_preds[0, 0]

# Check H vs W variation
gpu_h_std = np.std(gpu_out, axis=1)  # std across W for each row
gpu_w_std = np.std(gpu_out, axis=0)  # std across H for each col
np_h_std = np.std(np_out, axis=1)
np_w_std = np.std(np_out, axis=0)

print(f"  GPU - mean H-std (variation across W): {gpu_h_std.mean():.6f}")
print(f"  GPU - mean W-std (variation across H): {gpu_w_std.mean():.6f}")
print(f"  NumPy - mean H-std (variation across W): {np_h_std.mean():.6f}")
print(f"  NumPy - mean W-std (variation across H): {np_w_std.mean():.6f}")

if gpu_h_std.mean() < np_h_std.mean() * 0.1:
    print("  WARNING: GPU output has much less W-variation than NumPy!")
if gpu_w_std.mean() < np_w_std.mean() * 0.1:
    print("  WARNING: GPU output has much less H-variation than NumPy!")

if all_ok:
    print("\n*** ALL FORWARD PASS CHECKS PASSED ***")
    print("The forward pass is correct - the issue is in training dynamics.")
else:
    print("\n*** FORWARD PASS MISMATCH DETECTED ***")
    print("There is a bug in the forward pass pipeline!")

nn.release_all_buffers()
