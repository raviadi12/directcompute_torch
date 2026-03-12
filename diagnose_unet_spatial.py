"""
Diagnostic: test spatial correctness of UNet-relevant ops at real sizes.
Checks conv2d, upsample, concat, maxpool, sigmoid, dice_loss at 128x128 scale.
Compares DirectCompute results with numpy reference.
"""
import numpy as np
import nn_engine as nn
import scipy.signal

def check_close(a, b, name, rtol=1e-3, atol=1e-4):
    diff = np.abs(a - b)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        # Show spatial pattern of errors
        if a.ndim >= 3:
            err = diff
            while err.ndim > 2:
                err = err[0]  # take first batch, first channel
            h_err = np.mean(err, axis=1)  # avg error per row
            w_err = np.mean(err, axis=0)  # avg error per col
            print(f"FAILED: {name} | Max diff: {max_diff:.6f} | Mean diff: {mean_diff:.6f}")
            print(f"  H-error profile (first 10 rows): {h_err[:10]}")
            print(f"  W-error profile (first 10 cols): {w_err[:10]}")
            print(f"  H-error profile (last 10 rows):  {h_err[-10:]}")
            print(f"  W-error profile (last 10 cols):  {w_err[-10:]}")
        else:
            print(f"FAILED: {name} | Max diff: {max_diff:.6f} | Mean diff: {mean_diff:.6f}")
        return False
    else:
        print(f"PASSED: {name} | Max diff: {max_diff:.6f}")
        return True


def numpy_conv2d(x, w, b, stride=1, padding=0):
    """Reference conv2d: x(N,C,H,W), w(OC,IC,KH,KW), b(OC)"""
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = w.shape
    OH = (IH + 2*padding - KH) // stride + 1
    OW = (IW + 2*padding - KW) // stride + 1
    
    if padding > 0:
        x_pad = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
    else:
        x_pad = x
    
    out = np.zeros((N, OC, OH, OW), dtype=np.float32)
    for n in range(N):
        for oc in range(OC):
            for oh in range(OH):
                for ow in range(OW):
                    val = 0.0
                    for ic in range(IC):
                        for kh in range(KH):
                            for kw in range(KW):
                                ih = oh * stride + kh
                                iw = ow * stride + kw
                                val += x_pad[n, ic, ih, iw] * w[oc, ic, kh, kw]
                    out[n, oc, oh, ow] = val + b[oc]
    return out


def numpy_conv2d_fast(x, w, b, stride=1, padding=0):
    """Faster reference conv2d using im2col approach"""
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = w.shape
    OH = (IH + 2*padding - KH) // stride + 1
    OW = (IW + 2*padding - KW) // stride + 1
    
    if padding > 0:
        x_pad = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
    else:
        x_pad = x
    
    # im2col
    col = np.zeros((N, IC*KH*KW, OH*OW), dtype=np.float32)
    for n in range(N):
        idx = 0
        for oh in range(OH):
            for ow in range(OW):
                patch = x_pad[n, :, oh*stride:oh*stride+KH, ow*stride:ow*stride+KW]
                col[n, :, idx] = patch.reshape(-1)
                idx += 1
    
    # matmul: w_reshaped (OC, IC*KH*KW) @ col (IC*KH*KW, OH*OW) for each batch
    w_reshaped = w.reshape(OC, -1)
    out = np.zeros((N, OC, OH, OW), dtype=np.float32)
    for n in range(N):
        mm = w_reshaped @ col[n]  # (OC, OH*OW)
        out[n] = (mm + b.reshape(-1, 1)).reshape(OC, OH, OW)
    
    return out


print("=" * 70)
print("DIAGNOSTIC: Spatial Correctness at UNet Scale")
print("=" * 70)

rng = np.random.RandomState(123)

# ── Test 1: Conv2d at 128x128 with spatial pattern ──
print("\n--- Test 1: Conv2d 128x128 with spatial gradient pattern ---")
# Create input with a known spatial pattern (diagonal gradient)
H, W = 128, 128
x_np = np.zeros((1, 3, H, W), dtype=np.float32)
for h in range(H):
    for w in range(W):
        x_np[0, 0, h, w] = h / H          # vertical gradient
        x_np[0, 1, h, w] = w / W          # horizontal gradient  
        x_np[0, 2, h, w] = (h + w) / (H + W)  # diagonal gradient

w_np = rng.randn(16, 3, 3, 3).astype(np.float32) * 0.1
b_np = np.zeros(16, dtype=np.float32)

# NumPy reference
ref = numpy_conv2d_fast(x_np, w_np, b_np, stride=1, padding=1)

# DirectCompute
x_t = nn.Tensor(x_np)
w_t = nn.Tensor(w_np)
b_t = nn.Tensor(b_np)
y_t = nn.conv2d(x_t, w_t, b_t, stride=1, padding=1)
gpu_out = y_t.sync()

check_close(gpu_out, ref, "Conv2d 128x128 spatial pattern")

# Check spatial structure: are H and W patterns preserved?
# If H/W are swapped, the "vertical gradient" channel would show horizontal gradient
print(f"  GPU out[0,0,:,64] first 5: {gpu_out[0,0,:5,64]}")  # Should vary with h
print(f"  GPU out[0,0,64,:] first 5: {gpu_out[0,0,64,:5]}")  # Should vary with w
print(f"  Ref out[0,0,:,64] first 5: {ref[0,0,:5,64]}")
print(f"  Ref out[0,0,64,:] first 5: {ref[0,0,64,:5]}")
nn.release_all_buffers()

# ── Test 2: Conv2d backward gradient spatial check ──
print("\n--- Test 2: Conv2d backward spatial gradient ---")
x_np = rng.randn(2, 16, 64, 64).astype(np.float32) * 0.1
w_np = rng.randn(32, 16, 3, 3).astype(np.float32) * 0.1
b_np = np.zeros(32, dtype=np.float32)

x_t = nn.Tensor(x_np, requires_grad=True)
w_t = nn.Tensor(w_np, requires_grad=True)
b_t = nn.Tensor(b_np, requires_grad=True)
y_t = nn.conv2d(x_t, w_t, b_t, stride=1, padding=1)

# Create a spatially-structured gradient (circle pattern)
dy_np = np.zeros((2, 32, 64, 64), dtype=np.float32)
for h in range(64):
    for w in range(64):
        dist = ((h - 32)**2 + (w - 32)**2) ** 0.5
        dy_np[:, :, h, w] = 1.0 if dist < 20 else 0.0

y_t.backward(nn.Tensor(dy_np))
dx_gpu = x_t.grad.sync()

# The input gradient should also have spatial structure (not be all-constant or row-constant)
dx_sample = dx_gpu[0, 0]  # (64, 64)
h_var = np.var(dx_sample, axis=1)  # variance across W for each row
w_var = np.var(dx_sample, axis=0)  # variance across H for each column
print(f"  dx H-variance (across W) range: [{h_var.min():.6f}, {h_var.max():.6f}]")
print(f"  dx W-variance (across H) range: [{w_var.min():.6f}, {w_var.max():.6f}]")
if h_var.max() < 1e-6:
    print("  WARNING: gradient is constant across W dimension (spatial structure lost)")
if w_var.max() < 1e-6:
    print("  WARNING: gradient is constant across H dimension (spatial structure lost)")
nn.release_all_buffers()

# ── Test 3: Upsample 16x16 → 32x32 (UNet bottleneck scale) ──
print("\n--- Test 3: Upsample 16x16 → 32x32 ---")
x_np = rng.randn(2, 128, 16, 16).astype(np.float32)
x_t = nn.Tensor(x_np, requires_grad=True)
y_t = nn.upsample2d(x_t, 2, 2)
gpu_out = y_t.sync()

ref = x_np.repeat(2, axis=2).repeat(2, axis=3)  # (2, 128, 32, 32)
check_close(gpu_out, ref, "Upsample 16→32")

# Check backward
dy_np = rng.randn(2, 128, 32, 32).astype(np.float32)
y_t.backward(nn.Tensor(dy_np))
dx_gpu = x_t.grad.sync()

# Reference: each input pixel accumulates 4 gradient values (2x2 block)
dx_ref = np.zeros_like(x_np)
for h in range(16):
    for w in range(16):
        dx_ref[:, :, h, w] = (dy_np[:, :, 2*h, 2*w] + dy_np[:, :, 2*h+1, 2*w] + 
                               dy_np[:, :, 2*h, 2*w+1] + dy_np[:, :, 2*h+1, 2*w+1])
check_close(dx_gpu, dx_ref, "Upsample backward 32→16")
nn.release_all_buffers()

# ── Test 4: Full UNet-like forward pass spatial check ──
print("\n--- Test 4: UNet forward spatial pattern test ---")
# Create a spatial pattern that should be reconstructable by UNet
# If spatial indexing is wrong, the output won't match the expected pattern
x_np = np.zeros((1, 3, 128, 128), dtype=np.float32)
# Create a circle pattern
for h in range(128):
    for w in range(128):
        dist = ((h - 64)**2 + (w - 64)**2) ** 0.5
        x_np[0, :, h, w] = 1.0 if dist < 40 else 0.0

x_t = nn.Tensor(x_np)
# First conv
w1 = rng.randn(16, 3, 3, 3).astype(np.float32) * 0.1
b1 = np.zeros(16, dtype=np.float32)
y = nn.conv2d(x_t, nn.Tensor(w1), nn.Tensor(b1), padding=1)
y = nn.relu(y)

# Maxpool
p = nn.maxpool2d(y, 2, 2)  # 64x64

# Conv in encoder
w2 = rng.randn(32, 16, 3, 3).astype(np.float32) * 0.1
b2 = np.zeros(32, dtype=np.float32)
e = nn.conv2d(p, nn.Tensor(w2), nn.Tensor(b2), padding=1)
e = nn.relu(e)

# Upsample back
u = nn.upsample2d(e, 2, 2)  # 128x128

# Check that upsample output has spatial structure matching the circle
u_np = u.sync()
u_sample = u_np[0, 0]  # (128, 128)

# The center should be different from edges
center_val = np.mean(u_sample[50:78, 50:78])
edge_val = np.mean(u_sample[0:10, 0:10])
print(f"  Center region mean: {center_val:.6f}")
print(f"  Edge region mean:   {edge_val:.6f}")
print(f"  Difference: {abs(center_val - edge_val):.6f}")

# Check H vs W symmetry (the circle is symmetric, so output should be too)
h_profile = np.mean(u_sample, axis=1)  # avg across W for each H
w_profile = np.mean(u_sample, axis=0)  # avg across H for each W
h_w_corr = np.corrcoef(h_profile, w_profile)[0,1]
print(f"  H-profile vs W-profile correlation: {h_w_corr:.4f} (should be ~1.0 for circle)")

if abs(h_w_corr) < 0.8:
    print("  WARNING: H and W profiles are not correlated - spatial processing is asymmetric!")
nn.release_all_buffers()

# ── Test 5: Concat + slice at UNet scale ──
print("\n--- Test 5: Concat + backward at 128x128 ---")
a_np = rng.randn(2, 16, 128, 128).astype(np.float32)
b_np2 = rng.randn(2, 32, 128, 128).astype(np.float32)

a_t = nn.Tensor(a_np, requires_grad=True)
b_t2 = nn.Tensor(b_np2, requires_grad=True)
c_t = nn.concat([a_t, b_t2], axis=1)  # (2, 48, 128, 128)
c_out = c_t.sync()

ref = np.concatenate([a_np, b_np2], axis=1)
check_close(c_out, ref, "Concat 128x128")

# Check spatial structure preserved
dy = rng.randn(2, 48, 128, 128).astype(np.float32)
c_t.backward(nn.Tensor(dy))
da = a_t.grad.sync()
db = b_t2.grad.sync()
check_close(da, dy[:, :16], "Concat backward A 128x128")
check_close(db, dy[:, 16:], "Concat backward B 128x128")
nn.release_all_buffers()

# ── Test 6: Dice loss gradient spatial check ──
print("\n--- Test 6: Dice loss gradient spatial structure ---")
# Create prediction and target with known spatial structure
pred_np = np.zeros((1, 1, 128, 128), dtype=np.float32)
target_np = np.zeros((1, 1, 128, 128), dtype=np.float32)

# Prediction: horizontal band (what the model seems to learn)
pred_np[0, 0, 30:100, :] = 0.8

# Target: circle
for h in range(128):
    for w in range(128):
        if ((h-64)**2 + (w-64)**2) < 40**2:
            target_np[0, 0, h, w] = 1.0

pred_t = nn.Tensor(pred_np, requires_grad=True)
target_t = nn.Tensor(target_np)
loss_t = nn.dice_loss(pred_t, target_t)
loss_val = loss_t.sync()
print(f"  Dice loss value: {loss_val}")

loss_t.backward()
grad_np = pred_t.grad.sync()

# The gradient should have spatial structure matching the target
# Inside the circle: gradient should push prediction up
# Outside the circle: gradient should push prediction down
grad_inside = np.mean(grad_np[0, 0, 50:78, 50:78])  # Inside circle center
grad_outside = np.mean(grad_np[0, 0, 0:10, 0:10])    # Outside circle (top-left)
grad_band_outside_circle = np.mean(grad_np[0, 0, 30:40, 0:20])  # In band but outside circle

print(f"  Grad inside circle: {grad_inside:.6f}")
print(f"  Grad outside circle (top-left): {grad_outside:.6f}")
print(f"  Grad in band, outside circle: {grad_band_outside_circle:.6f}")

# Check that gradient varies spatially
grad_h_var = np.var(grad_np[0, 0], axis=1)  # variance across W per row
grad_w_var = np.var(grad_np[0, 0], axis=0)  # variance across H per col
print(f"  Grad H-variance range: [{grad_h_var.min():.8f}, {grad_h_var.max():.8f}]")
print(f"  Grad W-variance range: [{grad_w_var.min():.8f}, {grad_w_var.max():.8f}]")

if grad_w_var.max() < 1e-10:
    print("  WARNING: Dice gradient has NO variation across W (columns)!")
    print("  This would cause the model to learn row-constant predictions!")
nn.release_all_buffers()

# ── Test 7: End-to-end single conv backward check ──
print("\n--- Test 7: Conv backward numerical gradient check at 128x128 ---")
# Use finite differences to verify gradient
x_np = rng.randn(1, 3, 32, 32).astype(np.float32) * 0.1  # Smaller for speed
w_np = rng.randn(8, 3, 3, 3).astype(np.float32) * 0.1
b_np3 = np.zeros(8, dtype=np.float32)

# Analytical gradient  
x_t = nn.Tensor(x_np.copy(), requires_grad=True)
w_t = nn.Tensor(w_np.copy(), requires_grad=True)
b_t3 = nn.Tensor(b_np3.copy(), requires_grad=True)
y_t = nn.conv2d(x_t, w_t, b_t3, padding=1)
# Sum all outputs as loss
loss = y_t.sync().sum()
dy = np.ones_like(y_t.sync())
y_t.backward(nn.Tensor(dy))
dx_analytical = x_t.grad.sync().copy()
nn.release_all_buffers()

# Numerical gradient (finite differences)
eps = 1e-3
dx_numerical = np.zeros_like(x_np)
for h in [0, 15, 16, 31]:  # Sample a few positions
    for w in [0, 15, 16, 31]:
        for c in range(3):
            x_plus = x_np.copy()
            x_plus[0, c, h, w] += eps
            x_t_p = nn.Tensor(x_plus)
            y_p = nn.conv2d(x_t_p, nn.Tensor(w_np), nn.Tensor(b_np3), padding=1)
            loss_plus = y_p.sync().sum()
            nn.release_all_buffers()

            x_minus = x_np.copy()
            x_minus[0, c, h, w] -= eps
            x_t_m = nn.Tensor(x_minus)
            y_m = nn.conv2d(x_t_m, nn.Tensor(w_np), nn.Tensor(b_np3), padding=1)
            loss_minus = y_m.sync().sum()
            nn.release_all_buffers()

            dx_numerical[0, c, h, w] = (loss_plus - loss_minus) / (2 * eps)

for h in [0, 15, 16, 31]:
    for w in [0, 15, 16, 31]:
        for c in range(3):
            a = dx_analytical[0, c, h, w]
            n = dx_numerical[0, c, h, w]
            match = "OK" if abs(a - n) < 0.01 else "MISMATCH"
            if match == "MISMATCH":
                print(f"  [{c},{h},{w}] analytical={a:.4f} numerical={n:.4f} {match}")

print("  Numerical gradient check done (only mismatches shown)")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

nn.release_all_buffers()
