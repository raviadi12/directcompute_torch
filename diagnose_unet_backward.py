"""
Definitive test: Compare UNet backward pass parameter gradients
between DirectCompute and NumPy with identical weights and inputs.
"""
import numpy as np
import nn_engine as nn

def numpy_conv2d(x, w, b, padding=0):
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = w.shape
    OH = (IH + 2*padding - KH) + 1
    OW = (IW + 2*padding - KW) + 1
    if padding > 0:
        x_pad = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    else:
        x_pad = x
    col = np.zeros((N, IC*KH*KW, OH*OW), dtype=np.float32)
    for n in range(N):
        idx = 0
        for oh in range(OH):
            for ow in range(OW):
                col[n, :, idx] = x_pad[n, :, oh:oh+KH, ow:ow+KW].reshape(-1)
                idx += 1
    w_flat = w.reshape(OC, -1)
    out = np.zeros((N, OC, OH, OW), dtype=np.float32)
    for n in range(N):
        out[n] = (w_flat @ col[n] + b.reshape(-1, 1)).reshape(OC, OH, OW)
    return out, col

def numpy_conv2d_backward(grad_output, x, w, col, padding=0):
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = w.shape
    OH = (IH + 2*padding - KH) + 1
    OW = (IW + 2*padding - KW) + 1
    w_flat = w.reshape(OC, -1)
    
    # Reshape grad from NCHW to (OC, N*OH*OW) per batch
    dw = np.zeros_like(w_flat)
    db = np.zeros(OC, dtype=np.float32)
    dx_pad = np.zeros((N, IC, IH + 2*padding, IW + 2*padding), dtype=np.float32) if padding > 0 else np.zeros_like(x)
    
    for n in range(N):
        grad_flat = grad_output[n].reshape(OC, -1)  # (OC, OH*OW)
        dw += grad_flat @ col[n].T  # (OC, IC*KH*KW)
        db += grad_flat.sum(axis=1)
        
        # dInput via col2im
        dcol = w_flat.T @ grad_flat  # (IC*KH*KW, OH*OW)
        idx = 0
        for oh in range(OH):
            for ow in range(OW):
                patch = dcol[:, idx].reshape(IC, KH, KW)
                if padding > 0:
                    dx_pad[n, :, oh:oh+KH, ow:ow+KW] += patch
                else:
                    dx_pad[n, :, oh:oh+KH, ow:ow+KW] += patch
                idx += 1
    
    if padding > 0:
        dx = dx_pad[:, :, padding:-padding, padding:-padding]
    else:
        dx = dx_pad
    return dx, dw.reshape(w.shape), db

rng = np.random.RandomState(42)
SZ = 32  # Small size for speed (backward is O(n^2) in numpy)

# Weights for a 1-level enc + bottleneck + 1-level dec UNet
enc_w1 = rng.randn(8, 3, 3, 3).astype(np.float32) * 0.1
enc_b1 = np.zeros(8, np.float32)
enc_w2 = rng.randn(8, 8, 3, 3).astype(np.float32) * 0.1
enc_b2 = np.zeros(8, np.float32)

bot_w1 = rng.randn(16, 8, 3, 3).astype(np.float32) * 0.1
bot_b1 = np.zeros(16, np.float32)
bot_w2 = rng.randn(16, 16, 3, 3).astype(np.float32) * 0.1
bot_b2 = np.zeros(16, np.float32)

up_w1 = rng.randn(8, 24, 3, 3).astype(np.float32) * 0.1
up_b1 = np.zeros(8, np.float32)
up_w2 = rng.randn(8, 8, 3, 3).astype(np.float32) * 0.1
up_b2 = np.zeros(8, np.float32)

fin_w = rng.randn(1, 8, 1, 1).astype(np.float32) * 0.3
fin_b = np.zeros(1, np.float32)

# Input
x_np = rng.randn(1, 3, SZ, SZ).astype(np.float32) * 0.5

# === NumPy Forward + Backward ===
print("=" * 60)
print("NumPy Forward + Backward")
print("=" * 60)

# Forward
np_c1, col_c1 = numpy_conv2d(x_np, enc_w1, enc_b1, padding=1)
np_r1 = np.maximum(0, np_c1)
np_c2, col_c2 = numpy_conv2d(np_r1, enc_w2, enc_b2, padding=1)
np_e1 = np.maximum(0, np_c2)

# Maxpool
N, C, H, W = np_e1.shape
np_p1 = np.zeros((N, C, H//2, W//2), np.float32)
np_p1_mask = np.zeros_like(np_e1)
for n in range(N):
    for c in range(C):
        for h in range(H//2):
            for w in range(W//2):
                patch = np_e1[n, c, h*2:h*2+2, w*2:w*2+2]
                np_p1[n, c, h, w] = patch.max()
                idx = np.unravel_index(patch.argmax(), (2, 2))
                np_p1_mask[n, c, h*2+idx[0], w*2+idx[1]] = 1.0

# Bottleneck
np_bc1, col_bc1 = numpy_conv2d(np_p1, bot_w1, bot_b1, padding=1)
np_br1 = np.maximum(0, np_bc1)
np_bc2, col_bc2 = numpy_conv2d(np_br1, bot_w2, bot_b2, padding=1)
np_bot = np.maximum(0, np_bc2)

# Upsample + concat + decode
np_up = np_bot.repeat(2, axis=2).repeat(2, axis=3)
np_cat = np.concatenate([np_up, np_e1], axis=1)
np_dc1, col_dc1 = numpy_conv2d(np_cat, up_w1, up_b1, padding=1)
np_dr1 = np.maximum(0, np_dc1)
np_dc2, col_dc2 = numpy_conv2d(np_dr1, up_w2, up_b2, padding=1)
np_d1 = np.maximum(0, np_dc2)

# Final conv + sigmoid
np_logits, col_fin = numpy_conv2d(np_d1, fin_w, fin_b, padding=0)
np_preds = 1.0 / (1.0 + np.exp(-np_logits))

# Dice loss
batch = np_preds.shape[0]
size = np_preds.size // batch
for b in range(batch):
    p = np_preds[b].ravel()
    t_np = np.zeros(size, np.float32)
    for h in range(SZ):
        for w in range(SZ):
            if ((h-SZ//2)**2 + (w-SZ//2)**2) < (SZ//3)**2:
                t_np[h * SZ + w] = 1.0
    inter = np.sum(p * t_np)
    sp = np.sum(p); st = np.sum(t_np)
    denom = sp + st + 1e-6
    dice = 2 * inter / denom
    loss = 1 - dice
    print(f"  Batch {b}: dice={dice:.4f} loss={loss:.4f}")

# Target
target_np = np.zeros((1, 1, SZ, SZ), np.float32)
for h in range(SZ):
    for w in range(SZ):
        if ((h-SZ//2)**2 + (w-SZ//2)**2) < (SZ//3)**2:
            target_np[0, 0, h, w] = 1.0

# NumPy Backward
# Dice loss gradient
pred_flat = np_preds.reshape(batch, -1)
tgt_flat = target_np.reshape(batch, -1)
d_pred = np.zeros_like(np_preds)
for b in range(batch):
    inter = np.sum(pred_flat[b] * tgt_flat[b])
    denom = np.sum(pred_flat[b]) + np.sum(tgt_flat[b]) + 1e-6
    for i in range(pred_flat.shape[1]):
        t_i = tgt_flat[b, i]
        d_pred.ravel()[b * size + i] = -2.0 * (t_i * denom - inter) / (denom * denom) / batch

# Sigmoid backward
d_logits = d_pred * np_preds * (1.0 - np_preds)

# Final conv backward
dx_fin, dw_fin, db_fin = numpy_conv2d_backward(d_logits, np_d1, fin_w, col_fin, padding=0)

# Decoder backward: relu + conv + relu + conv
d_dc2 = dx_fin * (np_dc2 > 0)  # relu grad for d1
dx_dc2, dw_up2, db_up2 = numpy_conv2d_backward(d_dc2, np_dr1, up_w2, col_dc2, padding=1)
d_dc1 = dx_dc2 * (np_dc1 > 0)  # relu grad
dx_dc1, dw_up1, db_up1 = numpy_conv2d_backward(d_dc1, np_cat, up_w1, col_dc1, padding=1)

# Concat backward: split gradient
d_up = dx_dc1[:, :16, :, :]  # upsample part
d_e1_from_concat = dx_dc1[:, 16:, :, :]  # skip connection part

# Upsample backward
d_bot_from_up = np.zeros_like(np_bot)
for h in range(np_bot.shape[2]):
    for w in range(np_bot.shape[3]):
        d_bot_from_up[:, :, h, w] = (d_up[:, :, 2*h, 2*w] + d_up[:, :, 2*h+1, 2*w] +
                                      d_up[:, :, 2*h, 2*w+1] + d_up[:, :, 2*h+1, 2*w+1])

# Bottleneck backward
d_bc2 = d_bot_from_up * (np_bc2 > 0)
dx_bc2, dw_bot2, db_bot2 = numpy_conv2d_backward(d_bc2, np_br1, bot_w2, col_bc2, padding=1)
d_bc1 = dx_bc2 * (np_bc1 > 0)
dx_bc1, dw_bot1, db_bot1 = numpy_conv2d_backward(d_bc1, np_p1, bot_w1, col_bc1, padding=1)

# Maxpool backward
d_e1_from_pool = np.zeros_like(np_e1)
d_pool = dx_bc1
for n in range(N):
    for c in range(C):
        for h in range(H//2):
            for w in range(W//2):
                patch = np_e1[n, c, h*2:h*2+2, w*2:w*2+2]
                idx = np.unravel_index(patch.argmax(), (2, 2))
                d_e1_from_pool[n, c, h*2+idx[0], w*2+idx[1]] = d_pool[n, c, h, w]

# Total e1 gradient (from both skip connection and maxpool)
d_e1_total = d_e1_from_concat + d_e1_from_pool

# Encoder backward
d_c2 = d_e1_total * (np_c2 > 0)
dx_c2, dw_enc2, db_enc2 = numpy_conv2d_backward(d_c2, np_r1, enc_w2, col_c2, padding=1)
d_c1 = dx_c2 * (np_c1 > 0)
dx_c1, dw_enc1, db_enc1 = numpy_conv2d_backward(d_c1, x_np, enc_w1, col_c1, padding=1)

print(f"\nNumPy filter gradients computed.")

# === DirectCompute Forward + Backward ===
print("\n" + "=" * 60)
print("DirectCompute Forward + Backward")
print("=" * 60)

# Create tensors with shared weights
t_x = nn.Tensor(x_np, requires_grad=True)
t_enc_w1 = nn.Tensor(enc_w1, requires_grad=True, track=False)
t_enc_b1 = nn.Tensor(enc_b1, requires_grad=True, track=False)
t_enc_w2 = nn.Tensor(enc_w2, requires_grad=True, track=False)
t_enc_b2 = nn.Tensor(enc_b2, requires_grad=True, track=False)
t_bot_w1 = nn.Tensor(bot_w1, requires_grad=True, track=False)
t_bot_b1 = nn.Tensor(bot_b1, requires_grad=True, track=False)
t_bot_w2 = nn.Tensor(bot_w2, requires_grad=True, track=False)
t_bot_b2 = nn.Tensor(bot_b2, requires_grad=True, track=False)
t_up_w1 = nn.Tensor(up_w1, requires_grad=True, track=False)
t_up_b1 = nn.Tensor(up_b1, requires_grad=True, track=False)
t_up_w2 = nn.Tensor(up_w2, requires_grad=True, track=False)
t_up_b2 = nn.Tensor(up_b2, requires_grad=True, track=False)
t_fin_w = nn.Tensor(fin_w, requires_grad=True, track=False)
t_fin_b = nn.Tensor(fin_b, requires_grad=True, track=False)

# Forward
t_c1 = nn.conv2d(t_x, t_enc_w1, t_enc_b1, padding=1)
t_r1 = nn.relu(t_c1)
t_c2 = nn.conv2d(t_r1, t_enc_w2, t_enc_b2, padding=1)
t_e1 = nn.relu(t_c2)
t_p1 = nn.maxpool2d(t_e1, 2, 2)
t_bc1 = nn.conv2d(t_p1, t_bot_w1, t_bot_b1, padding=1)
t_br1 = nn.relu(t_bc1)
t_bc2 = nn.conv2d(t_br1, t_bot_w2, t_bot_b2, padding=1)
t_bot_out = nn.relu(t_bc2)
t_up_out = nn.upsample2d(t_bot_out, 2, 2)
t_cat = nn.concat([t_up_out, t_e1], axis=1)
t_dc1 = nn.conv2d(t_cat, t_up_w1, t_up_b1, padding=1)
t_dr1 = nn.relu(t_dc1)
t_dc2 = nn.conv2d(t_dr1, t_up_w2, t_up_b2, padding=1)
t_d1 = nn.relu(t_dc2)
t_logits = nn.conv2d(t_d1, t_fin_w, t_fin_b, padding=0)
t_preds = nn.sigmoid(t_logits)
t_loss = nn.dice_loss(t_preds, nn.Tensor(target_np))

t_loss_val = t_loss.sync()
print(f"  DC loss: {t_loss_val}")

# Backward
t_loss.backward()

# === Compare gradients ===
print("\n" + "=" * 60)
print("Gradient Comparison")
print("=" * 60)

pairs = [
    ("enc_w1", t_enc_w1, dw_enc1),
    ("enc_b1", t_enc_b1, db_enc1),
    ("enc_w2", t_enc_w2, dw_enc2),
    ("enc_b2", t_enc_b2, db_enc2),
    ("bot_w1", t_bot_w1, dw_bot1),
    ("bot_b1", t_bot_b1, db_bot1),
    ("bot_w2", t_bot_w2, dw_bot2),
    ("bot_b2", t_bot_b2, db_bot2),
    ("up_w1", t_up_w1, dw_up1),
    ("up_b1", t_up_b1, db_up1),
    ("up_w2", t_up_w2, dw_up2),
    ("up_b2", t_up_b2, db_up2),
    ("fin_w", t_fin_w, dw_fin),
    ("fin_b", t_fin_b, db_fin),
]

all_ok = True
for name, t_param, np_grad in pairs:
    gpu_grad = t_param.grad.sync()
    diff = np.abs(gpu_grad - np_grad)
    maxd = diff.max()
    meand = diff.mean()
    ok = np.allclose(gpu_grad, np_grad, atol=1e-3, rtol=1e-2)
    status = "OK" if ok else "MISMATCH"
    all_ok &= ok
    print(f"  {status}: {name:<10} max_diff={maxd:.6f} mean_diff={meand:.6f} |gpu|={np.abs(gpu_grad).max():.6f} |np|={np.abs(np_grad).max():.6f}")
    if not ok:
        # Show where the error is
        rel_err = diff / (np.abs(np_grad) + 1e-8)
        print(f"    Max relative error: {rel_err.max():.4f}")
        # Check if error is in a specific spatial pattern
        if gpu_grad.ndim == 4:
            for kh in range(gpu_grad.shape[2]):
                for kw in range(gpu_grad.shape[3]):
                    d = np.abs(gpu_grad[:,:,kh,kw] - np_grad[:,:,kh,kw]).max()
                    if d > 1e-3:
                        print(f"    Error at kernel pos [{kh},{kw}]: {d:.6f}")

# Also check input gradient (dx)
gpu_dx = t_x.grad.sync()
dx_diff = np.abs(gpu_dx - dx_c1)
print(f"\n  dx: max_diff={dx_diff.max():.6f} mean_diff={dx_diff.mean():.6f}")

if all_ok:
    print("\n*** ALL BACKWARD PASS GRADIENTS MATCH ***")
    print("Backward is correct. The rectangular band issue is caused by")
    print("training dynamics (clipping strategy / learning rate), not a code bug.")
else:
    print("\n*** BACKWARD PASS GRADIENT MISMATCH DETECTED ***")

nn.release_all_buffers()
