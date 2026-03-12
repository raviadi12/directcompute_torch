"""Check backward pass: compare gradients between DC and PyTorch with same weights."""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nn_engine as dc

np.random.seed(42)
torch.manual_seed(42)

B, C, H, W = 1, 3, 16, 16
base = 4

x_np = np.random.randn(B, C, H, W).astype(np.float32) * 0.1
y_np = np.zeros((B, 1, H, W), dtype=np.float32)
y_np[0, 0, 4:12, 4:12] = 1.0  # square mask

# === Build PyTorch model ===
class PTConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=True)
    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))))

class PTUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = PTConvBlock(3, base)
        self.enc2 = PTConvBlock(base, base*2)
        self.bot = PTConvBlock(base*2, base*4)
        self.up2 = PTConvBlock(base*4 + base*2, base*2)
        self.up1 = PTConvBlock(base*2 + base, base)
        self.final = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2)
        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2)
        b = self.bot(p2)
        u2 = F.interpolate(b, scale_factor=2, mode='nearest')
        d2 = self.up2(torch.cat([u2, e2], 1))
        u1 = F.interpolate(d2, scale_factor=2, mode='nearest')
        d1 = self.up1(torch.cat([u1, e1], 1))
        return torch.sigmoid(self.final(d1))

pt = PTUNet()

# === Copy PT weights to DC ===
def copy_conv(pt_conv):
    w = pt_conv.weight.data.numpy().copy()
    b = pt_conv.bias.data.numpy().copy()
    dw = dc.Tensor(w, requires_grad=True, track=False)
    db = dc.Tensor(b, requires_grad=True, track=False)
    return dw, db

enc1_w1, enc1_b1 = copy_conv(pt.enc1.conv1)
enc1_w2, enc1_b2 = copy_conv(pt.enc1.conv2)
enc2_w1, enc2_b1 = copy_conv(pt.enc2.conv1)
enc2_w2, enc2_b2 = copy_conv(pt.enc2.conv2)
bot_w1, bot_b1 = copy_conv(pt.bot.conv1)
bot_w2, bot_b2 = copy_conv(pt.bot.conv2)
up2_w1, up2_b1 = copy_conv(pt.up2.conv1)
up2_w2, up2_b2 = copy_conv(pt.up2.conv2)
up1_w1, up1_b1 = copy_conv(pt.up1.conv1)
up1_w2, up1_b2 = copy_conv(pt.up1.conv2)
final_w, final_b = copy_conv(pt.final)

all_dc_params = [enc1_w1, enc1_b1, enc1_w2, enc1_b2,
                 enc2_w1, enc2_b1, enc2_w2, enc2_b2,
                 bot_w1, bot_b1, bot_w2, bot_b2,
                 up2_w1, up2_b1, up2_w2, up2_b2,
                 up1_w1, up1_b1, up1_w2, up1_b2,
                 final_w, final_b]

# === PyTorch forward + backward ===
x_pt = torch.from_numpy(x_np.copy())
y_pt = torch.from_numpy(y_np.copy())
x_pt.requires_grad = True

pred_pt = pt(x_pt)
# Dice loss
inter = torch.sum(pred_pt * y_pt, dim=[1,2,3])
sum_p = torch.sum(pred_pt, dim=[1,2,3])
sum_t = torch.sum(y_pt, dim=[1,2,3])
loss_pt = torch.mean(1.0 - (2.0 * inter) / (sum_p + sum_t + 1e-6))
loss_pt.backward()

print("=" * 70)
print("  GRADIENT COMPARISON (same weights, same input)")
print("=" * 70)
print(f"  PT loss: {loss_pt.item():.6f}")

# === DC forward + backward ===
x_dc = dc.Tensor(x_np.copy(), requires_grad=True)
y_dc = dc.Tensor(y_np.copy(), requires_grad=False)

# Forward
e1 = dc.conv2d(x_dc, enc1_w1, enc1_b1, padding=1); e1 = dc.relu(e1)
e1 = dc.conv2d(e1, enc1_w2, enc1_b2, padding=1); e1 = dc.relu(e1)
p1 = dc.maxpool2d(e1)
e2 = dc.conv2d(p1, enc2_w1, enc2_b1, padding=1); e2 = dc.relu(e2)
e2 = dc.conv2d(e2, enc2_w2, enc2_b2, padding=1); e2 = dc.relu(e2)
p2 = dc.maxpool2d(e2)
b = dc.conv2d(p2, bot_w1, bot_b1, padding=1); b = dc.relu(b)
b = dc.conv2d(b, bot_w2, bot_b2, padding=1); b = dc.relu(b)
u2 = dc.upsample2d(b)
c2 = dc.concat([u2, e2])
d2 = dc.conv2d(c2, up2_w1, up2_b1, padding=1); d2 = dc.relu(d2)
d2 = dc.conv2d(d2, up2_w2, up2_b2, padding=1); d2 = dc.relu(d2)
u1 = dc.upsample2d(d2)
c1 = dc.concat([u1, e1])
d1 = dc.conv2d(c1, up1_w1, up1_b1, padding=1); d1 = dc.relu(d1)
d1 = dc.conv2d(d1, up1_w2, up1_b2, padding=1); d1 = dc.relu(d1)
logits = dc.conv2d(d1, final_w, final_b, padding=0)
pred_dc = dc.sigmoid(logits)
loss_dc = dc.dice_loss(pred_dc, y_dc)
loss_val = loss_dc.sync()[0]
print(f"  DC loss: {loss_val:.6f}")

# Clear grads
for p in all_dc_params:
    if p.grad is not None:
        dc.lib.ClearBuffer(p.grad.gpu_buf)

loss_dc.backward()

# Compare gradients for each layer
pt_params = list(pt.parameters())
param_names = [
    "enc1.conv1.w", "enc1.conv1.b", "enc1.conv2.w", "enc1.conv2.b",
    "enc2.conv1.w", "enc2.conv1.b", "enc2.conv2.w", "enc2.conv2.b",
    "bot.conv1.w", "bot.conv1.b", "bot.conv2.w", "bot.conv2.b",
    "up2.conv1.w", "up2.conv1.b", "up2.conv2.w", "up2.conv2.b",
    "up1.conv1.w", "up1.conv1.b", "up1.conv2.w", "up1.conv2.b",
    "final.w", "final.b"
]

print(f"\n  {'Parameter':<20s} {'PT grad norm':>12s} {'DC grad norm':>12s} {'Max diff':>12s} {'Status':>8s}")
print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")

all_ok = True
for name, pt_p, dc_p in zip(param_names, pt_params, all_dc_params):
    pt_g = pt_p.grad.numpy()
    dc_g = dc_p.grad.sync()
    pt_norm = np.linalg.norm(pt_g)
    dc_norm = np.linalg.norm(dc_g)
    max_diff = np.max(np.abs(pt_g - dc_g))
    rel_diff = max_diff / (pt_norm + 1e-10)
    status = "OK" if rel_diff < 0.01 else "FAIL"
    if status == "FAIL":
        all_ok = False
    print(f"  {name:<20s} {pt_norm:>12.6f} {dc_norm:>12.6f} {max_diff:>12.6e} {status:>8s}")

dc.release_all_buffers()

# Also check dx (input gradient)
if x_dc.grad is not None:
    dc_dx = x_dc.grad.sync()
    pt_dx = x_pt.grad.numpy()
    dx_diff = np.max(np.abs(dc_dx - pt_dx))
    dx_norm = np.linalg.norm(pt_dx)
    print(f"\n  Input gradient (dx):")
    print(f"    PT norm: {dx_norm:.6f}, DC norm: {np.linalg.norm(dc_dx):.6f}, Max diff: {dx_diff:.6e}")

print(f"\n  {'ALL GRADIENTS MATCH' if all_ok else 'GRADIENT MISMATCH DETECTED!'}")
print(f"{'='*70}")
