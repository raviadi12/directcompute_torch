"""
Trace forward pass through UNet at base=16, comparing every intermediate
tensor between PyTorch and DirectCompute to find where divergence occurs.
"""
import numpy as np
import ctypes

BATCH_SIZE = 2
IMG_SIZE = 128
BASE = 16

def generate_weights():
    rng = np.random.RandomState(42)
    weights = {}
    def make_conv_block(name, in_c, out_c):
        scale = np.sqrt(2.0 / (in_c * 3 * 3))
        weights[f"{name}.w1"] = rng.randn(out_c, in_c, 3, 3).astype(np.float32) * scale
        weights[f"{name}.b1"] = np.zeros(out_c, dtype=np.float32)
        scale2 = np.sqrt(2.0 / (out_c * 3 * 3))
        weights[f"{name}.w2"] = rng.randn(out_c, out_c, 3, 3).astype(np.float32) * scale2
        weights[f"{name}.b2"] = np.zeros(out_c, dtype=np.float32)
    make_conv_block("enc1", 3, BASE)
    make_conv_block("enc2", BASE, BASE*2)
    make_conv_block("enc3", BASE*2, BASE*4)
    make_conv_block("bot", BASE*4, BASE*8)
    make_conv_block("up3", BASE*8 + BASE*4, BASE*4)
    make_conv_block("up2", BASE*4 + BASE*2, BASE*2)
    make_conv_block("up1", BASE*2 + BASE, BASE)
    scale_f = np.sqrt(2.0 / (BASE * 1 * 1))
    weights["final.w"] = rng.randn(1, BASE, 1, 1).astype(np.float32) * scale_f
    weights["final.b"] = np.zeros(1, dtype=np.float32)
    return weights

# Use fixed random input
rng = np.random.RandomState(123)
X_in = rng.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).astype(np.float32) * 0.5
W = generate_weights()

# ══════════════════════════════════════════════════════
# PyTorch forward with intermediate captures
# ══════════════════════════════════════════════════════
import torch
import torch.nn as tnn
import torch.nn.functional as F

class ConvBlock(tnn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = tnn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = tnn.Conv2d(out_c, out_c, 3, padding=1)
    def forward(self, x):
        x = self.conv1(x); x = F.relu(x)
        x = self.conv2(x); x = F.relu(x)
        return x

class UNet(tnn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, BASE)
        self.enc2 = ConvBlock(BASE, BASE*2)
        self.enc3 = ConvBlock(BASE*2, BASE*4)
        self.bot = ConvBlock(BASE*4, BASE*8)
        self.up3_conv = ConvBlock(BASE*8+BASE*4, BASE*4)
        self.up2_conv = ConvBlock(BASE*4+BASE*2, BASE*2)
        self.up1_conv = ConvBlock(BASE*2+BASE, BASE)
        self.final_conv = tnn.Conv2d(BASE, 1, 1)

pt_model = UNet()
block_map = [
    ("enc1", pt_model.enc1), ("enc2", pt_model.enc2), ("enc3", pt_model.enc3),
    ("bot", pt_model.bot),
    ("up3", pt_model.up3_conv), ("up2", pt_model.up2_conv), ("up1", pt_model.up1_conv),
]
with torch.no_grad():
    for name, block in block_map:
        block.conv1.weight.copy_(torch.from_numpy(W[f"{name}.w1"]))
        block.conv1.bias.copy_(torch.from_numpy(W[f"{name}.b1"]))
        block.conv2.weight.copy_(torch.from_numpy(W[f"{name}.w2"]))
        block.conv2.bias.copy_(torch.from_numpy(W[f"{name}.b2"]))
    pt_model.final_conv.weight.copy_(torch.from_numpy(W["final.w"]))
    pt_model.final_conv.bias.copy_(torch.from_numpy(W["final.b"]))

# Step-by-step forward
with torch.no_grad():
    x = torch.from_numpy(X_in)
    pt = {}
    e1 = pt_model.enc1(x); pt["e1"] = e1.numpy().copy()
    p1 = F.max_pool2d(e1, 2); pt["p1"] = p1.numpy().copy()
    e2 = pt_model.enc2(p1); pt["e2"] = e2.numpy().copy()
    p2 = F.max_pool2d(e2, 2); pt["p2"] = p2.numpy().copy()
    e3 = pt_model.enc3(p2); pt["e3"] = e3.numpy().copy()
    p3 = F.max_pool2d(e3, 2); pt["p3"] = p3.numpy().copy()
    b = pt_model.bot(p3); pt["bot"] = b.numpy().copy()
    u3 = F.interpolate(b, scale_factor=2, mode='nearest'); pt["u3"] = u3.numpy().copy()
    cat3 = torch.cat([u3, e3], 1); pt["cat3"] = cat3.numpy().copy()
    d3 = pt_model.up3_conv(cat3); pt["d3"] = d3.numpy().copy()
    u2 = F.interpolate(d3, scale_factor=2, mode='nearest'); pt["u2"] = u2.numpy().copy()
    cat2 = torch.cat([u2, e2], 1); pt["cat2"] = cat2.numpy().copy()
    d2 = pt_model.up2_conv(cat2); pt["d2"] = d2.numpy().copy()
    u1 = F.interpolate(d2, scale_factor=2, mode='nearest'); pt["u1"] = u1.numpy().copy()
    cat1 = torch.cat([u1, e1], 1); pt["cat1"] = cat1.numpy().copy()
    d1 = pt_model.up1_conv(cat1); pt["d1"] = d1.numpy().copy()
    logits = pt_model.final_conv(d1); pt["logits"] = logits.numpy().copy()
    out = torch.sigmoid(logits); pt["out"] = out.numpy().copy()

# ══════════════════════════════════════════════════════
# DirectCompute forward with intermediate captures
# ══════════════════════════════════════════════════════
import nn_engine as nn

class DCConvBlock:
    def __init__(self, name):
        self.w1 = nn.Tensor(W[f"{name}.w1"].copy(), requires_grad=False, track=False)
        self.b1 = nn.Tensor(W[f"{name}.b1"].copy(), requires_grad=False, track=False)
        self.w2 = nn.Tensor(W[f"{name}.w2"].copy(), requires_grad=False, track=False)
        self.b2 = nn.Tensor(W[f"{name}.b2"].copy(), requires_grad=False, track=False)
    def __call__(self, x):
        x = nn.conv2d(x, self.w1, self.b1, padding=1); x = nn.relu(x)
        x = nn.conv2d(x, self.w2, self.b2, padding=1); x = nn.relu(x)
        return x

enc1 = DCConvBlock("enc1")
enc2 = DCConvBlock("enc2")
enc3 = DCConvBlock("enc3")
bot = DCConvBlock("bot")
up3_conv = DCConvBlock("up3")
up2_conv = DCConvBlock("up2")
up1_conv = DCConvBlock("up1")
final_w = nn.Tensor(W["final.w"].copy(), requires_grad=False, track=False)
final_b = nn.Tensor(W["final.b"].copy(), requires_grad=False, track=False)

x_t = nn.Tensor(X_in.copy(), requires_grad=False)
dc = {}
e1_dc = enc1(x_t); dc["e1"] = e1_dc.sync().copy()
p1_dc = nn.maxpool2d(e1_dc, 2, 2); dc["p1"] = p1_dc.sync().copy()
e2_dc = enc2(p1_dc); dc["e2"] = e2_dc.sync().copy()
p2_dc = nn.maxpool2d(e2_dc, 2, 2); dc["p2"] = p2_dc.sync().copy()
e3_dc = enc3(p2_dc); dc["e3"] = e3_dc.sync().copy()
p3_dc = nn.maxpool2d(e3_dc, 2, 2); dc["p3"] = p3_dc.sync().copy()
b_dc = bot(p3_dc); dc["bot"] = b_dc.sync().copy()
u3_dc = nn.upsample2d(b_dc, 2, 2); dc["u3"] = u3_dc.sync().copy()
cat3_dc = nn.concat([u3_dc, e3_dc], axis=1); dc["cat3"] = cat3_dc.sync().copy()
d3_dc = up3_conv(cat3_dc); dc["d3"] = d3_dc.sync().copy()
u2_dc = nn.upsample2d(d3_dc, 2, 2); dc["u2"] = u2_dc.sync().copy()
cat2_dc = nn.concat([u2_dc, e2_dc], axis=1); dc["cat2"] = cat2_dc.sync().copy()
d2_dc = up2_conv(cat2_dc); dc["d2"] = d2_dc.sync().copy()
u1_dc = nn.upsample2d(d2_dc, 2, 2); dc["u1"] = u1_dc.sync().copy()
cat1_dc = nn.concat([u1_dc, e1_dc], axis=1); dc["cat1"] = cat1_dc.sync().copy()
d1_dc = up1_conv(cat1_dc); dc["d1"] = d1_dc.sync().copy()
logits_dc = nn.conv2d(d1_dc, final_w, final_b, padding=0); dc["logits"] = logits_dc.sync().copy()
out_dc = nn.sigmoid(logits_dc); dc["out"] = out_dc.sync().copy()

# ══════════════════════════════════════════════════════
# Compare
# ══════════════════════════════════════════════════════
print("=" * 100)
print(f"{'Layer':<10} {'Shape':>20} | {'MaxDiff':>12} {'MeanDiff':>12} {'RelMax':>12} | {'PT range':>20} {'DC range':>20}")
print("-" * 100)

for key in ["e1", "p1", "e2", "p2", "e3", "p3", "bot", "u3", "cat3", "d3",
            "u2", "cat2", "d2", "u1", "cat1", "d1", "logits", "out"]:
    p = pt[key]
    d = dc[key]
    diff = np.abs(p - d)
    rel = diff.max() / (np.abs(p).max() + 1e-10)
    pt_range = f"[{p.min():.4f}, {p.max():.4f}]"
    dc_range = f"[{d.min():.4f}, {d.max():.4f}]"
    flag = " <-- LARGE" if diff.max() > 0.001 else ""
    print(f"{key:<10} {str(p.shape):>20} | {diff.max():>12.8f} {diff.mean():>12.8f} {rel:>12.8f} | {pt_range:>20} {dc_range:>20}{flag}")

nn.release_all_buffers()
