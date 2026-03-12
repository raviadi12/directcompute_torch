"""
Minimal test: one batch from the actual dataset, initial weights,
compare forward pass between PyTorch and DirectCompute.
"""
import os, glob, numpy as np
import PIL.Image as Image

IMG_SIZE = 128
BASE = 16
BATCH_SIZE = 2

# Load dataset
img_dir = os.path.join("segment", "png_images", "IMAGES")
mask_dir = os.path.join("segment", "png_masks", "MASKS")
img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")) + glob.glob(os.path.join(mask_dir, "*.jpg")))
X, Y = [], []
for i in range(min(50, len(img_paths))):  # only load 50 for speed
    img = Image.open(img_paths[i]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1))
    mask = Image.open(mask_paths[i]).convert("L").resize((IMG_SIZE, IMG_SIZE))
    m = np.where(np.expand_dims(np.array(mask, dtype=np.float32) / 255.0, 0) > 0.01, 1.0, 0.0).astype(np.float32)
    X.append(arr); Y.append(m)
X = np.stack(X); Y = np.stack(Y)
print(f"Loaded {X.shape[0]} samples, X range: [{X.min():.4f}, {X.max():.4f}]")

# Same batch indices as comparison
np.random.seed(0)
indices = np.random.permutation(X.shape[0])
batch_idx = indices[:BATCH_SIZE]
x_batch = X[batch_idx]
y_batch = Y[batch_idx]
print(f"Batch indices: {batch_idx}")
print(f"x_batch shape: {x_batch.shape}, range: [{x_batch.min():.4f}, {x_batch.max():.4f}]")
print(f"y_batch shape: {y_batch.shape}, unique: {np.unique(y_batch)}")

# Generate weights
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

W = generate_weights()

# ── PyTorch ──
import torch
import torch.nn as tnn
import torch.nn.functional as F

class ConvBlock(tnn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = tnn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = tnn.Conv2d(out_c, out_c, 3, padding=1)
    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))))

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
    def forward(self, x):
        e1 = self.enc1(x); p1 = F.max_pool2d(e1, 2)
        e2 = self.enc2(p1); p2 = F.max_pool2d(e2, 2)
        e3 = self.enc3(p2); p3 = F.max_pool2d(e3, 2)
        b = self.bot(p3)
        u3 = F.interpolate(b, scale_factor=2, mode='nearest')
        d3 = self.up3_conv(torch.cat([u3, e3], 1))
        u2 = F.interpolate(d3, scale_factor=2, mode='nearest')
        d2 = self.up2_conv(torch.cat([u2, e2], 1))
        u1 = F.interpolate(d2, scale_factor=2, mode='nearest')
        d1 = self.up1_conv(torch.cat([u1, e1], 1))
        return torch.sigmoid(self.final_conv(d1))

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

with torch.no_grad():
    pt_pred = pt_model(torch.from_numpy(x_batch)).numpy()

print(f"\nPyTorch pred shape: {pt_pred.shape}")
print(f"  range: [{pt_pred.min():.6f}, {pt_pred.max():.6f}], mean: {pt_pred.mean():.6f}")

# ── DirectCompute ──
import nn_engine as nn
import ctypes

class DCConvBlock:
    def __init__(self, name):
        self.w1 = nn.Tensor(W[f"{name}.w1"].copy(), requires_grad=True, track=False)
        self.b1 = nn.Tensor(W[f"{name}.b1"].copy(), requires_grad=True, track=False)
        self.w2 = nn.Tensor(W[f"{name}.w2"].copy(), requires_grad=True, track=False)
        self.b2 = nn.Tensor(W[f"{name}.b2"].copy(), requires_grad=True, track=False)
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
final_w = nn.Tensor(W["final.w"].copy(), requires_grad=True, track=False)
final_b = nn.Tensor(W["final.b"].copy(), requires_grad=True, track=False)

x_t = nn.Tensor(x_batch.copy(), requires_grad=False)
e1 = enc1(x_t); p1 = nn.maxpool2d(e1, 2, 2)
e2 = enc2(p1); p2 = nn.maxpool2d(e2, 2, 2)
e3 = enc3(p2); p3 = nn.maxpool2d(e3, 2, 2)
b = bot(p3)
u3 = nn.upsample2d(b, 2, 2); d3 = up3_conv(nn.concat([u3, e3], axis=1))
u2 = nn.upsample2d(d3, 2, 2); d2 = up2_conv(nn.concat([u2, e2], axis=1))
u1 = nn.upsample2d(d2, 2, 2); d1 = up1_conv(nn.concat([u1, e1], axis=1))
logits = nn.conv2d(d1, final_w, final_b, padding=0)
out = nn.sigmoid(logits)
dc_pred = out.sync().copy()

print(f"\nDirectCompute pred shape: {dc_pred.shape}")
print(f"  range: [{dc_pred.min():.6f}, {dc_pred.max():.6f}], mean: {dc_pred.mean():.6f}")

# ── Compare ──
diff = np.abs(pt_pred - dc_pred)
print(f"\nComparison:")
print(f"  Max diff:  {diff.max():.8f}")
print(f"  Mean diff: {diff.mean():.8f}")
print(f"  Matching (diff < 1e-5): {(diff < 1e-5).sum()} / {diff.size}")

# Also compare loss
pt_loss_fn = lambda pred, target: torch.mean(1.0 - (2.0 * torch.sum(pred * target, dim=[1,2,3])) / (torch.sum(pred, dim=[1,2,3]) + torch.sum(target, dim=[1,2,3]) + 1e-6))
with torch.no_grad():
    pt_loss = pt_loss_fn(torch.from_numpy(pt_pred), torch.from_numpy(y_batch)).item()
# For DC loss, compute from numpy for comparison
inter = np.sum(dc_pred * y_batch, axis=(1,2,3))
sp = np.sum(dc_pred, axis=(1,2,3))
st = np.sum(y_batch, axis=(1,2,3))
dc_dice = 2 * inter / (sp + st + 1e-6)
dc_loss = np.mean(1 - dc_dice)
print(f"\nPT loss: {pt_loss:.8f}")
print(f"DC loss (numpy-computed): {dc_loss:.8f}")
print(f"Loss diff: {abs(pt_loss - dc_loss):.8f}")

nn.release_all_buffers()
