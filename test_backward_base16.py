"""
Test backward pass with real data at base=16.
Compare gradients between PyTorch and DirectCompute.
"""
import os, glob, numpy as np
import PIL.Image as Image

IMG_SIZE = 128
BASE = 16
BATCH_SIZE = 2

# Load minimal dataset
img_dir = os.path.join("segment", "png_images", "IMAGES")
mask_dir = os.path.join("segment", "png_masks", "MASKS")
img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")) + glob.glob(os.path.join(mask_dir, "*.jpg")))
X, Y = [], []
for i in range(min(50, len(img_paths))):
    img = Image.open(img_paths[i]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1))
    mask = Image.open(mask_paths[i]).convert("L").resize((IMG_SIZE, IMG_SIZE))
    m = np.where(np.expand_dims(np.array(mask, dtype=np.float32) / 255.0, 0) > 0.01, 1.0, 0.0).astype(np.float32)
    X.append(arr); Y.append(m)
X = np.stack(X); Y = np.stack(Y)

np.random.seed(0)
indices = np.random.permutation(X.shape[0])
batch_idx = indices[:BATCH_SIZE]
x_batch = X[batch_idx].copy()
y_batch = Y[batch_idx].copy()
print(f"Batch indices: {batch_idx}")

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

# ═══ PyTorch with gradients ═══
import torch, torch.nn as tnn, torch.nn.functional as F

class PTConvBlock(tnn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = tnn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = tnn.Conv2d(out_c, out_c, 3, padding=1)
    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))))

class PTUNet(tnn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = PTConvBlock(3, BASE)
        self.enc2 = PTConvBlock(BASE, BASE*2)
        self.enc3 = PTConvBlock(BASE*2, BASE*4)
        self.bot = PTConvBlock(BASE*4, BASE*8)
        self.up3_conv = PTConvBlock(BASE*8+BASE*4, BASE*4)
        self.up2_conv = PTConvBlock(BASE*4+BASE*2, BASE*2)
        self.up1_conv = PTConvBlock(BASE*2+BASE, BASE)
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

pt_model = PTUNet()
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

class DiceLoss(tnn.Module):
    def forward(self, pred, target):
        intersection = torch.sum(pred * target, dim=[1, 2, 3])
        sum_pred = torch.sum(pred, dim=[1, 2, 3])
        sum_target = torch.sum(target, dim=[1, 2, 3])
        dice = (2.0 * intersection) / (sum_pred + sum_target + 1e-6)
        return torch.mean(1.0 - dice)

pt_pred = pt_model(torch.from_numpy(x_batch))
pt_loss = DiceLoss()(pt_pred, torch.from_numpy(y_batch))
pt_loss.backward()

print("\n=== PyTorch Gradients ===")
pt_grads = {}
for name, block in block_map:
    for pname, param in [("w1", block.conv1.weight), ("b1", block.conv1.bias),
                          ("w2", block.conv2.weight), ("b2", block.conv2.bias)]:
        key = f"{name}.{pname}"
        g = param.grad.numpy()
        pt_grads[key] = g.copy()
key = "final.w"; pt_grads[key] = pt_model.final_conv.weight.grad.numpy().copy()
key = "final.b"; pt_grads[key] = pt_model.final_conv.bias.grad.numpy().copy()

pt_total_norm = 0.0
for k, g in pt_grads.items():
    pt_total_norm += np.sum(g ** 2)
pt_total_norm = np.sqrt(pt_total_norm)
print(f"PT grad norm: {pt_total_norm:.6f}")
print(f"PT loss: {pt_loss.item():.8f}")

# ═══ DirectCompute with gradients ═══
import nn_engine as nn
import ctypes

class DCConvBlock:
    def __init__(self, name):
        self.w1 = nn.Tensor(W[f"{name}.w1"].copy(), requires_grad=True, track=False)
        self.b1 = nn.Tensor(W[f"{name}.b1"].copy(), requires_grad=True, track=False)
        self.w2 = nn.Tensor(W[f"{name}.w2"].copy(), requires_grad=True, track=False)
        self.b2 = nn.Tensor(W[f"{name}.b2"].copy(), requires_grad=True, track=False)
    def params(self):
        return [("w1", self.w1), ("b1", self.b1), ("w2", self.w2), ("b2", self.b2)]
    def __call__(self, x):
        x = nn.conv2d(x, self.w1, self.b1, padding=1); x = nn.relu(x)
        x = nn.conv2d(x, self.w2, self.b2, padding=1); x = nn.relu(x)
        return x

dc_blocks = {
    "enc1": DCConvBlock("enc1"), "enc2": DCConvBlock("enc2"), "enc3": DCConvBlock("enc3"),
    "bot": DCConvBlock("bot"),
    "up3": DCConvBlock("up3"), "up2": DCConvBlock("up2"), "up1": DCConvBlock("up1"),
}
final_w = nn.Tensor(W["final.w"].copy(), requires_grad=True, track=False)
final_b = nn.Tensor(W["final.b"].copy(), requires_grad=True, track=False)

x_t = nn.Tensor(x_batch.copy(), requires_grad=False)
y_t = nn.Tensor(y_batch.copy(), requires_grad=False)

e1 = dc_blocks["enc1"](x_t); p1 = nn.maxpool2d(e1, 2, 2)
e2 = dc_blocks["enc2"](p1); p2 = nn.maxpool2d(e2, 2, 2)
e3 = dc_blocks["enc3"](p2); p3 = nn.maxpool2d(e3, 2, 2)
b = dc_blocks["bot"](p3)
u3 = nn.upsample2d(b, 2, 2); d3 = dc_blocks["up3"](nn.concat([u3, e3], axis=1))
u2 = nn.upsample2d(d3, 2, 2); d2 = dc_blocks["up2"](nn.concat([u2, e2], axis=1))
u1 = nn.upsample2d(d2, 2, 2); d1 = dc_blocks["up1"](nn.concat([u1, e1], axis=1))
logits = nn.conv2d(d1, final_w, final_b, padding=0)
out = nn.sigmoid(logits)
loss = nn.dice_loss(out, y_t)

dc_loss = float(loss.sync()[0])
loss.backward()

print("\n=== DirectCompute Gradients ===")
dc_grads = {}
for name, block in dc_blocks.items():
    for pname, param in block.params():
        key = f"{name}.{pname}"
        dc_grads[key] = param.grad.sync().copy()
dc_grads["final.w"] = final_w.grad.sync().copy()
dc_grads["final.b"] = final_b.grad.sync().copy()

dc_total_norm = 0.0
for k, g in dc_grads.items():
    dc_total_norm += np.sum(g ** 2)
dc_total_norm = np.sqrt(dc_total_norm)
print(f"DC grad norm: {dc_total_norm:.6f}")
print(f"DC loss[0]: {dc_loss:.8f}")

# ═══ Compare gradients ═══
print("\n" + "=" * 100)
print(f"{'Param':<15} {'Shape':>18} | {'MaxDiff':>12} {'MeanDiff':>12} {'RelMax':>12} | {'PT norm':>10} {'DC norm':>10}")
print("-" * 100)

for key in sorted(pt_grads.keys()):
    pg = pt_grads[key]
    dg = dc_grads[key]
    diff = np.abs(pg - dg)
    rel = diff.max() / (np.abs(pg).max() + 1e-10)
    pn = np.sqrt(np.sum(pg**2))
    dn = np.sqrt(np.sum(dg**2))
    flag = " <--" if diff.max() > 1e-4 else ""
    print(f"{key:<15} {str(pg.shape):>18} | {diff.max():>12.8f} {diff.mean():>12.8f} {rel:>12.8f} | {pn:>10.6f} {dn:>10.6f}{flag}")

print(f"\nTotal grad norm: PT={pt_total_norm:.6f}  DC={dc_total_norm:.6f}  diff={abs(pt_total_norm-dc_total_norm):.6f}")

nn.release_all_buffers()
