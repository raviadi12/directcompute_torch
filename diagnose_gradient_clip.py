"""
Diagnose: Check gradient magnitudes and clipping impact in UNet training.
Tests if element-wise clipping at 1.0 distorts gradients vs PyTorch norm clipping.
"""
import numpy as np
import os, glob, ctypes
import PIL.Image as Image
import nn_engine as nn

IMG_SIZE = 128
BATCH_SIZE = 2

def load_dataset(dataset_dir="segment"):
    img_dir = os.path.join(dataset_dir, "png_images", "IMAGES")
    mask_dir = os.path.join(dataset_dir, "png_masks", "MASKS")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")) + glob.glob(os.path.join(mask_dir, "*.jpg")))
    X, Y = [], []
    for img_p, mask_p in zip(img_paths, mask_paths):
        img = Image.open(img_p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1))
        mask = Image.open(mask_p).convert("L").resize((IMG_SIZE, IMG_SIZE))
        m = np.expand_dims(np.array(mask, dtype=np.float32) / 255.0, 0)
        m = np.where(m > 0.01, 1.0, 0.0).astype(np.float32)
        X.append(arr); Y.append(m)
    return np.stack(X), np.stack(Y)

class ConvBlock:
    def __init__(self, in_c, out_c, rng):
        scale = np.sqrt(2.0 / (in_c * 3 * 3))
        self.w1 = nn.Tensor(rng.randn(out_c, in_c, 3, 3).astype(np.float32) * scale, requires_grad=True, track=False)
        self.b1 = nn.Tensor(np.zeros((out_c,), dtype=np.float32), requires_grad=True, track=False)
        scale2 = np.sqrt(2.0 / (out_c * 3 * 3))
        self.w2 = nn.Tensor(rng.randn(out_c, out_c, 3, 3).astype(np.float32) * scale2, requires_grad=True, track=False)
        self.b2 = nn.Tensor(np.zeros((out_c,), dtype=np.float32), requires_grad=True, track=False)
    def parameters(self): return [self.w1, self.b1, self.w2, self.b2]
    def __call__(self, x):
        x = nn.conv2d(x, self.w1, self.b1, padding=1); x = nn.relu(x)
        x = nn.conv2d(x, self.w2, self.b2, padding=1); x = nn.relu(x)
        return x

class UNet:
    def __init__(self):
        rng = np.random.RandomState(42)
        base = 16
        self.enc1 = ConvBlock(3, base, rng)
        self.enc2 = ConvBlock(base, base * 2, rng)
        self.enc3 = ConvBlock(base * 2, base * 4, rng)
        self.bot = ConvBlock(base * 4, base * 8, rng)
        self.up3_conv = ConvBlock(base * 8 + base * 4, base * 4, rng)
        self.up2_conv = ConvBlock(base * 4 + base * 2, base * 2, rng)
        self.up1_conv = ConvBlock(base * 2 + base, base, rng)
        scale = np.sqrt(2.0 / (base * 1 * 1))
        self.final_w = nn.Tensor(rng.randn(1, base, 1, 1).astype(np.float32) * scale, requires_grad=True, track=False)
        self.final_b = nn.Tensor(np.zeros((1,), dtype=np.float32), requires_grad=True, track=False)
        self.params = (
            self.enc1.parameters() + self.enc2.parameters() + self.enc3.parameters() +
            self.bot.parameters() +
            self.up3_conv.parameters() + self.up2_conv.parameters() + self.up1_conv.parameters() +
            [self.final_w, self.final_b]
        )
    def forward(self, x):
        e1 = self.enc1(x); p1 = nn.maxpool2d(e1, 2, 2)
        e2 = self.enc2(p1); p2 = nn.maxpool2d(e2, 2, 2)
        e3 = self.enc3(p2); p3 = nn.maxpool2d(e3, 2, 2)
        b = self.bot(p3)
        u3 = nn.upsample2d(b, 2, 2); d3 = self.up3_conv(nn.concat([u3, e3], axis=1))
        u2 = nn.upsample2d(d3, 2, 2); d2 = self.up2_conv(nn.concat([u2, e2], axis=1))
        u1 = nn.upsample2d(d2, 2, 2); d1 = self.up1_conv(nn.concat([u1, e1], axis=1))
        logits = nn.conv2d(d1, self.final_w, self.final_b, padding=0)
        return nn.sigmoid(logits)

# Use synthetic data for speed (avoid loading 1000 images)
rng_data = np.random.RandomState(0)
x_batch = rng_data.rand(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
y_batch = np.zeros((BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
for i in range(BATCH_SIZE):
    # Circle mask
    for h in range(IMG_SIZE):
        for w in range(IMG_SIZE):
            if ((h-64)**2 + (w-64)**2) < 40**2:
                y_batch[i, 0, h, w] = 1.0

model = UNet()

x_t = nn.Tensor(x_batch, requires_grad=False)
y_t = nn.Tensor(y_batch, requires_grad=False)
preds = model.forward(x_t)
loss = nn.dice_loss(preds, y_t)
loss_val = loss.sync()
print(f"Loss: {loss_val}")

# Clear grads
for p in model.params:
    if p.grad is not None:
        nn.lib.ClearBuffer(p.grad.gpu_buf)

loss.backward()

# Analyze gradient magnitudes
print("\n" + "=" * 80)
print("GRADIENT ANALYSIS")
print("=" * 80)
print(f"{'Layer':<30} {'Shape':<20} {'Max|g|':>10} {'Mean|g|':>10} {'Std':>10} {'%>1.0':>8} {'%>0.1':>8}")
print("-" * 100)

total_elements = 0
total_clipped = 0
all_grads = []

names = []
for i, block_name in enumerate(["enc1", "enc2", "enc3", "bot", "up3_conv", "up2_conv", "up1_conv"]):
    block = getattr(model, block_name)
    for j, (p, suffix) in enumerate(zip(block.parameters(), ["w1", "b1", "w2", "b2"])):
        names.append(f"{block_name}.{suffix}")
names.extend(["final_w", "final_b"])

for name, p in zip(names, model.params):
    g = p.grad.sync().copy()
    all_grads.append(g.ravel())
    abs_g = np.abs(g)
    max_g = abs_g.max()
    mean_g = abs_g.mean()
    std_g = g.std()
    pct_clip = 100.0 * np.mean(abs_g > 1.0)
    pct_large = 100.0 * np.mean(abs_g > 0.1)
    total_elements += g.size
    total_clipped += np.sum(abs_g > 1.0)
    print(f"{name:<30} {str(p.shape):<20} {max_g:>10.4f} {mean_g:>10.6f} {std_g:>10.6f} {pct_clip:>7.1f}% {pct_large:>7.1f}%")

all_grads_flat = np.concatenate(all_grads)
global_norm = np.sqrt(np.sum(all_grads_flat**2))

print("-" * 100)
print(f"\nGlobal gradient norm: {global_norm:.4f}")
print(f"Total parameters: {total_elements}")
print(f"Elements with |grad| > 1.0: {total_clipped} ({100*total_clipped/total_elements:.1f}%)")

# Simulate what PyTorch clip_grad_norm_ would do
if global_norm > 1.0:
    scale = 1.0 / global_norm
    print(f"\nPyTorch norm clipping would SCALE all grads by {scale:.6f}")
    print(f"  Effective LR with PyTorch: {1e-3 * scale:.6f}")
    print(f"  Effective LR with DirectCompute element-wise clip: ~{1e-3:.6f}")
    print(f"  RATIO: DirectCompute grads are {1/scale:.1f}x larger than PyTorch!")
else:
    print(f"\nPyTorch norm clipping would NOT trigger (norm {global_norm:.4f} <= 1.0)")

# Check: what fraction of the gradient signal is lost by element-wise clipping?
clipped_grads = np.clip(all_grads_flat, -1.0, 1.0)
original_norm = np.sqrt(np.sum(all_grads_flat**2))
clipped_norm = np.sqrt(np.sum(clipped_grads**2))
cos_sim = np.dot(all_grads_flat, clipped_grads) / (original_norm * clipped_norm + 1e-12)
print(f"\nCosine similarity between original and element-clipped gradients: {cos_sim:.6f}")
print(f"  (1.0 = perfect preservation, <1.0 = direction distorted)")

nn.release_all_buffers()
