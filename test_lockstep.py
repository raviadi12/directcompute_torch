"""
Step-by-step debugging: run both engines side by side for 3 steps,
comparing at every sub-step (forward, backward, clip, update).
"""
import os, glob, numpy as np, ctypes
import PIL.Image as Image

IMG_SIZE = 128
BASE = 16
BATCH_SIZE = 2
LR = 1e-3

# Load dataset (full 1000)
img_dir = os.path.join("segment", "png_images", "IMAGES")
mask_dir = os.path.join("segment", "png_masks", "MASKS")
img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")) + glob.glob(os.path.join(mask_dir, "*.jpg")))
X, Y = [], []
for img_p, mask_p in zip(img_paths, mask_paths):
    img = Image.open(img_p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1))
    mask = Image.open(mask_p).convert("L").resize((IMG_SIZE, IMG_SIZE))
    m = np.where(np.expand_dims(np.array(mask, dtype=np.float32) / 255.0, 0) > 0.01, 1.0, 0.0).astype(np.float32)
    X.append(arr); Y.append(m)
X = np.stack(X); Y = np.stack(Y)
print(f"Loaded {X.shape[0]} samples")

np.random.seed(0)
indices = np.random.permutation(X.shape[0])
batches = []
for i in range(3):
    batches.append(indices[i * BATCH_SIZE : (i + 1) * BATCH_SIZE])

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

# ═══ Set up PyTorch ═══
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

class DiceLoss(tnn.Module):
    def forward(self, pred, target):
        inter = torch.sum(pred * target, dim=[1, 2, 3])
        sp = torch.sum(pred, dim=[1, 2, 3])
        st = torch.sum(target, dim=[1, 2, 3])
        dice = (2.0 * inter) / (sp + st + 1e-6)
        return torch.mean(1.0 - dice)

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
pt_criterion = DiceLoss()
pt_optimizer = torch.optim.SGD(pt_model.parameters(), lr=LR)

# ═══ Set up DirectCompute ═══
import nn_engine as nn

class DCConvBlock:
    def __init__(self, name):
        self.w1 = nn.Tensor(W[f"{name}.w1"].copy(), requires_grad=True, track=False)
        self.b1 = nn.Tensor(W[f"{name}.b1"].copy(), requires_grad=True, track=False)
        self.w2 = nn.Tensor(W[f"{name}.w2"].copy(), requires_grad=True, track=False)
        self.b2 = nn.Tensor(W[f"{name}.b2"].copy(), requires_grad=True, track=False)
    def params_dict(self):
        return [("w1", self.w1), ("b1", self.b1), ("w2", self.w2), ("b2", self.b2)]
    def __call__(self, x):
        x = nn.conv2d(x, self.w1, self.b1, padding=1); x = nn.relu(x)
        x = nn.conv2d(x, self.w2, self.b2, padding=1); x = nn.relu(x)
        return x

dc_blocks = {}
for name in ["enc1", "enc2", "enc3", "bot", "up3", "up2", "up1"]:
    dc_blocks[name] = DCConvBlock(name)
dc_final_w = nn.Tensor(W["final.w"].copy(), requires_grad=True, track=False)
dc_final_b = nn.Tensor(W["final.b"].copy(), requires_grad=True, track=False)

dc_all_params = []
dc_param_names = []
for name in ["enc1", "enc2", "enc3", "bot", "up3", "up2", "up1"]:
    for pname, p in dc_blocks[name].params_dict():
        dc_all_params.append(p)
        dc_param_names.append(f"{name}.{pname}")
dc_all_params.append(dc_final_w); dc_param_names.append("final.w")
dc_all_params.append(dc_final_b); dc_param_names.append("final.b")

def dc_forward(x_np):
    x_t = nn.Tensor(x_np.copy(), requires_grad=False)
    e1 = dc_blocks["enc1"](x_t); p1 = nn.maxpool2d(e1, 2, 2)
    e2 = dc_blocks["enc2"](p1); p2 = nn.maxpool2d(e2, 2, 2)
    e3 = dc_blocks["enc3"](p2); p3 = nn.maxpool2d(e3, 2, 2)
    b = dc_blocks["bot"](p3)
    u3 = nn.upsample2d(b, 2, 2); d3 = dc_blocks["up3"](nn.concat([u3, e3], axis=1))
    u2 = nn.upsample2d(d3, 2, 2); d2 = dc_blocks["up2"](nn.concat([u2, e2], axis=1))
    u1 = nn.upsample2d(d2, 2, 2); d1 = dc_blocks["up1"](nn.concat([u1, e1], axis=1))
    logits = nn.conv2d(d1, dc_final_w, dc_final_b, padding=0)
    return nn.sigmoid(logits)

def get_pt_param(name):
    for n, block in block_map:
        if name == f"{n}.w1": return block.conv1.weight
        if name == f"{n}.b1": return block.conv1.bias
        if name == f"{n}.w2": return block.conv2.weight
        if name == f"{n}.b2": return block.conv2.bias
    if name == "final.w": return pt_model.final_conv.weight
    if name == "final.b": return pt_model.final_conv.bias

# ═══ Verify initial weights match ═══
print("\n=== Initial weight comparison ===")
max_w_diff = 0
for i, name in enumerate(dc_param_names):
    pt_w = get_pt_param(name).detach().numpy()
    dc_w = dc_all_params[i].sync()
    d = np.abs(pt_w - dc_w).max()
    max_w_diff = max(max_w_diff, d)
print(f"Max initial weight diff: {max_w_diff}")

# ═══ Run 3 steps side-by-side ═══
for step_i in range(3):
    bidx = batches[step_i]
    x_np = X[bidx]
    y_np = Y[bidx]
    
    print(f"\n{'='*80}")
    print(f"STEP {step_i}")
    print(f"{'='*80}")
    
    # ── Check weights before forward ──
    w_diffs = []
    for i, name in enumerate(dc_param_names):
        pt_w = get_pt_param(name).detach().numpy()
        dc_w = dc_all_params[i].sync()
        d = np.abs(pt_w - dc_w).max()
        w_diffs.append(d)
    max_wd = max(w_diffs)
    print(f"  Weight diff before forward: max={max_wd:.10f}")
    
    # ── PyTorch forward ──
    pt_pred = pt_model(torch.from_numpy(x_np))
    pt_loss = pt_criterion(pt_pred, torch.from_numpy(y_np))
    pt_pred_np = pt_pred.detach().numpy().copy()
    pt_loss_val = pt_loss.item()
    
    # ── DC forward ──
    dc_out = dc_forward(x_np)
    dc_loss = nn.dice_loss(dc_out, nn.Tensor(y_np.copy(), requires_grad=False))
    dc_pred_np = dc_out.sync().copy()
    dc_loss_vals = dc_loss.sync().copy()
    dc_loss_mean = float(np.mean(dc_loss_vals))
    
    pred_diff = np.abs(pt_pred_np - dc_pred_np)
    print(f"  Forward pred diff: max={pred_diff.max():.10f} mean={pred_diff.mean():.10f}")
    print(f"  Loss: PT={pt_loss_val:.8f}  DC_mean={dc_loss_mean:.8f}  diff={abs(pt_loss_val-dc_loss_mean):.8f}")
    
    # ── PyTorch backward ──
    pt_optimizer.zero_grad()
    pt_loss.backward()
    
    # Collect PT grads before clip
    pt_norm_sq = 0
    pt_grad_dict = {}
    for name in dc_param_names:
        g = get_pt_param(name).grad.numpy().copy()
        pt_grad_dict[name] = g
        pt_norm_sq += np.sum(g**2)
    pt_norm = np.sqrt(pt_norm_sq)
    
    # ── DC backward ──
    for p in dc_all_params:
        if p.grad is not None and p.grad.gpu_buf is not None:
            nn.lib.ClearBuffer(p.grad.gpu_buf)
    dc_loss.backward()
    
    # Collect DC grads before clip
    dc_norm_sq = 0
    dc_grad_dict = {}
    for i, name in enumerate(dc_param_names):
        g = dc_all_params[i].grad.sync().copy()
        dc_grad_dict[name] = g
        dc_norm_sq += np.sum(g**2)
    dc_norm = np.sqrt(dc_norm_sq)
    
    print(f"  Grad norm (pre-clip): PT={pt_norm:.8f}  DC={dc_norm:.8f}  diff={abs(pt_norm-dc_norm):.8f}")
    
    # Show gradient diff for each param
    grad_diffs = []
    for name in dc_param_names:
        gd = np.abs(pt_grad_dict[name] - dc_grad_dict[name]).max()
        grad_diffs.append((name, gd))
    grad_diffs.sort(key=lambda x: -x[1])
    print(f"  Top 5 grad diffs:")
    for name, d in grad_diffs[:5]:
        pg = pt_grad_dict[name]
        dg = dc_grad_dict[name]
        print(f"    {name:<15} maxdiff={d:.10f}  PT_max={np.abs(pg).max():.8f}  DC_max={np.abs(dg).max():.8f}")
    
    # ── PT clip + step ──
    pt_clip_norm = torch.nn.utils.clip_grad_norm_(pt_model.parameters(), max_norm=1.0)
    pt_optimizer.step()
    
    # ── DC clip + step ──
    dc_clip_norm = nn.clip_grad_norm(dc_all_params, max_norm=1.0)
    num_params = len(dc_all_params)
    params_arr = (ctypes.c_void_p * num_params)(*[p.gpu_buf for p in dc_all_params])
    grads_arr = (ctypes.c_void_p * num_params)(*[p.grad.gpu_buf for p in dc_all_params])
    sizes_arr = (ctypes.c_uint * num_params)(*[p.size for p in dc_all_params])
    nn.lib.SGDBatch(params_arr, grads_arr, sizes_arr, num_params, LR, 1e30)
    
    print(f"  Clip norm: PT={float(pt_clip_norm):.8f}  DC={dc_clip_norm:.8f}")
    
    # ── Weights after update ──
    w_diffs_after = []
    for i, name in enumerate(dc_param_names):
        pt_w = get_pt_param(name).detach().numpy()
        dc_w = dc_all_params[i].sync()
        d = np.abs(pt_w - dc_w).max()
        w_diffs_after.append((name, d))
    w_diffs_after.sort(key=lambda x: -x[1])
    print(f"  Weight diff after update: max={w_diffs_after[0][1]:.10f}")
    for name, d in w_diffs_after[:3]:
        print(f"    {name:<15} {d:.10f}")
    
    # Cleanup DC tracked tensors
    nn.release_all_buffers()
    nn.lib.FlushGPU()

print("\nDone!")
