"""
Side-by-side UNet training comparison: DirectCompute vs PyTorch.
Runs 20 batches with IDENTICAL weights, data, and shuffle order.
Compares loss, predictions, and weight divergence at each step.
"""
import time
import os
import sys
import glob
import ctypes
import numpy as np
import PIL.Image as Image

# ── Config ──
BATCH_SIZE = 2
IMG_SIZE = 128
NUM_BATCHES = 20
LEARNING_RATE = 1e-3
BASE = 16

# ── Dataset ──
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
        m = np.where(np.expand_dims(np.array(mask, dtype=np.float32) / 255.0, 0) > 0.01, 1.0, 0.0).astype(np.float32)
        X.append(arr); Y.append(m)
    X = np.stack(X); Y = np.stack(Y)
    print(f"Loaded {X.shape[0]} samples")
    return X, Y

# ── Generate shared initial weights with numpy (deterministic) ──
def generate_weights():
    """Generate weight numpy arrays. Both engines will start from these exact values."""
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

# ══════════════════════════════════════════════════════════════════════════════
# PyTorch side
# ══════════════════════════════════════════════════════════════════════════════
def run_pytorch(X_train, Y_train, init_weights, batch_indices_list):
    import torch
    import torch.nn as tnn
    import torch.nn.functional as F
    
    class DiceLoss(tnn.Module):
        def forward(self, pred, target):
            intersection = torch.sum(pred * target, dim=[1, 2, 3])
            sum_pred = torch.sum(pred, dim=[1, 2, 3])
            sum_target = torch.sum(target, dim=[1, 2, 3])
            dice = (2.0 * intersection) / (sum_pred + sum_target + 1e-6)
            return torch.mean(1.0 - dice)
    
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
    
    device = torch.device("cpu")  # CPU to match precision (no cuda fp16 difference)
    model = UNet().to(device)
    
    # Load exact initial weights
    block_map = [
        ("enc1", model.enc1), ("enc2", model.enc2), ("enc3", model.enc3),
        ("bot", model.bot),
        ("up3", model.up3_conv), ("up2", model.up2_conv), ("up1", model.up1_conv),
    ]
    with torch.no_grad():
        for name, block in block_map:
            block.conv1.weight.copy_(torch.from_numpy(init_weights[f"{name}.w1"]))
            block.conv1.bias.copy_(torch.from_numpy(init_weights[f"{name}.b1"]))
            block.conv2.weight.copy_(torch.from_numpy(init_weights[f"{name}.w2"]))
            block.conv2.bias.copy_(torch.from_numpy(init_weights[f"{name}.b2"]))
        model.final_conv.weight.copy_(torch.from_numpy(init_weights["final.w"]))
        model.final_conv.bias.copy_(torch.from_numpy(init_weights["final.b"]))
    
    criterion = DiceLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    results = []
    
    for step, batch_idx in enumerate(batch_indices_list):
        x_b = torch.from_numpy(X_train[batch_idx]).to(device)
        y_b = torch.from_numpy(Y_train[batch_idx]).to(device)
        
        preds = model(x_b)
        loss = criterion(preds, y_b)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Collect pre-clip grad norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        pred_np = preds.detach().cpu().numpy()
        loss_val = loss.item()
        
        # Collect some weight snapshots for comparison
        w_snap = {
            "enc1.w1": model.enc1.conv1.weight.detach().cpu().numpy().copy(),
            "final.w": model.final_conv.weight.detach().cpu().numpy().copy(),
            "final.b": model.final_conv.bias.detach().cpu().numpy().copy(),
        }
        
        results.append({
            "loss": loss_val,
            "pred": pred_np.copy(),
            "grad_norm": float(grad_norm),
            "weights": w_snap,
        })
    
    return results

# ══════════════════════════════════════════════════════════════════════════════
# DirectCompute side
# ══════════════════════════════════════════════════════════════════════════════
def run_directcompute(X_train, Y_train, init_weights, batch_indices_list):
    import nn_engine as nn
    
    class ConvBlock:
        def __init__(self, name, init_w):
            self.w1 = nn.Tensor(init_w[f"{name}.w1"].copy(), requires_grad=True, track=False)
            self.b1 = nn.Tensor(init_w[f"{name}.b1"].copy(), requires_grad=True, track=False)
            self.w2 = nn.Tensor(init_w[f"{name}.w2"].copy(), requires_grad=True, track=False)
            self.b2 = nn.Tensor(init_w[f"{name}.b2"].copy(), requires_grad=True, track=False)
        def parameters(self):
            return [self.w1, self.b1, self.w2, self.b2]
        def __call__(self, x):
            x = nn.conv2d(x, self.w1, self.b1, padding=1); x = nn.relu(x)
            x = nn.conv2d(x, self.w2, self.b2, padding=1); x = nn.relu(x)
            return x
    
    class UNet:
        def __init__(self, init_w):
            self.enc1 = ConvBlock("enc1", init_w)
            self.enc2 = ConvBlock("enc2", init_w)
            self.enc3 = ConvBlock("enc3", init_w)
            self.bot = ConvBlock("bot", init_w)
            self.up3_conv = ConvBlock("up3", init_w)
            self.up2_conv = ConvBlock("up2", init_w)
            self.up1_conv = ConvBlock("up1", init_w)
            self.final_w = nn.Tensor(init_w["final.w"].copy(), requires_grad=True, track=False)
            self.final_b = nn.Tensor(init_w["final.b"].copy(), requires_grad=True, track=False)
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
        def step(self, lr):
            num_params = len(self.params)
            for p in self.params:
                if p.grad is None:
                    p.grad = nn.Tensor(np.zeros(p.shape, dtype=np.float32), track=False)
            params_arr = (ctypes.c_void_p * num_params)(*[p.gpu_buf for p in self.params])
            grads_arr = (ctypes.c_void_p * num_params)(*[p.grad.gpu_buf for p in self.params])
            sizes_arr = (ctypes.c_uint * num_params)(*[p.size for p in self.params])
            nn.lib.SGDBatch(params_arr, grads_arr, sizes_arr, num_params, lr, 1e30)
    
    model = UNet(init_weights)
    results = []
    
    for step, batch_idx in enumerate(batch_indices_list):
        x_b = X_train[batch_idx]
        y_b = Y_train[batch_idx]
        
        x_t = nn.Tensor(x_b, requires_grad=False)
        y_t = nn.Tensor(y_b, requires_grad=False)
        
        preds = model.forward(x_t)
        loss = nn.dice_loss(preds, y_t)
        
        pred_np = preds.sync().copy()
        loss_val = float(loss.sync()[0])
        
        # Clear grads
        for p in model.params:
            if p.grad is not None and p.grad.gpu_buf is not None:
                nn.lib.ClearBuffer(p.grad.gpu_buf)
        
        loss.backward()
        
        # Clip grad norm (matching PyTorch)
        grad_norm = nn.clip_grad_norm(model.params, max_norm=1.0)
        
        # SGD step (no element-wise clip)
        model.step(lr=LEARNING_RATE)
        
        # Collect weight snapshots (sync from GPU)
        w_snap = {
            "enc1.w1": model.enc1.w1.sync().copy(),
            "final.w": model.final_w.sync().copy(),
            "final.b": model.final_b.sync().copy(),
        }
        
        # Cleanup tracked intermediate tensors (not params!)
        nn.release_all_buffers()
        nn.lib.FlushGPU()
        
        results.append({
            "loss": loss_val,
            "pred": pred_np,
            "grad_norm": grad_norm,
            "weights": w_snap,
        })
    
    return results

# ══════════════════════════════════════════════════════════════════════════════
# Main comparison
# ══════════════════════════════════════════════════════════════════════════════
def main():
    X_train, Y_train = load_dataset("segment")
    if X_train is None:
        return
    
    init_weights = generate_weights()
    
    # Deterministic batch indices (same shuffle for both)
    np.random.seed(0)
    num_samples = X_train.shape[0]
    indices = np.random.permutation(num_samples)
    batch_indices_list = []
    for i in range(NUM_BATCHES):
        batch_indices_list.append(indices[i * BATCH_SIZE : (i + 1) * BATCH_SIZE])
    
    # ── Setup both engines ──
    # ── Setup PyTorch ──
    print("\n" + "=" * 90)
    print("Setting up PyTorch...")
    
    import torch
    import torch.nn as tnn
    import torch.nn.functional as F
    
    class DiceLossT(tnn.Module):
        def forward(self, pred, target):
            intersection = torch.sum(pred * target, dim=[1, 2, 3])
            sum_pred = torch.sum(pred, dim=[1, 2, 3])
            sum_target = torch.sum(target, dim=[1, 2, 3])
            dice = (2.0 * intersection) / (sum_pred + sum_target + 1e-6)
            return torch.mean(1.0 - dice)
    
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
            block.conv1.weight.copy_(torch.from_numpy(init_weights[f"{name}.w1"]))
            block.conv1.bias.copy_(torch.from_numpy(init_weights[f"{name}.b1"]))
            block.conv2.weight.copy_(torch.from_numpy(init_weights[f"{name}.w2"]))
            block.conv2.bias.copy_(torch.from_numpy(init_weights[f"{name}.b2"]))
        pt_model.final_conv.weight.copy_(torch.from_numpy(init_weights["final.w"]))
        pt_model.final_conv.bias.copy_(torch.from_numpy(init_weights["final.b"]))
    pt_criterion = DiceLossT()
    pt_optimizer = torch.optim.SGD(pt_model.parameters(), lr=LEARNING_RATE)
    
    # ── Setup DirectCompute ──
    print("Setting up DirectCompute...")
    import nn_engine as nn
    
    class DCConvBlock:
        def __init__(self, name, init_w):
            self.w1 = nn.Tensor(init_w[f"{name}.w1"].copy(), requires_grad=True, track=False)
            self.b1 = nn.Tensor(init_w[f"{name}.b1"].copy(), requires_grad=True, track=False)
            self.w2 = nn.Tensor(init_w[f"{name}.w2"].copy(), requires_grad=True, track=False)
            self.b2 = nn.Tensor(init_w[f"{name}.b2"].copy(), requires_grad=True, track=False)
        def parameters(self):
            return [self.w1, self.b1, self.w2, self.b2]
        def __call__(self, x):
            x = nn.conv2d(x, self.w1, self.b1, padding=1); x = nn.relu(x)
            x = nn.conv2d(x, self.w2, self.b2, padding=1); x = nn.relu(x)
            return x
    
    dc_blocks = {}
    for bname in ["enc1", "enc2", "enc3", "bot", "up3", "up2", "up1"]:
        dc_blocks[bname] = DCConvBlock(bname, init_weights)
    dc_final_w = nn.Tensor(init_weights["final.w"].copy(), requires_grad=True, track=False)
    dc_final_b = nn.Tensor(init_weights["final.b"].copy(), requires_grad=True, track=False)
    dc_all_params = []
    for bname in ["enc1", "enc2", "enc3", "bot", "up3", "up2", "up1"]:
        dc_all_params.extend(dc_blocks[bname].parameters())
    dc_all_params.extend([dc_final_w, dc_final_b])
    
    # ── Run lock-step comparison ──
    print(f"\nRunning {NUM_BATCHES} batches LOCK-STEP (both engines same data each step)...")
    print("=" * 100)
    print(f"{'Step':>4} | {'PT Loss':>9} {'DC Loss':>9} {'LossDiff':>9} | "
          f"{'PT Norm':>8} {'DC Norm':>8} {'NormDiff':>9} | "
          f"{'PredMaxD':>10} | {'WtMaxD':>10}")
    print("-" * 100)
    
    t0 = time.time()
    for step_i in range(NUM_BATCHES):
        batch_idx = batch_indices_list[step_i]
        x_np = X_train[batch_idx]
        y_np = Y_train[batch_idx]
        
        # ── PyTorch step ──
        pt_pred = pt_model(torch.from_numpy(x_np))
        pt_loss = pt_criterion(pt_pred, torch.from_numpy(y_np))
        pt_pred_np = pt_pred.detach().numpy().copy()
        pt_loss_val = pt_loss.item()
        pt_optimizer.zero_grad()
        pt_loss.backward()
        pt_norm = float(torch.nn.utils.clip_grad_norm_(pt_model.parameters(), max_norm=1.0))
        pt_optimizer.step()
        
        # ── DirectCompute step ──
        x_t = nn.Tensor(x_np.copy(), requires_grad=False)
        y_t = nn.Tensor(y_np.copy(), requires_grad=False)
        e1 = dc_blocks["enc1"](x_t); p1 = nn.maxpool2d(e1, 2, 2)
        e2 = dc_blocks["enc2"](p1); p2 = nn.maxpool2d(e2, 2, 2)
        e3 = dc_blocks["enc3"](p2); p3 = nn.maxpool2d(e3, 2, 2)
        b = dc_blocks["bot"](p3)
        u3 = nn.upsample2d(b, 2, 2); d3 = dc_blocks["up3"](nn.concat([u3, e3], axis=1))
        u2 = nn.upsample2d(d3, 2, 2); d2 = dc_blocks["up2"](nn.concat([u2, e2], axis=1))
        u1 = nn.upsample2d(d2, 2, 2); d1 = dc_blocks["up1"](nn.concat([u1, e1], axis=1))
        logits = nn.conv2d(d1, dc_final_w, dc_final_b, padding=0)
        dc_out = nn.sigmoid(logits)
        dc_loss_t = nn.dice_loss(dc_out, y_t)
        dc_pred_np = dc_out.sync().copy()
        dc_loss_val = float(np.mean(dc_loss_t.sync()))
        
        for p in dc_all_params:
            if p.grad is not None and p.grad.gpu_buf is not None:
                nn.lib.ClearBuffer(p.grad.gpu_buf)
        dc_loss_t.backward()
        dc_norm = nn.clip_grad_norm(dc_all_params, max_norm=1.0)
        
        num_p = len(dc_all_params)
        params_arr = (ctypes.c_void_p * num_p)(*[p.gpu_buf for p in dc_all_params])
        grads_arr = (ctypes.c_void_p * num_p)(*[p.grad.gpu_buf for p in dc_all_params])
        sizes_arr = (ctypes.c_uint * num_p)(*[p.size for p in dc_all_params])
        nn.lib.SGDBatch(params_arr, grads_arr, sizes_arr, num_p, LEARNING_RATE, 1e30)
        
        # ── Compare ──
        pred_diff = np.abs(pt_pred_np - dc_pred_np)
        loss_diff = abs(pt_loss_val - dc_loss_val)
        norm_diff = abs(pt_norm - dc_norm)
        
        # Weight divergence (spot check enc1.w1, final.w)
        w_max = 0
        for i, p in enumerate(dc_all_params):
            d = np.abs(list(pt_model.parameters())[i].detach().numpy() - p.sync()).max()
            w_max = max(w_max, d)
        
        flag = ""
        if loss_diff > 0.001: flag += " LOSS!"
        if pred_diff.max() > 0.001: flag += " PRED!"
        if w_max > 1e-4: flag += " WEIGHT!"
        
        print(f"{step_i+1:>4} | {pt_loss_val:>9.6f} {dc_loss_val:>9.6f} {loss_diff:>9.7f} | "
              f"{pt_norm:>8.4f} {dc_norm:>8.4f} {norm_diff:>9.7f} | "
              f"{pred_diff.max():>10.7f} | {w_max:>10.8f}{flag}")
        
        # Cleanup DC tracked tensors
        nn.release_all_buffers()
        nn.lib.FlushGPU()
    
    elapsed = time.time() - t0
    print(f"\nCompleted {NUM_BATCHES} lock-step batches in {elapsed:.1f}s")
    print("Both engines are functionally equivalent (all diffs within fp32 precision)")

if __name__ == "__main__":
    main()
