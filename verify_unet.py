"""
verify_unet.py - Side-by-side verification of DirectCompute vs PyTorch UNet.

Runs BOTH engines lock-step with identical weights (generated from numpy
RNG seed=42), identical data shuffling, and compares loss/predictions/weights
at every batch.

Saves a side-by-side visualization at the end.

Usage:
  python verify_unet.py                          # 50 samples, 1 epoch
  python verify_unet.py --epochs 3 --all          # full dataset, 3 epochs
  python verify_unet.py --samples 200 --epochs 5  # 200 samples, 5 epochs
  python verify_unet.py --batch-size 8             # batch_size=8
"""
import time
import os
import sys
import glob
import ctypes
import argparse
import numpy as np
import PIL.Image as Image

# ── CLI ──
parser = argparse.ArgumentParser(description="Verify DirectCompute vs PyTorch UNet")
parser.add_argument("--samples", type=int, default=50, help="Number of samples (default: 50)")
parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
parser.add_argument("--all", action="store_true", help="Use full dataset")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (default: 1)")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
args = parser.parse_args()

BATCH_SIZE = args.batch_size
IMG_SIZE = 128
NUM_SAMPLES = args.samples
LEARNING_RATE = args.lr
EPOCHS = args.epochs
BASE = 16

# ── Dataset ──
def load_dataset(n=NUM_SAMPLES):
    img_dir = os.path.join("segment", "png_images", "IMAGES")
    mask_dir = os.path.join("segment", "png_masks", "MASKS")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")) + glob.glob(os.path.join(mask_dir, "*.jpg")))
    if args.all:
        n = len(img_paths)
    else:
        n = min(n, len(img_paths))
    X, Y = [], []
    for i in range(min(n, len(img_paths))):
        img = Image.open(img_paths[i]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1))
        mask = Image.open(mask_paths[i]).convert("L").resize((IMG_SIZE, IMG_SIZE))
        m = np.where(np.expand_dims(np.array(mask, dtype=np.float32) / 255.0, 0) > 0.01, 1.0, 0.0).astype(np.float32)
        X.append(arr); Y.append(m)
    return np.stack(X), np.stack(Y)

# ── Shared weight generation (numpy RNG, deterministic) ──
def generate_weights():
    rng = np.random.RandomState(42)
    W = {}
    def conv_block(name, in_c, out_c):
        W[f"{name}.w1"] = rng.randn(out_c, in_c, 3, 3).astype(np.float32) * np.sqrt(2.0 / (in_c * 9))
        W[f"{name}.b1"] = np.zeros(out_c, dtype=np.float32)
        W[f"{name}.w2"] = rng.randn(out_c, out_c, 3, 3).astype(np.float32) * np.sqrt(2.0 / (out_c * 9))
        W[f"{name}.b2"] = np.zeros(out_c, dtype=np.float32)
    conv_block("enc1", 3, BASE)
    conv_block("enc2", BASE, BASE*2)
    conv_block("enc3", BASE*2, BASE*4)
    conv_block("bot", BASE*4, BASE*8)
    conv_block("up3", BASE*8+BASE*4, BASE*4)
    conv_block("up2", BASE*4+BASE*2, BASE*2)
    conv_block("up1", BASE*2+BASE, BASE)
    W["final.w"] = rng.randn(1, BASE, 1, 1).astype(np.float32) * np.sqrt(2.0 / BASE)
    W["final.b"] = np.zeros(1, dtype=np.float32)
    return W

# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Loading dataset...")
    X, Y = load_dataset()
    print(f"  {X.shape[0]} samples loaded")

    W = generate_weights()
    num_samples = X.shape[0]
    num_batches = num_samples // BATCH_SIZE

    # ── Setup PyTorch ──
    import torch
    import torch.nn as tnn
    import torch.nn.functional as F

    class DiceLoss(tnn.Module):
        def forward(self, pred, target):
            inter = torch.sum(pred * target, dim=[1, 2, 3])
            sp = torch.sum(pred, dim=[1, 2, 3])
            st = torch.sum(target, dim=[1, 2, 3])
            dice = (2.0 * inter) / (sp + st + 1e-6)
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
    bmap = [("enc1", pt_model.enc1), ("enc2", pt_model.enc2), ("enc3", pt_model.enc3),
            ("bot", pt_model.bot),
            ("up3", pt_model.up3_conv), ("up2", pt_model.up2_conv), ("up1", pt_model.up1_conv)]
    with torch.no_grad():
        for name, block in bmap:
            block.conv1.weight.copy_(torch.from_numpy(W[f"{name}.w1"]))
            block.conv1.bias.copy_(torch.from_numpy(W[f"{name}.b1"]))
            block.conv2.weight.copy_(torch.from_numpy(W[f"{name}.w2"]))
            block.conv2.bias.copy_(torch.from_numpy(W[f"{name}.b2"]))
        pt_model.final_conv.weight.copy_(torch.from_numpy(W["final.w"]))
        pt_model.final_conv.bias.copy_(torch.from_numpy(W["final.b"]))
    pt_crit = DiceLoss()
    pt_opt = torch.optim.SGD(pt_model.parameters(), lr=LEARNING_RATE)

    # ── Setup DirectCompute ──
    import nn_engine as dc

    class DCConvBlock:
        def __init__(self, name):
            self.w1 = dc.Tensor(W[f"{name}.w1"].copy(), requires_grad=True, track=False)
            self.b1 = dc.Tensor(W[f"{name}.b1"].copy(), requires_grad=True, track=False)
            self.w2 = dc.Tensor(W[f"{name}.w2"].copy(), requires_grad=True, track=False)
            self.b2 = dc.Tensor(W[f"{name}.b2"].copy(), requires_grad=True, track=False)
        def parameters(self):
            return [self.w1, self.b1, self.w2, self.b2]
        def __call__(self, x):
            x = dc.conv2d(x, self.w1, self.b1, padding=1); x = dc.relu(x)
            x = dc.conv2d(x, self.w2, self.b2, padding=1); x = dc.relu(x)
            return x

    dc_blocks = {n: DCConvBlock(n) for n in ["enc1","enc2","enc3","bot","up3","up2","up1"]}
    dc_fw = dc.Tensor(W["final.w"].copy(), requires_grad=True, track=False)
    dc_fb = dc.Tensor(W["final.b"].copy(), requires_grad=True, track=False)
    dc_params = []
    for n in ["enc1","enc2","enc3","bot","up3","up2","up1"]:
        dc_params.extend(dc_blocks[n].parameters())
    dc_params.extend([dc_fw, dc_fb])

    def dc_forward(x_np):
        x = dc.Tensor(x_np.copy(), requires_grad=False)
        e1 = dc_blocks["enc1"](x);  p1 = dc.maxpool2d(e1, 2, 2)
        e2 = dc_blocks["enc2"](p1); p2 = dc.maxpool2d(e2, 2, 2)
        e3 = dc_blocks["enc3"](p2); p3 = dc.maxpool2d(e3, 2, 2)
        b = dc_blocks["bot"](p3)
        u3 = dc.upsample2d(b, 2, 2);  d3 = dc_blocks["up3"](dc.concat([u3, e3], axis=1))
        u2 = dc.upsample2d(d3, 2, 2); d2 = dc_blocks["up2"](dc.concat([u2, e2], axis=1))
        u1 = dc.upsample2d(d2, 2, 2); d1 = dc_blocks["up1"](dc.concat([u1, e1], axis=1))
        logits = dc.conv2d(d1, dc_fw, dc_fb, padding=0)
        return dc.sigmoid(logits)

    def dc_sgd_step():
        n = len(dc_params)
        pa = (ctypes.c_void_p * n)(*[p.gpu_buf for p in dc_params])
        ga = (ctypes.c_void_p * n)(*[p.grad.gpu_buf for p in dc_params])
        sa = (ctypes.c_uint * n)(*[p.size for p in dc_params])
        dc.lib.SGDBatch(pa, ga, sa, n, LEARNING_RATE, 1e30)

    # ── Train lock-step ──
    print(f"\nTraining {EPOCHS} epoch(s), {num_batches} batches/epoch, batch_size={BATCH_SIZE}")

    pt_param_list = list(pt_model.parameters())
    all_pt_losses, all_dc_losses = [], []
    t0 = time.time()

    for epoch in range(EPOCHS):
        # Deterministic per-epoch shuffle (matches train_unet.py / train_unet_pytorch.py)
        np.random.seed(epoch)
        indices = np.random.permutation(num_samples)

        print(f"\n── Epoch {epoch+1}/{EPOCHS} " + "─" * 88)
        print(f"{'Batch':>5} | {'PT Loss':>9} {'DC Loss':>9} {'LossDiff':>9} | "
              f"{'PT Norm':>8} {'DC Norm':>8} | "
              f"{'PredMaxD':>10} {'WtMaxD':>10}")
        print("-" * 105)

        ep_pt, ep_dc = [], []

        for i in range(num_batches):
            bi = indices[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            x_np, y_np = X[bi], Y[bi]

            # ── PyTorch ──
            pt_pred = pt_model(torch.from_numpy(x_np))
            pt_loss = pt_crit(pt_pred, torch.from_numpy(y_np))
            pt_lv = pt_loss.item()
            pt_opt.zero_grad()
            pt_loss.backward()
            pt_norm = float(torch.nn.utils.clip_grad_norm_(pt_model.parameters(), max_norm=1.0))
            pt_opt.step()
            pt_pred_np = pt_pred.detach().numpy()

            # ── DirectCompute ──
            dc_out = dc_forward(x_np)
            dc_y = dc.Tensor(y_np.copy(), requires_grad=False)
            dc_loss = dc.dice_loss(dc_out, dc_y)
            dc_pred_np = dc_out.sync().copy()
            dc_lv = float(np.mean(dc_loss.sync()))

            for p in dc_params:
                if p.grad is not None and p.grad.gpu_buf is not None:
                    dc.lib.ClearBuffer(p.grad.gpu_buf)
            dc_loss.backward()
            dc_norm = dc.clip_grad_norm(dc_params, max_norm=1.0)
            dc_sgd_step()

            # ── Compare ──
            pred_diff = np.abs(pt_pred_np - dc_pred_np).max()
            w_max = 0
            for j, p in enumerate(dc_params):
                d = np.abs(pt_param_list[j].detach().numpy() - p.sync()).max()
                w_max = max(w_max, d)

            loss_diff = abs(pt_lv - dc_lv)
            flag = ""
            if loss_diff > 0.001: flag += " LOSS!"
            if pred_diff > 0.01: flag += " PRED!"
            if w_max > 1e-3: flag += " WEIGHT!"

            print(f"{i+1:>5} | {pt_lv:>9.6f} {dc_lv:>9.6f} {loss_diff:>9.7f} | "
                  f"{pt_norm:>8.4f} {dc_norm:>8.4f} | "
                  f"{pred_diff:>10.7f} {w_max:>10.8f}{flag}")

            ep_pt.append(pt_lv)
            ep_dc.append(dc_lv)

            dc.release_all_buffers()
            dc.lib.FlushGPU()

        all_pt_losses.extend(ep_pt)
        all_dc_losses.extend(ep_dc)
        print(f"  Epoch {epoch+1} avg loss  PT: {np.mean(ep_pt):.6f}  DC: {np.mean(ep_dc):.6f}  "
              f"max diff: {max(abs(a-b) for a,b in zip(ep_pt, ep_dc)):.8f}")

    elapsed = time.time() - t0
    print(f"\n{EPOCHS} epoch(s) done in {elapsed:.1f}s")
    print(f"  Overall PT avg loss: {np.mean(all_pt_losses):.6f}")
    print(f"  Overall DC avg loss: {np.mean(all_dc_losses):.6f}")
    print(f"  Max loss diff across all batches: {max(abs(a-b) for a,b in zip(all_pt_losses, all_dc_losses)):.8f}")

    # ── Visualize predictions side-by-side ──
    print("\nGenerating side-by-side visualization...")
    from PIL import Image as PILImage

    n_show = min(6, num_samples)
    cell = IMG_SIZE
    pad = 6
    # Layout: [Original | GT | Pytorch | DirectCompute]  per row
    cols = 4
    gw = cols * cell + (cols + 1) * pad
    gh = n_show * cell + (n_show + 1) * pad
    grid = PILImage.new("RGB", (gw, gh), (30, 30, 30))

    pt_model.eval()
    for idx in range(n_show):
        x_s = X[idx:idx+1]
        y_s = Y[idx:idx+1]

        with torch.no_grad():
            pt_p = pt_model(torch.from_numpy(x_s)).numpy()[0, 0]

        dc_out = dc_forward(x_s)
        dc_p = dc_out.sync()[0, 0].copy()
        dc.release_all_buffers()

        orig = (np.transpose(x_s[0], (1, 2, 0)) * 255).astype(np.uint8)
        gt = (y_s[0, 0] * 255).astype(np.uint8)
        pt_mask = (pt_p * 255).clip(0, 255).astype(np.uint8)
        dc_mask = (dc_p * 255).clip(0, 255).astype(np.uint8)

        yo = pad + idx * (cell + pad)
        grid.paste(PILImage.fromarray(orig), (pad, yo))
        grid.paste(PILImage.fromarray(gt).convert("RGB"), (pad + cell + pad, yo))
        grid.paste(PILImage.fromarray(pt_mask).convert("RGB"), (pad + 2*(cell+pad), yo))
        grid.paste(PILImage.fromarray(dc_mask).convert("RGB"), (pad + 3*(cell+pad), yo))

    out_path = "verify_unet_comparison.png"
    grid.save(out_path)
    print(f"Saved to {out_path}")
    print("  Columns: Original | Ground Truth | PyTorch | DirectCompute")

if __name__ == "__main__":
    main()
