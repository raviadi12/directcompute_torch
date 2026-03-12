"""
train_unet_dc.py - Standalone DirectCompute UNet training script.

Uses the same architecture, weight initialisation (numpy RNG seed=42),
data shuffling, and SGD optimiser as verify_unet.py — but DirectCompute only.

Usage:
  python train_unet_dc.py                          # 50 samples, 1 epoch
  python train_unet_dc.py --epochs 3 --all          # full dataset, 3 epochs
  python train_unet_dc.py --samples 200 --epochs 5  # 200 samples, 5 epochs
  python train_unet_dc.py --batch-size 8             # batch_size=8
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
parser = argparse.ArgumentParser(description="Train UNet with DirectCompute")
parser.add_argument("--samples",    type=int,   default=50,   help="Number of samples (default: 50)")
parser.add_argument("--batch-size", type=int,   default=2,    help="Batch size (default: 2)")
parser.add_argument("--all",        action="store_true",      help="Use full dataset")
parser.add_argument("--epochs",     type=int,   default=1,    help="Number of epochs (default: 1)")
parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate (default: 1e-3)")
parser.add_argument("--renderdoc",  action="store_true",      help="Capture first batch with RenderDoc then exit")
args = parser.parse_args()

BATCH_SIZE    = args.batch_size
IMG_SIZE      = 128
NUM_SAMPLES   = args.samples
LEARNING_RATE = args.lr
EPOCHS        = args.epochs
BASE          = 16

# ── Dataset ──
def load_dataset():
    img_dir   = os.path.join("segment", "png_images", "IMAGES")
    mask_dir  = os.path.join("segment", "png_masks",  "MASKS")
    img_paths  = sorted(glob.glob(os.path.join(img_dir,  "*.png")) +
                        glob.glob(os.path.join(img_dir,  "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")) +
                        glob.glob(os.path.join(mask_dir, "*.jpg")))
    n = len(img_paths) if args.all else min(NUM_SAMPLES, len(img_paths))
    X, Y = [], []
    for i in range(min(n, len(img_paths))):
        img  = Image.open(img_paths[i]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr  = np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1))
        mask = Image.open(mask_paths[i]).convert("L").resize((IMG_SIZE, IMG_SIZE))
        m    = np.where(
            np.expand_dims(np.array(mask, dtype=np.float32) / 255.0, 0) > 0.01,
            1.0, 0.0
        ).astype(np.float32)
        X.append(arr); Y.append(m)
    return np.stack(X), np.stack(Y)

# ── Weight initialisation (numpy RNG seed=42, matches verify_unet.py) ──
def generate_weights():
    rng = np.random.RandomState(42)
    W = {}
    def conv_block(name, in_c, out_c):
        W[f"{name}.w1"] = rng.randn(out_c, in_c,   3, 3).astype(np.float32) * np.sqrt(2.0 / (in_c * 9))
        W[f"{name}.b1"] = np.zeros(out_c, dtype=np.float32)
        W[f"{name}.w2"] = rng.randn(out_c, out_c,  3, 3).astype(np.float32) * np.sqrt(2.0 / (out_c * 9))
        W[f"{name}.b2"] = np.zeros(out_c, dtype=np.float32)
    conv_block("enc1", 3,          BASE)
    conv_block("enc2", BASE,       BASE*2)
    conv_block("enc3", BASE*2,     BASE*4)
    conv_block("bot",  BASE*4,     BASE*8)
    conv_block("up3",  BASE*8+BASE*4, BASE*4)
    conv_block("up2",  BASE*4+BASE*2, BASE*2)
    conv_block("up1",  BASE*2+BASE,   BASE)
    W["final.w"] = rng.randn(1, BASE, 1, 1).astype(np.float32) * np.sqrt(2.0 / BASE)
    W["final.b"] = np.zeros(1, dtype=np.float32)
    return W

# ── Main ──
def main():
    print("Loading dataset...")
    X, Y = load_dataset()
    print(f"  {X.shape[0]} samples loaded")

    W = generate_weights()
    num_samples = X.shape[0]
    num_batches = num_samples // BATCH_SIZE

    import nn_engine as dc

    # ── RenderDoc hook (optional) ──
    rdoc = None
    if args.renderdoc:
        import rdoc_helper
        rdoc = rdoc_helper.get_rdoc_api()
        if rdoc is None:
            print("[RDoc] RenderDoc not available — run this script launched from RenderDoc UI.")
            sys.exit(1)

    # ── Build model on GPU ──
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

    blocks = {n: DCConvBlock(n) for n in ["enc1", "enc2", "enc3", "bot", "up3", "up2", "up1"]}
    fw = dc.Tensor(W["final.w"].copy(), requires_grad=True, track=False)
    fb = dc.Tensor(W["final.b"].copy(), requires_grad=True, track=False)

    params = []
    for n in ["enc1", "enc2", "enc3", "bot", "up3", "up2", "up1"]:
        params.extend(blocks[n].parameters())
    params.extend([fw, fb])

    def forward(x_np):
        x  = dc.Tensor(x_np.copy(), requires_grad=False)
        e1 = blocks["enc1"](x);  p1 = dc.maxpool2d(e1, 2, 2)
        e2 = blocks["enc2"](p1); p2 = dc.maxpool2d(e2, 2, 2)
        e3 = blocks["enc3"](p2); p3 = dc.maxpool2d(e3, 2, 2)
        b  = blocks["bot"](p3)
        u3 = dc.upsample2d(b,  2, 2); d3 = blocks["up3"](dc.concat([u3, e3], axis=1))
        u2 = dc.upsample2d(d3, 2, 2); d2 = blocks["up2"](dc.concat([u2, e2], axis=1))
        u1 = dc.upsample2d(d2, 2, 2); d1 = blocks["up1"](dc.concat([u1, e1], axis=1))
        logits = dc.conv2d(d1, fw, fb, padding=0)
        return dc.sigmoid(logits)

    def sgd_step():
        n  = len(params)
        pa = (ctypes.c_void_p * n)(*[p.gpu_buf       for p in params])
        ga = (ctypes.c_void_p * n)(*[p.grad.gpu_buf  for p in params])
        sa = (ctypes.c_uint   * n)(*[p.size           for p in params])
        dc.lib.SGDBatch(pa, ga, sa, n, LEARNING_RATE, 1e30)

    def clip_and_sgd_step():
        """Fused GPU-side gradient clipping + SGD (no CPU round-trips)."""
        return dc.sgd_step_clipped(params, LEARNING_RATE, max_norm=1.0)

    # ── Training loop ──
    print(f"\nTraining {EPOCHS} epoch(s), {num_batches} batches/epoch, "
          f"batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")

    all_losses = []
    t0 = time.time()

    for epoch in range(EPOCHS):
        np.random.seed(epoch)                          # deterministic shuffle per epoch
        indices = np.random.permutation(num_samples)

        print(f"\n── Epoch {epoch+1}/{EPOCHS} " + "─" * 60)
        print(f"{'Batch':>5} | {'Loss':>10} | {'GradNorm':>10}")
        print("-" * 35)

        ep_losses = []

        for i in range(num_batches):
            bi   = indices[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            x_np = X[bi]
            y_np = Y[bi]

            # RenderDoc: start capture on first batch
            if rdoc and epoch == 0 and i == 0:
                print("[RDoc] >>> STARTING RENDERDOC CAPTURE <<<")
                rdoc.StartFrameCapture(None, None)

            # Forward
            pred    = forward(x_np)
            dc_y    = dc.Tensor(y_np.copy(), requires_grad=False)
            loss    = dc.dice_loss(pred, dc_y)
            loss_val = float(np.mean(loss.sync()))

            # Backward
            for p in params:
                if p.grad is not None and p.grad.gpu_buf is not None:
                    dc.lib.ClearBuffer(p.grad.gpu_buf)
            loss.backward()

            # Gradient clipping + SGD update (fused on GPU)
            grad_norm = clip_and_sgd_step()

            # RenderDoc: end capture after first batch and exit
            if rdoc and epoch == 0 and i == 0:
                rdoc.EndFrameCapture(None, None)
                print("[RDoc] >>> CAPTURE COMPLETE — check the RenderDoc UI <<<")
                dc.release_all_buffers()
                dc.lib.FlushGPU()
                sys.exit(0)

            print(f"{i+1:>5} | {loss_val:>10.6f} | {grad_norm:>10.4f}")
            ep_losses.append(loss_val)

            dc.release_all_buffers()
            dc.lib.FlushGPU()

        avg = np.mean(ep_losses)
        all_losses.extend(ep_losses)
        print(f"  Epoch {epoch+1} avg loss: {avg:.6f}  "
              f"min: {min(ep_losses):.6f}  max: {max(ep_losses):.6f}")

    elapsed = time.time() - t0
    print(f"\n{EPOCHS} epoch(s) done in {elapsed:.1f}s  "
          f"({elapsed / (EPOCHS * num_batches) * 1000:.1f} ms/batch)")
    print(f"  Overall avg loss : {np.mean(all_losses):.6f}")
    print(f"  Final batch loss : {all_losses[-1]:.6f}")

    # ── Save a quick prediction visualisation ──
    print("\nGenerating prediction visualisation...")
    from PIL import Image as PILImage

    n_show = min(6, num_samples)
    cell, pad = IMG_SIZE, 6
    cols = 3  # Original | Ground Truth | Prediction
    gw = cols * cell + (cols + 1) * pad
    gh = n_show * cell + (n_show + 1) * pad
    grid = PILImage.new("RGB", (gw, gh), (30, 30, 30))

    for idx in range(n_show):
        x_s = X[idx:idx+1]
        y_s = Y[idx:idx+1]

        out = forward(x_s)
        pred_np = out.sync()[0, 0].copy()
        dc.release_all_buffers()

        orig    = (np.transpose(x_s[0], (1, 2, 0)) * 255).astype(np.uint8)
        gt      = (y_s[0, 0] * 255).astype(np.uint8)
        pred_img = (pred_np * 255).clip(0, 255).astype(np.uint8)

        yo = pad + idx * (cell + pad)
        grid.paste(PILImage.fromarray(orig),                (pad,                  yo))
        grid.paste(PILImage.fromarray(gt).convert("RGB"),   (pad + cell + pad,     yo))
        grid.paste(PILImage.fromarray(pred_img).convert("RGB"), (pad + 2*(cell+pad), yo))

    out_path = "train_unet_dc_preview.png"
    grid.save(out_path)
    print(f"Saved to {out_path}")
    print("  Columns: Original | Ground Truth | Prediction")

if __name__ == "__main__":
    main()
