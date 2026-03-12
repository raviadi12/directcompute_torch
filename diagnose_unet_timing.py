"""Diagnose per-batch timing variance in U-Net training."""
import time
import os
import glob
import ctypes
import numpy as np
import PIL.Image as Image
import nn_engine as nn

BATCH_SIZE = 2
IMG_SIZE = 128
NUM_BATCHES = 30  # Just enough to see the pattern

def load_dataset(dataset_dir="segment"):
    img_dir = os.path.join(dataset_dir, "png_images", "IMAGES")
    mask_dir = os.path.join(dataset_dir, "png_masks", "MASKS")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")) + glob.glob(os.path.join(mask_dir, "*.jpg")))
    X_list, Y_list = [], []
    for img_p, mask_p in zip(img_paths, mask_paths):
        img = Image.open(img_p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_arr = np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1))
        mask = Image.open(mask_p).convert("L").resize((IMG_SIZE, IMG_SIZE))
        mask_arr = np.expand_dims(np.where(np.array(mask, dtype=np.float32) / 255.0 > 0.01, 1.0, 0.0).astype(np.float32), axis=0)
        X_list.append(img_arr)
        Y_list.append(mask_arr)
    return np.stack(X_list, axis=0), np.stack(Y_list, axis=0)

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
        self.enc2 = ConvBlock(base, base*2, rng)
        self.enc3 = ConvBlock(base*2, base*4, rng)
        self.bot = ConvBlock(base*4, base*8, rng)
        self.up3_conv = ConvBlock(base*8+base*4, base*4, rng)
        self.up2_conv = ConvBlock(base*4+base*2, base*2, rng)
        self.up1_conv = ConvBlock(base*2+base, base, rng)
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
        e1 = self.enc1(x);     p1 = nn.maxpool2d(e1, 2, 2)
        e2 = self.enc2(p1);    p2 = nn.maxpool2d(e2, 2, 2)
        e3 = self.enc3(p2);    p3 = nn.maxpool2d(e3, 2, 2)
        b = self.bot(p3)
        u3 = nn.upsample2d(b, 2, 2); c3 = nn.concat([u3, e3], axis=1); d3 = self.up3_conv(c3)
        u2 = nn.upsample2d(d3, 2, 2); c2 = nn.concat([u2, e2], axis=1); d2 = self.up2_conv(c2)
        u1 = nn.upsample2d(d2, 2, 2); c1 = nn.concat([u1, e1], axis=1); d1 = self.up1_conv(c1)
        logits = nn.conv2d(d1, self.final_w, self.final_b, padding=0)
        return nn.relu6(logits)
    def step(self, lr):
        num_params = len(self.params)
        for p in self.params:
            if p.grad is None:
                p.grad = nn.Tensor(np.zeros_like(p.data), track=False)
        params_a = (ctypes.c_void_p * num_params)(*[p.gpu_buf for p in self.params])
        grads_a = (ctypes.c_void_p * num_params)(*[p.grad.gpu_buf for p in self.params])
        sizes_a = (ctypes.c_uint * num_params)(*[p.size for p in self.params])
        nn.lib.SGDBatch(params_a, grads_a, sizes_a, num_params, lr, 1.0)

X_train, Y_train = load_dataset()
model = UNet()

# Get pool stats function
nn.lib.GetPoolStats.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
nn.lib.GetPoolMemory.restype = ctypes.c_uint64

print(f"\n{'='*90}")
print(f"  {'Batch':>5} | {'Upload':>8} | {'Forward':>8} | {'Sync':>8} | {'Clear':>8} | {'Backward':>8} | {'SGD':>8} | {'Release':>8} | {'Flush':>8} | {'TOTAL':>8} | Pool Hit/Miss")
print(f"{'='*90}")

indices = np.arange(X_train.shape[0])

for i in range(NUM_BATCHES):
    batch_indices = indices[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    x_batch = X_train[batch_indices]
    y_batch = Y_train[batch_indices]
    
    # Reset pool stats each batch
    nn.lib.ResetPoolStats()
    
    # 1. Upload
    t0 = time.perf_counter()
    x_tensor = nn.Tensor(x_batch, requires_grad=False)
    y_tensor = nn.Tensor(y_batch, requires_grad=False)
    t_upload = time.perf_counter() - t0
    
    # 2. Forward
    t0 = time.perf_counter()
    preds = model.forward(x_tensor)
    loss = nn.dice_loss(preds, y_tensor)
    t_fwd = time.perf_counter() - t0
    
    # 3. Sync (GPU readback)
    t0 = time.perf_counter()
    batch_loss_val = loss.sync()[0]
    t_sync = time.perf_counter() - t0
    
    # 4. Clear grads
    t0 = time.perf_counter()
    for p in model.params:
        if p.grad is not None:
            nn.lib.ClearBuffer(p.grad.gpu_buf)
    t_clear = time.perf_counter() - t0
    
    # 5. Backward
    t0 = time.perf_counter()
    loss.backward()
    t_bwd = time.perf_counter() - t0
    
    # 6. SGD
    t0 = time.perf_counter()
    model.step(lr=1e-3)
    t_sgd = time.perf_counter() - t0
    
    # 7. Release
    t0 = time.perf_counter()
    nn.release_all_buffers()
    t_rel = time.perf_counter() - t0
    
    # 8. Flush
    t0 = time.perf_counter()
    nn.lib.FlushGPU()
    t_flush = time.perf_counter() - t0
    
    total = t_upload + t_fwd + t_sync + t_clear + t_bwd + t_sgd + t_rel + t_flush
    
    # Pool stats
    hits = ctypes.c_uint64(0)
    misses = ctypes.c_uint64(0)
    nn.lib.GetPoolStats(ctypes.byref(hits), ctypes.byref(misses))
    pool_mem = nn.lib.GetPoolMemory()
    
    print(f"  {i+1:>5} | {t_upload*1000:>7.1f}ms | {t_fwd*1000:>7.1f}ms | {t_sync*1000:>7.1f}ms | {t_clear*1000:>7.1f}ms | {t_bwd*1000:>7.1f}ms | {t_sgd*1000:>7.1f}ms | {t_rel*1000:>7.1f}ms | {t_flush*1000:>7.1f}ms | {total*1000:>7.1f}ms | {hits.value}/{misses.value} Pool:{pool_mem/1024/1024:.1f}MB")

print(f"{'='*90}")
