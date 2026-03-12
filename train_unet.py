import time
import os
import sys
import glob
import numpy as np
import PIL.Image as Image
import nn_engine as nn
import rdoc_helper

# --- Hyperparameters ---
BATCH_SIZE = 2      # Small batch size to fit in VRAM (adjust based on GPU memory)
IMG_SIZE = 128      # 128x128 images for training
EPOCHS = 3
LEARNING_RATE = 1e-3

# --- Dataset Loader ---
# Assumes folder structure:
# segment/
#   images/
#     img1.png, img2.png, ...
#   masks/
#     img1.png, img2.png, ... 

def load_dataset(dataset_dir="segment"):
    print(f"Loading dataset from {dataset_dir}...")
    
    img_dir = os.path.join(dataset_dir, "png_images", "IMAGES")
    mask_dir = os.path.join(dataset_dir, "png_masks", "MASKS")
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f"Dataset folders missing. Please create {img_dir} and {mask_dir}.")
        return None, None
        
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")) + glob.glob(os.path.join(mask_dir, "*.jpg")))
    
    if len(img_paths) == 0:
        print("No images found. Please add training images.")
        return None, None
        
    if len(img_paths) != len(mask_paths):
        print(f"Warning: Count mismatch! Found {len(img_paths)} images and {len(mask_paths)} masks.")
        # Try to match by filename if possible, otherwise zip truncates
        
    X_list = []
    Y_list = []
    
    for img_p, mask_p in zip(img_paths, mask_paths):
        # Load Image (rgb)
        img = Image.open(img_p).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        img_arr = np.transpose(img_arr, (2, 0, 1))         # (3, H, W)
        
        # Load Mask (grayscale)
        mask = Image.open(mask_p).convert("L")
        mask = mask.resize((IMG_SIZE, IMG_SIZE))
        mask_arr = np.array(mask, dtype=np.float32) / 255.0 # (H, W)
        mask_arr = np.expand_dims(mask_arr, axis=0)         # (1, H, W)
        # Binarize mask (Any non-black pixel becomes foreground)
        mask_arr = np.where(mask_arr > 0.01, 1.0, 0.0).astype(np.float32)
        
        X_list.append(img_arr)
        Y_list.append(mask_arr)
        
    X = np.stack(X_list, axis=0) # (Batch, 3, H, W)
    Y = np.stack(Y_list, axis=0) # (Batch, 1, H, W)
    
    print(f"Loaded {X.shape[0]} samples. Input: {X.shape}, Targets: {Y.shape}")
    return X, Y

# --- U-Net Model Definition ---
class ConvBlock:
    def __init__(self, in_c, out_c, rng):
        # He Initialization
        scale = np.sqrt(2.0 / (in_c * 3 * 3))
        self.w1 = nn.Tensor(rng.randn(out_c, in_c, 3, 3).astype(np.float32) * scale, requires_grad=True, track=False)
        self.b1 = nn.Tensor(np.zeros((out_c,), dtype=np.float32), requires_grad=True, track=False)
        
        scale2 = np.sqrt(2.0 / (out_c * 3 * 3))
        self.w2 = nn.Tensor(rng.randn(out_c, out_c, 3, 3).astype(np.float32) * scale2, requires_grad=True, track=False)
        self.b2 = nn.Tensor(np.zeros((out_c,), dtype=np.float32), requires_grad=True, track=False)
        
    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2]
        
    def __call__(self, x):
        x = nn.conv2d(x, self.w1, self.b1, padding=1)
        x = nn.relu(x)
        x = nn.conv2d(x, self.w2, self.b2, padding=1)
        x = nn.relu(x)
        return x

class UNet:
    def __init__(self):
        rng = np.random.RandomState(42)
        
        # Base filters (e.g., 16 or 32 due to engine memory limits on iGPU)
        base = 16 
        
        # Encoder
        self.enc1 = ConvBlock(3, base, rng)
        self.enc2 = ConvBlock(base, base * 2, rng)
        self.enc3 = ConvBlock(base * 2, base * 4, rng)
        
        # Bottleneck
        self.bot = ConvBlock(base * 4, base * 8, rng)
        
        # Decoder (Using UpSample2D instead of Transpose Convolutions)
        self.up3_conv = ConvBlock(base * 8 + base * 4, base * 4, rng)
        self.up2_conv = ConvBlock(base * 4 + base * 2, base * 2, rng)
        self.up1_conv = ConvBlock(base * 2 + base, base, rng)
        
        # Final Segmentation Head (1x1 conv to map to classes)
        # Using 1 output channel for Binary classification (sigmoid approximation via dice)
        scale = np.sqrt(2.0 / (base * 1 * 1))
        self.final_w = nn.Tensor(rng.randn(1, base, 1, 1).astype(np.float32) * scale, requires_grad=True, track=False)
        self.final_b = nn.Tensor(np.zeros((1,), dtype=np.float32), requires_grad=True, track=False)
        
        self.params = (
            self.enc1.parameters() + self.enc2.parameters() + self.enc3.parameters() +
            self.bot.parameters() +
            self.up3_conv.parameters() + self.up2_conv.parameters() + self.up1_conv.parameters() +
            [self.final_w, self.final_b]
        )
        
        print(f"U-Net Initialized with {len(self.params)} parameter tensors.")

    def forward(self, x):
        # Encoder Path
        e1 = self.enc1(x)                               # 128x128
        p1 = nn.maxpool2d(e1, pool_size=2, stride=2)    # 64x64
        
        e2 = self.enc2(p1)                              # 64x64
        p2 = nn.maxpool2d(e2, pool_size=2, stride=2)    # 32x32
        
        e3 = self.enc3(p2)                              # 32x32
        p3 = nn.maxpool2d(e3, pool_size=2, stride=2)    # 16x16
        
        # Bottleneck
        b = self.bot(p3)                                # 16x16
        
        # Decoder Path (UpSample -> Concat -> Conv)
        u3 = nn.upsample2d(b, scaleH=2, scaleW=2)       # 32x32
        c3 = nn.concat([u3, e3], axis=1)
        d3 = self.up3_conv(c3)
        
        u2 = nn.upsample2d(d3, scaleH=2, scaleW=2)      # 64x64
        c2 = nn.concat([u2, e2], axis=1)
        d2 = self.up2_conv(c2)
        
        u1 = nn.upsample2d(d2, scaleH=2, scaleW=2)      # 128x128
        c1 = nn.concat([u1, e1], axis=1)
        d1 = self.up1_conv(c1)
        
        # Output Head
        logits = nn.conv2d(d1, self.final_w, self.final_b, padding=0)
        
        # Sigmoid bounds output to [0, 1] which Dice Loss requires
        preds = nn.sigmoid(logits)
        
        return preds
        
    def step(self, lr):
        # Call the fused C++ SGD engine manually for all parameters
        num_params = len(self.params)
        
        # Ensure all params have initialized grad buffers (can happen on first step if gradients sum to 0 trivially but shouldn't)
        for p in self.params:
            if p.grad is None:
                p.grad = nn.Tensor(np.zeros_like(p.data), track=False)
                
        params_array = (ctypes.c_void_p * num_params)(*[p.gpu_buf for p in self.params])
        grads_array = (ctypes.c_void_p * num_params)(*[p.grad.gpu_buf for p in self.params])
        sizes_array = (ctypes.c_uint * num_params)(*[p.size for p in self.params])
        nn.lib.SGDBatch(params_array, grads_array, sizes_array, num_params, lr, 1e30) # No element-wise clip (norm clip already applied)
        
import ctypes

def train():
    X_train, Y_train = load_dataset("segment")
    if X_train is None:
        return
        
    num_samples = X_train.shape[0]
    num_batches = num_samples // BATCH_SIZE
    
    model = UNet()
    
    rdoc = rdoc_helper.get_rdoc_api()
    
    print("\nStarting Training Loop...")
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        start_time = time.time()
        
        # Shuffle dataset (deterministic per epoch to match PyTorch script)
        np.random.seed(epoch)
        indices = np.random.permutation(num_samples)
        
        for i in range(num_batches):
            batch_indices = indices[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            x_batch = X_train[batch_indices]
            y_batch = Y_train[batch_indices]
            
            # 1. Upload to GPU
            x_tensor = nn.Tensor(x_batch, requires_grad=False)
            y_tensor = nn.Tensor(y_batch, requires_grad=False)
            
            # RenderDoc capture first batch
            if epoch == 0 and i == 0 and rdoc:
                print(">>> STARTING RENDERDOC CAPTURE <<<")
                rdoc.StartFrameCapture(None, None)
            
            # 2. Forward Pass
            preds = model.forward(x_tensor)
            
            # 3. Dice Loss calculation
            loss = nn.dice_loss(preds, y_tensor)
            batch_loss_val = float(np.mean(loss.sync()))  # batch mean (matches PyTorch)
            total_loss += batch_loss_val
            
            # 4. Backward Pass (Clear previous grads first)
            for p in model.params:
                if p.grad is not None and p.grad.gpu_buf is not None:
                    nn.lib.ClearBuffer(p.grad.gpu_buf)
                    
            loss.backward()
            
            # 5. Clip gradient norm (matches PyTorch clip_grad_norm_)
            nn.clip_grad_norm(model.params, max_norm=1.0)
            
            # 6. Optimizer Step
            model.step(lr=LEARNING_RATE)
            
            # 7. Memory Cleanup
            nn.release_all_buffers()
            nn.lib.FlushGPU()
            
            # RenderDoc end capture
            if epoch == 0 and i == 0 and rdoc:
                rdoc.EndFrameCapture(None, None)
                print(">>> CAPTURE COMPLETE! Check the RenderDoc UI. <<<")
                sys.exit(0)
            
            print(f"  Batch {i+1}/{num_batches} - Loss: {batch_loss_val:.4f}", end="\r")
            
        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(1, num_batches)
        print(f"\nEpoch {epoch+1}/{EPOCHS} | Avg Dice Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
    
    # ── Visualize Results ──
    print("\nGenerating prediction visualizations...")
    num_samples_to_show = min(6, num_samples)
    
    fig_width = 3 * 3  # 3 columns
    fig_height = num_samples_to_show * 3
    
    from PIL import Image as PILImage
    
    # Create a grid image: each row = [Original, Ground Truth, Prediction]
    grid_w = 3 * IMG_SIZE + 4 * 10  # 3 images + padding
    grid_h = num_samples_to_show * IMG_SIZE + (num_samples_to_show + 1) * 10
    grid = PILImage.new("RGB", (grid_w, grid_h), (30, 30, 30))
    
    for idx in range(num_samples_to_show):
        x_sample = X_train[idx:idx+1]
        y_sample = Y_train[idx:idx+1]
        
        x_tensor = nn.Tensor(x_sample, requires_grad=False)
        pred = model.forward(x_tensor)
        pred_np = pred.sync()
        nn.release_all_buffers()
        
        # Original image: (3, H, W) -> (H, W, 3)
        orig = (np.transpose(x_sample[0], (1, 2, 0)) * 255).astype(np.uint8)
        # Ground truth mask: (1, H, W) -> (H, W) -> RGB
        gt = (y_sample[0, 0] * 255).astype(np.uint8)
        # Predicted mask: (1, H, W) -> (H, W) -> binarized -> RGB
        pred_mask = (pred_np[0, 0] * 255).clip(0, 255).astype(np.uint8)
        
        y_off = 10 + idx * (IMG_SIZE + 10)
        grid.paste(PILImage.fromarray(orig), (10, y_off))
        grid.paste(PILImage.fromarray(gt).convert("RGB"), (10 + IMG_SIZE + 10, y_off))
        grid.paste(PILImage.fromarray(pred_mask).convert("RGB"), (10 + 2 * (IMG_SIZE + 10), y_off))
    
    out_path = "unet_results_directcompute.png"
    grid.save(out_path)
    print(f"Saved visualization to {out_path}")
    print("  Columns: Original | Ground Truth | Prediction")

if __name__ == "__main__":
    train()
