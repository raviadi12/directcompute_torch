import time
import os
import glob
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Hyperparameters ---
BATCH_SIZE = 2
IMG_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 1e-3

# --- Dataset Loader (Exact same logic) ---
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
        print("No images found.")
        return None, None
        
    X_list = []
    Y_list = []
    
    for img_p, mask_p in zip(img_paths, mask_paths):
        img = Image.open(img_p).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_arr = np.transpose(img_arr, (2, 0, 1))
        
        mask = Image.open(mask_p).convert("L")
        mask = mask.resize((IMG_SIZE, IMG_SIZE))
        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        mask_arr = np.expand_dims(mask_arr, axis=0)
        # Any non-black is foreground
        mask_arr = np.where(mask_arr > 0.01, 1.0, 0.0).astype(np.float32)
        
        X_list.append(img_arr)
        Y_list.append(mask_arr)
        
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    
    print(f"Loaded {X.shape[0]} samples. Input: {X.shape}, Targets: {Y.shape}")
    return X, Y

# --- Dice Loss ---
class DiceLoss(nn.Module):
    def forward(self, pred, target):
        batch = pred.shape[0]
        # pred = pred.view(batch, -1)
        # target = target.view(batch, -1)
        
        intersection = torch.sum(pred * target, dim=[1, 2, 3])
        sum_pred = torch.sum(pred, dim=[1, 2, 3])
        sum_target = torch.sum(target, dim=[1, 2, 3])
        
        dice = (2.0 * intersection) / (sum_pred + sum_target + 1e-6)
        return torch.mean(1.0 - dice)

# --- U-Net Model Definition ---
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)

    def init_from_rng(self, rng, in_c, out_c):
        """Initialize weights from numpy RNG to match DirectCompute exactly."""
        with torch.no_grad():
            self.conv1.weight.copy_(torch.from_numpy(rng.randn(out_c, in_c, 3, 3).astype(np.float32) * np.sqrt(2.0 / (in_c * 9))))
            self.conv1.bias.zero_()
            self.conv2.weight.copy_(torch.from_numpy(rng.randn(out_c, out_c, 3, 3).astype(np.float32) * np.sqrt(2.0 / (out_c * 9))))
            self.conv2.bias.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x

class UNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(3, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        
        # Bottleneck
        self.bot = ConvBlock(base * 4, base * 8)
        
        # Decoder
        self.up3_conv = ConvBlock(base * 8 + base * 4, base * 4)
        self.up2_conv = ConvBlock(base * 4 + base * 2, base * 2)
        self.up1_conv = ConvBlock(base * 2 + base, base)
        
        # Final Segmentation Head
        self.final_conv = nn.Conv2d(base, 1, kernel_size=1, padding=0)
        
        # Initialize ALL weights from numpy RNG (same sequence as DirectCompute)
        rng = np.random.RandomState(42)
        self.enc1.init_from_rng(rng, 3, base)
        self.enc2.init_from_rng(rng, base, base * 2)
        self.enc3.init_from_rng(rng, base * 2, base * 4)
        self.bot.init_from_rng(rng, base * 4, base * 8)
        self.up3_conv.init_from_rng(rng, base * 8 + base * 4, base * 4)
        self.up2_conv.init_from_rng(rng, base * 4 + base * 2, base * 2)
        self.up1_conv.init_from_rng(rng, base * 2 + base, base)
        with torch.no_grad():
            self.final_conv.weight.copy_(torch.from_numpy(rng.randn(1, base, 1, 1).astype(np.float32) * np.sqrt(2.0 / base)))
            self.final_conv.bias.zero_()

    def forward(self, x):
        # Encoder Path
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2)
        
        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2)
        
        e3 = self.enc3(p2)
        p3 = F.max_pool2d(e3, 2)
        
        # Bottleneck
        b = self.bot(p3)
        
        # Decoder Path (UpSample -> Concat -> Conv)
        # Using nearest neighbor upsampling
        u3 = F.interpolate(b, scale_factor=2, mode='nearest')
        c3 = torch.cat([u3, e3], dim=1)
        d3 = self.up3_conv(c3)
        
        u2 = F.interpolate(d3, scale_factor=2, mode='nearest')
        c2 = torch.cat([u2, e2], dim=1)
        d2 = self.up2_conv(c2)
        
        u1 = F.interpolate(d2, scale_factor=2, mode='nearest')
        c1 = torch.cat([u1, e1], dim=1)
        d1 = self.up1_conv(c1)
        
        # Output Head
        logits = self.final_conv(d1)
        preds = torch.sigmoid(logits)
        
        return preds

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} backend")
    
    X_train, Y_train = load_dataset("segment")
    if X_train is None:
        return
        
    num_samples = X_train.shape[0]
    num_batches = num_samples // BATCH_SIZE
    
    # Init model (weights from numpy RNG seed=42 to match DirectCompute exactly)
    model = UNet().to(device)
    
    criterion = DiceLoss()
    # DirectCompute implementation uses basic SGD with clip=1.0 and no momentum/weight-decay
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting Training Loop...")
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        start_time = time.time()
        
        # Match python numpy RNG for shuffling
        np.random.seed(epoch) # deterministic shuffle per epoch based on numpy just to be 100% 1-to-1 if desired
        indices = np.random.permutation(num_samples)
        
        for i in range(num_batches):
            batch_indices = indices[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            
            x_batch = torch.from_numpy(X_train[batch_indices]).to(device)
            y_batch = torch.from_numpy(Y_train[batch_indices]).to(device)
            
            # Forward
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Apply same gradient clipping as DirectCompute engine (1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step
            optimizer.step()
            
            batch_loss_val = loss.item()
            total_loss += batch_loss_val
            
            print(f"  Batch {i+1}/{num_batches} - Loss: {batch_loss_val:.4f}", end="\r")
            
        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(1, num_batches)
        print(f"\nEpoch {epoch+1}/{EPOCHS} | Avg Dice Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
    
    # ── Visualize Results ──
    print("\nGenerating prediction visualizations...")
    num_samples_to_show = min(6, num_samples)
    
    model.eval()
    
    from PIL import Image as PILImage
    
    # Create a grid image: each row = [Original, Ground Truth, Prediction]
    grid_w = 3 * IMG_SIZE + 4 * 10
    grid_h = num_samples_to_show * IMG_SIZE + (num_samples_to_show + 1) * 10
    grid = PILImage.new("RGB", (grid_w, grid_h), (30, 30, 30))
    
    with torch.no_grad():
        for idx in range(num_samples_to_show):
            x_sample = torch.from_numpy(X_train[idx:idx+1]).to(device)
            y_sample = Y_train[idx:idx+1]
            
            pred = model(x_sample).cpu().numpy()
            
            # Original image: (3, H, W) -> (H, W, 3)
            orig = (np.transpose(X_train[idx], (1, 2, 0)) * 255).astype(np.uint8)
            # Ground truth mask: (1, H, W) -> (H, W)
            gt = (y_sample[0, 0] * 255).astype(np.uint8)
            # Predicted mask: (1, H, W) -> (H, W)
            pred_mask = (pred[0, 0] * 255).clip(0, 255).astype(np.uint8)
            
            y_off = 10 + idx * (IMG_SIZE + 10)
            grid.paste(PILImage.fromarray(orig), (10, y_off))
            grid.paste(PILImage.fromarray(gt).convert("RGB"), (10 + IMG_SIZE + 10, y_off))
            grid.paste(PILImage.fromarray(pred_mask).convert("RGB"), (10 + 2 * (IMG_SIZE + 10), y_off))
    
    out_path = "unet_results_pytorch.png"
    grid.save(out_path)
    print(f"Saved visualization to {out_path}")
    print("  Columns: Original | Ground Truth | Prediction")

if __name__ == "__main__":
    train()
