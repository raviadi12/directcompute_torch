"""PyTorch version of MobileNetV2 transfer learning on PetImages (Cat vs Dog).
Same pipeline as train_petnet_transfer.py for direct comparison.

Usage: python train_petnet_transfer_pytorch.py
Requires: pip install torch torchvision pillow
"""
import os, time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[PyTorch] Using device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ── ImageNet normalization (same as DirectCompute version) ────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def load_pets(limit_per_class=200, size=224, skip_per_class=0):
    images, labels = [], []
    classes = ['Cat', 'Dog']
    print(f"Loading {limit_per_class * 2} images at {size}x{size} (skip={skip_per_class})...")
    for i, cls in enumerate(classes):
        folder = f"PetImages/{cls}"
        if not os.path.exists(folder):
            continue
        skipped, count = 0, 0
        for f in os.listdir(folder):
            if count >= limit_per_class:
                break
            if skipped < skip_per_class:
                skipped += 1
                continue
            try:
                path = os.path.join(folder, f)
                img = Image.open(path).convert('RGB')
                img = img.resize((size, size))
                arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
                arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
                images.append(arr)
                labels.append(i)
                count += 1
            except Exception:
                continue
    return np.array(images), np.array(labels)


def main():
    train_limit = 800
    X, Y = load_pets(limit_per_class=train_limit, size=224)
    if len(X) == 0:
        print("No images found in PetImages/ directory!")
        return

    idx = np.random.permutation(len(X))
    X, Y = X[idx], Y[idx]
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    print(f"Dataset: Train={len(X_train)}, Val={len(X_val)}")

    test_limit = 200
    X_test, Y_test = load_pets(limit_per_class=test_limit, size=224,
                                skip_per_class=train_limit)
    print(f"Test set: {len(X_test)} unseen images")

    # ── Build frozen backbone ─────────────────────────────────────────────────
    print("\nBuilding MobileNetV2...")
    try:
        from torchvision.models import MobileNet_V2_Weights
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    except (ImportError, TypeError):
        model = models.mobilenet_v2(pretrained=True)

    # Remove classifier, keep feature extractor
    backbone = model.features
    pool = nn.AdaptiveAvgPool2d((1, 1))
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    backbone = backbone.to(device)
    pool = pool.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")

    # ── Phase 1: Extract features ─────────────────────────────────────────────
    print("\nExtracting features through frozen backbone...")
    extract_bs = 8
    start = time.time()

    def extract_all(images, label=""):
        feats = []
        with torch.no_grad():
            for i in range(0, len(images), extract_bs):
                end = min(i + extract_bs, len(images))
                xb = torch.from_numpy(images[i:end]).to(device)
                fb = pool(backbone(xb)).flatten(1)
                feats.append(fb.cpu().numpy())
                if (i // extract_bs) % 20 == 0:
                    print(f"  {label} {i}/{len(images)}...", end="\r")
        print(f"  {label} {len(images)}/{len(images)} done")
        return np.concatenate(feats, axis=0)

    F_train = extract_all(X_train, "train")
    F_val   = extract_all(X_val,   "val")
    F_test  = extract_all(X_test,  "test")
    extract_time = time.time() - start
    print(f"Feature extraction done in {extract_time:.1f}s  ({F_train.shape[1]}-dim features)")

    del backbone, pool, model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Phase 2: Train linear classifier ─────────────────────────────────────
    classifier = nn.Linear(1280, 2).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    F_tr = torch.from_numpy(F_train)
    Y_tr = torch.from_numpy(Y_train).long()
    F_vl = torch.from_numpy(F_val)
    Y_vl = torch.from_numpy(Y_val).long()

    train_ds = TensorDataset(F_tr, Y_tr)
    val_ds   = TensorDataset(F_vl, Y_vl)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

    batch_size = 32
    epochs = 30
    print(f"\nTraining classifier on extracted features (batch_size={batch_size})...")
    train_start = time.time()

    for epoch in range(epochs):
        # ── Train ──
        classifier.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = classifier(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
        train_loss = total_loss / total
        train_acc  = correct / total

        # ── Validate ──
        classifier.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = classifier(xb)
                loss = criterion(logits, yb)
                total_loss += loss.item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)
        val_loss = total_loss / total
        val_acc  = correct / total
        elapsed = time.time() - train_start
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Loss: {train_loss:.4f} | Train: {train_acc:.1%} | Val: {val_acc:.1%} | "
              f"Time: {elapsed:.1f}s")

    total_time = time.time() - start
    print(f"\nTotal time: {total_time:.1f}s (extract: {extract_time:.1f}s + train: {time.time()-train_start:.1f}s)")

    # ── Phase 3: Evaluate on unseen test data ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Evaluating on {len(F_test)} UNSEEN test images...")
    classifier.eval()
    F_ts = torch.from_numpy(F_test)
    Y_ts = torch.from_numpy(Y_test).long()
    test_loader = DataLoader(TensorDataset(F_ts, Y_ts), batch_size=32, shuffle=False)

    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = classifier(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
    test_loss = total_loss / total
    test_acc  = correct / total
    n_cats = int((Y_test == 0).sum())
    n_dogs = int((Y_test == 1).sum())
    print(f"Test set: {len(F_test)} images ({n_cats} cats, {n_dogs} dogs)")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.1%}")


if __name__ == "__main__":
    main()
