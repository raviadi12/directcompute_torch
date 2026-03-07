"""Transfer learning with pretrained MobileNetV2 on PetImages (Cat vs Dog).

Usage:
  1. python download_mobilenet.py        (once, requires torch)
  2. python train_petnet_transfer.py     (uses DirectCompute engine only)

Loads ImageNet-pretrained MobileNetV2, freezes the backbone,
and fine-tunes a 2-class linear classifier head on PetImages.
"""
import os, time, ctypes
import numpy as np
from PIL import Image
from nn_engine import (Tensor, Linear, ConvLayer, DepthwiseConvLayer, Model,
                       AdamW, Metrics, BatchNorm2d,
                       relu6, softmax_ce, global_avg_pool2d, add, flatten,
                       end_batch, get_pool_stats, get_pool_memory, lib)


# ── Architecture ──────────────────────────────────────────────────────────────

class InvertedResidual:
    """MobileNetV2 inverted residual block."""
    def __init__(self, inp, oup, stride, expand_ratio):
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res = (stride == 1 and inp == oup)
        self.expand_ratio = expand_ratio

        if expand_ratio != 1:
            self.expand_conv = ConvLayer(inp, hidden_dim, ks=1)
            self.expand_bn = BatchNorm2d(hidden_dim)

        self.dw = DepthwiseConvLayer(hidden_dim, ks=3, stride=stride, padding=1)
        self.dw_bn = BatchNorm2d(hidden_dim)
        self.project = ConvLayer(hidden_dim, oup, ks=1)
        self.project_bn = BatchNorm2d(oup)

    def __call__(self, x):
        identity = x
        if self.expand_ratio != 1:
            x = relu6(self.expand_bn(self.expand_conv(x)))
        x = relu6(self.dw_bn(self.dw(x)))
        x = self.project_bn(self.project(x))
        if self.use_res:
            x = add(x, identity)
        return x

    def parameters(self):
        params = []
        if self.expand_ratio != 1:
            params.extend([self.expand_conv.filters, self.expand_conv.bias,
                           self.expand_bn.gamma, self.expand_bn.beta])
        params.extend([self.dw.filters, self.dw.bias,
                       self.dw_bn.gamma, self.dw_bn.beta,
                       self.project.filters, self.project.bias,
                       self.project_bn.gamma, self.project_bn.beta])
        return params

    def train(self, mode=True):
        if self.expand_ratio != 1:
            self.expand_bn.training = mode
        self.dw_bn.training = mode
        self.project_bn.training = mode


# MobileNetV2 inverted residual settings: (expand_ratio, channels, repeats, stride)
MOBILENET_SETTINGS = [
    (1,  16,  1, 1),
    (6,  24,  2, 2),
    (6,  32,  3, 2),
    (6,  64,  4, 2),
    (6,  96,  3, 1),
    (6, 160,  3, 2),
    (6, 320,  1, 1),
]


class MobileNetV2(Model):
    def __init__(self, num_classes=2):
        super().__init__()
        # Initial conv: 3 -> 32, stride 2
        self.conv0 = ConvLayer(3, 32, ks=3, stride=2, padding=1)
        self.bn0 = BatchNorm2d(32)

        # Build inverted residual blocks
        input_channel = 32
        block_idx = 0
        for t, c, n, s in MOBILENET_SETTINGS:
            for i in range(n):
                stride = s if i == 0 else 1
                setattr(self, f'b{block_idx}', InvertedResidual(input_channel, c, stride, t))
                block_idx += 1
                input_channel = c
        self.num_blocks = block_idx  # 17

        # Final 1x1 conv: 320 -> 1280
        self.conv_last = ConvLayer(320, 1280, ks=1)
        self.bn_last = BatchNorm2d(1280)

        # Classifier head
        self.classifier = Linear(1280, num_classes)

    def forward(self, x):
        x = relu6(self.bn0(self.conv0(x)))
        for i in range(self.num_blocks):
            x = getattr(self, f'b{i}')(x)
        x = relu6(self.bn_last(self.conv_last(x)))
        x = global_avg_pool2d(x)  # (N, 1280, 1, 1) -> (N, 1280)
        x = flatten(x)
        return self.classifier(x)

    def extract_features(self, x):
        """Forward through backbone only, returns 1280-dim features (no classifier)."""
        x = relu6(self.bn0(self.conv0(x)))
        for i in range(self.num_blocks):
            x = getattr(self, f'b{i}')(x)
        x = relu6(self.bn_last(self.conv_last(x)))
        x = global_avg_pool2d(x)
        x = flatten(x)
        return x

    def parameters(self):
        params = [self.conv0.filters, self.conv0.bias,
                  self.bn0.gamma, self.bn0.beta]
        for i in range(self.num_blocks):
            params.extend(getattr(self, f'b{i}').parameters())
        params.extend([self.conv_last.filters, self.conv_last.bias,
                       self.bn_last.gamma, self.bn_last.beta,
                       self.classifier.w, self.classifier.b])
        return params

    def train(self, mode=True):
        self.bn0.training = mode
        for i in range(self.num_blocks):
            getattr(self, f'b{i}').train(mode)
        self.bn_last.training = mode

    def load_pretrained(self, npz_path):
        """Load pretrained weights from .npz exported by download_mobilenet.py."""
        weights = np.load(npz_path)

        def upload(tensor, key):
            data = weights[key].astype(np.float32)
            assert data.shape == tensor.shape, \
                f"Shape mismatch for {key}: expected {tensor.shape}, got {data.shape}"
            tensor.data = data.copy()
            lib.UpdateBuffer(tensor.gpu_buf,
                             data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        def load_conv_bn(conv_key, bn_key, conv_layer, bn_layer):
            upload(conv_layer.filters, f"{conv_key}.weight")
            upload(bn_layer.gamma,        f"{bn_key}.weight")
            upload(bn_layer.beta,         f"{bn_key}.bias")
            upload(bn_layer.running_mean, f"{bn_key}.running_mean")
            upload(bn_layer.running_var,  f"{bn_key}.running_var")

        # Initial conv + BN
        load_conv_bn("features.0.0", "features.0.1", self.conv0, self.bn0)

        # Inverted residual blocks
        block_idx = 0
        feature_idx = 1
        for t, c, n, s in MOBILENET_SETTINGS:
            for _ in range(n):
                blk = getattr(self, f'b{block_idx}')
                p = f"features.{feature_idx}"
                if t != 1:
                    # expand + DW + project
                    load_conv_bn(f"{p}.conv.0.0", f"{p}.conv.0.1",
                                 blk.expand_conv, blk.expand_bn)
                    upload(blk.dw.filters,        f"{p}.conv.1.0.weight")
                    upload(blk.dw_bn.gamma,       f"{p}.conv.1.1.weight")
                    upload(blk.dw_bn.beta,        f"{p}.conv.1.1.bias")
                    upload(blk.dw_bn.running_mean,f"{p}.conv.1.1.running_mean")
                    upload(blk.dw_bn.running_var, f"{p}.conv.1.1.running_var")
                    upload(blk.project.filters,        f"{p}.conv.2.weight")
                    upload(blk.project_bn.gamma,       f"{p}.conv.3.weight")
                    upload(blk.project_bn.beta,        f"{p}.conv.3.bias")
                    upload(blk.project_bn.running_mean,f"{p}.conv.3.running_mean")
                    upload(blk.project_bn.running_var, f"{p}.conv.3.running_var")
                else:
                    # DW + project only (expand_ratio=1)
                    upload(blk.dw.filters,        f"{p}.conv.0.0.weight")
                    upload(blk.dw_bn.gamma,       f"{p}.conv.0.1.weight")
                    upload(blk.dw_bn.beta,        f"{p}.conv.0.1.bias")
                    upload(blk.dw_bn.running_mean,f"{p}.conv.0.1.running_mean")
                    upload(blk.dw_bn.running_var, f"{p}.conv.0.1.running_var")
                    upload(blk.project.filters,        f"{p}.conv.1.weight")
                    upload(blk.project_bn.gamma,       f"{p}.conv.2.weight")
                    upload(blk.project_bn.beta,        f"{p}.conv.2.bias")
                    upload(blk.project_bn.running_mean,f"{p}.conv.2.running_mean")
                    upload(blk.project_bn.running_var, f"{p}.conv.2.running_var")
                block_idx += 1
                feature_idx += 1

        # Final conv + BN
        load_conv_bn("features.18.0", "features.18.1", self.conv_last, self.bn_last)

        # Skip classifier (we use our own 2-class head)
        loaded = feature_idx + 1  # +1 for initial conv
        total_params = sum(p.data.size for p in self.parameters())
        print(f"Loaded {loaded} layer groups from {npz_path}")
        print(f"Model has {total_params:,} parameters")


# ── Data loading ──────────────────────────────────────────────────────────────

# ImageNet normalization used during MobileNetV2 pretraining
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


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    weights_path = "mobilenet_v2_weights.npz"
    if not os.path.exists(weights_path):
        print(f"ERROR: {weights_path} not found!")
        print("Run 'python download_mobilenet.py' first (requires torch).")
        return

    # Load data
    train_limit = 800
    X, Y = load_pets(limit_per_class=train_limit, size=224)
    if len(X) == 0:
        print("No images found in PetImages/ directory!")
        return

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X, Y = X[idx], Y[idx]
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    print(f"Dataset: Train={len(X_train)}, Val={len(X_val)}")

    # Load unseen test data (images the model has never seen)
    test_limit = 200
    X_test, Y_test = load_pets(limit_per_class=test_limit, size=224,
                               skip_per_class=train_limit)
    print(f"Test set: {len(X_test)} unseen images")

    # Build model and load pretrained backbone
    print("\nBuilding MobileNetV2...")
    model = MobileNetV2(num_classes=2)
    model.load_pretrained(weights_path)

    # Freeze entire backbone
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # ── Phase 1: Extract ALL features once with frozen backbone ──────────
    print("\nExtracting features through frozen backbone...")
    extract_bs = 2  # small batch to fit iGPU memory
    start = time.time()

    def extract_all(images, label=""):
        feats = []
        for i in range(0, len(images), extract_bs):
            end = min(i + extract_bs, len(images))
            xb = Tensor(images[i:end])
            fb = model.extract_features(xb)
            feats.append(fb.sync().copy())
            end_batch()
            if (i // extract_bs) % 20 == 0:
                print(f"  {label} {i}/{len(images)}...", end="\r")
        print(f"  {label} {len(images)}/{len(images)} done")
        return np.concatenate(feats, axis=0)

    F_train = extract_all(X_train, "train")
    F_val = extract_all(X_val, "val")
    F_test = extract_all(X_test, "test")
    extract_time = time.time() - start
    print(f"Feature extraction done in {extract_time:.1f}s  "
          f"({F_train.shape[1]}-dim features)")

    # Free backbone memory + images — only features needed from here
    del model, X, X_train, X_val, X_test
    end_batch()

    # ── Phase 2: Train linear classifier on cached features ──────────────
    classifier = Linear(1280, 2)
    train_params = [classifier.w, classifier.b]
    optimizer = AdamW(train_params, lr=0.001, weight_decay=0.01)
    metrics = Metrics()

    batch_size = 32
    epochs = 30

    print(f"\nTraining classifier on extracted features (batch_size={batch_size})...")
    train_start = time.time()

    for epoch in range(epochs):
        # Shuffle training data each epoch
        perm = np.random.permutation(len(F_train))
        F_shuf, Y_shuf = F_train[perm], Y_train[perm]

        # ── Train ──
        metrics.reset()
        for i in range(0, len(F_shuf), batch_size):
            end = min(i + batch_size, len(F_shuf))
            optimizer.zero_grad()
            xb = Tensor(F_shuf[i:end])
            yb = Tensor(Y_shuf[i:end])
            logits = classifier(xb)
            loss = softmax_ce(logits, yb)
            metrics.update(loss, logits, yb)
            loss.backward()
            optimizer.step(clip=1.0)
            end_batch()

        train_loss, train_acc = metrics.collect(len(F_train))

        # ── Validate ──
        metrics.reset()
        for i in range(0, len(F_val), batch_size):
            end = min(i + batch_size, len(F_val))
            xb = Tensor(F_val[i:end])
            yb = Tensor(Y_val[i:end])
            logits = classifier(xb)
            loss = softmax_ce(logits, yb)
            metrics.update(loss, logits, yb)
            end_batch()

        val_loss, val_acc = metrics.collect(len(F_val))
        elapsed = time.time() - train_start
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Loss: {train_loss:.4f} | Train: {train_acc:.1%} | Val: {val_acc:.1%} | "
              f"Time: {elapsed:.1f}s")

    total = time.time() - start
    hits, misses = get_pool_stats()
    vram = get_pool_memory() / (1024 * 1024)
    print(f"\nTotal time: {total:.1f}s (extract: {extract_time:.1f}s + train: {time.time()-train_start:.1f}s)")
    print(f"Pool stats: {hits} hits, {misses} misses, {vram:.1f}MB VRAM in pool")

    # ── Phase 3: Evaluate on unseen test data ────────────────────────────
    print(f"\n{'='*60}")
    print(f"Evaluating on {len(F_test)} UNSEEN test images...")
    metrics.reset()
    for i in range(0, len(F_test), batch_size):
        end = min(i + batch_size, len(F_test))
        xb = Tensor(F_test[i:end])
        yb = Tensor(Y_test[i:end])
        logits = classifier(xb)
        loss = softmax_ce(logits, yb)
        metrics.update(loss, logits, yb)
        end_batch()

    test_loss, test_acc = metrics.collect(len(F_test))
    n_cats = int((Y_test == 0).sum())
    n_dogs = int((Y_test == 1).sum())
    print(f"Test set: {len(F_test)} images ({n_cats} cats, {n_dogs} dogs)")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.1%}")


if __name__ == "__main__":
    main()
