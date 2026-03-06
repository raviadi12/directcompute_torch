import os
import numpy as np
from PIL import Image
import time
from nn_engine import (Tensor, Linear, ConvLayer, SGD, Metrics,
                       relu, softmax_ce, maxpool2d, flatten, end_batch,
                       bias_relu, get_pool_stats, get_pool_memory)

def load_pets(limit_per_class=100, size=224):
    images, labels = [], []
    classes = ['Cat', 'Dog']
    for i, cls in enumerate(classes):
        folder = f"PetImages/{cls}"
        files = os.listdir(folder)[:limit_per_class]
        print(f"Loading {cls} images...")
        for f in files:
            try:
                img = Image.open(os.path.join(folder, f)).convert('RGB')
                img = img.resize((size, size))
                images.append(np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0)
                labels.append(i)
            except:
                continue
    return np.array(images), np.array(labels)

def train_alexnet():
    X, Y = load_pets(limit_per_class=2000)
    idx = np.arange(len(X)); np.random.shuffle(idx); X, Y = X[idx], Y[idx]
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    print(f"Dataset Split: Train={len(X_train)}, Val={len(X_val)}")

    # AlexNet Architecture
    c1 = ConvLayer(3, 64, 11, stride=4, padding=2)
    c2 = ConvLayer(64, 192, 5, padding=2)
    c3 = ConvLayer(192, 384, 3, padding=1)
    c4 = ConvLayer(384, 256, 3, padding=1)
    c5 = ConvLayer(256, 256, 3, padding=1)
    l1 = Linear(256 * 6 * 6, 512)
    l2 = Linear(512, 512)
    l3 = Linear(512, 2)

    params = [c1.filters, c1.bias, c2.filters, c2.bias, c3.filters, c3.bias,
              c4.filters, c4.bias, c5.filters, c5.bias,
              l1.w, l1.b, l2.w, l2.b, l3.w, l3.b]
    optimizer = SGD(params, lr=0.01)
    metrics = Metrics()

    batch_size = 64
    accum_steps = 4              
    epochs = 50

    def forward(xb):
        x = c1(xb, relu=True)
        x = maxpool2d(x, pool_size=3, stride=2)
        x = c2(x, relu=True)
        x = maxpool2d(x, pool_size=3, stride=2)
        x = c3(x, relu=True)
        x = c4(x, relu=True)
        x = c5(x, relu=True)
        x = maxpool2d(x, pool_size=3, stride=2)
        x = flatten(x)
        x = l1(x, relu=True)
        x = l2(x, relu=True)
        return l3(x)

    # AlexNet creates huge intermediate buffers (im2col > 60MB each).
    # On iGPU (128MB dedicated VRAM): flush every batch, no pool warming,
    # no ReusableTensor — keep VRAM free for active compute.
    end_batch.flush_interval = 1

    print(f"\nStarting AlexNet training on DirectCompute GPU...")
    print(f"  Batch size: {batch_size}, Accum steps: {accum_steps}, Effective batch: {batch_size * accum_steps}")
    start = time.time()

    for epoch in range(epochs):
        # ── Train ──
        metrics.reset()
        step_count = 0
        t0 = time.perf_counter()

        for i in range(0, len(X_train), batch_size):
            end = min(i + batch_size, len(X_train))
            if step_count % accum_steps == 0:
                optimizer.zero_grad()

            xb = Tensor(X_train[i:end], track=True)
            yb = Tensor(Y_train[i:end], track=True)

            logits = forward(xb)
            loss = softmax_ce(logits, yb)
            metrics.update(loss, logits, yb)

            loss.backward()
            step_count += 1
            if step_count % accum_steps == 0 or end >= len(X_train):
                optimizer.step(clip=1.0)
            end_batch()

        train_loss, train_acc = metrics.collect(len(X_train))

        # ── Validate ──
        metrics.reset()
        for i in range(0, len(X_val), batch_size):
            end = min(i + batch_size, len(X_val))
            xb = Tensor(X_val[i:end])
            yb = Tensor(Y_val[i:end])
            logits = forward(xb)
            loss = softmax_ce(logits, yb)
            metrics.update(loss, logits, yb)
            end_batch()

        val_loss, val_acc = metrics.collect(len(X_val))

        epoch_ms = (time.perf_counter() - t0) * 1000
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | {epoch_ms:.0f} ms")

    total = time.time() - start
    hits, misses = get_pool_stats()
    pool_mb = get_pool_memory() / (1024*1024)
    print(f"Total AlexNet DirectCompute Training Time: {total:.2f}s")
    print(f"Pool stats: {hits} hits, {misses} misses, {pool_mb:.1f}MB pooled")

if __name__ == "__main__":
    train_alexnet()
