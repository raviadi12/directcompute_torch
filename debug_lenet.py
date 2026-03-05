"""LeNet GPU shader debug — per-shader profiling + matrix shape analysis."""
import os, numpy as np, time
from PIL import Image
from nn_engine import (Tensor, Linear, ConvLayer, SGD, Metrics,
                       relu, softmax_ce, maxpool2d, flatten,
                       end_batch, profiler)

# ── Data ─────────────────────────────────────────────────────────────────────
def load_mnist(limit_per_class=500):
    images, labels = [], []
    for digit in range(10):
        folder = f"mnist/{digit}"
        for f in os.listdir(folder)[:limit_per_class]:
            img = Image.open(os.path.join(folder, f)).convert('L')
            images.append(np.array(img).reshape(1, 28, 28).astype(np.float32) / 255.0)
            labels.append(digit)
    return np.array(images), np.array(labels)

# ── Model ────────────────────────────────────────────────────────────────────
X, Y = load_mnist()
idx = np.arange(len(X)); np.random.shuffle(idx); X, Y = X[idx], Y[idx]
split = int(0.9 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_val,   Y_val   = X[split:], Y[split:]
print(f"Dataset: Train={len(X_train)}, Val={len(X_val)}")

c1 = ConvLayer(1, 6, 5)
c2 = ConvLayer(6, 16, 5)
l1 = Linear(16*4*4, 120)
l2 = Linear(120, 84)
l3 = Linear(84, 10)
params = [c1.filters, c1.bias, c2.filters, c2.bias,
          l1.w, l1.b, l2.w, l2.b, l3.w, l3.b]
opt = SGD(params, lr=0.01)
metrics = Metrics()
batch_size = 64

# ── Analyze conv layer matrix shapes ─────────────────────────────────────────
print(f"\n{'='*80}")
print(f"  CONV LAYER MATRIX SHAPE ANALYSIS (batch_size={batch_size})")
print(f"{'='*80}")

_COARSEN_THRESHOLD = 64  # from nn_engine.py

def analyze_conv(name, inC, outC, kH, kW, inH, inW, stride, padding):
    outH = (inH + 2*padding - kH) // stride + 1
    outW = (inW + 2*padding - kW) // stride + 1
    totalRow = inC * kH * kW
    totalCol = batch_size * outH * outW
    
    # Forward matmul: outC × totalRow × totalCol
    M, K, N = outC, totalRow, totalCol
    kernel = "coarsened" if M >= _COARSEN_THRESHOLD and N >= _COARSEN_THRESHOLD else "universal"
    im2col_size = totalRow * totalCol * 4 / 1024 / 1024  # MB
    
    print(f"\n  {name}: input ({batch_size}x{inC}x{inH}x{inW}) -> output ({batch_size}x{outC}x{outH}x{outW})")
    print(f"    Forward matmul:  M={M}, K={K}, N={N}  -> {kernel}")
    print(f"    im2col buffer:   {totalRow}x{totalCol} = {totalRow*totalCol:,} floats ({im2col_size:.1f} MB)")
    
    # Backward dF matmul: outC × totalCol × totalRow (transposed)
    M2, K2, N2 = outC, totalCol, totalRow
    kernel2 = "coarsened" if M2 >= _COARSEN_THRESHOLD and N2 >= _COARSEN_THRESHOLD else "universal"
    print(f"    Backward dF:     M={M2}, K={K2}, N={N2}  -> {kernel2}")
    
    # Backward dI matmul: totalRow × outC × totalCol
    M3, K3, N3 = totalRow, outC, totalCol
    kernel3 = "coarsened" if M3 >= _COARSEN_THRESHOLD and N3 >= _COARSEN_THRESHOLD else "universal"
    print(f"    Backward dInput: M={M3}, K={K3}, N={N3}  -> {kernel3}")
    
    return outH, outW

print("\n  -- LeNet --")
h, w = analyze_conv("Conv1", 1, 6, 5, 5, 28, 28, 1, 0)     # 28→24
h, w = 12, 12  # after maxpool
h, w = analyze_conv("Conv2", 6, 16, 5, 5, h, w, 1, 0)       # 12→8

print("\n  -- AlexNet (for comparison) --")
h, w = analyze_conv("Conv1", 3, 64, 11, 11, 224, 224, 4, 2)  # 224→55
h, w = (h-3)//2+1, (w-3)//2+1  # maxpool 3×3 stride 2 → 27
h, w = analyze_conv("Conv2", 64, 192, 5, 5, h, w, 1, 2)      # 27→27
h, w = (h-3)//2+1, (w-3)//2+1  # maxpool → 13
h, w = analyze_conv("Conv3", 192, 384, 3, 3, h, w, 1, 1)     # 13→13
h, w = analyze_conv("Conv4", 384, 256, 3, 3, h, w, 1, 1)     # 13→13
h, w = analyze_conv("Conv5", 256, 256, 3, 3, h, w, 1, 1)     # 13→13

print(f"\n{'='*80}")

# ── Install GPU profiler ─────────────────────────────────────────────────────
profiler.install()
profiler.enable()

# ── Train 2 epochs ───────────────────────────────────────────────────────────
for epoch in range(2):
    profiler.reset()
    metrics.reset()
    t0 = time.perf_counter()

    for i in range(0, len(X_train), batch_size):
        end = min(i + batch_size, len(X_train))
        opt.zero_grad()
        xb = Tensor(X_train[i:end], track=True)
        yb = Tensor(Y_train[i:end], track=True)

        x = relu(c1(xb))
        x = maxpool2d(x)
        x = relu(c2(x))
        x = maxpool2d(x)
        x = flatten(x)
        x = relu(l1(x))
        x = relu(l2(x))
        logits = l3(x)
        loss = softmax_ce(logits, yb)
        metrics.update(loss, logits, yb)

        loss.backward()
        opt.step(clip=1.0)
        end_batch()

    train_loss, train_acc = metrics.collect(len(X_train))
    wall = (time.perf_counter() - t0) * 1000
    print(f"Epoch {epoch+1}/2 | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Wall: {wall:.0f} ms")
    profiler.report(epoch_label=f"Epoch {epoch+1}")

profiler.disable()
profiler.uninstall()
