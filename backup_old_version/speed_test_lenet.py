"""LeNet speed test — coarse-grained timing to find actual bottleneck."""
import ctypes, os, numpy as np, time
from nn_engine import (Tensor, Linear, ConvLayer, SGD, lib,
                       relu, softmax_ce, maxpool2d, flatten,
                       release_all_buffers,
                       gpu_argmax_correct, gpu_accumulate_loss)
from PIL import Image

def load_mnist(limit_per_class=500):
    images, labels = [], []
    for digit in range(10):
        folder = f"mnist/{digit}"
        for f in os.listdir(folder)[:limit_per_class]:
            img = Image.open(os.path.join(folder, f)).convert('L')
            images.append(np.array(img).reshape(1, 28, 28).astype(np.float32) / 255.0)
            labels.append(digit)
    return np.array(images), np.array(labels)

X, Y = load_mnist()
idx = np.arange(len(X)); np.random.shuffle(idx); X, Y = X[idx], Y[idx]
split = int(0.9 * len(X))
X_train, Y_train = X[:split], Y[:split]
print(f"Dataset: Train={len(X_train)}")

c1 = ConvLayer(1, 6, 5)
c2 = ConvLayer(6, 16, 5)
l1 = Linear(16*4*4, 120)
l2 = Linear(120, 84)
l3 = Linear(84, 10)
params = [c1.filters, c1.bias, c2.filters, c2.bias,
          l1.w, l1.b, l2.w, l2.b, l3.w, l3.b]
opt = SGD(params, lr=0.01)
batch_size = 64

correct_buf = lib.CreateBuffer(None, 1)
loss_accum_buf = lib.CreateBuffer(None, 1)
_f1 = np.empty(1, dtype=np.float32)
_f1_ptr = _f1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# Warmup
for i in range(0, len(X_train), batch_size):
    end = min(i + batch_size, len(X_train))
    opt.zero_grad()
    xb = Tensor(X_train[i:end], track=True)
    yb = Tensor(Y_train[i:end], track=True)
    x = relu(c1(xb)); x = maxpool2d(x); x = relu(c2(x)); x = maxpool2d(x)
    x = flatten(x); x = relu(l1(x)); x = relu(l2(x)); logits = l3(x)
    loss = softmax_ce(logits, yb); loss.backward()
    opt.step(clip=1.0); release_all_buffers()
print("Warmup done")

# Test 1: Full epoch with GPU metrics
t0 = time.perf_counter()
lib.ClearBuffer(correct_buf); lib.ClearBuffer(loss_accum_buf)
for i in range(0, len(X_train), batch_size):
    end = min(i + batch_size, len(X_train))
    opt.zero_grad()
    xb = Tensor(X_train[i:end], track=True)
    yb = Tensor(Y_train[i:end], track=True)
    x = relu(c1(xb)); x = maxpool2d(x); x = relu(c2(x)); x = maxpool2d(x)
    x = flatten(x); x = relu(l1(x)); x = relu(l2(x)); logits = l3(x)
    loss = softmax_ce(logits, yb)
    gpu_accumulate_loss(loss, loss_accum_buf)
    gpu_argmax_correct(logits, yb, correct_buf)
    loss.backward(); opt.step(clip=1.0); release_all_buffers()
t_loop = (time.perf_counter() - t0) * 1000

t1 = time.perf_counter()
lib.ReadBuffer(loss_accum_buf, _f1_ptr); total_loss = float(_f1[0])
lib.ReadBuffer(correct_buf, _f1_ptr)
correct = int(np.frombuffer(_f1.tobytes(), dtype=np.uint32)[0])
t_sync = (time.perf_counter() - t1) * 1000

n_batches = (len(X_train) + batch_size - 1) // batch_size
print(f"\nGPU metrics: Loss={total_loss/n_batches:.4f} Acc={correct/len(X_train):.4f}")
print(f"  Loop time:  {t_loop:.1f} ms")
print(f"  Sync time:  {t_sync:.1f} ms  (Flush + ReadBuffer)")
print(f"  TOTAL:      {t_loop + t_sync:.1f} ms")

# Test 2: Loop only (no sync at all, skip metrics)
t0 = time.perf_counter()
for i in range(0, len(X_train), batch_size):
    end = min(i + batch_size, len(X_train))
    opt.zero_grad()
    xb = Tensor(X_train[i:end], track=True)
    yb = Tensor(Y_train[i:end], track=True)
    x = relu(c1(xb)); x = maxpool2d(x); x = relu(c2(x)); x = maxpool2d(x)
    x = flatten(x); x = relu(l1(x)); x = relu(l2(x)); logits = l3(x)
    loss = softmax_ce(logits, yb)
    loss.backward(); opt.step(clip=1.0); release_all_buffers()
t_loop_only = (time.perf_counter() - t0) * 1000
print(f"\nNo-metrics loop: {t_loop_only:.1f} ms (pure Python dispatch overhead)")

lib.ReleaseBuffer(correct_buf)
lib.ReleaseBuffer(loss_accum_buf)
