"""Benchmark LeNet with matmul_reduce optimization."""
import os
import numpy as np
from PIL import Image
import time
from nn_engine import (Tensor, Linear, ConvLayer, Model, SGD, Metrics,
                       relu, softmax_ce, maxpool2d, flatten, end_batch,
                       bias_relu, auto_warm, get_pool_stats,
                       BatchNorm2d)

def load_mnist(limit_per_class=100):
    images, labels = [], []
    for digit in range(10):
        folder = f"mnist/{digit}"
        for f in os.listdir(folder)[:limit_per_class]:
            img = Image.open(os.path.join(folder, f)).convert('L')
            images.append(np.array(img).reshape(1, 28, 28).astype(np.float32) / 255.0)
            labels.append(digit)
    return np.array(images), np.array(labels)

class LeNet(Model):
    def __init__(self):
        super().__init__()
        self.c1 = ConvLayer(1, 6, 5)
        self.bn1 = BatchNorm2d(6)
        self.c2 = ConvLayer(6, 16, 5)
        self.bn2 = BatchNorm2d(16)
        self.l1 = Linear(16*4*4, 120)
        self.l2 = Linear(120, 84)
        self.l3 = Linear(84, 10)

    def forward(self, xb):
        x = self.c1(xb); x = self.bn1(x); x = relu(x); x = maxpool2d(x)
        x = self.c2(x);  x = self.bn2(x); x = relu(x); x = maxpool2d(x)
        x = flatten(x)
        x = self.l1(x, relu=True)
        x = self.l2(x, relu=True)
        return self.l3(x)

X, Y = load_mnist(limit_per_class=500)
idx = np.arange(len(X)); np.random.shuffle(idx); X, Y = X[idx], Y[idx]
split = int(0.9 * len(X))
X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]

model = LeNet()
params = model.parameters()
optimizer = SGD(params, lr=0.01)
metrics = Metrics()
batch_size = 32
epochs = 25

sizes = auto_warm(model.forward, params, X_train[:batch_size], Y_train[:batch_size], optimizer, copies=3)

print(f"\nStarting LeNet benchmark (matmul_reduce optimization)...")
start = time.time()

for epoch in range(epochs):
    model.train(); metrics.reset()
    for i in range(0, len(X_train), batch_size):
        end = min(i + batch_size, len(X_train))
        optimizer.zero_grad()
        xb = Tensor(X_train[i:end])
        yb = Tensor(Y_train[i:end])
        logits = model(xb)
        loss = softmax_ce(logits, yb)
        metrics.update(loss, logits, yb)
        loss.backward()
        optimizer.step(clip=1.0)
        end_batch()
    train_loss, train_acc = metrics.collect(len(X_train))

    model.eval(); metrics.reset()
    for i in range(0, len(X_val), batch_size):
        end = min(i + batch_size, len(X_val))
        xb = Tensor(X_val[i:end])
        yb = Tensor(Y_val[i:end])
        logits = model(xb)
        loss = softmax_ce(logits, yb)
        metrics.update(loss, logits, yb)
        end_batch()
    val_loss, val_acc = metrics.collect(len(X_val))

    print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

total = time.time() - start
hits, misses = get_pool_stats()
print(f"\nTotal Training Time: {total:.2f}s")
print(f"Pool stats: {hits} hits, {misses} misses ({hits/(hits+misses)*100:.1f}% hit rate)")
