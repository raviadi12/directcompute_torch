import os
import numpy as np
from PIL import Image
import time
from nn_engine import (Tensor, Linear, ConvLayer, SGD, Metrics,
                       relu, softmax_ce, maxpool2d, flatten, end_batch,
                       bias_relu)

def load_mnist(limit_per_class=100):
    images, labels = [], []
    for digit in range(10):
        folder = f"mnist/{digit}"
        for f in os.listdir(folder)[:limit_per_class]:
            img = Image.open(os.path.join(folder, f)).convert('L')
            images.append(np.array(img).reshape(1, 28, 28).astype(np.float32) / 255.0)
            labels.append(digit)
    return np.array(images), np.array(labels)

def train_lenet():
    X, Y = load_mnist(limit_per_class=500)
    idx = np.arange(len(X)); np.random.shuffle(idx); X, Y = X[idx], Y[idx]
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    print(f"Dataset Split: Train={len(X_train)}, Val={len(X_val)}")

    # Model
    c1 = ConvLayer(1, 6, 5)
    c2 = ConvLayer(6, 16, 5)
    l1 = Linear(16*4*4, 120)
    l2 = Linear(120, 84)
    l3 = Linear(84, 10)

    params = [c1.filters, c1.bias, c2.filters, c2.bias,
              l1.w, l1.b, l2.w, l2.b, l3.w, l3.b]
    optimizer = SGD(params, lr=0.01)
    metrics = Metrics()
    batch_size = 4096
    epochs = 25

    def forward(xb):
        x = c1(xb, relu=True)       # fused conv+relu (1 fewer dispatch)
        x = maxpool2d(x)
        x = c2(x, relu=True)        # fused conv+relu
        x = maxpool2d(x)
        x = flatten(x)
        x = l1(x, relu=True)        # fused bias+relu (1 fewer dispatch)
        x = l2(x, relu=True)        # fused bias+relu
        return l3(x)

    print("\nStarting LeNet training on DirectCompute GPU...")
    start = time.time()

    for epoch in range(epochs):
        # ── Train ──
        metrics.reset()
        for i in range(0, len(X_train), batch_size):
            end = min(i + batch_size, len(X_train))
            optimizer.zero_grad()
            xb = Tensor(X_train[i:end], track=True)
            yb = Tensor(Y_train[i:end], track=True)

            logits = forward(xb)
            loss = softmax_ce(logits, yb)
            metrics.update(loss, logits, yb)

            loss.backward()
            optimizer.step(clip=1.0)
            end_batch()

        train_loss, train_acc = metrics.collect(len(X_train))

        # ── Validate ──
        metrics.reset()
        for i in range(0, len(X_val), batch_size):
            end = min(i + batch_size, len(X_val))
            xb = Tensor(X_val[i:end], track=True)
            yb = Tensor(Y_val[i:end], track=True)
            logits = forward(xb)
            loss = softmax_ce(logits, yb)
            metrics.update(loss, logits, yb)
            end_batch()

        val_loss, val_acc = metrics.collect(len(X_val))

        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    print(f"Total LeNet DirectCompute Training Time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    train_lenet()
