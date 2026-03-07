import os
import math
import numpy as np
from PIL import Image
import time
from tinygrad import Tensor, dtypes, Device
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters

def load_mnist(limit_per_class=500):
    images, labels = [], []
    for digit in range(10):
        folder = f"mnist/{digit}"
        for f in os.listdir(folder)[:limit_per_class]:
            img = Image.open(os.path.join(folder, f)).convert('L')
            images.append(np.array(img).reshape(1, 28, 28).astype(np.float32) / 255.0)
            labels.append(digit)
    return np.array(images), np.array(labels)

def xavier_uniform(shape, fan_in, fan_out):
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    return Tensor.uniform(*shape, low=-bound, high=bound)

class LeNet:
    def __init__(self):
        # Conv2d with Xavier uniform init (matching DirectCompute engine)
        self.c1_w = xavier_uniform((6, 1, 5, 5), 1*5*5, 6*5*5)
        self.c1_b = Tensor.zeros(6)
        self.c2_w = xavier_uniform((16, 6, 5, 5), 6*5*5, 16*5*5)
        self.c2_b = Tensor.zeros(16)
        # Linear layers with Xavier uniform init
        self.l1_w = xavier_uniform((256, 120), 256, 120)
        self.l1_b = Tensor.zeros(120)
        self.l2_w = xavier_uniform((120, 84), 120, 84)
        self.l2_b = Tensor.zeros(84)
        self.l3_w = xavier_uniform((84, 10), 84, 10)
        self.l3_b = Tensor.zeros(10)

    def __call__(self, x):
        x = x.conv2d(self.c1_w, self.c1_b).relu().max_pool2d((2, 2))
        x = x.conv2d(self.c2_w, self.c2_b).relu().max_pool2d((2, 2))
        x = x.reshape(x.shape[0], -1)
        x = x.linear(self.l1_w, self.l1_b).relu()
        x = x.linear(self.l2_w, self.l2_b).relu()
        return x.linear(self.l3_w, self.l3_b)

def train_lenet():
    print(f"Tinygrad device: {Device.DEFAULT}")

    X, Y = load_mnist(limit_per_class=500)
    idx = np.arange(len(X)); np.random.shuffle(idx); X, Y = X[idx], Y[idx]
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    print(f"Dataset Split: Train={len(X_train)}, Val={len(X_val)}")

    Tensor.training = True

    model = LeNet()
    optimizer = SGD(get_parameters(model), lr=0.01)

    batch_size = 32
    epochs = 25

    print(f"\nStarting LeNet training on Tinygrad ({Device.DEFAULT})...")
    start = time.time()

    for epoch in range(epochs):
        # ── Train ──
        total_loss, correct, total = 0.0, 0, 0
        for i in range(0, len(X_train), batch_size):
            end = min(i + batch_size, len(X_train))
            xb = Tensor(X_train[i:end])
            yb = Tensor(Y_train[i:end].astype(np.int32))

            optimizer.zero_grad()
            logits = model(xb)
            loss = logits.sparse_categorical_crossentropy(yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * (end - i)
            preds = logits.argmax(axis=1).numpy()
            correct += (preds == Y_train[i:end]).sum()
            total += end - i

        train_loss = total_loss / total
        train_acc = correct / total

        # ── Validate ──
        val_correct, val_total = 0, 0
        for i in range(0, len(X_val), batch_size):
            end = min(i + batch_size, len(X_val))
            xb = Tensor(X_val[i:end])
            logits = model(xb)
            preds = logits.argmax(axis=1).numpy()
            val_correct += (preds == Y_val[i:end]).sum()
            val_total += end - i

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    print(f"Total LeNet Tinygrad Training Time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    train_lenet()
