import os
import numpy as np
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def load_pets(limit_per_class=100, size=224):
    images = []
    labels = []
    classes = ['Cat', 'Dog']
    for i, cls in enumerate(classes):
        folder = f"PetImages/{cls}"
        files = os.listdir(folder)[:limit_per_class]
        print(f"Loading {cls} images...")
        for f in files:
            try:
                img = Image.open(os.path.join(folder, f)).convert('RGB')
                img = img.resize((size, size))
                img_data = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
                images.append(img_data)
                labels.append(i)
            except:
                continue
    return np.array(images), np.array(labels)

# ── AlexNet (matching DirectCompute architecture exactly) ────────────────────
class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),   # c1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),            # c2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),           # c3
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),           # c4
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),           # c5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),                             # l1
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),                                     # l2
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),                             # l3
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_alexnet_pytorch():
    X, Y = load_pets(limit_per_class=2000)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X, Y = X[idx], Y[idx]

    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    print(f"Dataset Split: Train={len(X_train)}, Val={len(X_val)}")

    device = torch.device('cpu')
    model = AlexNet(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    batch_size = 64
    epochs = 10

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train).long())
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val).long())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("\nStarting AlexNet training on PyTorch CPU...")
    start_total = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        t0 = time.perf_counter()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            # Gradient clipping to match DirectCompute version
            nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        wall = (time.perf_counter() - t0) * 1000

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        train_acc = correct / total
        val_acc = val_correct / val_total
        avg_loss = epoch_loss / total
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Wall: {wall:.0f} ms")

    end_total = time.time()
    print(f"Total AlexNet PyTorch CPU Training Time: {end_total - start_total:.2f}s")

if __name__ == "__main__":
    train_alexnet_pytorch()
