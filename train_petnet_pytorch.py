import os
import numpy as np
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Use CUDA if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_pets(limit_per_class=200, size=112):
    images, labels = [], []
    classes = ['Cat', 'Dog']
    print(f"Loading {limit_per_class * 2} images at {size}x{size}...")
    for i, cls in enumerate(classes):
        folder = f"PetImages/{cls}"
        if not os.path.exists(folder): continue
        files = os.listdir(folder)[:limit_per_class]
        for f in files:
            try:
                img = Image.open(os.path.join(folder, f)).convert('RGB')
                img = img.resize((size, size))
                # PyTorch expects (C, H, W)
                images.append(np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0)
                labels.append(i)
            except:
                continue
    return np.array(images), np.array(labels)

class PetNet(nn.Module):
    def __init__(self):
        super(PetNet, self).__init__()
        # Input: 3 x 112 x 112
        self.c1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3) # -> 32 x 56 x 56
        self.bn1 = nn.BatchNorm2d(32)
        
        self.c2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # -> 64 x 56 x 56
        self.bn2 = nn.BatchNorm2d(64)
        
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # -> 128 x 28 x 28 (after pool)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # -> 256 x 14 x 14 (after pool)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Head
        self.l1 = nn.Linear(256 * 7 * 7, 256) 
        self.l2 = nn.Linear(256, 2)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.bn1(self.c1(x)))
        x = self.maxpool(self.relu(self.bn2(self.c2(x))))
        x = self.maxpool(self.relu(self.bn3(self.c3(x))))
        x = self.maxpool(self.relu(self.bn4(self.c4(x))))
        
        x = torch.flatten(x, 1)
        x = self.relu(self.l1(x))
        return self.l2(x)

def train_petnet_pytorch():
    X, Y = load_pets(limit_per_class=200, size=112)
    if len(X) == 0:
        print("No images found!")
        return
        
    idx = np.arange(len(X)); np.random.shuffle(idx); X, Y = X[idx], Y[idx]
    split = int(0.9 * len(X))
    
    X_train = torch.from_numpy(X[:split]).to(device)
    Y_train = torch.from_numpy(Y[:split]).long().to(device)
    X_val = torch.from_numpy(X[split:]).to(device)
    Y_val = torch.from_numpy(Y[split:]).long().to(device)
    
    print(f"Dataset Split: Train={len(X_train)}, Val={len(X_val)}")

    model = PetNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 16
    epochs = 20

    print("\nStarting PetNet (AlexNet-v2) training on PyTorch...")
    start_time = time.time()

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_loss = 0
        train_correct = 0
        
        for i in range(0, len(X_train), batch_size):
            end = min(i + batch_size, len(X_train))
            optimizer.zero_grad()
            
            xb = X_train[i:end]
            yb = Y_train[i:end]

            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * (end - i)
            train_correct += (logits.argmax(1) == yb).sum().item()

        # ── Validate ──
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                end = min(i + batch_size, len(X_val))
                xb = X_val[i:end]
                yb = Y_val[i:end]
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * (end - i)
                val_correct += (logits.argmax(1) == yb).sum().item()

        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss/len(X_train):.4f} | Train Acc: {train_correct/len(X_train):.4f} | Val Acc: {val_correct/len(X_val):.4f}")

    total_time = time.time() - start_time
    print(f"Total PetNet PyTorch Training Time: {total_time:.2f}s")

if __name__ == "__main__":
    train_petnet_pytorch()
