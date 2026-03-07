import os
import numpy as np
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.optim as optim

def load_mnist(limit_per_class=100):
    images = []
    labels = []
    for digit in range(10):
        folder = f"mnist/{digit}"
        files = os.listdir(folder)[:limit_per_class]
        for f in files:
            img = Image.open(os.path.join(folder, f)).convert('L')
            img_data = np.array(img).reshape(1, 28, 28).astype(np.float32) / 255.0
            images.append(img_data)
            labels.append(digit)
    return np.array(images), np.array(labels)

def get_accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return torch.sum(preds == targets).item()

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(6, 16, 5)
        self.l1 = nn.Linear(16 * 4 * 4, 120)
        self.l2 = nn.Linear(120, 84)
        self.l3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        # Xavier uniform init (matches DirectCompute engine)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.pool(x)
        x = self.relu(self.c2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x

def train_lenet():
    # Force PyTorch to use CPU
    device = torch.device('cpu')
    
    X, Y = load_mnist(limit_per_class=500)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X, Y = X[idx], Y[idx]
    split = int(0.9 * len(X))
    
    X_train = torch.tensor(X[:split], dtype=torch.float32).to(device)
    X_val = torch.tensor(X[split:], dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y[:split], dtype=torch.long).to(device)
    Y_val = torch.tensor(Y[split:], dtype=torch.long).to(device)
    
    print(f"Dataset Split: Train={len(X_train)}, Val={len(X_val)}")
    
    model = LeNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 64
    epochs = 25
    
    print("\nStarting LeNet training on PyTorch CPU...")
    start_total = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        train_correct = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_end = min(i + batch_size, len(X_train))
            optimizer.zero_grad()
            
            xb = X_train[i:batch_end]
            yb = Y_train[i:batch_end]
            
            logits = model(xb)
            loss = criterion(logits, yb)
            
            epoch_loss += loss.item() * (batch_end - i)
            
            loss.backward()
            
            # Gradient clipping to match clip=1.0 in DirectCompute custom code
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            
            optimizer.step()
            
            train_correct += get_accuracy(logits, yb)
            
        train_acc = train_correct / len(X_train)
        avg_loss = epoch_loss / len(X_train)
        
        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_end = min(i + batch_size, len(X_val))
                xb = X_val[i:batch_end]
                yb = Y_val[i:batch_end]
                logits = model(xb)
                val_correct += get_accuracy(logits, yb)
        val_acc = val_correct / len(X_val)
        
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
    end_total = time.time()
    print(f"Total LeNet PyTorch CPU Training Time: {end_total - start_total:.2f}s")

if __name__ == "__main__":
    train_lenet()
