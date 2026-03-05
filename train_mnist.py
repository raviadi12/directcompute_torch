import os
import numpy as np
from PIL import Image
import time
from nn_engine import Tensor, Linear, SGD, relu, softmax_ce

def load_mnist(limit_per_class=100):
    images = []
    labels = []
    for digit in range(10):
        folder = f"mnist/{digit}"
        files = os.listdir(folder)[:limit_per_class]
        for f in files:
            img = Image.open(os.path.join(folder, f)).convert('L')
            img_data = np.array(img).flatten().astype(np.float32) / 255.0
            images.append(img_data)
            labels.append(digit)
    return np.array(images), np.array(labels)

def train():
    X_train, Y_train = load_mnist(limit_per_class=1000) # Use more data
    
    # Shuffle
    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    X_train, Y_train = X_train[idx], Y_train[idx]
    
    # Model: 784 -> 128 -> 10
    # Use He initialization
    l1 = Linear(784, 128)
    l1.w = Tensor(np.random.randn(784, 128).astype(np.float32) * np.sqrt(2/784), requires_grad=True)
    l2 = Linear(128, 10)
    l2.w = Tensor(np.random.randn(128, 10).astype(np.float32) * np.sqrt(2/128), requires_grad=True)
    
    optimizer = SGD([l1.w, l1.b, l2.w, l2.b], lr=0.1)
    
    batch_size = 128 # Larger batch
    epochs = 10
    
    print("\nStarting training on DirectCompute GPU...")
    start_total = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        for i in range(0, len(X_train), batch_size):
            xb = Tensor(X_train[i:i+batch_size])
            yb = Tensor(Y_train[i:i+batch_size])
            
            # Forward
            x1 = relu(l1(xb))
            logits = l2(x1)
            
            # Loss
            loss_tensor = softmax_ce(logits, yb)
            loss_tensor.sync()
            epoch_loss += loss_tensor.data[0]
            
            # Backward
            loss_tensor.backward()
            
            # Update
            optimizer.step()
            
            # Accuracy
            logits.sync()
            preds = np.argmax(logits.data, axis=1)
            correct += np.sum(preds == Y_train[i:i+batch_size])
            
        acc = correct / len(X_train)
        avg_loss = epoch_loss / (len(X_train) / batch_size)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
        
    end_total = time.time()
    print(f"Total DirectCompute Training Time: {end_total - start_total:.2f}s")

    # PyTorch CPU Comparison
    import torch
    import torch.nn as nn
    import torch.optim as optim

    print("\nStarting comparison with PyTorch (CPU)...")
    
    # Use exact same data
    tX = torch.tensor(X_train)
    tY = torch.tensor(Y_train).long()

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.l1 = nn.Linear(784, 128)
            self.l2 = nn.Linear(128, 10)
            # Initialize with same values if possible, but let's just use defaults or match our init
            with torch.no_grad():
                # We used He init in DX version
                nn.init.kaiming_normal_(self.l1.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(self.l1.bias)
                nn.init.kaiming_normal_(self.l2.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(self.l2.bias)

        def forward(self, x):
            x = torch.relu(self.l1(x))
            x = self.l2(x)
            return x

    tmodel = Net()
    toptimizer = optim.SGD(tmodel.parameters(), lr=0.1)
    tcriterion = nn.CrossEntropyLoss()

    start_torch = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        for i in range(0, len(X_train), batch_size):
            toptimizer.zero_grad()
            inputs = tX[i:i+batch_size]
            targets = tY[i:i+batch_size]
            
            outputs = tmodel(inputs)
            tloss = tcriterion(outputs, targets)
            tloss.backward()
            toptimizer.step()
            
            epoch_loss += tloss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            
        acc = correct / len(X_train)
        avg_loss = epoch_loss / (len(X_train) / batch_size)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
        
    end_torch = time.time()
    print(f"Total PyTorch CPU Training Time: {end_torch - start_torch:.2f}s")
    print(f"\nSummary:")
    print(f"DirectCompute (GPU) Time: {end_total - start_total:.2f}s | Final Acc: {acc:.4f}")
    print(f"PyTorch (CPU) Time:       {end_torch - start_torch:.2f}s | Final Acc: {acc:.4f}")
    print(f"Speedup: { (end_torch - start_torch) / (end_total - start_total):.2f}x")

if __name__ == "__main__":
    train()
