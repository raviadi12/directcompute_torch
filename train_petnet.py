import os
import numpy as np
from PIL import Image
import time
from nn_engine import (Tensor, Linear, ConvLayer, Model, AdamW, Metrics,
                       relu, softmax_ce, maxpool2d, flatten, end_batch,
                       BatchNorm2d, get_pool_stats, get_pool_memory)

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
                # Transpose to (C, H, W) for ConvLayer
                images.append(np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0)
                labels.append(i)
            except:
                continue
    return np.array(images), np.array(labels)

class PetNet(Model):
    def __init__(self):
        super().__init__()
        # Input: 3 x 112 x 112
        self.c1 = ConvLayer(3, 32, ks=7, stride=2, padding=3) # -> 32 x 56 x 56
        self.bn1 = BatchNorm2d(32)
        
        self.c2 = ConvLayer(32, 64, ks=3, padding=1) # -> 64 x 56 x 56
        self.bn2 = BatchNorm2d(64)
        
        self.c3 = ConvLayer(64, 128, ks=3, padding=1) # -> 128 x 28 x 28 (after pool)
        self.bn3 = BatchNorm2d(128)
        
        self.c4 = ConvLayer(128, 256, ks=3, padding=1) # -> 256 x 14 x 14 (after pool)
        self.bn4 = BatchNorm2d(256)
        
        # Head
        self.l1 = Linear(256 * 7 * 7, 256) 
        self.l2 = Linear(256, 2)

    def forward(self, x):
        # Layer 1: 112 -> 56
        x = relu(self.bn1(self.c1(x)))
        
        # Layer 2: 56 -> 28
        x = maxpool2d(relu(self.bn2(self.c2(x))), pool_size=2, stride=2)
        
        # Layer 3: 28 -> 14
        x = maxpool2d(relu(self.bn3(self.c3(x))), pool_size=2, stride=2)
        
        # Layer 4: 14 -> 7
        x = maxpool2d(relu(self.bn4(self.c4(x))), pool_size=2, stride=2)
        
        x = flatten(x)
        x = relu(self.l1(x))
        return self.l2(x)

    def _onnx_graph(self, input_shape, helper, TensorProto, numpy_helper):
        nodes, initializers = [], []
        # Simple graph export for PetNet
        def _bn_onnx(bn, name, inp, out):
            m = bn.running_mean.sync()
            v = bn.running_var.sync()
            g = bn.gamma.sync()
            b = bn.beta.sync()
            initializers.extend([
                numpy_helper.from_array(g, f"{name}_g"),
                numpy_helper.from_array(b, f"{name}_b"),
                numpy_helper.from_array(m, f"{name}_m"),
                numpy_helper.from_array(v, f"{name}_v")
            ])
            nodes.append(helper.make_node("BatchNormalization", 
                [inp, f"{name}_g", f"{name}_b", f"{name}_m", f"{name}_v"], [out], epsilon=bn.eps))

        Model._conv_onnx(self.c1, "c1", "input", "c1_out", helper, numpy_helper, initializers, nodes)
        _bn_onnx(self.bn1, "bn1", "c1_out", "bn1_out")
        nodes.append(helper.make_node("Relu", ["bn1_out"], ["r1"]))
        
        Model._conv_onnx(self.c2, "c2", "r1", "c2_out", helper, numpy_helper, initializers, nodes)
        _bn_onnx(self.bn2, "bn2", "c2_out", "bn2_out")
        nodes.append(helper.make_node("Relu", ["bn2_out"], ["r2"]))
        nodes.append(helper.make_node("MaxPool", ["r2"], ["p2"], kernel_shape=[2,2], strides=[2,2]))

        Model._conv_onnx(self.c3, "c3", "p2", "c3_out", helper, numpy_helper, initializers, nodes)
        _bn_onnx(self.bn3, "bn3", "c3_out", "bn3_out")
        nodes.append(helper.make_node("Relu", ["bn3_out"], ["r3"]))
        nodes.append(helper.make_node("MaxPool", ["r3"], ["p3"], kernel_shape=[2,2], strides=[2,2]))

        Model._conv_onnx(self.c4, "c4", "p3", "c4_out", helper, numpy_helper, initializers, nodes)
        _bn_onnx(self.bn4, "bn4", "c4_out", "bn4_out")
        nodes.append(helper.make_node("Relu", ["bn4_out"], ["r4"]))
        nodes.append(helper.make_node("MaxPool", ["r4"], ["p4"], kernel_shape=[2,2], strides=[2,2]))

        nodes.append(helper.make_node("Flatten", ["p4"], ["flat"]))
        Model._linear_onnx(self.l1, "l1", "flat", "fc1_out", helper, numpy_helper, initializers, nodes, with_relu=True)
        Model._linear_onnx(self.l2, "l2", "fc1_out", "output", helper, numpy_helper, initializers, nodes)
        
        return nodes, initializers, "output", [1, 2]

def train_petnet():
    # Load images
    X, Y = load_pets(limit_per_class=200, size=112)
    if len(X) == 0:
        print("No images found in PetImages directory!")
        return
        
    idx = np.arange(len(X)); np.random.shuffle(idx); X, Y = X[idx], Y[idx]
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    print(f"Dataset Split: Train={len(X_train)}, Val={len(X_val)}")

    model = PetNet()
    params = model.parameters()
    # Using AdamW for better generalization
    optimizer = AdamW(params, lr=0.0005, weight_decay=0.01)
    metrics = Metrics()
    
    batch_size = 16
    accum_steps = 2 # Effective batch size 32
    epochs = 20

    # Ensure memory is released aggressively for iGPU
    end_batch.flush_interval = 1 

    print("\nStarting PetNet (AlexNet-v2) training on DirectCompute GPU...")
    start = time.time()

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        metrics.reset()
        step_count = 0
        for i in range(0, len(X_train), batch_size):
            end = min(i + batch_size, len(X_train))
            if step_count % accum_steps == 0:
                optimizer.zero_grad()
                
            xb = Tensor(X_train[i:end])
            yb = Tensor(Y_train[i:end])

            logits = model(xb)
            loss = softmax_ce(logits, yb)
            metrics.update(loss, logits, yb)

            loss.backward()
            step_count += 1
            if step_count % accum_steps == 0 or end >= len(X_train):
                optimizer.step(clip=1.0)
                end_batch()

        train_loss, train_acc = metrics.collect(len(X_train))

        # ── Validate ──
        model.eval()
        metrics.reset()
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
    vram = get_pool_memory() / (1024 * 1024)
    print(f"Total PetNet (AlexNet-v2) Training Time: {total:.2f}s")
    print(f"Pool stats: {hits} hits, {misses} misses, {vram:.1f}MB VRAM in pool")

    # ONNX export
    model.export("petnet.onnx", input_shape=[1, 3, 112, 112])

if __name__ == "__main__":
    train_petnet()
