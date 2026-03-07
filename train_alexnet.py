import os
import numpy as np
from PIL import Image
import time
from nn_engine import (Tensor, Linear, ConvLayer, Model, SGD, Metrics,
                       relu, softmax_ce, maxpool2d, flatten, end_batch,
                       bias_relu, get_pool_stats, get_pool_memory)

def load_pets(limit_per_class=100, size=224):
    images, labels = [], []
    classes = ['Cat', 'Dog']
    for i, cls in enumerate(classes):
        folder = f"PetImages/{cls}"
        files = os.listdir(folder)[:limit_per_class]
        print(f"Loading {cls} images...")
        for f in files:
            try:
                img = Image.open(os.path.join(folder, f)).convert('RGB')
                img = img.resize((size, size))
                images.append(np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0)
                labels.append(i)
            except:
                continue
    return np.array(images), np.array(labels)

class AlexNet(Model):
    def __init__(self):
        super().__init__()
        self.c1 = ConvLayer(3, 64, 11, stride=4, padding=2)
        self.c2 = ConvLayer(64, 192, 5, padding=2)
        self.c3 = ConvLayer(192, 384, 3, padding=1)
        self.c4 = ConvLayer(384, 256, 3, padding=1)
        self.c5 = ConvLayer(256, 256, 3, padding=1)
        self.l1 = Linear(256 * 6 * 6, 512)
        self.l2 = Linear(512, 512)
        self.l3 = Linear(512, 2)

    def forward(self, xb):
        x = self.c1(xb, relu=True)
        x = maxpool2d(x, pool_size=3, stride=2)
        x = self.c2(x, relu=True)
        x = maxpool2d(x, pool_size=3, stride=2)
        x = self.c3(x, relu=True)
        x = self.c4(x, relu=True)
        x = self.c5(x, relu=True)
        x = maxpool2d(x, pool_size=3, stride=2)
        x = flatten(x)
        x = self.l1(x, relu=True)
        x = self.l2(x, relu=True)
        return self.l3(x)

    def _onnx_graph(self, input_shape, helper, TensorProto, numpy_helper):
        nodes, initializers = [], []
        h = Model._conv_onnx
        g = Model._linear_onnx
        h(self.c1, "c1", "input", "pool1", helper, numpy_helper, initializers, nodes, with_relu=True, pool=(3, 2))
        h(self.c2, "c2", "pool1", "pool2", helper, numpy_helper, initializers, nodes, with_relu=True, pool=(3, 2))
        h(self.c3, "c3", "pool2", "c3_out", helper, numpy_helper, initializers, nodes, with_relu=True)
        h(self.c4, "c4", "c3_out", "c4_out", helper, numpy_helper, initializers, nodes, with_relu=True)
        h(self.c5, "c5", "c4_out", "pool5", helper, numpy_helper, initializers, nodes, with_relu=True, pool=(3, 2))
        nodes.append(helper.make_node("Flatten", ["pool5"], ["flat"], axis=1))
        g(self.l1, "l1", "flat", "fc1", helper, numpy_helper, initializers, nodes, with_relu=True)
        g(self.l2, "l2", "fc1", "fc2", helper, numpy_helper, initializers, nodes, with_relu=True)
        g(self.l3, "l3", "fc2", "output", helper, numpy_helper, initializers, nodes)
        return nodes, initializers, "output", [1, 2]

def train_alexnet():
    X, Y = load_pets(limit_per_class=2000)
    idx = np.arange(len(X)); np.random.shuffle(idx); X, Y = X[idx], Y[idx]
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    print(f"Dataset Split: Train={len(X_train)}, Val={len(X_val)}")

    model = AlexNet()
    params = model.parameters()
    optimizer = SGD(params, lr=0.01)
    metrics = Metrics()

    batch_size = 64
    accum_steps = 4              
    epochs = 50

    def forward(xb):
        return model(xb)

    # AlexNet creates huge intermediate buffers (im2col > 60MB each).
    # On iGPU (128MB dedicated VRAM): flush every batch, no pool warming.
    end_batch.flush_interval = 1

    print(f"\nStarting AlexNet training on DirectCompute GPU...")
    print(f"  Batch size: {batch_size}, Accum steps: {accum_steps}, Effective batch: {batch_size * accum_steps}")
    start = time.time()

    for epoch in range(epochs):
        # ── Train ──
        metrics.reset()
        step_count = 0
        t0 = time.perf_counter()

        for i in range(0, len(X_train), batch_size):
            end = min(i + batch_size, len(X_train))
            if step_count % accum_steps == 0:
                optimizer.zero_grad()

            xb = Tensor(X_train[i:end], track=True)
            yb = Tensor(Y_train[i:end], track=True)

            logits = forward(xb)
            loss = softmax_ce(logits, yb)
            metrics.update(loss, logits, yb)

            loss.backward()
            step_count += 1
            if step_count % accum_steps == 0 or end >= len(X_train):
                optimizer.step(clip=1.0)
                end_batch()

        train_loss, train_acc = metrics.collect(len(X_train))

        # ── Validate ──
        metrics.reset()
        for i in range(0, len(X_val), batch_size):
            end = min(i + batch_size, len(X_val))
            xb = Tensor(X_val[i:end])
            yb = Tensor(Y_val[i:end])
            logits = forward(xb)
            loss = softmax_ce(logits, yb)
            metrics.update(loss, logits, yb)
            end_batch()

        val_loss, val_acc = metrics.collect(len(X_val))

        epoch_ms = (time.perf_counter() - t0) * 1000
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | {epoch_ms:.0f} ms")

    total = time.time() - start
    hits, misses = get_pool_stats()
    pool_mb = get_pool_memory() / (1024*1024)
    print(f"Total AlexNet DirectCompute Training Time: {total:.2f}s")
    print(f"Pool stats: {hits} hits, {misses} misses, {pool_mb:.1f}MB pooled")

    # ONNX export
    model.export("alexnet.onnx", input_shape=[1, 3, 224, 224])

if __name__ == "__main__":
    train_alexnet()
