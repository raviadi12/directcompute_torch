# DirectCompute Neural Network Engine

A high-performance, from-scratch neural network training framework that runs entirely on the GPU using **DirectCompute (D3D11 Compute Shaders)**. No CUDA, no cuDNN â€” just raw HLSL shaders dispatched through a thin C++ runtime, driven from Python.

## Quick Install (Windows Only)

```bash
pip install directcompute-nn
```

The Windows wheel includes everything: pre-compiled `engine.dll`, bundled HLSL shaders, and the Python API. No C++ compiler required.

## Key Features

- **Pure DirectCompute**: Uses standard D3D11 compute shaders (HLSL) â€” runs on any DirectX 11 GPU (Intel, AMD, NVIDIA) on Windows.
- **Full Autograd**: Python-based automatic differentiation with GPU-accelerated gradient accumulation.
- **Complete Layer Library**: Linear, Conv2D, **DepthwiseConv2D**, BatchNorm2d, MaxPool, **GlobalAvgPool2D**, and more.
- **Differentiable Skip Connections**: `add()` is fully differentiable â€” build ResNet and MobileNet-style residual blocks.
- **Modern Optimizers**: SGD, Adam, AdamW, and the state-of-the-art **Muon** optimizer.
- **Transfer Learning**: Load pretrained weights, freeze backbone layers, fine-tune on custom data.
- **ONNX Inference**: Load and run ONNX models on the DirectCompute engine.

---

## Usage Examples

### 1. Minimal Forward Pass

```python
import numpy as np
from nn_engine import Tensor, Linear, relu

x = Tensor(np.random.randn(32, 128).astype(np.float32))
layer = Linear(128, 64)
y = relu(layer(x))
print(y.shape)  # (32, 64)
```

### 2. Standard Training Loop (LeNet)

```python
import numpy as np
from nn_engine import (Tensor, ConvLayer, Linear, Model, BatchNorm2d,
                       maxpool2d, flatten, relu, softmax_ce,
                       AdamW, Metrics, end_batch)

class LeNet(Model):
    def __init__(self):
        super().__init__()
        self.c1 = ConvLayer(1, 6, 5)
        self.c2 = ConvLayer(6, 16, 5)
        self.l1 = Linear(16*4*4, 120)
        self.l2 = Linear(120, 10)

    def forward(self, x):
        x = maxpool2d(relu(self.c1(x)))
        x = maxpool2d(relu(self.c2(x)))
        x = flatten(x)
        x = relu(self.l1(x))
        return self.l2(x)

model = LeNet()
optimizer = AdamW(model.parameters(), lr=0.001)
metrics = Metrics()

for epoch in range(10):
    for xb_np, yb_np in your_dataloader():          # yield (np.float32, np.int32)
        optimizer.zero_grad()
        xb, yb = Tensor(xb_np), Tensor(yb_np)
        loss = softmax_ce(model(xb), yb)
        metrics.update(loss, model(xb), yb)
        loss.backward()
        optimizer.step(clip=1.0)
        end_batch()                                  # flushes GPU, frees intermediates
```

### 3. ResNet-style Residual Block

The `add()` function is fully differentiable and supports skip connections:

```python
from nn_engine import (Tensor, ConvLayer, BatchNorm2d, Linear,
                       relu, add, flatten, softmax_ce, AdamW, end_batch)
import numpy as np

class ResBlock:
    def __init__(self, channels):
        self.conv1 = ConvLayer(channels, channels, ks=3, padding=1)
        self.bn1 = BatchNorm2d(channels)
        self.conv2 = ConvLayer(channels, channels, ks=3, padding=1)
        self.bn2 = BatchNorm2d(channels)

    def __call__(self, x):
        identity = x
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return relu(add(out, identity))  # differentiable skip connection

    def parameters(self):
        return [self.conv1.filters, self.conv1.bias,
                self.bn1.gamma, self.bn1.beta,
                self.conv2.filters, self.conv2.bias,
                self.bn2.gamma, self.bn2.beta]
```

### 4. MobileNet-style Inverted Residual Block

`DepthwiseConvLayer` and `relu6` enable full MobileNetV2-style blocks:

```python
from nn_engine import (Tensor, ConvLayer, DepthwiseConvLayer, BatchNorm2d,
                       relu6, add, global_avg_pool2d, flatten,
                       Linear, softmax_ce, AdamW, end_batch)
import numpy as np

class InvertedResidual:
    """MobileNetV2 inverted residual block."""
    def __init__(self, inp, oup, stride, expand_ratio):
        hidden = int(round(inp * expand_ratio))
        self.use_res = (stride == 1 and inp == oup)
        self.expand_ratio = expand_ratio
        if expand_ratio != 1:
            self.expand_conv = ConvLayer(inp, hidden, ks=1)
            self.expand_bn   = BatchNorm2d(hidden)
        self.dw      = DepthwiseConvLayer(hidden, ks=3, stride=stride, padding=1)
        self.dw_bn   = BatchNorm2d(hidden)
        self.project = ConvLayer(hidden, oup, ks=1)
        self.project_bn = BatchNorm2d(oup)

    def __call__(self, x):
        identity = x
        if self.expand_ratio != 1:
            x = relu6(self.expand_bn(self.expand_conv(x)))
        x = relu6(self.dw_bn(self.dw(x)))
        x = self.project_bn(self.project(x))
        if self.use_res:
            x = add(x, identity)
        return x
```

---

## Transfer Learning with Pretrained MobileNetV2

The engine supports loading pretrained weights and performing feature extraction. Below is a complete working example â€” **training a Cat/Dog classifier on PetImages in ~42 seconds using just a 128MB Intel iGPU**, matching PyTorch CPU performance.

### Step 1 â€” Download pretrained weights (one-time, requires `torch`)

```bash
pip install torch torchvision
```

```python
# download_weights.py
import numpy as np

try:
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
except ImportError:
    from torchvision.models import mobilenet_v2
    model = mobilenet_v2(pretrained=True)

model.eval()
arrays = {k: v.cpu().numpy() for k, v in model.state_dict().items()
          if 'num_batches_tracked' not in k}
np.savez("mobilenet_v2_weights.npz", **arrays)
print(f"Saved {len(arrays)} tensors")
```

### Step 2 â€” Build MobileNetV2 and load ImageNet weights

```python
import ctypes, numpy as np
from nn_engine import (Tensor, ConvLayer, DepthwiseConvLayer, BatchNorm2d,
                       relu6, add, global_avg_pool2d, flatten, Linear,
                       softmax_ce, AdamW, Metrics, end_batch, lib)

MOBILENET_SETTINGS = [
    (1,  16,  1, 1), (6,  24,  2, 2), (6,  32,  3, 2),
    (6,  64,  4, 2), (6,  96,  3, 1), (6, 160,  3, 2), (6, 320,  1, 1),
]

class InvertedResidual:
    def __init__(self, inp, oup, stride, t):
        hidden = int(round(inp * t))
        self.use_res = (stride == 1 and inp == oup)
        self.t = t
        if t != 1:
            self.expand_conv = ConvLayer(inp, hidden, ks=1)
            self.expand_bn   = BatchNorm2d(hidden)
        self.dw = DepthwiseConvLayer(hidden, ks=3, stride=stride, padding=1)
        self.dw_bn   = BatchNorm2d(hidden)
        self.project = ConvLayer(hidden, oup, ks=1)
        self.project_bn = BatchNorm2d(oup)

    def __call__(self, x):
        identity = x
        if self.t != 1:
            x = relu6(self.expand_bn(self.expand_conv(x)))
        x = relu6(self.dw_bn(self.dw(x)))
        x = self.project_bn(self.project(x))
        return add(x, identity) if self.use_res else x

    def parameters(self):
        p = []
        if self.t != 1:
            p += [self.expand_conv.filters, self.expand_conv.bias,
                  self.expand_bn.gamma,     self.expand_bn.beta]
        return p + [self.dw.filters, self.dw.bias,
                    self.dw_bn.gamma, self.dw_bn.beta,
                    self.project.filters, self.project.bias,
                    self.project_bn.gamma, self.project_bn.beta]

    def set_training(self, mode):
        if self.t != 1: self.expand_bn.training = mode
        self.dw_bn.training = mode
        self.project_bn.training = mode


class MobileNetV2:
    def __init__(self, num_classes=2):
        self.conv0 = ConvLayer(3, 32, ks=3, stride=2, padding=1)
        self.bn0   = BatchNorm2d(32)
        inp = 32
        idx = 0
        for t, c, n, s in MOBILENET_SETTINGS:
            for i in range(n):
                setattr(self, f'b{idx}', InvertedResidual(inp, c, s if i == 0 else 1, t))
                idx += 1
                inp = c
        self.num_blocks = idx           # 17
        self.conv_last  = ConvLayer(320, 1280, ks=1)
        self.bn_last    = BatchNorm2d(1280)
        self.classifier = Linear(1280, num_classes)

    def forward(self, x):
        x = relu6(self.bn0(self.conv0(x)))
        for i in range(self.num_blocks):
            x = getattr(self, f'b{i}')(x)
        x = relu6(self.bn_last(self.conv_last(x)))
        x = global_avg_pool2d(x)
        return self.classifier(flatten(x))

    def extract_features(self, x):
        """Backbone only â€” returns 1280-dim features without classifier."""
        x = relu6(self.bn0(self.conv0(x)))
        for i in range(self.num_blocks):
            x = getattr(self, f'b{i}')(x)
        x = relu6(self.bn_last(self.conv_last(x)))
        x = global_avg_pool2d(x)
        return flatten(x)

    def parameters(self):
        p = [self.conv0.filters, self.conv0.bias, self.bn0.gamma, self.bn0.beta]
        for i in range(self.num_blocks): p.extend(getattr(self, f'b{i}').parameters())
        return p + [self.conv_last.filters, self.conv_last.bias,
                    self.bn_last.gamma, self.bn_last.beta,
                    self.classifier.w, self.classifier.b]

    def eval(self):
        self.bn0.training = False
        for i in range(self.num_blocks): getattr(self, f'b{i}').set_training(False)
        self.bn_last.training = False

    def load_pretrained(self, npz_path):
        weights = np.load(npz_path)

        def upload(tensor, key):
            data = weights[key].astype(np.float32)
            assert data.shape == tensor.shape, f"Shape mismatch {key}: {data.shape} vs {tensor.shape}"
            tensor.data = data.copy()
            lib.UpdateBuffer(tensor.gpu_buf, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        def load_conv_bn(ck, bk, cl, bl):
            upload(cl.filters, f"{ck}.weight")
            upload(bl.gamma, f"{bk}.weight"); upload(bl.beta, f"{bk}.bias")
            upload(bl.running_mean, f"{bk}.running_mean")
            upload(bl.running_var,  f"{bk}.running_var")

        load_conv_bn("features.0.0", "features.0.1", self.conv0, self.bn0)

        bidx, fidx = 0, 1
        for t, c, n, s in MOBILENET_SETTINGS:
            for _ in range(n):
                blk, p = getattr(self, f'b{bidx}'), f"features.{fidx}"
                if t != 1:
                    load_conv_bn(f"{p}.conv.0.0", f"{p}.conv.0.1", blk.expand_conv, blk.expand_bn)
                    upload(blk.dw.filters, f"{p}.conv.1.0.weight")
                    for attr, key in [("gamma", "weight"), ("beta", "bias"),
                                      ("running_mean", "running_mean"), ("running_var", "running_var")]:
                        upload(getattr(blk.dw_bn, attr), f"{p}.conv.1.1.{key}")
                    upload(blk.project.filters, f"{p}.conv.2.weight")
                    for attr, key in [("gamma", "weight"), ("beta", "bias"),
                                      ("running_mean", "running_mean"), ("running_var", "running_var")]:
                        upload(getattr(blk.project_bn, attr), f"{p}.conv.3.{key}")
                else:
                    upload(blk.dw.filters, f"{p}.conv.0.0.weight")
                    for attr, key in [("gamma", "weight"), ("beta", "bias"),
                                      ("running_mean", "running_mean"), ("running_var", "running_var")]:
                        upload(getattr(blk.dw_bn, attr), f"{p}.conv.0.1.{key}")
                    upload(blk.project.filters, f"{p}.conv.1.weight")
                    for attr, key in [("gamma", "weight"), ("beta", "bias"),
                                      ("running_mean", "running_mean"), ("running_var", "running_var")]:
                        upload(getattr(blk.project_bn, attr), f"{p}.conv.2.{key}")
                bidx += 1; fidx += 1

        load_conv_bn("features.18.0", "features.18.1", self.conv_last, self.bn_last)
        print(f"Loaded pretrained weights from {npz_path}")
```

### Step 3 â€” Feature extraction + classifier training

```python
import os, time
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

def load_images(folder, label, limit=500):
    images, labels = [], []
    for f in list(os.listdir(folder))[:limit + 200]:   # buffer for corrupt files
        if len(images) >= limit: break
        try:
            img = Image.open(os.path.join(folder, f)).convert('RGB').resize((224, 224))
            arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
            images.append((arr - IMAGENET_MEAN) / IMAGENET_STD)
            labels.append(label)
        except Exception:
            continue
    return images, labels

# Load data
cats, cat_labels = load_images("PetImages/Cat", 0, limit=500)
dogs, dog_labels = load_images("PetImages/Dog", 1, limit=500)
X = np.array(cats + dogs, dtype=np.float32)
Y = np.array(cat_labels + dog_labels, dtype=np.int32)

idx = np.random.permutation(len(X)); X, Y = X[idx], Y[idx]
split = int(0.9 * len(X))
X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]

# Build model + load weights
model = MobileNetV2(num_classes=2)
model.load_pretrained("mobilenet_v2_weights.npz")
for p in model.parameters(): p.requires_grad = False
model.eval()

# Phase 1: Extract features once (runs frozen backbone on GPU)
def extract(images, label=""):
    feats = []
    for i in range(0, len(images), 2):
        xb = Tensor(images[i:i+2])
        feats.append(model.extract_features(xb).sync().copy())
        end_batch()
    return np.concatenate(feats)

print("Extracting features...")
t0 = time.time()
F_train = extract(X_train, "train")
F_val   = extract(X_val,   "val")
print(f"Done in {time.time()-t0:.1f}s  ({F_train.shape[1]}-dim features)")

# Phase 2: Train linear head on cached features
clf = Linear(1280, 2)
opt = AdamW([clf.w, clf.b], lr=0.001, weight_decay=0.01)

for epoch in range(30):
    perm = np.random.permutation(len(F_train))
    opt.zero_grad()
    for i in range(0, len(F_train), 32):
        xb = Tensor(F_train[perm[i:i+32]])
        yb = Tensor(Y_train[perm[i:i+32]])
        loss = softmax_ce(clf(xb), yb)
        loss.backward(); opt.step(); opt.zero_grad()
        end_batch()

print("Training complete!")
```

### Benchmark Results (PetImages Cat vs Dog, Intel Xe iGPU)

---

## Changelog

### v0.2.0 — GPU-Side Performance Optimizations

Major GPU performance improvements reducing per-batch time on segmentation workloads:

**Dice Loss — 2-pass parallel backward (was O(N²), now O(1) per thread)**
- `nn_dice_loss.hlsl` rewritten with 256-thread shared-memory reduction; dispatch is `(batch, 1, 1)` instead of a single serial thread
- `nn_dice_loss_grad.hlsl` rewritten to a two-pass approach: pre-compute per-batch `{intersection, sum_pred, sum_target}` sums in a new Pass 1 shader (`nn_dice_loss_sums.hlsl`), then each backward thread reads directly from that buffer — eliminating the ~19ms O(N²) loop entirely

**GPU-side gradient clipping — eliminates 30+ CopyResource stalls**
- New `sgd_step_clipped(params, lr, max_norm)` function: gradient norm computation and SGD weight update are fully GPU-resident
- Three new shaders: `nn_grad_sq_reduce.hlsl` (per-param ² partial sums), `nn_grad_norm_final.hlsl` (total norm + clip scale), `nn_sgd_clipped.hlsl` (scaled weight update)
- New C++ function `SGDBatchClipped` in `engine.cpp`: only 1 GPU→CPU readback (the norm scalar) vs. 1 readback per parameter previously

**New and updated shaders in this release:**
`nn_dice_loss_sums.hlsl`, `nn_grad_sq_reduce.hlsl`, `nn_grad_norm_final.hlsl`, `nn_sgd_clipped.hlsl`, plus updated `nn_dice_loss.hlsl` and `nn_dice_loss_grad.hlsl`

### v0.1.9 and earlier
See [GitHub releases](https://github.com/raviadi12/directcompute_torch/releases) for full history.

| | DirectCompute (iGPU) | PyTorch (CPU) |
|--|--|--|
| Feature extraction (1600 imgs) | **41.7s** | 43.1s |
| Classifier training (30 epochs) | **0.9s** | 1.4s |
| **Total** | **42.7s** | 44.5s |
| **Test accuracy (400 unseen)** | **98.2%** | 98.0% |

The DirectCompute engine runs feature extraction faster than PyTorch CPU, on a 128MB integrated GPU with no dedicated VRAM.

---

## Full API Reference

### Layers

| Class | Description |
|-------|-------------|
| `Linear(in, out)` | Fully-connected layer. Weights: `(in, out)`, bias: `(out,)` |
| `ConvLayer(inC, outC, ks, stride, padding)` | 2D convolution via im2col+matmul. Skips im2col when `requires_grad=False` (frozen layers save GPU memory) |
| `DepthwiseConvLayer(channels, ks, stride, padding)` | Depthwise separable convolution â€” one filter per input channel. Used in MobileNet-style blocks |
| `BatchNorm2d(num_features)` | Batch normalization with running stats. Set `.training=False` for eval/frozen mode |
| `maxpool2d(x, pool_size, stride)` | Max pooling with saved indices for backward |
| `global_avg_pool2d(x)` | Global average pool: `(N, C, H, W)` â†’ `(N, C, 1, 1)` |
| `flatten(x)` | Flatten spatial dims: `(N, C, H, W)` â†’ `(N, C*H*W)` |

### Differentiable Operations

| Function | Description |
|----------|-------------|
| `relu(x)` | ReLU activation |
| `relu6(x)` | Clamped ReLU: `min(max(x, 0), 6)` â€” used in MobileNetV2 |
| `add(a, b)` | Element-wise add with full gradient support â€” enables residual/skip connections |
| `softmax_ce(logits, labels)` | Fused softmax + cross-entropy loss |
| `matmul(A, B, transA, transB)` | Matrix multiply with optional transpose flags |

### Optimizers

| Class | Description |
|-------|-------------|
| `SGD(params, lr)` | Stochastic Gradient Descent with gradient clipping |
| `Adam(params, lr, weight_decay)` | Adaptive Moment Estimation |
| `AdamW(params, lr, weight_decay)` | Adam with decoupled weight decay (recommended for most tasks) |
| `Muon(params, lr)` | Orthogonal gradient optimizer via Newton-Schulz iteration |

### Tensor

```python
t = Tensor(np_array, requires_grad=True)
t.sync()          # read data back from GPU â†’ numpy array
t.upload(data)    # push new data into existing GPU buffer (no realloc)
t.backward()      # run autograd backward from this tensor
t.shape           # tuple
t.size            # total element count
t.grad            # gradient Tensor (set after backward)
```

### Model Base Class

```python
class MyModel(Model):
    def forward(self, x): ...          # implement forward pass
    def parameters(self): ...          # return list of Tensors

model.parameters()                     # auto-discovered from Linear/ConvLayer/BatchNorm2d
model.train(); model.eval()            # toggle BN training mode
model.export("model.onnx", [1,3,224,224])  # ONNX export
```

### Training Utilities

```python
end_batch()          # flush GPU pipeline + bulk-free intermediate tensors
Metrics()            # tracks loss and accuracy across batches
get_pool_stats()     # (hits, misses) for GPU buffer pool
get_pool_memory()    # bytes currently in pool
```

---

## Contributing and Source Code

Full source code, C++ engine, and HLSL shader implementation:

**[https://github.com/raviadi12/directcompute_torch](https://github.com/raviadi12/directcompute_torch)**

## Future Roadmap

- **Vulkan & DX12 Backends**: Cross-platform GPU support.
- **UMA Optimizations**: Better memory paths for integrated GPUs.
- **Transposed Convolutions**: For upsampling / generative models.
- **Enhanced ONNX**: Full graph import for pretrained model deployment.
