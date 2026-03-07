# DirectCompute Neural Network Engine

A from-scratch neural network training framework that runs entirely on the GPU using **DirectCompute (D3D11 Compute Shaders)**. No CUDA, no cuDNN — just raw HLSL shaders dispatched through a thin C++ runtime, driven from Python via ctypes.

## Platform Support

This project currently works on **Windows only**.

Reason: the runtime is built on **DirectCompute / Direct3D 11**, which is part of the Windows graphics stack.

If you need cross-platform support, that is future work.

## How To Use

If you want the ready-to-use version, install from PyPI on Windows:

```bash
pip install directcompute-nn
```

The Windows wheel already includes:

- `engine.dll`
- bundled `nn_*.hlsl` shaders
- the Python runtime API

So you can use the engine immediately without compiling C++.

### Quick Example

```python
import numpy as np
from nn_engine import Tensor, Linear, relu

x = Tensor(np.random.randn(32, 128).astype(np.float32))
layer = Linear(128, 64)
y = relu(layer(x))

print(y.shape)  # (32, 64)
```

### Train And Export Example

```python
from nn_engine import ConvLayer, Linear, Model, maxpool2d, flatten

class LeNet(Model):
    def __init__(self):
        super().__init__()
        self.c1 = ConvLayer(1, 6, 5)
        self.c2 = ConvLayer(6, 16, 5)
        self.l1 = Linear(16*4*4, 120)
        self.l2 = Linear(120, 84)
        self.l3 = Linear(84, 10)

    def forward(self, x):
        x = self.c1(x, relu=True)
        x = maxpool2d(x)
        x = self.c2(x, relu=True)
        x = maxpool2d(x)
        x = flatten(x)
        x = self.l1(x, relu=True)
        x = self.l2(x, relu=True)
        return self.l3(x)

    def _onnx_graph(self, input_shape, helper, TensorProto, numpy_helper):
        nodes, initializers = [], []
        h = Model._conv_onnx
        g = Model._linear_onnx
        h(self.c1, "c1", "input", "pool1", helper, numpy_helper, initializers, nodes, with_relu=True, pool=(2, 2))
        h(self.c2, "c2", "pool1", "pool2", helper, numpy_helper, initializers, nodes, with_relu=True, pool=(2, 2))
        nodes.append(helper.make_node("Flatten", ["pool2"], ["flat"], axis=1))
        g(self.l1, "l1", "flat", "fc1", helper, numpy_helper, initializers, nodes, with_relu=True)
        g(self.l2, "l2", "fc1", "fc2", helper, numpy_helper, initializers, nodes, with_relu=True)
        g(self.l3, "l3", "fc2", "output", helper, numpy_helper, initializers, nodes)
        return nodes, initializers, "output", [1, 10]

model = LeNet()
model.export("lenet.onnx", input_shape=[1, 1, 28, 28])
```

### Available API Surface

Current layers:

- `Linear`
- `ConvLayer`
- `DepthwiseConvLayer` — depthwise separable convolution (MobileNet-style)
- `BatchNorm2d`
- `maxpool2d`
- `global_avg_pool2d` — global average pooling (needed for ResNet/MobileNet heads)
- `flatten`

Current activations and losses:

- `relu`
- `relu6` — clamped ReLU used in MobileNetV2
- `add` — differentiable element-wise add (skip connections / residual blocks)
- `softmax_ce`

Current optimizers:

- `SGD`
- `Adam`
- `AdamW`
- `Muon`

Current model utilities:

- `Model.parameters()`
- `Model.export()`
- `ONNXModel`
- `Metrics`

### Transfer Learning Example

The engine can load pretrained ImageNet weights and run feature extraction. See [`train_petnet_transfer.py`](train_petnet_transfer.py) for a complete example that:

1. Downloads MobileNetV2 pretrained weights from torchvision (once, via `download_mobilenet.py`)
2. Freezes the backbone — Conv2D automatically skips im2col allocation when `requires_grad=False`, saving GPU memory
3. Extracts 1280-dim features once for all images
4. Trains a linear classifier on the cached features
5. Evaluates on a held-out unseen test set

```
python download_mobilenet.py   # requires torch, one-time download
python train_petnet_transfer.py
```

Results on PetImages (Cat vs Dog), 400 unseen test images, Intel Xe iGPU:

| Phase | Time |
|-------|------|
| Feature extraction (1600 images) | 41.7s |
| Classifier training (30 epochs) | 0.9s |
| **Total** | **42.7s** |

**Test accuracy: 98.2%** — matching PyTorch CPU (43.1s, 98.0%) while running on a 128MB iGPU.

## Contributing from GitHub

We welcome contributions! If you want to work on the engine itself, modify shaders, or optimize the C++ runtime:

1.  **Clone the Repo**: `git clone https://github.com/raviadi12/directcompute_torch.git`
2.  **Environment**: Ensure you have the [Build Prerequisites](#build-prerequisites) for C++ compilation.
3.  **Refactor & PR**: Feel free to submit Pull Requests for new layers, optimized kernels, or bug fixes.
4.  **Issue Tracking**: Report any DirectCompute-specific driver issues or performance regressions in the Issues tab.

## Architecture Overview

```
Python (nn_engine.py)          ←  Autograd, layers, optimizer
    │
    ▼  ctypes FFI
C++ DLL (engine.dll)           ←  D3D11 device, shader dispatch, buffer management
    │
    ▼  ID3D11DeviceContext::Dispatch()
HLSL Compute Shaders (nn_*.hlsl)  ←  GPU kernels for every operation
```

### Layer Stack

| Component | File | Role |
|-----------|------|------|
| **Runtime** | `engine.cpp` → `engine.dll` | D3D11 device init, buffer create/read/release, shader compile & dispatch |
| **Framework** | `nn_engine.py` | Tensor class, autograd (topological backward), layers (Linear, ConvLayer, DepthwiseConvLayer, MaxPool2D, GlobalAvgPool2D, BatchNorm2d, Flatten), optimizers (SGD, Adam, AdamW, Muon) |
| **Training scripts** | `train_lenet.py`, `train_alexnet.py` | End-to-end training loops with validation |
| **Shaders** | `nn_*.hlsl` | One HLSL file per GPU kernel (see full list below) |

## Build Prerequisites

- **Windows 10/11** with a DirectX 11-capable GPU
- **Visual Studio 2022+** (need `cl.exe` and `vcvarsall.bat` for compiling the DLL)
- **Python 3.10+** with `numpy` and `Pillow`

```
pip install numpy pillow
```

## Building

### Compile the engine DLL

```bat
compile_engine.bat
```

This runs:
```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
cl.exe /EHsc /O2 /LD engine.cpp /link /OUT:engine.dll
```

> **Note:** Adjust the Visual Studio path in `compile_engine.bat` to match your installation. The key requirement is `vcvarsall.bat` to set up the MSVC x64 toolchain.

Output: `engine.dll` in the project root (loaded by `nn_engine.py` at import time).

### HLSL shaders

Shaders are **compiled at runtime** by `D3DCompileFromFile()` when `nn_engine.py` is imported — no separate shader compilation step needed. Each shader must have entry point `CSMain` and target `cs_5_0`.

## Running

### LeNet on MNIST

```
python train_lenet.py
```

Expects `mnist/0/`, `mnist/1/`, ..., `mnist/9/` folders containing grayscale digit images.

### AlexNet on PetImages

```
python train_alexnet.py
```

Expects `PetImages/Cat/` and `PetImages/Dog/` folders containing JPEG images (resized to 224×224 at load time).

## Future Work

- **Higher-Level Backends**: Implement **Vulkan** and **DirectX 12** backends for cross-platform support and better multi-queue orchestration.
- **Unified Memory Architecture (UMA) Optimizations**: Tailored memory mapping and `memcpy` paths for integrated GPUs (iGPUs) to minimize staging buffer overhead.
- **Sparse Operations**: GPU Kernels for sparse matrix computations and efficient pruning techniques.
- **Advanced ONNX Compatibility**: Full-scale ONNX graph importing and optimized deployment paths.
- **Fused Operators**: More fused HLSL kernels (e.g., Fused Conv-BN-ReLU) to reduce dispatch-bound bottlenecks.

## How the Engine Works

### engine.cpp — The D3D11 Runtime

The C++ DLL exposes 8 functions via `extern "C" __declspec(dllexport)`:

| Function | Purpose |
|----------|---------|
| `InitEngine()` | Creates D3D11 device + immediate context |
| `CompileShader(name, path)` | Compiles HLSL from file, stores `ID3D11ComputeShader*` by name |
| `CreateBuffer(data, count)` | Allocates a `StructuredBuffer<float>` on GPU with SRV + UAV views |
| `AddRefBuffer(handle)` | Increments ref count (for shared buffers like Flatten) |
| `ReleaseBuffer(handle)` | Decrements ref count; frees GPU resources at zero |
| `ReadBuffer(handle, dst)` | Copies GPU → CPU via staging buffer (triggers `Flush()`) |
| `ClearBuffer(handle)` | Zeros a UAV buffer (used before scatter ops like MaxPool backward) |
| `RunShader(name, srvs, srvCount, uavs, uavCount, threads, cb, cbSize)` | The main dispatch call — binds resources, sets constant buffer, calls `Dispatch()` |

#### Buffer Layout

Every GPU buffer is a `StructuredBuffer<float>` wrapped in a `GPUBuffer` struct:
```cpp
struct GPUBuffer {
    ID3D11Buffer* buffer;              // The raw D3D11 buffer
    ID3D11ShaderResourceView* srv;     // For reading in shaders (register t0, t1, ...)
    ID3D11UnorderedAccessView* uav;    // For writing in shaders (register u0, u1, ...)
    uint32_t size;                     // Element count (floats)
    int refCount;                      // Manual ref counting for shared ownership
};
```

#### Constant Buffer Convention

Shader parameters are passed through a constant buffer at `register(b0)`. Two layouts are used:

**ParamsCB** — 9 × uint32 (36 bytes max):
```hlsl
cbuffer Params : register(b0) {
    uint u1, u2, u3, u4, u5, u6, u7, u8, u9;
};
```
Each shader interprets these fields differently (e.g., M/K/N/flags for matmul, batch/inC/inH/inW/kH/stride/padding/outH/outW for im2col).

**SGDParamsCB** — 16 bytes:
```hlsl
cbuffer Params : register(b0) {
    uint count; float lr; float clip; uint pad;
};
```

### nn_engine.py — The Python Autograd Framework

#### Tensor
- Wraps a NumPy array + a GPU buffer handle
- `requires_grad=True` enables gradient tracking
- `track=True` registers for bulk cleanup via `release_all_buffers()`
- `.sync()` reads GPU data back to CPU (triggers D3D11 `Flush()`)
- `.backward()` builds topological order and runs backward pass

#### Autograd
Each differentiable operation is a `Function` subclass with `forward()` and `backward()` methods. The forward pass stores `self.inputs` and attaches `res._ctx = self` to the output tensor. The backward pass receives `grad_output` and calls `input._accumulate_grad(grad)`.

Gradient accumulation uses the `grad_accum` shader to add in-place on GPU when a tensor receives gradients from multiple paths.

#### Convolution: The im2col Approach

Convolutions are implemented via the **im2col** transformation, which converts convolution into matrix multiplication:

**Forward:**
```
input → [im2col] → col_matrix → [matmul] filters × col_matrix → [conv_reshape] → output
```
1. `im2col` shader extracts patches into a `(inC*kH*kW) × (batch*outH*outW)` matrix
2. `matmul` computes `filters(outC × inC*kH*kW) × col_matrix` → `(outC × batch*outH*outW)`
3. `conv_reshape` adds bias and reshapes to `(batch, outC, outH, outW)`

**Backward:**
```
grad_output → [conv_grad_reshape] → grad_reshaped (outC × batch*outH*outW)

Filter gradients:   grad_reshaped × im2col_matrix^T  (reuses saved im2col from forward)
Input gradients:    filters^T × grad_reshaped → [col2im] → grad_input
Bias gradients:     sum over spatial dims of grad_reshaped
```

The transpose operations use the matmul `flags` parameter instead of explicit transpose shaders.

## HLSL Shader Reference

### Matmul Shaders

| Shader | Tile Size | Description |
|--------|-----------|-------------|
| `nn_matmul_universal.hlsl` | 16×16 | Tiled matmul with transpose flags. Used for small matrices (M or N < 128) |
| `nn_matmul_coarsened.hlsl` | 64×64, 4×4 WPT | Coarsened tiled matmul. 16×16 thread groups, each thread computes a 4×4 output block. Used for large matrices |

Both shaders share the same constant buffer: `{M, K, N, flags}` where:
- `flags & 1` → transpose A (read A as column-major)
- `flags & 2` → transpose B (read B as column-major)

The Python helper `_run_mm()` automatically selects the shader based on matrix dimensions (threshold: 128).

### Convolution Shaders

| Shader | Purpose |
|--------|---------|
| `nn_im2col.hlsl` | Extracts image patches into column matrix for convolution |
| `nn_col2im.hlsl` | Scatters column matrix back to image format (backward pass) |
| `nn_conv_reshape.hlsl` | Reshapes matmul output to (batch, C, H, W) + adds bias |
| `nn_conv_grad_reshape.hlsl` | Reshapes (batch, C, H, W) gradient to matrix form for backward matmul |

### Activation / Loss Shaders

| Shader | Purpose |
|--------|---------|
| `nn_relu.hlsl` | ReLU forward: `max(0, x)` |
| `nn_relu_grad.hlsl` | ReLU backward: `grad * (x > 0)` |
| `nn_relu6.hlsl` | ReLU6 forward: `min(max(0, x), 6)` |
| `nn_relu6_grad.hlsl` | ReLU6 backward: `grad * (0 < x < 6)` |
| `nn_softmax.hlsl` | Numerically stable softmax (per-row max subtraction) |
| `nn_softmax_ce_grad.hlsl` | Combined softmax + cross-entropy gradient: `softmax - one_hot` |
| `nn_loss.hlsl` | Cross-entropy loss computation |
| `nn_add.hlsl` | Element-wise add for differentiable skip connections |

### Pooling Shaders

| Shader | Purpose |
|--------|---------|
| `nn_maxpool_forward.hlsl` | Max pooling with index tracking (stores argmax for backward) |
| `nn_maxpool_backward.hlsl` | Scatter gradients to max positions using saved indices |
| `nn_global_avg_pool.hlsl` | Global average pooling: mean over H×W → (N, C) |
| `nn_global_avg_pool_backward.hlsl` | Broadcast gradient back to H×W |

### Depthwise Convolution Shaders

| Shader | Purpose |
|--------|---------|
| `nn_depthwise_conv_forward.hlsl` | Depthwise conv forward: one filter per channel |
| `nn_depthwise_conv_backward_input.hlsl` | Depthwise conv input gradient |
| `nn_depthwise_conv_backward_filter.hlsl` | Depthwise conv filter gradient |

### Utility Shaders

| Shader | Purpose |
|--------|---------|
| `nn_add_bias.hlsl` | Adds bias vector to each row: `out[i] = A[i] + B[i % cols]` |
| `nn_bias_grad.hlsl` | Sum-reduces rows to compute bias gradient |
| `nn_sgd.hlsl` | SGD update: `param -= lr * clamp(grad, -clip, clip)` |
| `nn_grad_accum.hlsl` | In-place gradient accumulation: `accum += grad` |

### Legacy / Unused Shaders

These exist in the repo but are **not used** by the current engine — the im2col approach replaced them:

| Shader | Note |
|--------|------|
| `nn_conv_forward.hlsl` | Direct convolution (replaced by im2col + matmul) |
| `nn_conv_forward_tiled.hlsl` | Tiled direct convolution (replaced) |
| `nn_conv_backprop_filters.hlsl` | Direct filter gradient (replaced by matmul transpose) |
| `nn_conv_backprop_filters_tiled.hlsl` | Tiled version (still compiled but unused by im2col path) |
| `nn_conv_backprop_input.hlsl` | Direct input gradient (replaced by matmul transpose + col2im) |
| `nn_conv_backprop_input_fused.hlsl` | Fused version (still compiled but unused) |
| `nn_conv_reduce_filters.hlsl` | Filter gradient reduction (unused) |
| `nn_matmul.hlsl` | Original matmul without transpose (superseded by universal) |
| `nn_matmul_transpose_a.hlsl` | Dedicated transpose-A matmul (superseded by flags) |
| `nn_matmul_transpose_b.hlsl` | Dedicated transpose-B matmul (superseded by flags) |

## Optimizations

### Constant Buffer Caching

The engine reuses a single `D3D11_USAGE_DYNAMIC` constant buffer across all dispatches via `MAP_WRITE_DISCARD`. This avoids creating and destroying a CB per shader call. The cache grows if a larger CB is needed but never shrinks.

### Adaptive Matmul Selection

```python
_COARSEN_THRESHOLD = 128
if M >= 128 and N >= 128:
    use nn_matmul_coarsened   # 64×64 tiles, 4×4 work-per-thread
else:
    use nn_matmul_universal   # 16×16 tiles, 1×1 work-per-thread
```

The coarsened shader is ~2× faster for large matrices but has tile overhead that hurts small ones (like LeNet's 84×10 FC layer).

### Transpose via Flags (No Extra Buffers)

Instead of explicit transpose operations or separate transpose shaders, both matmul shaders accept a `flags` field in the constant buffer. The shader adjusts its indexing at load time:
```hlsl
uint idxA = (flags & 1) ? (col * M + row) : (row * K + col);  // transpose A
uint idxB = (flags & 2) ? (col * K + row) : (row * N + col);  // transpose B
```
This saves GPU memory and dispatch overhead for backward-pass transposes.

### SRV/UAV Unbinding

After every `Dispatch()`, the engine unbinds all SRVs and UAVs:
```cpp
g_context->CSSetShaderResources(0, srvCount, nullSRVs);
g_context->CSSetUnorderedAccessViews(0, uavCount, nullUAVs, nullptr);
```
This is **critical** in D3D11. Without it, a buffer written as UAV in one dispatch and read as SRV in the next will silently return stale/zero data, because D3D11 detects the binding conflict and unbinds the SRV automatically.

### Per-Dispatch Flush

Each `RunShader()` call ends with `g_context->Flush()`. While counter-intuitive (batching dispatches seems better), removing Flush causes a **2× slowdown** on D3D11 because:
- Without Flush, the CPU queues many dispatches but the GPU sits idle until `ReadBuffer()` triggers execution
- With Flush, the GPU starts executing immediately while the CPU prepares the next dispatch
- This enables CPU-GPU pipelining, which is essential for the many small dispatches in a neural network training step

## Memory Management

- **Tracked tensors**: Intermediate tensors created during forward/backward are registered via `track=True` and bulk-freed by `release_all_buffers()` at the end of each batch
- **Persistent tensors**: Weights and biases use `track=False` and survive across batches
- **Ref counting**: `AddRefBuffer`/`ReleaseBuffer` handle shared ownership (e.g., `Flatten` shares its input's GPU buffer)
- **im2col buffer**: Saved during forward for reuse in backward, then explicitly released

## File Structure

```
engine.cpp                     # D3D11 compute shader runtime (→ engine.dll)
compile_engine.bat             # Build script for engine.dll
nn_engine.py                   # Python autograd framework + ctypes bindings
train_lenet.py                 # LeNet-5 on MNIST
train_alexnet.py               # AlexNet on PetImages
train_petnet_transfer.py       # MobileNetV2 transfer learning on PetImages
download_mobilenet.py          # Download pretrained MobileNetV2 weights (requires torch, once)
train_petnet_transfer_pytorch.py  # PyTorch equivalent for benchmarking
nn_*.hlsl                      # HLSL compute shaders (one per operation)
mnist/                         # MNIST digit images (0-9 subfolders)
PetImages/                     # Cat/Dog image dataset
```
