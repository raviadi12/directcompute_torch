import ctypes
import numpy as np
import os
import sys
from PIL import Image
import struct
import time

def float_to_uint(f):
    return struct.unpack('I', struct.pack('f', f))[0]

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resource_dirs():
    dirs = [
        _MODULE_DIR,
        os.path.join(_MODULE_DIR, "directcompute_nn"),
        os.getcwd(),
    ]
    seen = set()
    out = []
    for path in dirs:
        norm = os.path.normcase(os.path.abspath(path))
        if norm not in seen and os.path.isdir(path):
            seen.add(norm)
            out.append(path)
    return out


def _find_resource(name):
    for base in _resource_dirs():
        candidate = os.path.join(base, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find resource '{name}' in: {_resource_dirs()}")


# Load the engine
lib = ctypes.CDLL(_find_resource("engine.dll"))
lib.InitEngine.restype = ctypes.c_bool
lib.CompileShader.argtypes = [ctypes.c_char_p, ctypes.c_wchar_p]
lib.CreateBuffer.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint]
lib.CreateBuffer.restype = ctypes.c_void_p
lib.AddRefBuffer.argtypes = [ctypes.c_void_p]
lib.ReadBuffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
lib.ReleaseBuffer.argtypes = [ctypes.c_void_p]
lib.ClearBuffer.argtypes = [ctypes.c_void_p]
lib.UpdateBuffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
lib.RunShader.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p, ctypes.c_uint]
lib.RunShaderRaw.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p, ctypes.c_uint]

# Fused operations (reduce Python→C++ crossings)
lib.RunShaderAlloc.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p, ctypes.c_uint]
lib.RunShaderAlloc.restype = ctypes.c_void_p
lib.RunShaderAlloc2.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p, ctypes.c_uint]
lib.RunShaderAlloc2.restype = ctypes.c_void_p
lib.SGDBatch.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_float, ctypes.c_float]
lib.SGDBatchClipped.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_float, ctypes.c_float]
lib.SGDBatchClipped.restype = ctypes.c_float
lib.ReleaseBufferBatch.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
lib.FlushGPU.argtypes = []
lib.FlushGPU.restype = None

# Fused C++ operations (eliminate multiple DLL round-trips per layer)
lib.MatMulAlloc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
lib.MatMulAlloc.restype = ctypes.c_void_p
lib.ConvForwardFused.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # x, filters, bias
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # batch, inC, inH, inW
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # outC, kH, kW
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # stride, padding, actType
    ctypes.POINTER(ctypes.c_void_p)]  # pIm2col
lib.ConvForwardFused.restype = ctypes.c_void_p
lib.ConvBackwardFused.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,  # grad_output, fwd_output (nullable)
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # x, filters, im2col
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # batch, inC, inH, inW
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # outC, kH, kW
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # stride, padding, actType
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # x/filters/bias requires_grad
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]  # dF, dB, dIn
lib.ConvBackwardFused.restype = None
lib.MaxPoolForwardFused.argtypes = [
    ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # x, batch, inC, inH, inW
    ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]  # pool_size, stride, pIndices
lib.MaxPoolForwardFused.restype = ctypes.c_void_p
lib.MaxPoolBackwardFused.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,  # grad_output, indices
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # batch, inC, outH, outW
    ctypes.c_uint]  # x_size
lib.MaxPoolBackwardFused.restype = ctypes.c_void_p

# Pool management APIs
lib.WarmPool.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
lib.WarmPool.restype = None
lib.GetPoolStats.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
lib.GetPoolStats.restype = None
lib.ResetPoolStats.argtypes = []
lib.ResetPoolStats.restype = None
lib.GetPoolMemory.argtypes = []
lib.GetPoolMemory.restype = ctypes.c_uint64
lib.DrainPool.argtypes = []
lib.DrainPool.restype = None

# Graph replay APIs
lib.ResolveShader.argtypes = [ctypes.c_char_p]
lib.ResolveShader.restype = ctypes.c_void_p
lib.RunSequence.argtypes = [ctypes.c_void_p, ctypes.c_uint]
lib.RunSequence.restype = None
lib.RunSequenceWithSGD.argtypes = [
    ctypes.c_void_p, ctypes.c_uint,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_float, ctypes.c_float]
lib.RunSequenceWithSGD.restype = None

lib.InitEngine()

# ── GPU Detection ──
class GPUInfo(ctypes.Structure):
    _fields_ = [
        ("vendorId", ctypes.c_uint),
        ("deviceId", ctypes.c_uint),
        ("dedicatedVRAM", ctypes.c_uint64),
        ("sharedMemory", ctypes.c_uint64),
        ("isIntegrated", ctypes.c_uint),
        ("description", ctypes.c_char * 128),
    ]
lib.GetGPUInfo.argtypes = [ctypes.POINTER(GPUInfo)]
lib.GetGPUInfo.restype = None

_gpu_info = GPUInfo()
lib.GetGPUInfo(ctypes.byref(_gpu_info))
_IS_IGPU = bool(_gpu_info.isIntegrated)
_GPU_VENDOR = {0x8086: "Intel", 0x10DE: "NVIDIA", 0x1002: "AMD"}.get(_gpu_info.vendorId, f"0x{_gpu_info.vendorId:04X}")
_GPU_NAME = _gpu_info.description.decode("ascii", errors="replace").strip("\x00")
_VRAM_MB = _gpu_info.dedicatedVRAM // (1024 * 1024)
_SHARED_MB = _gpu_info.sharedMemory // (1024 * 1024)
print(f"[GPU] {_GPU_NAME} ({_GPU_VENDOR})")
print(f"      VRAM: {_VRAM_MB}MB dedicated, {_SHARED_MB}MB shared -> {'iGPU' if _IS_IGPU else 'dGPU'}")

# ── Shader Compilation ──
shaders = ["matmul_universal", "matmul_coarsened", "add_bias", "relu", "relu_grad", "softmax", "softmax_ce_grad", "bias_grad", "sgd", "loss", 
           "conv_forward_tiled", "conv_backprop_filters_tiled", "conv_backprop_input_fused",
           "maxpool_forward", "maxpool_backward", "grad_accum", "im2col", "conv_reshape", "conv_grad_reshape", "col2im",
           "argmax_correct", "accumulate_scalar",
           "bias_relu", "conv_grad_reshape_relu",
           "batchnorm_mean_var", "batchnorm_forward", "batchnorm_backward_reduce", "batchnorm_backward_dx",
           "adam", "scale_add", "rms",
           "depthwise_conv_forward", "depthwise_conv_backward_input", "depthwise_conv_backward_filter",
           "global_avg_pool", "global_avg_pool_backward", "add",
           "relu6", "relu6_grad", "leaky_relu", "leaky_relu_grad",
           "upsample", "upsample_backward", "concat_insert", "slice",
           "dice_loss", "dice_loss_grad", "sigmoid", "sigmoid_grad",
           "matmul_reduce",
           "dice_loss_sums", "grad_sq_reduce", "grad_norm_final", "sgd_clipped"]
for s in shaders:
    lib.CompileShader(s.encode(), _find_resource(f"nn_{s}.hlsl"))

# Compile GPU-specific matmul variant
if not _IS_IGPU:
    lib.CompileShader(b"matmul_dgpu", _find_resource("nn_matmul_dgpu.hlsl"))
    print("      Compiled dGPU matmul variant (TS=64, WPT=4, K_STEP=16)")
else:
    print("      Using iGPU matmul variant (TS=64, WPT=4, K_STEP=32)")

class ParamsCB(ctypes.Structure):
    _fields_ = [("u1", ctypes.c_uint), ("u2", ctypes.c_uint), ("u3", ctypes.c_uint), ("u4", ctypes.c_uint),
                ("u5", ctypes.c_uint), ("u6", ctypes.c_uint), ("u7", ctypes.c_uint), ("u8", ctypes.c_uint),
                ("u9", ctypes.c_uint)]

class SGDParamsCB(ctypes.Structure):
    _fields_ = [("count", ctypes.c_uint), ("lr", ctypes.c_float), ("clip", ctypes.c_float), ("pad", ctypes.c_uint)]

# ── GPU Profiler ─────────────────────────────────────────────────────────────
class GPUProfiler:
    """Hooks into lib calls to track per-shader timing and VRAM allocations."""
    def __init__(self):
        self.enabled = False
        self._shader_time = {}    # shader_name -> total ms this epoch
        self._shader_calls = {}   # shader_name -> call count
        self._vram_bytes = 0      # current live VRAM
        self._vram_peak = 0       # peak VRAM this epoch
        self._alloc_count = 0
        self._free_count = 0
        self._epoch_alloc_bytes = 0
        # save originals
        self._orig_RunShader = lib.RunShader
        self._orig_CreateBuffer = lib.CreateBuffer
        self._orig_ReleaseBuffer = lib.ReleaseBuffer
        self._buf_sizes = {}      # handle -> size_bytes

    def enable(self):
        self.enabled = True
        self.reset()

    def disable(self):
        self.enabled = False

    def reset(self):
        self._shader_time.clear()
        self._shader_calls.clear()
        self._vram_peak = self._vram_bytes
        self._alloc_count = 0
        self._free_count = 0
        self._epoch_alloc_bytes = 0

    def track_run_shader(self, name, srvs, srvCount, uavs, uavCount, threads, cb, cbSize):
        if self.enabled:
            sname = name.decode() if isinstance(name, bytes) else name
            t0 = time.perf_counter()
            self._orig_RunShader(name, srvs, srvCount, uavs, uavCount, threads, cb, cbSize)
            dt = (time.perf_counter() - t0) * 1000.0
            self._shader_time[sname] = self._shader_time.get(sname, 0.0) + dt
            self._shader_calls[sname] = self._shader_calls.get(sname, 0) + 1
        else:
            self._orig_RunShader(name, srvs, srvCount, uavs, uavCount, threads, cb, cbSize)

    def track_create_buffer(self, data, count):
        handle = self._orig_CreateBuffer(data, count)
        if self.enabled and handle:
            sz = int(count) * 4
            self._buf_sizes[handle] = sz
            self._vram_bytes += sz
            self._epoch_alloc_bytes += sz
            self._alloc_count += 1
            if self._vram_bytes > self._vram_peak:
                self._vram_peak = self._vram_bytes
        return handle

    def track_release_buffer(self, handle):
        if self.enabled and handle and handle in self._buf_sizes:
            self._vram_bytes -= self._buf_sizes.pop(handle)
            self._free_count += 1
        self._orig_ReleaseBuffer(handle)

    def track_release_buffer_batch(self, handles, count):
        if self.enabled:
            for i in range(count):
                h = handles[i]
                if h and h in self._buf_sizes:
                    self._vram_bytes -= self._buf_sizes.pop(h)
                    self._free_count += 1
        self._orig_ReleaseBufferBatch(handles, count)

    def track_sgd_batch(self, params, grads, sizes, numParams, lr, clip):
        if self.enabled:
            t0 = time.perf_counter()
            self._orig_SGDBatch(params, grads, sizes, numParams, lr, clip)
            dt = (time.perf_counter() - t0) * 1000
            self._shader_time["sgd"] = self._shader_time.get("sgd", 0.0) + dt
            self._shader_calls["sgd"] = self._shader_calls.get("sgd", 0) + int(numParams)
        else:
            self._orig_SGDBatch(params, grads, sizes, numParams, lr, clip)

    def install(self):
        """Replace lib.RunShader/CreateBuffer/ReleaseBuffer with profiled versions."""
        self._orig_ReleaseBufferBatch = lib.ReleaseBufferBatch
        self._orig_SGDBatch = lib.SGDBatch
        lib.RunShader = self.track_run_shader
        lib.CreateBuffer = self.track_create_buffer
        lib.ReleaseBuffer = self.track_release_buffer
        lib.ReleaseBufferBatch = self.track_release_buffer_batch
        lib.SGDBatch = self.track_sgd_batch

    def uninstall(self):
        lib.RunShader = self._orig_RunShader
        lib.CreateBuffer = self._orig_CreateBuffer
        lib.ReleaseBuffer = self._orig_ReleaseBuffer
        lib.ReleaseBufferBatch = self._orig_ReleaseBufferBatch
        lib.SGDBatch = self._orig_SGDBatch

    def report(self, epoch_label=""):
        if not self.enabled:
            return
        total_ms = sum(self._shader_time.values())
        total_calls = sum(self._shader_calls.values())
        print(f"\n{'='*80}")
        print(f"  GPU DEBUG {epoch_label}")
        print(f"{'='*80}")
        print(f"  {'Shader':<28} {'Calls':>6} {'Total ms':>10} {'Avg ms':>10} {'% Time':>8}")
        print(f"  {'-'*28} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
        for name in sorted(self._shader_time, key=self._shader_time.get, reverse=True):
            ms = self._shader_time[name]
            calls = self._shader_calls[name]
            avg = ms / calls if calls else 0
            pct = (ms / total_ms * 100) if total_ms else 0
            print(f"  {name:<28} {calls:>6} {ms:>10.2f} {avg:>10.3f} {pct:>7.1f}%")
        print(f"  {'-'*28} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
        print(f"  {'TOTAL':<28} {total_calls:>6} {total_ms:>10.2f}")
        print()
        print(f"  VRAM live:  {self._vram_bytes / 1024 / 1024:>8.2f} MB")
        print(f"  VRAM peak:  {self._vram_peak / 1024 / 1024:>8.2f} MB")
        print(f"  Allocated:  {self._epoch_alloc_bytes / 1024 / 1024:>8.2f} MB  ({self._alloc_count} buffers)")
        print(f"  Freed:      {self._free_count} buffers")
        print(f"{'='*80}\n")

profiler = GPUProfiler()

# ── CPU Overhead Profiler ────────────────────────────────────────────────────
class CPUProfiler:
    """Tracks where CPU time goes: tensor creation, forward, backward, etc."""
    def __init__(self):
        self._timers = {}
        self._counts = {}
        self._t0 = None
        self._label = None

    def reset(self):
        self._timers.clear()
        self._counts.clear()

    def begin(self, label):
        self._t0 = time.perf_counter()
        self._label = label

    def end(self):
        if self._t0 is not None and self._label is not None:
            dt = (time.perf_counter() - self._t0) * 1000
            self._timers[self._label] = self._timers.get(self._label, 0.0) + dt
            self._counts[self._label] = self._counts.get(self._label, 0) + 1
            self._t0 = None
            self._label = None

    def report(self, wall_ms, epoch_label=""):
        total_tracked = sum(self._timers.values())
        print(f"\n{'='*80}")
        print(f"  CPU OVERHEAD {epoch_label}")
        print(f"{'='*80}")
        print(f"  {'Phase':<28} {'Calls':>6} {'Total ms':>10} {'Avg ms':>10} {'% Wall':>8}")
        print(f"  {'-'*28} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
        for name in sorted(self._timers, key=self._timers.get, reverse=True):
            ms = self._timers[name]
            calls = self._counts[name]
            avg = ms / calls if calls else 0
            pct = (ms / wall_ms * 100) if wall_ms else 0
            print(f"  {name:<28} {calls:>6} {ms:>10.2f} {avg:>10.3f} {pct:>7.1f}%")
        print(f"  {'-'*28} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
        untracked = wall_ms - total_tracked
        print(f"  {'Tracked':<28} {'':>6} {total_tracked:>10.2f} {'':>10} {total_tracked/wall_ms*100:>7.1f}%")
        print(f"  {'Untracked (interp/loop)':<28} {'':>6} {untracked:>10.2f} {'':>10} {untracked/wall_ms*100:>7.1f}%")
        print(f"  {'Wall total':<28} {'':>6} {wall_ms:>10.2f}")
        print(f"{'='*80}\n")

cpu_profiler = CPUProfiler()
# ─────────────────────────────────────────────────────────────────────────────

# ── Pre-allocated ctypes arrays for fast dispatch (avoid per-call allocation) ──
_srv_arr = (ctypes.c_void_p * 8)()
_uav_arr = (ctypes.c_void_p * 8)()
_threads = (ctypes.c_uint * 3)()
_cb = ParamsCB()

def _dispatch(name, srvs, uavs, tx, ty, tz, u1=0, u2=0, u3=0, u4=0, u5=0, u6=0, u7=0, u8=0, u9=0, cbSize=16):
    """Fast shader dispatch with pre-allocated ctypes arrays (typed SRV/UAV)."""
    nS = len(srvs)
    for i in range(nS): _srv_arr[i] = srvs[i]
    nU = len(uavs)
    for i in range(nU): _uav_arr[i] = uavs[i]
    _threads[0] = tx; _threads[1] = ty; _threads[2] = tz
    if cbSize > 0:
        _cb.u1 = u1; _cb.u2 = u2; _cb.u3 = u3; _cb.u4 = u4
        _cb.u5 = u5; _cb.u6 = u6; _cb.u7 = u7; _cb.u8 = u8; _cb.u9 = u9
        lib.RunShader(name, _srv_arr, nS, _uav_arr, nU, _threads, ctypes.byref(_cb), cbSize)
    else:
        lib.RunShader(name, _srv_arr, nS, _uav_arr, nU, _threads, None, 0)

def _dispatch_raw(name, srvs, uavs, tx, ty, tz, u1=0, u2=0, u3=0, u4=0, u5=0, u6=0, u7=0, u8=0, u9=0, cbSize=16):
    """Dispatch using raw SRV/UAV (ByteAddressBuffer/RWByteAddressBuffer)."""
    nS = len(srvs)
    for i in range(nS): _srv_arr[i] = srvs[i]
    nU = len(uavs)
    for i in range(nU): _uav_arr[i] = uavs[i]
    _threads[0] = tx; _threads[1] = ty; _threads[2] = tz
    if cbSize > 0:
        _cb.u1 = u1; _cb.u2 = u2; _cb.u3 = u3; _cb.u4 = u4
        _cb.u5 = u5; _cb.u6 = u6; _cb.u7 = u7; _cb.u8 = u8; _cb.u9 = u9
        lib.RunShaderRaw(name, _srv_arr, nS, _uav_arr, nU, _threads, ctypes.byref(_cb), cbSize)
    else:
        lib.RunShaderRaw(name, _srv_arr, nS, _uav_arr, nU, _threads, None, 0)

_all_tensors = []

def release_all_buffers():
    global _all_tensors
    # Collect GPU handles for batch release
    bufs_to_release = []
    for t in _all_tensors:
        if t._ctx:
            t._ctx.inputs = ()
            t._ctx = None
        t.grad = None
        if t.gpu_buf:
            bufs_to_release.append(t.gpu_buf)
            t.gpu_buf = None
    _all_tensors = []
    # Batch release: one Python→C++ call for all buffers
    if bufs_to_release:
        n = len(bufs_to_release)
        arr = (ctypes.c_void_p * n)(*bufs_to_release)
        lib.ReleaseBufferBatch(arr, n)
    # Non-blocking flush: let GPU start executing and free buffer memory.
    # Called separately in training loops so it can be timed independently.
    # For large models like AlexNet, call lib.FlushGPU() after this function.

def _shape_size(shape):
    """Pure Python product — faster than np.prod for small tuples."""
    s = 1
    for d in shape:
        s *= d
    return s

class Tensor:
    __slots__ = ['data', 'shape', 'size', 'requires_grad', 'grad', 'gpu_buf', '_ctx']

    def __init__(self, data, requires_grad=False, track=True):
        if isinstance(data, np.ndarray):
            self.data = data if data.dtype == np.float32 else data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.requires_grad = requires_grad
        self.grad = None
        self.gpu_buf = lib.CreateBuffer(self.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.size)
        self._ctx = None
        if track: _all_tensors.append(self)

    def __del__(self):
        if self.gpu_buf:
            lib.ReleaseBuffer(self.gpu_buf)
            self.gpu_buf = None

    @classmethod
    def from_gpu(cls, handle, shape, requires_grad=False, track=True):
        t = object.__new__(cls)
        t.data = None
        t.shape = shape
        t.size = _shape_size(shape)
        t.requires_grad = requires_grad
        t.grad = None
        t.gpu_buf = handle
        t._ctx = None
        if track: _all_tensors.append(t)
        return t

    def sync(self):
        if self.gpu_buf:
            if self.data is None:
                self.data = np.empty(self.shape, dtype=np.float32)
            lib.ReadBuffer(self.gpu_buf, self.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return self.data

    def upload(self, data):
        """Upload new data to an existing GPU buffer (no buffer creation)."""
        if data.dtype != np.float32: data = data.astype(np.float32)
        self.data = data
        self.shape = data.shape
        self.size = data.size
        lib.UpdateBuffer(self.gpu_buf, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def backward(self, grad=None):
        if grad is None:
            grad = Tensor(np.ones(self.shape, dtype=np.float32), track=False)
        self.grad = grad
        # Iterative topological sort (avoids Python recursion overhead)
        topo = []
        visited = set()
        stack = [self]
        while stack:
            v = stack[-1]
            if v in visited:
                stack.pop()
                continue
            # Check if all children visited
            children_done = True
            if v._ctx:
                for i in v._ctx.inputs:
                    if isinstance(i, Tensor) and i.requires_grad and i not in visited:
                        stack.append(i)
                        children_done = False
            if children_done:
                visited.add(v)
                stack.pop()
                topo.append(v)
        for v in reversed(topo):
            if v._ctx and v.grad is not None: v._ctx.backward(v.grad)

    def _accumulate_grad(self, grad):
        if self.grad is None or self.grad.gpu_buf is None:
            self.grad = grad
        else:
            _dispatch(b"grad_accum", (grad.gpu_buf,), (self.grad.gpu_buf,),
                      (self.size + 255)//256, 1, 1, u1=self.size)

class Function:
    def __init__(self, *inputs): self.inputs = inputs
    def backward(self, grad_output): raise NotImplementedError

# Matmul dispatch helper — GPU-adaptive
_TILE_U = 16  # Universal shader tile size (always 16)
if _IS_IGPU:
    _TILE_C = 64
    _COARSEN_THRESHOLD = 64
    _MM_COARSE_SHADER = b"matmul_coarsened"
else:
    _TILE_C = 64
    _COARSEN_THRESHOLD = 64
    _MM_COARSE_SHADER = b"matmul_dgpu"

def _run_mm(a_buf, b_buf, out_buf, M, K, N, flags=0):
    if M >= _COARSEN_THRESHOLD and N >= _COARSEN_THRESHOLD:
        _dispatch(_MM_COARSE_SHADER, (a_buf, b_buf), (out_buf,),
                  (N+_TILE_C-1)//_TILE_C, (M+_TILE_C-1)//_TILE_C, 1, u1=M, u2=K, u3=N, u4=flags)
    else:
        _dispatch(b"matmul_universal", (a_buf, b_buf), (out_buf,),
                  (N+_TILE_U-1)//_TILE_U, (M+_TILE_U-1)//_TILE_U, 1, u1=M, u2=K, u3=N, u4=flags)

class MatMul(Function):
    def forward(self, A, B, transA=False, transB=False):
        self.inputs = (A, B)
        self.transA = transA
        self.transB = transB
        flags = (1 if transA else 0) | (2 if transB else 0)
        
        M = A.shape[1] if transA else A.shape[0]
        K = A.shape[0] if transA else A.shape[1]
        N = B.shape[0] if transB else B.shape[1]
        
        out_handle = lib.MatMulAlloc(A.gpu_buf, B.gpu_buf, M, K, N, flags)
        res = Tensor.from_gpu(out_handle, (M, N), requires_grad=A.requires_grad or B.requires_grad)
        res._ctx = self
        return res
        
    def backward(self, grad_output):
        A, B = self.inputs
        if A.requires_grad:
            # If C = A @ B, then dA = dC @ B^T
            # If C = A^T @ B, then dA = B @ dC^T
            flags_A = 2 if not self.transB else 0
            M_A, K_A, N_A = grad_output.shape[0], B.shape[1], B.shape[0]
            out_handle = lib.MatMulAlloc(grad_output.gpu_buf, B.gpu_buf, M_A, K_A, N_A, flags_A)
            A._accumulate_grad(Tensor.from_gpu(out_handle, A.shape, track=True))
            
        if B.requires_grad:
            # If C = A @ B, then dB = A^T @ dC
            flags_B = 1 if not self.transA else 0
            M_B, K_B, N_B = A.shape[1], A.shape[0], grad_output.shape[1]
            out_handle = lib.MatMulAlloc(A.gpu_buf, grad_output.gpu_buf, M_B, K_B, N_B, flags_B)
            B._accumulate_grad(Tensor.from_gpu(out_handle, B.shape, track=True))

class Conv2D(Function):
    def __del__(self):
        if hasattr(self, 'im2col_handle') and self.im2col_handle:
            lib.ReleaseBuffer(self.im2col_handle)
            self.im2col_handle = None

    def forward(self, x, filters, bias, stride=1, padding=0, actType=0):
        self.inputs = (x, filters, bias)
        self.stride, self.padding, self.actType = stride, padding, actType
        batch, inC, inH, inW = x.shape
        outC, _, kH, kW = filters.shape
        outH = (inH + 2 * padding - kH) // stride + 1
        outW = (inW + 2 * padding - kW) // stride + 1

        needs_backward = x.requires_grad or filters.requires_grad or bias.requires_grad
        if needs_backward:
            pIm2col = ctypes.c_void_p()
            out_handle = lib.ConvForwardFused(
                x.gpu_buf, filters.gpu_buf, bias.gpu_buf,
                batch, inC, inH, inW, outC, kH, kW,
                stride, padding, actType, ctypes.byref(pIm2col))
            self.im2col_handle = pIm2col.value
        else:
            out_handle = lib.ConvForwardFused(
                x.gpu_buf, filters.gpu_buf, bias.gpu_buf,
                batch, inC, inH, inW, outC, kH, kW,
                stride, padding, actType, None)
            self.im2col_handle = None

        res = Tensor.from_gpu(out_handle, (batch, outC, outH, outW), requires_grad=needs_backward)
        self.fwd_output = res if actType != 0 and needs_backward else None
        res._ctx = self
        return res

    def backward(self, grad_output):
        x, filters, bias = self.inputs
        batch, inC, inH, inW = x.shape
        outC, _, kH, kW = filters.shape

        pDFilters = ctypes.c_void_p()
        pDBias = ctypes.c_void_p()
        pDInput = ctypes.c_void_p()

        fwd_buf = self.fwd_output.gpu_buf if self.fwd_output else None
        lib.ConvBackwardFused(
            grad_output.gpu_buf, fwd_buf,
            x.gpu_buf, filters.gpu_buf, self.im2col_handle,
            batch, inC, inH, inW, outC, kH, kW,
            self.stride, self.padding, self.actType,
            1 if x.requires_grad else 0,
            1 if filters.requires_grad else 0,
            1 if bias.requires_grad else 0,
            ctypes.byref(pDFilters), ctypes.byref(pDBias), ctypes.byref(pDInput))

        self.fwd_output = None
        self.im2col_handle = None  # released by ConvBackwardFused

        if pDFilters.value:
            filters._accumulate_grad(Tensor.from_gpu(pDFilters.value, filters.shape, track=True))
        if pDBias.value:
            bias._accumulate_grad(Tensor.from_gpu(pDBias.value, bias.shape, track=True))
        if pDInput.value:
            x._accumulate_grad(Tensor.from_gpu(pDInput.value, x.shape, track=True))

class MaxPool2D(Function):
    def __del__(self):
        if hasattr(self, 'indices_handle') and self.indices_handle:
            lib.ReleaseBuffer(self.indices_handle)
            self.indices_handle = None

    def forward(self, x, pool_size=2, stride=2):
        self.inputs = (x,)
        batch, inC, inH, inW = x.shape
        outH = (inH - pool_size) // stride + 1
        outW = (inW - pool_size) // stride + 1

        pIndices = ctypes.c_void_p()
        out_handle = lib.MaxPoolForwardFused(
            x.gpu_buf, batch, inC, inH, inW, pool_size, stride,
            ctypes.byref(pIndices))
        self.indices_handle = pIndices.value

        res = Tensor.from_gpu(out_handle, (batch, inC, outH, outW), requires_grad=x.requires_grad)
        res._ctx = self
        return res

    def backward(self, grad_output):
        x = self.inputs[0]
        batch, inC, outH, outW = grad_output.shape

        dInput = lib.MaxPoolBackwardFused(
            grad_output.gpu_buf, self.indices_handle,
            batch, inC, outH, outW, x.size)
        self.indices_handle = None  # released by MaxPoolBackwardFused

        x._accumulate_grad(Tensor.from_gpu(dInput, x.shape, track=True))

class Flatten(Function):
    def forward(self, x):
        self.inputs = (x,)
        lib.AddRefBuffer(x.gpu_buf)
        res = Tensor.from_gpu(x.gpu_buf, (x.shape[0], x.size // x.shape[0]), requires_grad=x.requires_grad, track=False)
        res._ctx = self
        return res
    def backward(self, grad_output):
        lib.AddRefBuffer(grad_output.gpu_buf)
        self.inputs[0]._accumulate_grad(Tensor.from_gpu(grad_output.gpu_buf, self.inputs[0].shape, track=False))

class AddBias(Function):
    def forward(self, A, B):
        self.inputs = (A, B)
        out_handle = lib.CreateBuffer(None, A.size)
        _dispatch(b"add_bias", (A.gpu_buf, B.gpu_buf), (out_handle,),
                  (A.size+255)//256, 1, 1, u1=A.shape[0], u2=A.shape[1])
        res = Tensor.from_gpu(out_handle, A.shape, requires_grad=A.requires_grad or B.requires_grad)
        res._ctx = self
        return res
    def backward(self, grad_output):
        A, B = self.inputs
        if A.requires_grad: A._accumulate_grad(grad_output)
        if B.requires_grad:
            dB_handle = lib.CreateBuffer(None, B.size)
            _dispatch(b"bias_grad", (grad_output.gpu_buf,), (dB_handle,),
                      B.size, 1, 1, u1=grad_output.shape[0], u2=B.size, u3=B.size, u4=1)
            B._accumulate_grad(Tensor.from_gpu(dB_handle, B.shape, track=True))

class BiasReLUFunc(Function):
    """Fused add_bias + ReLU: saves 1 dispatch + 1 buffer vs separate ops."""
    def forward(self, A, B):
        self.inputs = (A, B)
        out_handle = lib.CreateBuffer(None, A.size)
        _dispatch(b"bias_relu", (A.gpu_buf, B.gpu_buf), (out_handle,),
                  (A.size+255)//256, 1, 1, u1=A.shape[0], u2=A.shape[1])
        res = Tensor.from_gpu(out_handle, A.shape, requires_grad=A.requires_grad or B.requires_grad)
        self.output = res  # store for relu backward
        res._ctx = self
        return res
    def backward(self, grad_output):
        A, B = self.inputs
        out = self.output
        # Apply relu gradient: effective_grad = grad * (output > 0)
        eff_grad_handle = lib.CreateBuffer(None, grad_output.size)
        _dispatch(b"relu_grad", (grad_output.gpu_buf, out.gpu_buf), (eff_grad_handle,),
                  (grad_output.size+255)//256, 1, 1, u1=grad_output.size)
        eff_grad = Tensor.from_gpu(eff_grad_handle, grad_output.shape, track=True)
        self.output = None
        if A.requires_grad:
            A._accumulate_grad(eff_grad)
        if B.requires_grad:
            dB_handle = lib.CreateBuffer(None, B.size)
            _dispatch(b"bias_grad", (eff_grad.gpu_buf,), (dB_handle,),
                      B.size, 1, 1, u1=eff_grad.shape[0], u2=B.size, u3=B.size, u4=1)
            B._accumulate_grad(Tensor.from_gpu(dB_handle, B.shape, track=True))

class ReLUFunc(Function):
    def forward(self, x):
        self.inputs = (x,)
        out_handle = lib.CreateBuffer(None, x.size)
        _dispatch(b"relu", (x.gpu_buf,), (out_handle,),
                  (x.size+255)//256, 1, 1, cbSize=0)
        res = Tensor.from_gpu(out_handle, x.shape, requires_grad=x.requires_grad)
        res._ctx = self
        return res
    def backward(self, grad_output):
        x = self.inputs[0]
        if x.requires_grad:
            out_handle = lib.CreateBuffer(None, x.size)
            _dispatch(b"relu_grad", (grad_output.gpu_buf, x.gpu_buf), (out_handle,),
                      (x.size+255)//256, 1, 1, u1=x.size)
            x._accumulate_grad(Tensor.from_gpu(out_handle, x.shape, track=True))

class ReLU6Func(Function):
    def forward(self, x):
        self.inputs = (x,)
        out_handle = lib.CreateBuffer(None, x.size)
        _dispatch(b"relu6", (x.gpu_buf,), (out_handle,),
                  (x.size+255)//256, 1, 1, u1=x.size)
        res = Tensor.from_gpu(out_handle, x.shape, requires_grad=x.requires_grad)
        res._ctx = self
        return res
    def backward(self, grad_output):
        x = self.inputs[0]
        if x.requires_grad:
            out_handle = lib.CreateBuffer(None, x.size)
            _dispatch(b"relu6_grad", (grad_output.gpu_buf, x.gpu_buf), (out_handle,),
                      (x.size+255)//256, 1, 1, u1=x.size)
            x._accumulate_grad(Tensor.from_gpu(out_handle, x.shape, track=True))

class SigmoidFunc(Function):
    def forward(self, x):
        self.inputs = (x,)
        out_handle = lib.CreateBuffer(None, x.size)
        _dispatch(b"sigmoid", (x.gpu_buf,), (out_handle,),
                  (x.size+255)//256, 1, 1, u1=x.size)
        res = Tensor.from_gpu(out_handle, x.shape, requires_grad=x.requires_grad)
        res._ctx = self
        self.out_tensor = res
        return res
    def backward(self, grad_output):
        x = self.inputs[0]
        if x.requires_grad:
            out_handle = lib.CreateBuffer(None, x.size)
            _dispatch(b"sigmoid_grad", (grad_output.gpu_buf, self.out_tensor.gpu_buf), (out_handle,),
                      (x.size+255)//256, 1, 1, u1=x.size)
            x._accumulate_grad(Tensor.from_gpu(out_handle, x.shape, track=True))

class LeakyReLUFunc(Function):
    def forward(self, x, alpha=0.1):
        self.inputs = (x,)
        self.alpha = alpha
        out_handle = lib.CreateBuffer(None, x.size)
        _dispatch(b"leaky_relu", (x.gpu_buf,), (out_handle,),
                  (x.size+255)//256, 1, 1, u1=x.size, u2=float_to_uint(alpha))
        res = Tensor.from_gpu(out_handle, x.shape, requires_grad=x.requires_grad)
        res._ctx = self
        return res
    def backward(self, grad_output):
        x = self.inputs[0]
        if x.requires_grad:
            out_handle = lib.CreateBuffer(None, x.size)
            _dispatch(b"leaky_relu_grad", (grad_output.gpu_buf, x.gpu_buf), (out_handle,),
                      (x.size+255)//256, 1, 1, u1=x.size, u2=float_to_uint(self.alpha))
            x._accumulate_grad(Tensor.from_gpu(out_handle, x.shape, track=True))

class UpSample2D(Function):
    def forward(self, x, scaleH=2, scaleW=2):
        self.inputs = (x,)
        self.scaleH = scaleH
        self.scaleW = scaleW
        batch, C, inH, inW = x.shape
        outH, outW = inH * scaleH, inW * scaleW
        out_size = batch * C * outH * outW
        out_handle = lib.CreateBuffer(None, out_size)
        
        _dispatch(b"upsample", (x.gpu_buf,), (out_handle,),
                  (out_size+255)//256, 1, 1, 
                  u1=batch, u2=C, u3=inH, u4=inW, u5=outH, u6=outW, u7=scaleH, u8=scaleW, cbSize=32)
        
        res = Tensor.from_gpu(out_handle, (batch, C, outH, outW), requires_grad=x.requires_grad)
        res._ctx = self
        return res
    
    def backward(self, grad_output):
        x = self.inputs[0]
        if x.requires_grad:
            batch, C, inH, inW = x.shape
            outH, outW = inH * self.scaleH, inW * self.scaleW
            out_handle = lib.CreateBuffer(None, x.size)
            _dispatch(b"upsample_backward", (grad_output.gpu_buf,), (out_handle,),
                      (x.size+255)//256, 1, 1,
                      u1=batch, u2=C, u3=inH, u4=inW, u5=outH, u6=outW, u7=self.scaleH, u8=self.scaleW, cbSize=32)
            x._accumulate_grad(Tensor.from_gpu(out_handle, x.shape, track=True))

class Concat(Function):
    def forward(self, tensors, axis=1):
        if axis != 1:
            raise NotImplementedError("Concat currently only supported along channel dimension (axis 1)")
        self.inputs = tuple(tensors)
        batch = tensors[0].shape[0]
        H = tensors[0].shape[2]
        W = tensors[0].shape[3]
        spatial_size = H * W
        
        total_C = sum(t.shape[1] for t in tensors)
        out_size = batch * total_C * spatial_size
        out_handle = lib.CreateBuffer(None, out_size)
        
        c_offset = 0
        self.c_offsets = []
        for t in tensors:
            self.c_offsets.append(c_offset)
            c_in = t.shape[1]
            t_size = batch * c_in * spatial_size
            _dispatch(b"concat_insert", (t.gpu_buf,), (out_handle,),
                      (t_size+255)//256, 1, 1,
                      u1=batch, u2=c_in, u3=spatial_size, u4=total_C, u5=c_offset, cbSize=20)
            c_offset += c_in
            
        res = Tensor.from_gpu(out_handle, (batch, total_C, H, W), requires_grad=any(t.requires_grad for t in tensors))
        res._ctx = self
        return res
        
    def backward(self, grad_output):
        batch, total_C, H, W = grad_output.shape
        spatial_size = H * W
        
        for t, c_offset in zip(self.inputs, self.c_offsets):
            if t.requires_grad:
                c_in = t.shape[1]
                t_size = batch * c_in * spatial_size
                out_handle = lib.CreateBuffer(None, t_size)
                _dispatch(b"slice", (grad_output.gpu_buf,), (out_handle,),
                          (t_size+255)//256, 1, 1,
                          u1=batch, u2=c_in, u3=spatial_size, u4=total_C, u5=c_offset, cbSize=20)
                t._accumulate_grad(Tensor.from_gpu(out_handle, t.shape, track=True))

class SoftmaxCEFunc(Function):
    def forward(self, x, labels):
        self.inputs = (x, labels)
        batch_size, num_classes = x.shape
        softmax_handle = lib.CreateBuffer(None, x.size)
        _dispatch(b"softmax", (x.gpu_buf,), (softmax_handle,),
                  (batch_size+15)//16, 1, 1, u1=batch_size, u2=num_classes)
        self.softmax_out = Tensor.from_gpu(softmax_handle, x.shape)
        loss_handle = lib.CreateBuffer(None, 1)
        _dispatch(b"loss", (x.gpu_buf, labels.gpu_buf), (loss_handle,),
                  1, 1, 1, u1=batch_size, u2=num_classes)
        res = Tensor.from_gpu(loss_handle, (1,), requires_grad=True)
        res._ctx = self
        return res
    def backward(self, grad_output):
        x, labels = self.inputs
        batch_size, num_classes = x.shape
        out_handle = lib.CreateBuffer(None, x.size)
        _dispatch(b"softmax_ce_grad", (self.softmax_out.gpu_buf, labels.gpu_buf), (out_handle,),
                  (num_classes+15)//16, (batch_size+15)//16, 1, u1=batch_size, u2=num_classes)
        x._accumulate_grad(Tensor.from_gpu(out_handle, x.shape, track=True))

class DiceLossFunc(Function):
    def forward(self, pred, target):
        self.inputs = (pred, target)
        batch = pred.shape[0]
        size = pred.size // batch
        out_handle = lib.CreateBuffer(None, batch)
        # Parallel reduction: one thread group per batch element
        _dispatch(b"dice_loss", (pred.gpu_buf, target.gpu_buf), (out_handle,),
                  batch, 1, 1, u1=batch, u2=size)
        res = Tensor.from_gpu(out_handle, (batch,), requires_grad=True)
        res._ctx = self
        return res

    def backward(self, grad_output):
        pred, target = self.inputs
        batch = pred.shape[0]
        size = pred.size // batch
        # Pass 1: compute per-batch sums (intersection, sum_pred, sum_target)
        sums_handle = lib.CreateBuffer(None, batch * 3)
        _dispatch(b"dice_loss_sums", (pred.gpu_buf, target.gpu_buf), (sums_handle,),
                  batch, 1, 1, u1=batch, u2=size)
        # Pass 2: compute per-element gradients using precomputed sums
        out_handle = lib.CreateBuffer(None, pred.size)
        _dispatch(b"dice_loss_grad", (pred.gpu_buf, target.gpu_buf, sums_handle), (out_handle,),
                  (pred.size+255)//256, 1, 1, u1=batch, u2=size)
        lib.ReleaseBuffer(sums_handle)
        grad_tensor = Tensor.from_gpu(out_handle, pred.shape, track=True)
        pred._accumulate_grad(grad_tensor)

def matmul(A, B, transA=False, transB=False): return MatMul().forward(A, B, transA, transB)
def add_bias(A, B): return AddBias().forward(A, B)
def bias_relu(A, B): return BiasReLUFunc().forward(A, B)
def relu(x): return ReLUFunc().forward(x)
def relu6(x): return ReLU6Func().forward(x)
def sigmoid(x): return SigmoidFunc().forward(x)
def softmax_ce(x, labels): return SoftmaxCEFunc().forward(x, labels)
def conv2d(x, f, b, stride=1, padding=0, actType=0): return Conv2D().forward(x, f, b, stride, padding, actType)
def maxpool2d(x, pool_size=2, stride=2): return MaxPool2D().forward(x, pool_size, stride)
def flatten(x): return Flatten().forward(x)
def leaky_relu(x, alpha=0.1): return LeakyReLUFunc().forward(x, alpha)
def upsample2d(x, scaleH=2, scaleW=2): return UpSample2D().forward(x, scaleH, scaleW)
def concat(tensors, axis=1): return Concat().forward(tensors, axis)
def dice_loss(pred, target): return DiceLossFunc().forward(pred, target)

def clip_grad_norm(params, max_norm=1.0):
    """Clips gradient norm (matching PyTorch clip_grad_norm_).
    CPU fallback: syncs all grads to CPU, computes norm, re-uploads.
    Use sgd_step_clipped() instead for GPU-side fused clip+SGD.
    """
    total_norm_sq = 0.0
    grad_data = []
    for p in params:
        if p.grad is not None and p.grad.gpu_buf is not None:
            g = p.grad.sync()
            grad_data.append(g)
            total_norm_sq += float(np.sum(g * g))
        else:
            grad_data.append(None)
    total_norm = float(np.sqrt(total_norm_sq))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p, g in zip(params, grad_data):
            if g is not None:
                p.grad.upload(g * scale)
    return total_norm

def sgd_step_clipped(params, lr, max_norm=1.0):
    """Fused GPU-side gradient clipping + SGD update.
    Computes grad norm, clips, and updates weights entirely on GPU.
    Only reads back 1 float (the norm) instead of all parameters.
    Returns the total gradient norm.
    """
    n = len(params)
    pa = (ctypes.c_void_p * n)(*[p.gpu_buf for p in params])
    ga = (ctypes.c_void_p * n)(*[p.grad.gpu_buf for p in params])
    sa = (ctypes.c_uint * n)(*[p.size for p in params])
    return float(lib.SGDBatchClipped(pa, ga, sa, n, lr, max_norm))

def scale_add(A, B, alpha=1.0, beta=1.0):
    out_handle = lib.CreateBuffer(None, A.size)
    _dispatch(b"scale_add", (A.gpu_buf, B.gpu_buf), (out_handle,),
              (A.size + 255) // 256, 1, 1,
              u1=A.size, u2=float_to_uint(alpha), u3=float_to_uint(beta), cbSize=16)
    res = Tensor.from_gpu(out_handle, A.shape, track=True)
    return res

# ── Differentiable Element-wise Add (skip / residual connections) ────────────

class AddFunc(Function):
    def forward(self, A, B):
        self.inputs = (A, B)
        out_handle = lib.CreateBuffer(None, A.size)
        M_N = A.size
        _dispatch(b"add", (A.gpu_buf, B.gpu_buf), (out_handle,),
                  (M_N + 255) // 256, 1, 1, u1=M_N, u2=1)
        res = Tensor.from_gpu(out_handle, A.shape,
                              requires_grad=A.requires_grad or B.requires_grad)
        res._ctx = self
        return res
    def backward(self, grad_output):
        A, B = self.inputs
        if A.requires_grad:
            lib.AddRefBuffer(grad_output.gpu_buf)
            A._accumulate_grad(Tensor.from_gpu(grad_output.gpu_buf, A.shape, track=False))
        if B.requires_grad:
            lib.AddRefBuffer(grad_output.gpu_buf)
            B._accumulate_grad(Tensor.from_gpu(grad_output.gpu_buf, B.shape, track=False))

def add(A, B): return AddFunc().forward(A, B)

# ── Depthwise Conv2d ─────────────────────────────────────────────────────────

class DepthwiseConv2DFunc(Function):
    def forward(self, x, filters, bias, stride=1, padding=0):
        self.inputs = (x, filters, bias)
        self.stride, self.padding = stride, padding
        batch, C, inH, inW = x.shape
        _, _, kH, kW = filters.shape
        outH = (inH + 2 * padding - kH) // stride + 1
        outW = (inW + 2 * padding - kW) // stride + 1
        out_size = batch * C * outH * outW

        out_handle = lib.CreateBuffer(None, out_size)
        _dispatch(b"depthwise_conv_forward",
                  (x.gpu_buf, filters.gpu_buf, bias.gpu_buf), (out_handle,),
                  (out_size + 255) // 256, 1, 1,
                  u1=batch, u2=C, u3=inH, u4=inW,
                  u5=kH, u6=kW, u7=stride, u8=padding, u9=outH,
                  cbSize=36)
        res = Tensor.from_gpu(out_handle, (batch, C, outH, outW),
                              requires_grad=x.requires_grad or filters.requires_grad or bias.requires_grad)
        res._ctx = self
        return res

    def backward(self, grad_output):
        x, filters, bias = self.inputs
        batch, C, inH, inW = x.shape
        _, _, kH, kW = filters.shape
        outH = grad_output.shape[2]

        # dx
        if x.requires_grad:
            dx_handle = lib.CreateBuffer(None, x.size)
            _dispatch(b"depthwise_conv_backward_input",
                      (grad_output.gpu_buf, filters.gpu_buf), (dx_handle,),
                      (x.size + 255) // 256, 1, 1,
                      u1=batch, u2=C, u3=inH, u4=inW,
                      u5=kH, u6=kW, u7=self.stride, u8=self.padding, u9=outH,
                      cbSize=36)
            x._accumulate_grad(Tensor.from_gpu(dx_handle, x.shape, track=True))

        # dfilter + dbias (single dispatch)
        if filters.requires_grad or bias.requires_grad:
            df_handle = lib.CreateBuffer(None, filters.size)
            db_handle = lib.CreateBuffer(None, bias.size)
            _dispatch(b"depthwise_conv_backward_filter",
                      (grad_output.gpu_buf, x.gpu_buf), (df_handle, db_handle),
                      C, 1, 1,
                      u1=batch, u2=C, u3=inH, u4=inW,
                      u5=kH, u6=kW, u7=self.stride, u8=self.padding, u9=outH,
                      cbSize=36)
            if filters.requires_grad:
                filters._accumulate_grad(Tensor.from_gpu(df_handle, filters.shape, track=True))
            else:
                lib.ReleaseBuffer(df_handle)
            if bias.requires_grad:
                bias._accumulate_grad(Tensor.from_gpu(db_handle, bias.shape, track=True))
            else:
                lib.ReleaseBuffer(db_handle)

def depthwise_conv2d(x, f, b, stride=1, padding=0):
    return DepthwiseConv2DFunc().forward(x, f, b, stride, padding)

# ── Global Average Pooling ───────────────────────────────────────────────────

class GlobalAvgPool2DFunc(Function):
    def forward(self, x):
        self.inputs = (x,)
        batch, C, H, W = x.shape
        self.S = H * W
        out_handle = lib.CreateBuffer(None, batch * C)
        _dispatch(b"global_avg_pool", (x.gpu_buf,), (out_handle,),
                  batch * C, 1, 1,
                  u1=batch, u2=C, u3=self.S)
        res = Tensor.from_gpu(out_handle, (batch, C), requires_grad=x.requires_grad)
        res._ctx = self
        return res

    def backward(self, grad_output):
        x = self.inputs[0]
        if x.requires_grad:
            dx_handle = lib.CreateBuffer(None, x.size)
            batch, C = grad_output.shape
            _dispatch(b"global_avg_pool_backward",
                      (grad_output.gpu_buf,), (dx_handle,),
                      (x.size + 255) // 256, 1, 1,
                      u1=batch, u2=C, u3=self.S)
            x._accumulate_grad(Tensor.from_gpu(dx_handle, x.shape, track=True))

def global_avg_pool2d(x): return GlobalAvgPool2DFunc().forward(x)

def rms(A):
    out_handle = lib.CreateBuffer(None, 1)
    _dispatch(b"rms", (A.gpu_buf,), (out_handle,),
              1, 1, 1, u1=A.size, cbSize=16)
    v = np.empty(1, dtype=np.float32)
    lib.ReadBuffer(out_handle, v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    lib.ReleaseBuffer(out_handle)
    return max(float(v[0]), 1e-8)

class BatchNorm2dFunc(Function):
    def forward(self, x, gamma, beta, running_mean, running_var, eps=1e-5, momentum=0.1, training=True):
        self.inputs = (x, gamma, beta)
        self.batch, self.channels, self.h, self.w = x.shape
        self.S = self.h * self.w
        self.eps = eps
        
        if training:
            self.batch_mean_handle = lib.CreateBuffer(None, self.channels)
            self.batch_var_handle = lib.CreateBuffer(None, self.channels)
            
            _dispatch(b"batchnorm_mean_var", 
                      (x.gpu_buf,), 
                      (self.batch_mean_handle, self.batch_var_handle, running_mean.gpu_buf, running_var.gpu_buf),
                      self.channels, 1, 1, 
                      u1=self.batch, u2=self.channels, u3=self.S, u4=int(training), u5=float_to_uint(momentum), cbSize=20)
            
            used_mean_buf = self.batch_mean_handle
            used_var_buf = self.batch_var_handle
        else:
            used_mean_buf = running_mean.gpu_buf
            used_var_buf = running_var.gpu_buf
            self.batch_mean_handle = None
            self.batch_var_handle = None

        out_handle = lib.CreateBuffer(None, x.size)
        _dispatch(b"batchnorm_forward", 
                  (x.gpu_buf, used_mean_buf, used_var_buf, gamma.gpu_buf, beta.gpu_buf), 
                  (out_handle,),
                  (x.size + 255) // 256, 1, 1,
                  u1=self.batch, u2=self.channels, u3=self.S, u4=float_to_uint(eps), cbSize=16)

        res = Tensor.from_gpu(out_handle, x.shape, requires_grad=x.requires_grad or gamma.requires_grad or beta.requires_grad)
        res._ctx = self
        return res

    def backward(self, grad_output):
        x, gamma, beta = self.inputs
        
        if self.batch_mean_handle is None:
            raise RuntimeError("BatchNorm backward requires training=True during forward")
        
        dgamma_handle = lib.CreateBuffer(None, self.channels)
        dbeta_handle = lib.CreateBuffer(None, self.channels)
        
        _dispatch(b"batchnorm_backward_reduce",
                  (x.gpu_buf, grad_output.gpu_buf, self.batch_mean_handle, self.batch_var_handle),
                  (dgamma_handle, dbeta_handle),
                  self.channels, 1, 1,
                  u1=self.batch, u2=self.channels, u3=self.S, u4=float_to_uint(self.eps))

        dx_handle = lib.CreateBuffer(None, x.size)
        _dispatch(b"batchnorm_backward_dx",
                  (x.gpu_buf, grad_output.gpu_buf, self.batch_mean_handle, self.batch_var_handle, gamma.gpu_buf, dgamma_handle, dbeta_handle),
                  (dx_handle,),
                  (x.size + 255) // 256, 1, 1,
                  u1=self.batch, u2=self.channels, u3=self.S, u4=float_to_uint(self.eps))

        if gamma.requires_grad:
            gamma._accumulate_grad(Tensor.from_gpu(dgamma_handle, gamma.shape, track=True))
        else:
            lib.ReleaseBuffer(dgamma_handle)
            
        if beta.requires_grad:
            beta._accumulate_grad(Tensor.from_gpu(dbeta_handle, beta.shape, track=True))
        else:
            lib.ReleaseBuffer(dbeta_handle)
            
        if x.requires_grad:
            x._accumulate_grad(Tensor.from_gpu(dx_handle, x.shape, track=True))
        else:
            lib.ReleaseBuffer(dx_handle)
            
        lib.ReleaseBuffer(self.batch_mean_handle)
        lib.ReleaseBuffer(self.batch_var_handle)

def batchnorm2d(x, gamma, beta, rm, rv, eps=1e-5, momentum=0.1, training=True):
    return BatchNorm2dFunc().forward(x, gamma, beta, rm, rv, eps, momentum, training)

class Linear:
    def __init__(self, in_features, out_features):
        limit = np.sqrt(6 / (in_features + out_features))
        self.w = Tensor(np.random.uniform(-limit, limit, (in_features, out_features)), requires_grad=True, track=False)
        self.b = Tensor(np.zeros(out_features), requires_grad=True, track=False)
    def __call__(self, x, relu=False):
        m = matmul(x, self.w)
        return bias_relu(m, self.b) if relu else add_bias(m, self.b)

class ConvLayer:
    def __init__(self, inC, outC, ks, stride=1, padding=0):
        fan_in, fan_out = inC * ks * ks, outC * ks * ks
        limit = np.sqrt(6 / (fan_in + fan_out))
        self.filters = Tensor(np.random.uniform(-limit, limit, (outC, inC, ks, ks)), requires_grad=True, track=False)
        self.bias = Tensor(np.zeros(outC), requires_grad=True, track=False)
        self.stride, self.padding = stride, padding
    def __call__(self, x, actType=0, relu=False):
        act = 1 if relu else actType
        return conv2d(x, self.filters, self.bias, self.stride, self.padding, act)

class DepthwiseConvLayer:
    """Depthwise convolution: one filter per input channel.
    filters shape: (C, 1, kH, kW), bias shape: (C,)"""
    def __init__(self, channels, ks, stride=1, padding=0):
        limit = np.sqrt(6 / (ks * ks * 2))
        self.filters = Tensor(np.random.uniform(-limit, limit, (channels, 1, ks, ks)), requires_grad=True, track=False)
        self.bias = Tensor(np.zeros(channels), requires_grad=True, track=False)
        self.stride, self.padding = stride, padding
    def __call__(self, x):
        return depthwise_conv2d(x, self.filters, self.bias, self.stride, self.padding)

class BatchNorm2d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True, track=False)
        self.beta = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True, track=False)
        self.running_mean = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=False, track=False)
        self.running_var = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=False, track=False)
        self.training = True
        
    def __call__(self, x):
        return batchnorm2d(x, self.gamma, self.beta, self.running_mean, self.running_var, self.eps, self.momentum, self.training)


class Model:
    """Base class for models. Provides parameter discovery, export, and save/load.

    Usage:
        class LeNet(Model):
            def __init__(self):
                super().__init__()
                self.c1 = ConvLayer(1, 6, 5)
                self.l1 = Linear(256, 120)
            def forward(self, x):
                ...

        model = LeNet()
        model.parameters()          # auto-discovers all layer params
        model.export("lenet.onnx")  # exports to ONNX after training
    """

    def __init__(self):
        self._forward_trace = []   # populated by export()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def parameters(self):
        """Auto-discover all trainable parameters from ConvLayer, DepthwiseConvLayer, Linear, and BatchNorm2d attributes."""
        params = []
        for name in dir(self):
            attr = getattr(self, name)
            if isinstance(attr, (ConvLayer, DepthwiseConvLayer)):
                params.extend([attr.filters, attr.bias])
            elif isinstance(attr, Linear):
                params.extend([attr.w, attr.b])
            elif hasattr(attr, '__class__') and attr.__class__.__name__ == 'BatchNorm2d':
                params.extend([attr.gamma, attr.beta])
        return params

    def train(self, mode=True):
        for name in dir(self):
            attr = getattr(self, name)
            if hasattr(attr, 'training'):
                attr.training = mode

    def eval(self):
        self.train(False)

    def _get_layers(self):
        """Return ordered list of (name, layer) tuples."""
        layers = []
        for name in sorted(dir(self)):
            attr = getattr(self, name)
            if isinstance(attr, (ConvLayer, DepthwiseConvLayer, Linear)):
                layers.append((name, attr))
        return layers

    def export(self, path, input_shape=None, model_name=None):
        """Export trained model to ONNX format.

        Args:
            path: output .onnx file path
            input_shape: e.g. [1, 1, 28, 28] — required for graph construction.
                         Batch dim is made dynamic automatically.
            model_name: name embedded in the ONNX model (defaults to class name)
        """
        try:
            import onnx
            from onnx import helper, TensorProto, numpy_helper
        except ImportError:
            print("Install 'onnx' package to export: pip install onnx")
            return

        if input_shape is None:
            raise ValueError("input_shape required, e.g. [1, 1, 28, 28]")
        if model_name is None:
            model_name = self.__class__.__name__

        # Trace the forward pass to discover the ONNX graph structure
        nodes, initializers = [], []
        tensor_id = [0]
        def _next_name(prefix="t"):
            tensor_id[0] += 1
            return f"{prefix}_{tensor_id[0]}"

        # Walk through the forward pass by running a dummy trace
        # We inspect the model structure from layer attributes + forward() definition
        # This uses the _onnx_graph() hook if defined, otherwise auto-traces from layers
        if hasattr(self, '_onnx_graph'):
            nodes, initializers, output_name, output_shape = self._onnx_graph(
                input_shape, helper, TensorProto, numpy_helper)
        else:
            raise NotImplementedError(
                "Define _onnx_graph(self, input_shape, helper, TensorProto, numpy_helper) "
                "or use the pre-built LeNet/AlexNet models")

        # Build ONNX model
        input_shape_dynamic = [None] + list(input_shape[1:])
        output_shape_dynamic = [None] + list(output_shape[1:])

        graph = helper.make_graph(
            nodes,
            model_name,
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape_dynamic)],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape_dynamic)],
            initializer=initializers,
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        onnx.save(model, path)

        size = os.path.getsize(path)
        if size < 1024:
            sz = f"{size} B"
        elif size < 1024 * 1024:
            sz = f"{size / 1024:.1f} KB"
        else:
            sz = f"{size / (1024 * 1024):.1f} MB"
        print(f"Model exported to {path} ({sz})")

    @staticmethod
    def _conv_onnx(layer, name, input_name, output_name, helper, numpy_helper, initializers, nodes, with_relu=False, pool=None):
        """Helper: add Conv (+ optional Relu + MaxPool) ONNX nodes."""
        w = layer.filters.sync()  # (outC, inC, kH, kW) — matches ONNX
        b = layer.bias.sync()
        initializers.append(numpy_helper.from_array(w, f"{name}_w"))
        initializers.append(numpy_helper.from_array(b, f"{name}_b"))

        conv_out = f"{name}_conv"
        ks = [w.shape[2], w.shape[3]]
        nodes.append(helper.make_node("Conv", [input_name, f"{name}_w", f"{name}_b"], [conv_out],
                                      kernel_shape=ks,
                                      strides=[layer.stride, layer.stride],
                                      pads=[layer.padding, layer.padding, layer.padding, layer.padding]))
        current = conv_out
        if with_relu:
            relu_out = f"{name}_relu"
            nodes.append(helper.make_node("Relu", [current], [relu_out]))
            current = relu_out
        if pool:
            pool_out = output_name
            nodes.append(helper.make_node("MaxPool", [current], [pool_out],
                                          kernel_shape=[pool[0], pool[0]],
                                          strides=[pool[1], pool[1]]))
            current = pool_out
        else:
            # rename last output
            if current != output_name:
                nodes[-1].output[0] = output_name
        return output_name

    @staticmethod
    def _linear_onnx(layer, name, input_name, output_name, helper, numpy_helper, initializers, nodes, with_relu=False):
        """Helper: add Gemm (+ optional Relu) ONNX nodes."""
        w = layer.w.sync().T  # (in, out) → (out, in) for ONNX Gemm transB=1
        b = layer.b.sync()
        initializers.append(numpy_helper.from_array(w, f"{name}_w"))
        initializers.append(numpy_helper.from_array(b, f"{name}_b"))

        gemm_out = f"{name}_gemm" if with_relu else output_name
        nodes.append(helper.make_node("Gemm", [input_name, f"{name}_w", f"{name}_b"], [gemm_out], transB=1))
        if with_relu:
            nodes.append(helper.make_node("Relu", [gemm_out], [output_name]))
        return output_name

class ONNXModel(Model):
    """Dynamically parses and executes an ONNX model.
    Usage:
        model = ONNXModel("lenet.onnx")
        logits = model(xb)
    """
    def __init__(self, path):
        super().__init__()
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError:
            raise ImportError("Install 'onnx' package to load models: pip install onnx")
        
        self.onnx_model = onnx.load(path)
        self.graph = self.onnx_model.graph
        
        self.initializers = {}
        for init in self.graph.initializer:
            data = numpy_helper.to_array(init)
            self.initializers[init.name] = Tensor(data, track=False, requires_grad=False)
            
    def parameters(self):
        # By default, ONNX models are loaded for inference
        return []

    def forward(self, x):
        tensors = {self.graph.input[0].name: x}
        tensors.update(self.initializers)
        
        for node in self.graph.node:
            inputs = [tensors[name] for name in node.input]
            op = node.op_type
            
            if op == "Conv":
                w, b = inputs[1], inputs[2] if len(inputs) > 2 else Tensor(np.zeros(w.shape[0]))
                attrs = {a.name: a for a in node.attribute}
                stride = attrs['strides'].ints[0] if 'strides' in attrs else 1
                padding = attrs['pads'].ints[0] if 'pads' in attrs else 0
                out = conv2d(inputs[0], w, b, stride=stride, padding=padding)
            elif op == "Relu":
                out = relu(inputs[0])
            elif op == "MaxPool":
                attrs = {a.name: a for a in node.attribute}
                pool_size = attrs['kernel_shape'].ints[0] if 'kernel_shape' in attrs else 2
                stride = attrs['strides'].ints[0] if 'strides' in attrs else 2
                out = maxpool2d(inputs[0], pool_size=pool_size, stride=stride)
            elif op == "Flatten":
                out = flatten(inputs[0])
            elif op == "Gemm":
                attrs = {a.name: a for a in node.attribute}
                w, b = inputs[1], inputs[2]
                # Gemm in ONNX usually has transB=1
                # Our matmul assumes (N, K) @ (K, M).
                # If ONNX w is (M, K) and transB=1, we need to transpose w.
                # However, ONNX initializers are loaded as Tensors. We don't have transpose yet,
                # but we can reuse the linear logic.
                # Since we don't have dynamic transpose, we do it in numpy and upload? No, w is static!
                # Wait, Gemm weight is static.
                if attrs.get('transB', None) and attrs['transB'].i == 1:
                    w_data = w.sync()
                    w = Tensor(w_data.T, track=False)
                out = add_bias(matmul(inputs[0], w), b)
            else:
                raise NotImplementedError(f"ONNX operator '{op}' not supported yet.")
                
            for out_name in node.output:
                tensors[out_name] = out
                
        return tensors[self.graph.output[0].name]



class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
        # Pre-allocate ctypes arrays for SGDBatch (avoid per-call allocation)
        n = len(params)
        self._param_arr = (ctypes.c_void_p * n)()
        self._grad_arr = (ctypes.c_void_p * n)()
        self._size_arr = (ctypes.c_uint * n)()

    def zero_grad(self):
        for p in self.params:
            if p.grad: p.grad = None

    def step(self, clip=1.0):
        n = 0
        for p in self.params:
            if p.grad:
                self._param_arr[n] = p.gpu_buf
                self._grad_arr[n] = p.grad.gpu_buf
                self._size_arr[n] = p.size
                n += 1
        if n > 0:
            lib.SGDBatch(self._param_arr, self._grad_arr, self._size_arr,
                         n, self.lr, clip)


class AdamW:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [Tensor(np.zeros(p.shape), track=False) for p in params]
        self.v = [Tensor(np.zeros(p.shape), track=False) for p in params]
        # Keep references alive
        self._m_bufs = [x.gpu_buf for x in self.m]
        self._v_bufs = [x.gpu_buf for x in self.v]

    def zero_grad(self):
        for p in self.params:
            if p.grad: p.grad = None

    def step(self, clip=1.0):
        self.t += 1
        # bias correction
        step_size = self.lr * (1 - self.beta2**self.t)**0.5 / (1 - self.beta1**self.t)
        
        # We process in python loop, each dispatch takes ~0.005ms
        for i, p in enumerate(self.params):
            if not p.grad: continue
            _dispatch_raw(
                b"adam", 
                (p.grad.gpu_buf,), 
                (p.gpu_buf, self._m_bufs[i], self._v_bufs[i]),
                (p.size + 255) // 256, 1, 1,
                u1=p.size, 
                u2=float_to_uint(step_size), 
                u3=float_to_uint(self.beta1), 
                u4=float_to_uint(self.beta2),
                u5=float_to_uint(self.eps),
                u6=float_to_uint(self.lr * self.weight_decay),
                u7=float_to_uint(clip),
                cbSize=32
            )

class Adam(AdamW):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, betas, eps, weight_decay=weight_decay)

class Muon:
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01, ns_steps=5):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ns_steps = ns_steps
        self.m = [Tensor(np.zeros(p.shape, dtype=np.float32), track=False) for p in params]
        # AdamW objects internally for 1D variables
        self.adam_fallback = AdamW(params, lr=lr*0.1, weight_decay=weight_decay)
        
    def zero_grad(self):
        for p in self.params:
            if p.grad: p.grad = None

    def step(self, clip=1.0):
        # 1D params use AdamW
        self.adam_fallback.t += 1
        step_size_adam = self.adam_fallback.lr * (1 - self.adam_fallback.beta2**self.adam_fallback.t)**0.5 / (1 - self.adam_fallback.beta1**self.adam_fallback.t)

        for i, p in enumerate(self.params):
            if not p.grad: continue
            
            if len(p.shape) < 2:
                # SGD / AdamW fallback for 1D
                _dispatch_raw(
                    b"adam", 
                    (p.grad.gpu_buf,), 
                    (p.gpu_buf, self.adam_fallback._m_bufs[i], self.adam_fallback._v_bufs[i]),
                    (p.size + 255) // 256, 1, 1,
                    u1=p.size, u2=float_to_uint(step_size_adam), u3=float_to_uint(self.adam_fallback.beta1), 
                    u4=float_to_uint(self.adam_fallback.beta2), u5=float_to_uint(self.adam_fallback.eps),
                    u6=float_to_uint(self.adam_fallback.lr * self.adam_fallback.weight_decay),
                    u7=float_to_uint(clip), cbSize=32
                )
                continue
                
            # 2D variables (or flattened ND) use Muon
            # 1. Update momentum: m = momentum * m + grad
            _dispatch_raw(
                b"scale_add",
                (self.m[i].gpu_buf, p.grad.gpu_buf),
                (self.m[i].gpu_buf,),
                (p.size + 255) // 256, 1, 1,
                u1=p.size, u2=float_to_uint(self.momentum), u3=float_to_uint(1.0), cbSize=16
            )
            
            # 2. View as 2D for Newton-Schulz
            M = p.shape[0]
            K = p.size // M
            m_2d = Tensor.from_gpu(self.m[i].gpu_buf, (M, K), track=False)
            
            # 3. RMS Norm
            g_rms = rms(m_2d)
            X = scale_add(m_2d, m_2d, 1.0 / g_rms, 0.0)
            
            # 4. Newton Schulz iteration
            for _ in range(self.ns_steps):
                if M >= K:
                    A = matmul(X, X, transA=True)  # X^T @ X
                    B = matmul(X, A)               # X @ (X^T X)
                else:
                    A = matmul(X, X, transB=True)  # X @ X^T
                    B = matmul(A, X)               # (X X^T) @ X
                
                # X = 1.5 * X - 0.5 * B
                X = scale_add(X, B, 1.5, -0.5)

            # 5. Apply weight decay & update: w = w * (1 - lr*wd) - lr * X
            _dispatch_raw(
                b"scale_add",
                (p.gpu_buf, X.gpu_buf),
                (p.gpu_buf,),
                (p.size + 255) // 256, 1, 1,
                u1=p.size, u2=float_to_uint(1.0 - self.lr * self.weight_decay), u3=float_to_uint(-self.lr), cbSize=16
            )

# ── Internal GPU helpers (used by Metrics class) ─────────────────────────────
def gpu_argmax_correct(logits, labels, correct_buf):
    """Counts correct predictions on GPU."""
    batch_size, num_classes = logits.shape
    _dispatch_raw(b"argmax_correct", (logits.gpu_buf, labels.gpu_buf), (correct_buf,),
              (batch_size + 255) // 256, 1, 1, u1=batch_size, u2=num_classes)

def gpu_accumulate_loss(loss_tensor, accum_buf):
    """Adds loss_tensor[0] to accum_buf[0] on GPU."""
    _dispatch(b"accumulate_scalar", (loss_tensor.gpu_buf,), (accum_buf,),
              1, 1, 1, cbSize=0)


# ── Clean Public API ────────────────────────────────────────────────────────

class Metrics:
    """GPU-side metrics tracker. Accumulates loss and accuracy on GPU,
    syncs to CPU only when you call collect(). PyTorch-like usage:

        metrics = Metrics()
        for epoch in range(epochs):
            metrics.reset()
            for xb, yb in batches:
                logits = model(xb)
                loss = softmax_ce(logits, yb)
                metrics.update(loss, logits, yb)
                loss.backward()
                optimizer.step()
                end_batch()
            avg_loss, accuracy = metrics.collect(num_samples)
    """
    def __init__(self):
        self._correct_buf = lib.CreateBuffer(None, 1)
        self._loss_buf = lib.CreateBuffer(None, 1)
        self._f1 = np.empty(1, dtype=np.float32)
        self._f1_ptr = self._f1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._n_batches = 0
        self.reset()

    def reset(self):
        """Clear accumulators for a new epoch."""
        lib.ClearBuffer(self._correct_buf)
        lib.ClearBuffer(self._loss_buf)
        self._n_batches = 0

    def update(self, loss, logits, labels):
        """Track loss and accuracy for one batch (entirely on GPU, no sync)."""
        gpu_accumulate_loss(loss, self._loss_buf)
        gpu_argmax_correct(logits, labels, self._correct_buf)
        self._n_batches += 1

    def collect(self, num_samples):
        """Sync GPU and return (avg_loss, accuracy). Called once per epoch."""
        lib.ReadBuffer(self._loss_buf, self._f1_ptr)
        avg_loss = float(self._f1[0]) / self._n_batches if self._n_batches else 0.0
        lib.ReadBuffer(self._correct_buf, self._f1_ptr)
        correct = int(np.frombuffer(self._f1.tobytes(), dtype=np.uint32)[0])
        accuracy = correct / num_samples if num_samples else 0.0
        return avg_loss, accuracy

    def __del__(self):
        if self._correct_buf:
            lib.ReleaseBuffer(self._correct_buf)
        if self._loss_buf:
            lib.ReleaseBuffer(self._loss_buf)


def end_batch():
    """End-of-batch cleanup: release intermediate tensors and periodically flush GPU.
    Call this after optimizer.step() in every training batch.
    
    Flush strategy: submit GPU commands every N batches instead of every batch.
    Intel iGPU Flush() has ~4ms driver overhead per call, so reducing flush
    frequency from every batch to every ~10 batches saves significant time.
    """
    release_all_buffers()
    end_batch._count += 1
    if end_batch._count % end_batch.flush_interval == 0:
        lib.FlushGPU()

end_batch._count = 0
end_batch.flush_interval = 10  # flush every N batches (default: 10)


# ── Buffer Pool Warm-up ─────────────────────────────────────────────────────

def warm_pool(sizes, copies=2):
    """Pre-allocate GPU buffers of given sizes into the pool.
    
    Args:
        sizes: list/set of element counts (ints) to pre-warm
        copies: how many copies of each size to keep in pool
    
    Example:
        warm_pool(discover_buffer_sizes(forward, batch=32), copies=2)
    """
    sizes_arr = (ctypes.c_uint * len(sizes))(*sizes)
    lib.WarmPool(sizes_arr, len(sizes), copies)


def discover_buffer_sizes(forward_fn, params, X_batch, Y_batch, optimizer):
    """Run 1 training batch and discover all buffer sizes created.
    Returns a set of element counts suitable for warm_pool().
    
    Args:
        forward_fn: callable(xb) -> logits
        params: list of parameter Tensors
        X_batch, Y_batch: numpy arrays for a single batch  
        optimizer: SGD instance
    """
    sizes = set()
    orig_create = lib.CreateBuffer
    def _track_create(data, count):
        sizes.add(int(count))
        return orig_create(data, count)
    lib.CreateBuffer = _track_create
    
    try:
        optimizer.zero_grad()
        xb = Tensor(X_batch, track=True)
        yb = Tensor(Y_batch, track=True)
        logits = forward_fn(xb)
        loss = softmax_ce(logits, yb)
        loss.backward()
        optimizer.step(clip=1.0)
        end_batch()
    finally:
        lib.CreateBuffer = orig_create
    
    return sizes


def auto_warm(forward_fn, params, X_sample, Y_sample, optimizer, copies=2):
    """Discover buffer sizes and warm the pool in one call.
    Call this once before training to eliminate cold-start allocation overhead.
    
    Args:
        forward_fn: callable(xb) -> logits
        params: list of parameter Tensors
        X_sample, Y_sample: one batch of data (numpy arrays)
        optimizer: SGD instance
        copies: pool copies per size
    """
    sizes = discover_buffer_sizes(forward_fn, params, X_sample, Y_sample, optimizer)
    warm_pool(sizes, copies)
    lib.ResetPoolStats()
    return sizes


def get_pool_stats():
    """Return (hits, misses) from the engine buffer pool."""
    hits = ctypes.c_uint64(0)
    misses = ctypes.c_uint64(0)
    lib.GetPoolStats(ctypes.byref(hits), ctypes.byref(misses))
    return hits.value, misses.value


def get_pool_memory():
    """Return total bytes currently held in the buffer pool."""
    return lib.GetPoolMemory()


def drain_pool():
    """Free all pooled buffers, reclaiming VRAM."""
    lib.DrainPool()


# ── Compute Graph: Record & Replay ──────────────────────────────────────────

class _SeqCmd(ctypes.Structure):
    """Matches C++ SeqCmd struct — 256-byte dispatch command."""
    _fields_ = [
        ("shader",   ctypes.c_void_p),
        ("srvs",     ctypes.c_void_p * 8),
        ("srvCount", ctypes.c_uint),
        ("uavs",     ctypes.c_void_p * 4),
        ("uavCount", ctypes.c_uint),
        ("threads",  ctypes.c_uint * 3),
        ("cbData",   ctypes.c_ubyte * 64),
        ("cbSize",   ctypes.c_uint),
        ("pad",      ctypes.c_uint * 1),
    ]


class ComputeGraph:
    """Record one forward+backward pass, then replay from C++ with zero Python overhead.
    
    For fixed-shape models (same batch_size every iteration), all dispatch parameters
    are identical across batches — only input data changes. Record once, replay N times.
    
    Usage:
        graph = ComputeGraph()
        graph.trace(forward_fn, optimizer, X_batch, Y_batch, params, clip=1.0)
        # Training loop — one DLL call per batch instead of ~40:
        for epoch in ...:
            for xb, yb in batches:
                loss_val, logits_data = graph.replay(xb, yb)
    """
    
    def __init__(self):
        self._cmds = []              # list of _SeqCmd during recording
        self._cmd_array = None       # contiguous ctypes array after compile
        self._cmd_count = 0
        self._intermediates = []     # pre-allocated GPU buffers (kept alive)
        self._input_handle = None    # GPU buffer for input data
        self._label_handle = None    # GPU buffer for labels
        self._loss_handle = None     # GPU buffer to read loss from
        self._logits_handle = None   # GPU buffer to read logits from
        self._param_bufs = None      # ctypes array of param handles
        self._grad_bufs = None       # ctypes array of grad handles
        self._param_sizes = None     # ctypes array of param sizes
        self._num_params = 0
        self._lr = 0.0
        self._clip = 0.0
        self._compiled = False
        self._shader_cache = {}      # name -> resolved shader pointer
        self._loss_size = 0
        self._logits_shape = None
    
    def _resolve_shader(self, name):
        """Resolve shader name to pointer, with caching."""
        if name not in self._shader_cache:
            self._shader_cache[name] = lib.ResolveShader(name)
        return self._shader_cache[name]
    
    def trace(self, forward_fn, optimizer, X_batch, Y_batch, params, clip=1.0):
        """Run one batch (forward + backward + SGD) while recording all dispatches.
        
        After this call, the graph is compiled and ready for replay().
        Intermediate buffers are pre-allocated and kept alive.
        
        Args:
            forward_fn: callable(Tensor) -> Tensor (logits)
            optimizer: SGD instance
            X_batch, Y_batch: numpy arrays for one batch
            params: list of parameter Tensors
            clip: gradient clipping value
        """
        self._lr = optimizer.lr
        self._clip = clip
        self._cmds = []
        self._intermediates = []
        
        # Pre-allocate persistent input/label buffers
        X = X_batch if X_batch.dtype == np.float32 else X_batch.astype(np.float32)
        Y = Y_batch if Y_batch.dtype == np.float32 else Y_batch.astype(np.float32)
        self._input_handle = lib.CreateBuffer(
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), X.size)
        self._label_handle = lib.CreateBuffer(
            Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), Y.size)
        self._input_size = X.size
        self._label_size = Y.size
        self._input_shape = X_batch.shape
        self._label_shape = Y_batch.shape
        
        # Install dispatch hooks to capture commands
        orig_RunShader = lib.RunShader
        orig_RunShaderRaw = lib.RunShaderRaw
        captured = self  # closure reference
        
        def _hook_run(name, srvs, srvCount, uavs, uavCount, threads, cb, cbSize):
            """Capture dispatch command AND execute it."""
            cmd = _SeqCmd()
            cmd.shader = captured._resolve_shader(name)
            for i in range(srvCount):
                cmd.srvs[i] = srvs[i]
            cmd.srvCount = srvCount
            for i in range(uavCount):
                cmd.uavs[i] = uavs[i]
            cmd.uavCount = uavCount
            cmd.threads[0] = threads[0]
            cmd.threads[1] = threads[1]
            cmd.threads[2] = threads[2]
            if cb and cbSize > 0:
                cb_bytes = (ctypes.c_ubyte * cbSize).from_address(
                    ctypes.cast(cb, ctypes.c_void_p).value if not isinstance(cb, int) 
                    else cb)
                ctypes.memmove(cmd.cbData, cb_bytes, min(cbSize, 64))
            cmd.cbSize = cbSize
            captured._cmds.append(cmd)
            # Still execute normally during trace
            orig_RunShader(name, srvs, srvCount, uavs, uavCount, threads, cb, cbSize)
        
        # Hook both RunShader and RunShaderRaw
        lib.RunShader = _hook_run
        lib.RunShaderRaw = _hook_run
        
        # Track buffer creations to keep intermediates alive
        created_bufs = []
        orig_CreateBuffer = lib.CreateBuffer
        
        def _hook_create(data, count):
            handle = orig_CreateBuffer(data, count)
            created_bufs.append(handle)
            return handle
        lib.CreateBuffer = _hook_create
        
        # Also intercept batched SGD to capture param/grad handles
        sgd_captured = [False]
        orig_SGDBatch = lib.SGDBatch
        def _hook_sgd(p, g, s, n, lr, clip):
            sgd_captured[0] = True
            orig_SGDBatch(p, g, s, n, lr, clip)
        lib.SGDBatch = _hook_sgd
        
        try:
            # Run one complete training step
            optimizer.zero_grad()
            
            # Create input tensors using our persistent buffers
            lib.AddRefBuffer(self._input_handle)
            xb = Tensor.from_gpu(self._input_handle, X_batch.shape, 
                                 requires_grad=True, track=True)
            xb.data = X
            
            lib.AddRefBuffer(self._label_handle)
            yb = Tensor.from_gpu(self._label_handle, Y_batch.shape,
                                 requires_grad=False, track=True)
            yb.data = Y
            
            logits = forward_fn(xb)
            loss = softmax_ce(logits, yb)
            
            # Save handles for readback during replay
            self._loss_handle = loss.gpu_buf
            lib.AddRefBuffer(self._loss_handle)
            self._loss_size = loss.size
            
            self._logits_handle = logits.gpu_buf
            lib.AddRefBuffer(self._logits_handle)
            self._logits_shape = logits.shape
            
            loss.backward()
            optimizer.step(clip=clip)
            
            # DON'T call end_batch — we need buffers alive for replay
        finally:
            lib.RunShader = orig_RunShader
            lib.RunShaderRaw = orig_RunShaderRaw
            lib.CreateBuffer = orig_CreateBuffer
            lib.SGDBatch = orig_SGDBatch
        
        # Keep ALL intermediate buffers alive (AddRef) so handles stay valid
        for handle in created_bufs:
            if handle:
                lib.AddRefBuffer(handle)
                self._intermediates.append(handle)
        
        # Set up SGD parameter arrays
        self._num_params = len(params)
        ParamArr = ctypes.c_void_p * self._num_params
        SizeArr = ctypes.c_uint * self._num_params
        GradArr = ctypes.c_void_p * self._num_params
        self._param_bufs = ParamArr(*[p.gpu_buf for p in params])
        self._param_sizes = SizeArr(*[p.size for p in params])
        # Save gradient buffer handles NOW (before release_all_buffers destroys them)
        self._grad_bufs = GradArr(*[p.grad.gpu_buf if p.grad else None for p in params])
        # AddRef grad buffers to keep them alive
        for i in range(self._num_params):
            if self._grad_bufs[i]:
                lib.AddRefBuffer(self._grad_bufs[i])
        self._params_ref = params
        
        # Compile command array
        self._compile()
        
        # Release trace-batch tensors (intermediates kept alive by our AddRef above)
        release_all_buffers()
        
        return loss, logits
    
    def _compile(self):
        """Build contiguous ctypes array from recorded commands."""
        n = len(self._cmds)
        CmdArray = _SeqCmd * n
        self._cmd_array = CmdArray(*self._cmds)
        self._cmd_count = n
        self._compiled = True
    
    def replay(self, X_batch, Y_batch):
        """Execute the recorded graph with new input data.
        
        Returns (loss_value, logits_array) — no Python dispatch overhead.
        One DLL call for all ~39 dispatches + SGD.
        """
        if not self._compiled:
            raise RuntimeError("Graph not compiled. Call trace() first.")
        
        # Update input buffers (2 DLL calls)
        X = X_batch if X_batch.dtype == np.float32 else X_batch.astype(np.float32)
        Y = Y_batch if Y_batch.dtype == np.float32 else Y_batch.astype(np.float32)
        lib.UpdateBuffer(self._input_handle, 
                        X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        lib.UpdateBuffer(self._label_handle,
                        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        
        # Zero gradients using saved handles
        for i in range(self._num_params):
            if self._grad_bufs[i]:
                lib.ClearBuffer(self._grad_bufs[i])
        
        # Execute ALL dispatches in one C++ call (forward + backward)
        lib.RunSequence(ctypes.cast(self._cmd_array, ctypes.c_void_p),
                       self._cmd_count)
        
        # SGD update using saved gradient handles
        lib.SGDBatch(self._param_bufs, self._grad_bufs, self._param_sizes,
                    self._num_params, self._lr, self._clip)
    
    def read_loss(self):
        """Read loss value from GPU (requires flush)."""
        out = np.empty(self._loss_size, dtype=np.float32)
        lib.ReadBuffer(self._loss_handle, 
                      out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return float(out[0])
    
    def read_logits(self):
        """Read logits from GPU (requires flush)."""
        out = np.empty(self._logits_shape, dtype=np.float32)
        lib.ReadBuffer(self._logits_handle,
                      out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return out
    
    def release(self):
        """Free all pre-allocated intermediate buffers."""
        for handle in self._intermediates:
            if handle:
                lib.ReleaseBuffer(handle)
        self._intermediates = []
        if self._input_handle:
            lib.ReleaseBuffer(self._input_handle)
            self._input_handle = None
        if self._label_handle:
            lib.ReleaseBuffer(self._label_handle)
            self._label_handle = None
        if self._loss_handle:
            lib.ReleaseBuffer(self._loss_handle)
            self._loss_handle = None
        if self._logits_handle:
            lib.ReleaseBuffer(self._logits_handle)
            self._logits_handle = None
        if self._grad_bufs:
            for i in range(self._num_params):
                if self._grad_bufs[i]:
                    lib.ReleaseBuffer(self._grad_bufs[i])
            self._grad_bufs = None
        self._compiled = False
    
    def __del__(self):
        self.release()
        self.release()
