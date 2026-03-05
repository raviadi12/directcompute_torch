import ctypes
import numpy as np
import os
import sys
from PIL import Image
import time

# Load the engine
lib = ctypes.CDLL("./engine.dll")
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
lib.ReleaseBufferBatch.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
lib.FlushGPU.argtypes = []
lib.FlushGPU.restype = None

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
           "bias_relu", "conv_grad_reshape_relu"]
for s in shaders:
    lib.CompileShader(s.encode(), f"nn_{s}.hlsl")

# Compile GPU-specific matmul variant
if not _IS_IGPU:
    lib.CompileShader(b"matmul_dgpu", "nn_matmul_dgpu.hlsl")
    print("      Compiled dGPU matmul variant (TS=32, WPT=8, K_STEP=16)")
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
_srv1 = (ctypes.c_void_p * 1)()
_srv2 = (ctypes.c_void_p * 2)()
_uav1 = (ctypes.c_void_p * 1)()
_uav2 = (ctypes.c_void_p * 2)()
_threads = (ctypes.c_uint * 3)()
_cb = ParamsCB()

def _dispatch(name, srvs, uavs, tx, ty, tz, u1=0, u2=0, u3=0, u4=0, u5=0, u6=0, u7=0, u8=0, u9=0, cbSize=16):
    """Fast shader dispatch with pre-allocated ctypes arrays (typed SRV/UAV)."""
    nS = len(srvs)
    if nS == 1:
        _srv1[0] = srvs[0]
        srv_arr = _srv1
    else:
        _srv2[0] = srvs[0]; _srv2[1] = srvs[1]
        srv_arr = _srv2
    nU = len(uavs)
    if nU == 1:
        _uav1[0] = uavs[0]
        uav_arr = _uav1
    else:
        _uav2[0] = uavs[0]; _uav2[1] = uavs[1]
        uav_arr = _uav2
    _threads[0] = tx; _threads[1] = ty; _threads[2] = tz
    if cbSize > 0:
        _cb.u1 = u1; _cb.u2 = u2; _cb.u3 = u3; _cb.u4 = u4
        _cb.u5 = u5; _cb.u6 = u6; _cb.u7 = u7; _cb.u8 = u8; _cb.u9 = u9
        lib.RunShader(name, srv_arr, nS, uav_arr, nU, _threads, ctypes.byref(_cb), cbSize)
    else:
        lib.RunShader(name, srv_arr, nS, uav_arr, nU, _threads, None, 0)

def _dispatch_raw(name, srvs, uavs, tx, ty, tz, u1=0, u2=0, u3=0, u4=0, u5=0, u6=0, u7=0, u8=0, u9=0, cbSize=16):
    """Dispatch using raw SRV/UAV (ByteAddressBuffer/RWByteAddressBuffer)."""
    nS = len(srvs)
    if nS == 1:
        _srv1[0] = srvs[0]
        srv_arr = _srv1
    else:
        _srv2[0] = srvs[0]; _srv2[1] = srvs[1]
        srv_arr = _srv2
    nU = len(uavs)
    if nU == 1:
        _uav1[0] = uavs[0]
        uav_arr = _uav1
    else:
        _uav2[0] = uavs[0]; _uav2[1] = uavs[1]
        uav_arr = _uav2
    _threads[0] = tx; _threads[1] = ty; _threads[2] = tz
    if cbSize > 0:
        _cb.u1 = u1; _cb.u2 = u2; _cb.u3 = u3; _cb.u4 = u4
        _cb.u5 = u5; _cb.u6 = u6; _cb.u7 = u7; _cb.u8 = u8; _cb.u9 = u9
        lib.RunShaderRaw(name, srv_arr, nS, uav_arr, nU, _threads, ctypes.byref(_cb), cbSize)
    else:
        lib.RunShaderRaw(name, srv_arr, nS, uav_arr, nU, _threads, None, 0)

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
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v._ctx:
                    for i in v._ctx.inputs:
                        if isinstance(i, Tensor): build_topo(i)
                topo.append(v)
        build_topo(self)
        for v in reversed(topo):
            if v._ctx: v._ctx.backward(v.grad)

    def _accumulate_grad(self, grad):
        if self.grad is None:
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
    # iGPU: TS=64, WPT=4 — maximize compute-to-load ratio for limited bandwidth
    _TILE_C = 64
    _COARSEN_THRESHOLD = 64
    _MM_COARSE_SHADER = b"matmul_coarsened"
else:
    # dGPU: TS=32, WPT=8 — smaller tiles for more workgroups across many SMs
    _TILE_C = 32
    _COARSEN_THRESHOLD = 32
    _MM_COARSE_SHADER = b"matmul_dgpu"

def _run_mm(a_buf, b_buf, out_buf, M, K, N, flags=0):
    if M >= _COARSEN_THRESHOLD and N >= _COARSEN_THRESHOLD:
        _dispatch(_MM_COARSE_SHADER, (a_buf, b_buf), (out_buf,),
                  (N+_TILE_C-1)//_TILE_C, (M+_TILE_C-1)//_TILE_C, 1, u1=M, u2=K, u3=N, u4=flags)
    else:
        _dispatch(b"matmul_universal", (a_buf, b_buf), (out_buf,),
                  (N+_TILE_U-1)//_TILE_U, (M+_TILE_U-1)//_TILE_U, 1, u1=M, u2=K, u3=N, u4=flags)

class MatMul(Function):
    def forward(self, A, B):
        self.inputs = (A, B)
        M, K, N = A.shape[0], A.shape[1], B.shape[1]
        out_handle = lib.CreateBuffer(None, M * N)
        _run_mm(A.gpu_buf, B.gpu_buf, out_handle, M, K, N, 0)
        res = Tensor.from_gpu(out_handle, (M, N), requires_grad=A.requires_grad or B.requires_grad)
        res._ctx = self
        return res
    def backward(self, grad_output):
        A, B = self.inputs
        if A.requires_grad:
            out_handle = lib.CreateBuffer(None, A.size)
            _run_mm(grad_output.gpu_buf, B.gpu_buf, out_handle, A.shape[0], B.shape[1], B.shape[0], 2)
            A._accumulate_grad(Tensor.from_gpu(out_handle, A.shape, track=True))
        if B.requires_grad:
            out_handle = lib.CreateBuffer(None, B.size)
            _run_mm(A.gpu_buf, grad_output.gpu_buf, out_handle, A.shape[1], A.shape[0], grad_output.shape[1], 1)
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
        outH, outW = (inH + 2 * padding - kH) // stride + 1, (inW + 2 * padding - kW) // stride + 1
        
        totalRow, totalCol = inC * kH * kW, batch * outH * outW
        im2col_handle = lib.CreateBuffer(None, totalRow * totalCol)
        self.im2col_handle = im2col_handle
        
        _dispatch(b"im2col", (x.gpu_buf,), (im2col_handle,),
                  (totalCol + 15)//16, (totalRow + 15)//16, 1,
                  u1=batch, u2=inC, u3=inH, u4=inW, u5=kH, u6=stride, u7=padding, u8=outH, u9=outW, cbSize=36)
        
        mm_handle = lib.CreateBuffer(None, outC * totalCol)
        _run_mm(filters.gpu_buf, im2col_handle, mm_handle, outC, totalRow, totalCol, 0)
        
        out_handle = lib.CreateBuffer(None, batch * outC * outH * outW)
        _dispatch(b"conv_reshape", (mm_handle, bias.gpu_buf), (out_handle,),
                  (totalCol+15)//16, (outC+15)//16, 1,
                  u1=batch, u2=outC, u3=outH, u4=outW, u5=actType, cbSize=20)
        
        lib.ReleaseBuffer(mm_handle)
        res = Tensor.from_gpu(out_handle, (batch, outC, outH, outW), requires_grad=x.requires_grad or filters.requires_grad or bias.requires_grad)
        # Store output reference for fused relu backward
        self.fwd_output = res if actType != 0 else None
        res._ctx = self
        return res

    def backward(self, grad_output):
        x, filters, bias = self.inputs
        batch, inC, inH, inW = x.shape
        outC, _, kH, kW = filters.shape
        outH, outW = grad_output.shape[2], grad_output.shape[3]
        totalRow, totalCol = inC * kH * kW, batch * outH * outW
        
        # Fused grad reshape + ReLU gradient (saves 1 dispatch + 1 buffer)
        grad_reshaped = lib.CreateBuffer(None, outC * totalCol)
        if self.actType == 1 and self.fwd_output is not None:
            _dispatch(b"conv_grad_reshape_relu",
                      (grad_output.gpu_buf, self.fwd_output.gpu_buf), (grad_reshaped,),
                      (totalCol + 15)//16, (outC + 15)//16, 1,
                      u1=batch, u5=outC, u8=outH, u9=outW, cbSize=36)
            self.fwd_output = None
        else:
            _dispatch(b"conv_grad_reshape", (grad_output.gpu_buf,), (grad_reshaped,),
                      (totalCol + 15)//16, (outC + 15)//16, 1,
                      u1=batch, u5=outC, u8=outH, u9=outW, cbSize=36)
        
        if filters.requires_grad:
            dF_handle = lib.CreateBuffer(None, outC * totalRow)
            _run_mm(grad_reshaped, self.im2col_handle, dF_handle, outC, totalCol, totalRow, 2)
            filters._accumulate_grad(Tensor.from_gpu(dF_handle, filters.shape, track=True))
            
        if bias.requires_grad:
            dB_handle = lib.CreateBuffer(None, outC)
            _dispatch(b"bias_grad", (grad_reshaped,), (dB_handle,),
                      outC, 1, 1, u1=totalCol, u2=outC, u3=1, u4=totalCol)
            bias._accumulate_grad(Tensor.from_gpu(dB_handle, bias.shape, track=True))
            
        if x.requires_grad:
            dIcol_handle = lib.CreateBuffer(None, totalRow * totalCol)
            _run_mm(filters.gpu_buf, grad_reshaped, dIcol_handle, totalRow, outC, totalCol, 1)
            dIn_handle = lib.CreateBuffer(None, x.size)
            _dispatch(b"col2im", (dIcol_handle,), (dIn_handle,),
                      (x.size + 255)//256, 1, 1,
                      u1=batch, u2=inC, u3=inH, u4=inW, u5=kH, u6=self.stride, u7=self.padding, u8=outH, u9=outW, cbSize=36)
            lib.ReleaseBuffer(dIcol_handle)
            x._accumulate_grad(Tensor.from_gpu(dIn_handle, x.shape, track=True))
            
        lib.ReleaseBuffer(grad_reshaped)
        lib.ReleaseBuffer(self.im2col_handle)
        self.im2col_handle = None

class MaxPool2D(Function):
    def __del__(self):
        if hasattr(self, 'indices_handle') and self.indices_handle:
            lib.ReleaseBuffer(self.indices_handle)
            self.indices_handle = None

    def forward(self, x, pool_size=2, stride=2):
        self.inputs = (x,)
        batch, inC, inH, inW = x.shape
        outH, outW = (inH - pool_size)//stride + 1, (inW - pool_size)//stride + 1
        out_handle = lib.CreateBuffer(None, batch * inC * outH * outW)
        self.indices_handle = lib.CreateBuffer(None, batch * inC * outH * outW)
        _dispatch(b"maxpool_forward", (x.gpu_buf,), (out_handle, self.indices_handle),
                  (outW + 15)//16, (outH + 15)//16, batch * inC,
                  u1=batch, u2=inC, u3=inH, u4=inW, u5=pool_size, u6=stride, u7=outH, u8=outW, cbSize=32)
        res = Tensor.from_gpu(out_handle, (batch, inC, outH, outW), requires_grad=x.requires_grad)
        res._ctx = self
        return res

    def backward(self, grad_output):
        x = self.inputs[0]
        batch, inC, outH, outW = grad_output.shape
        dInput = lib.CreateBuffer(None, x.size)
        lib.ClearBuffer(dInput)
        _dispatch(b"maxpool_backward", (grad_output.gpu_buf, self.indices_handle), (dInput,),
                  (outW + 15)//16, (outH + 15)//16, batch * inC,
                  u1=batch, u2=inC, u3=outH, u4=outW)
        lib.ReleaseBuffer(self.indices_handle)
        self.indices_handle = None
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

def matmul(A, B): return MatMul().forward(A, B)
def add_bias(A, B): return AddBias().forward(A, B)
def bias_relu(A, B): return BiasReLUFunc().forward(A, B)
def relu(x): return ReLUFunc().forward(x)
def softmax_ce(x, labels): return SoftmaxCEFunc().forward(x, labels)
def conv2d(x, f, b, stride=1, padding=0, actType=0): return Conv2D().forward(x, f, b, stride, padding, actType)
def maxpool2d(x, pool_size=2, stride=2): return MaxPool2D().forward(x, pool_size, stride)
def flatten(x): return Flatten().forward(x)

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
    """End-of-batch cleanup: release intermediate tensors and flush GPU.
    Call this after optimizer.step() in every training batch."""
    release_all_buffers()
    lib.FlushGPU()
