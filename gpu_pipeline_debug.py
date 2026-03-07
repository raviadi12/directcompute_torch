"""
gpu_pipeline_debug.py — Deep GPU pipeline debugger for DirectCompute NN engine.

Intercepts EVERY GPU call (shader dispatch, buffer create/release/clear/read/update,
flush) and records a full execution trace with:
  - Exact chronological order of all GPU operations
  - Per-operation memory allocation/deallocation (bytes, pool hit/miss)
  - Live VRAM watermark (current, peak, per-phase)
  - Data travel tracking (CPU→GPU uploads, GPU→CPU readbacks, GPU→GPU copies)
  - Backward pass dissection (which backward Function produced which shader calls)
  - Thread group geometry for every dispatch
  - Buffer lifetime analysis (alloc→last-use→free, leak detection)
  - Memory bandwidth estimation (bytes read/written per shader)
  - Phase breakdown (forward / backward / optimizer / metrics / cleanup)

Usage:
    python gpu_pipeline_debug.py          # runs LeNet 1 batch, full trace
    python gpu_pipeline_debug.py --model alexnet --batches 2
    python gpu_pipeline_debug.py --summary   # compact summary only
"""

import ctypes
import numpy as np
import os
import sys
import time
import argparse
from collections import defaultdict, OrderedDict

# Force UTF-8 output on Windows (cp1252 can't handle Unicode arrows)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── Import engine ──
from nn_engine import (
    lib, Tensor, Linear, ConvLayer, SGD, Metrics,
    relu, softmax_ce, maxpool2d, flatten, end_batch,
    release_all_buffers, bias_relu, _all_tensors,
    _IS_IGPU, _GPU_NAME, _GPU_VENDOR, _VRAM_MB, _SHARED_MB,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE TRACER — intercepts all engine.dll calls
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineTracer:
    """
    Low-level interceptor that monkey-patches every engine.dll function to record
    a full execution trace. Each event is a dict with timestamp, operation type,
    and operation-specific metadata.
    """

    def __init__(self):
        # ── Save originals ──
        self._orig = {
            "RunShader":        lib.RunShader,
            "RunShaderRaw":     lib.RunShaderRaw,
            "CreateBuffer":     lib.CreateBuffer,
            "ReleaseBuffer":    lib.ReleaseBuffer,
            "ReleaseBufferBatch": lib.ReleaseBufferBatch,
            "ReadBuffer":       lib.ReadBuffer,
            "UpdateBuffer":     lib.UpdateBuffer,
            "ClearBuffer":      lib.ClearBuffer,
            "FlushGPU":         lib.FlushGPU,
            "SGDBatch":         lib.SGDBatch,
            "AddRefBuffer":     lib.AddRefBuffer,
        }

        # ── Trace state ──
        self.events = []          # list of event dicts
        self.buf_meta = {}        # handle -> {size, alloc_time, alloc_idx, pool_hit, phase}
        self.buf_last_use = {}    # handle -> event_idx (last shader that touched it)
        self.live_bytes = 0
        self.peak_bytes = 0
        self.total_alloc_bytes = 0
        self.total_free_bytes = 0
        self.pool_hits = 0
        self.pool_misses = 0
        self.phase = "init"       # current phase: forward/backward/optimizer/metrics/cleanup
        self._t0 = time.perf_counter()
        self._phase_bytes = defaultdict(int)  # phase -> peak bytes
        self._phase_alloc = defaultdict(int)  # phase -> total allocated
        self._phase_free = defaultdict(int)   # phase -> total freed
        self._data_travel = {
            "cpu_to_gpu": 0,      # bytes uploaded (CreateBuffer w/ data, UpdateBuffer)
            "gpu_to_cpu": 0,      # bytes read back (ReadBuffer)
            "gpu_to_gpu": 0,      # bytes moved between GPU buffers (shader read+write)
        }
        self._shader_bw = defaultdict(lambda: {"read_bytes": 0, "write_bytes": 0, "calls": 0})
        self._installed = False
        self._freed_handles = set()  # track individually freed handles to avoid double-counting

        # Known buffer sizes for bandwidth estimation
        self._buf_sizes = {}  # handle -> float count

    def _ts(self):
        return (time.perf_counter() - self._t0) * 1000  # ms

    def _event(self, **kwargs):
        kwargs["idx"] = len(self.events)
        kwargs["ts_ms"] = self._ts()
        kwargs["phase"] = self.phase
        kwargs["live_bytes"] = self.live_bytes
        self.events.append(kwargs)
        return kwargs["idx"]

    # ── Intercept: CreateBuffer ──
    def _trace_CreateBuffer(self, data, count):
        count_int = int(count) if count else 0
        size_bytes = count_int * 4
        t0 = time.perf_counter()
        handle = self._orig["CreateBuffer"](data, count)
        dt = (time.perf_counter() - t0) * 1000

        # Heuristic: if the buffer was returned very fast and matches a pooled size, 
        # it's likely a pool hit. We can't know for sure without engine-side tracking,
        # but we can track by checking if we've seen this handle before.
        is_pool_hit = handle in self.buf_meta  # reused handle = pool hit
        if is_pool_hit:
            self.pool_hits += 1
            self._freed_handles.discard(handle)  # re-entering live tracking
        else:
            self.pool_misses += 1

        self.live_bytes += size_bytes
        self.total_alloc_bytes += size_bytes
        self._phase_alloc[self.phase] += size_bytes
        if self.live_bytes > self.peak_bytes:
            self.peak_bytes = self.live_bytes
        if self.live_bytes > self._phase_bytes[self.phase]:
            self._phase_bytes[self.phase] = self.live_bytes

        has_data = bool(data)
        if has_data:
            self._data_travel["cpu_to_gpu"] += size_bytes

        self.buf_meta[handle] = {
            "size": size_bytes,
            "count": count_int,
            "alloc_time": self._ts(),
            "alloc_idx": len(self.events),
            "pool_hit": is_pool_hit,
            "phase": self.phase,
        }
        self._buf_sizes[handle] = count_int

        self._event(
            op="CreateBuffer", handle=id(handle) if handle else 0,
            handle_raw=handle,
            count=count_int, size_bytes=size_bytes,
            has_data=has_data, pool_hit=is_pool_hit,
            time_ms=dt,
        )
        return handle

    # ── Intercept: ReleaseBuffer ──
    def _trace_ReleaseBuffer(self, handle):
        size_bytes = 0
        if handle and handle in self.buf_meta and handle not in self._freed_handles:
            size_bytes = self.buf_meta[handle]["size"]
            self.live_bytes -= size_bytes
            self.total_free_bytes += size_bytes
            self._phase_free[self.phase] += size_bytes
            self._freed_handles.add(handle)  # mark as freed

        self._event(
            op="ReleaseBuffer", handle=id(handle) if handle else 0,
            size_bytes=size_bytes,
        )
        self._orig["ReleaseBuffer"](handle)

    # ── Intercept: ReleaseBufferBatch ──
    def _trace_ReleaseBufferBatch(self, handles, count):
        total_bytes = 0
        count_int = int(count)
        for i in range(count_int):
            h = handles[i]
            if h and h in self.buf_meta and h not in self._freed_handles:
                sz = self.buf_meta[h]["size"]
                self.live_bytes -= sz
                total_bytes += sz
                self.total_free_bytes += sz
                self._phase_free[self.phase] += sz
                self._freed_handles.add(h)

        self._event(
            op="ReleaseBufferBatch", count=count_int,
            total_bytes=total_bytes,
        )
        self._orig["ReleaseBufferBatch"](handles, count)

    # ── Intercept: ReadBuffer ──
    def _trace_ReadBuffer(self, handle, dst):
        size_bytes = 0
        if handle and handle in self.buf_meta:
            size_bytes = self.buf_meta[handle]["size"]
        self._data_travel["gpu_to_cpu"] += size_bytes

        t0 = time.perf_counter()
        self._orig["ReadBuffer"](handle, dst)
        dt = (time.perf_counter() - t0) * 1000

        bw_gbps = (size_bytes / (dt / 1000 + 1e-9)) / 1e9 if dt > 0 else 0

        self._event(
            op="ReadBuffer", handle=id(handle) if handle else 0,
            size_bytes=size_bytes, time_ms=dt,
            bandwidth_gbps=bw_gbps,
        )

    # ── Intercept: UpdateBuffer ──
    def _trace_UpdateBuffer(self, handle, data):
        size_bytes = 0
        if handle and handle in self.buf_meta:
            size_bytes = self.buf_meta[handle]["size"]
        self._data_travel["cpu_to_gpu"] += size_bytes

        t0 = time.perf_counter()
        self._orig["UpdateBuffer"](handle, data)
        dt = (time.perf_counter() - t0) * 1000

        self._event(
            op="UpdateBuffer", handle=id(handle) if handle else 0,
            size_bytes=size_bytes, time_ms=dt,
        )

    # ── Intercept: ClearBuffer ──
    def _trace_ClearBuffer(self, handle):
        size_bytes = 0
        if handle and handle in self.buf_meta:
            size_bytes = self.buf_meta[handle]["size"]

        self._orig["ClearBuffer"](handle)
        self._event(
            op="ClearBuffer", handle=id(handle) if handle else 0,
            size_bytes=size_bytes,
        )

    # ── Intercept: FlushGPU ──
    def _trace_FlushGPU(self):
        t0 = time.perf_counter()
        self._orig["FlushGPU"]()
        dt = (time.perf_counter() - t0) * 1000
        self._event(op="FlushGPU", time_ms=dt)

    # ── Intercept: AddRefBuffer ──
    def _trace_AddRefBuffer(self, handle):
        self._orig["AddRefBuffer"](handle)
        self._event(
            op="AddRefBuffer", handle=id(handle) if handle else 0,
            size_bytes=self.buf_meta.get(handle, {}).get("size", 0),
        )

    # ── Intercept: RunShader ──
    def _trace_RunShader(self, name, srvs, srvCount, uavs, uavCount, threads, cb, cbSize):
        sname = name.decode() if isinstance(name, bytes) else name
        sc = int(srvCount)
        uc = int(uavCount)
        tx, ty, tz = int(threads[0]), int(threads[1]), int(threads[2])
        total_groups = tx * ty * tz

        # Estimate bandwidth: sum sizes of SRV (read) and UAV (read+write) buffers
        read_bytes = 0
        write_bytes = 0
        srv_handles = []
        uav_handles = []
        for i in range(sc):
            h = srvs[i]
            if h and h in self._buf_sizes:
                read_bytes += self._buf_sizes[h] * 4
                srv_handles.append(h)
                self.buf_last_use[h] = len(self.events)
        for i in range(uc):
            h = uavs[i]
            if h and h in self._buf_sizes:
                write_bytes += self._buf_sizes[h] * 4
                read_bytes += self._buf_sizes[h] * 4  # UAV is read+write
                uav_handles.append(h)
                self.buf_last_use[h] = len(self.events)

        self._data_travel["gpu_to_gpu"] += read_bytes + write_bytes
        self._shader_bw[sname]["read_bytes"] += read_bytes
        self._shader_bw[sname]["write_bytes"] += write_bytes
        self._shader_bw[sname]["calls"] += 1

        # Extract CB params if present
        cb_params = {}
        if cb and int(cbSize) > 0:
            from nn_engine import ParamsCB
            try:
                cb_struct = ctypes.cast(cb, ctypes.POINTER(ParamsCB)).contents
                cb_params = {f"u{i+1}": getattr(cb_struct, f"u{i+1}") for i in range(9)
                             if getattr(cb_struct, f"u{i+1}") != 0}
            except:
                pass

        t0 = time.perf_counter()
        self._orig["RunShader"](name, srvs, srvCount, uavs, uavCount, threads, cb, cbSize)
        dt = (time.perf_counter() - t0) * 1000

        # Estimate bandwidth (GB/s)
        total_bw_bytes = read_bytes + write_bytes
        bw_gbps = (total_bw_bytes / (dt / 1000 + 1e-9)) / 1e9 if dt > 0.001 else 0

        self._event(
            op="Dispatch", shader=sname,
            threads=(tx, ty, tz), total_groups=total_groups,
            srv_count=sc, uav_count=uc,
            read_bytes=read_bytes, write_bytes=write_bytes,
            cb_params=cb_params, time_ms=dt,
            est_bandwidth_gbps=bw_gbps,
        )

    # ── Intercept: RunShaderRaw (same tracing logic) ──
    def _trace_RunShaderRaw(self, name, srvs, srvCount, uavs, uavCount, threads, cb, cbSize):
        # Reuse same tracing, just call original RunShaderRaw
        sname = name.decode() if isinstance(name, bytes) else name
        sc = int(srvCount)
        uc = int(uavCount)
        tx, ty, tz = int(threads[0]), int(threads[1]), int(threads[2])
        total_groups = tx * ty * tz

        read_bytes = 0
        write_bytes = 0
        for i in range(sc):
            h = srvs[i]
            if h and h in self._buf_sizes:
                read_bytes += self._buf_sizes[h] * 4
                self.buf_last_use[h] = len(self.events)
        for i in range(uc):
            h = uavs[i]
            if h and h in self._buf_sizes:
                write_bytes += self._buf_sizes[h] * 4
                read_bytes += self._buf_sizes[h] * 4
                self.buf_last_use[h] = len(self.events)

        self._data_travel["gpu_to_gpu"] += read_bytes + write_bytes
        self._shader_bw[sname]["read_bytes"] += read_bytes
        self._shader_bw[sname]["write_bytes"] += write_bytes
        self._shader_bw[sname]["calls"] += 1

        t0 = time.perf_counter()
        self._orig["RunShaderRaw"](name, srvs, srvCount, uavs, uavCount, threads, cb, cbSize)
        dt = (time.perf_counter() - t0) * 1000

        self._event(
            op="Dispatch", shader=sname,
            threads=(tx, ty, tz), total_groups=total_groups,
            srv_count=sc, uav_count=uc,
            read_bytes=read_bytes, write_bytes=write_bytes,
            time_ms=dt,
        )

    # ── Intercept: SGDBatch ──
    def _trace_SGDBatch(self, params, grads, sizes, numParams, lr, clip):
        n = int(numParams)
        total_param_bytes = 0
        for i in range(n):
            total_param_bytes += int(sizes[i]) * 4

        t0 = time.perf_counter()
        self._orig["SGDBatch"](params, grads, sizes, numParams, lr, clip)
        dt = (time.perf_counter() - t0) * 1000

        self._event(
            op="SGDBatch", num_params=n,
            total_param_bytes=total_param_bytes,
            lr=float(lr), clip=float(clip), time_ms=dt,
        )

    # ── Install / Uninstall ──
    def install(self):
        """Monkey-patch all engine.dll functions with tracing wrappers."""
        if self._installed:
            return
        lib.RunShader = self._trace_RunShader
        lib.RunShaderRaw = self._trace_RunShaderRaw
        lib.CreateBuffer = self._trace_CreateBuffer
        lib.ReleaseBuffer = self._trace_ReleaseBuffer
        lib.ReleaseBufferBatch = self._trace_ReleaseBufferBatch
        lib.ReadBuffer = self._trace_ReadBuffer
        lib.UpdateBuffer = self._trace_UpdateBuffer
        lib.ClearBuffer = self._trace_ClearBuffer
        lib.FlushGPU = self._trace_FlushGPU
        lib.SGDBatch = self._trace_SGDBatch
        lib.AddRefBuffer = self._trace_AddRefBuffer
        self._installed = True

    def uninstall(self):
        """Restore original engine.dll functions."""
        if not self._installed:
            return
        for name, fn in self._orig.items():
            setattr(lib, name, fn)
        self._installed = False

    def set_phase(self, phase):
        """Set current execution phase for grouping."""
        self.phase = phase
        self._event(op="PhaseChange", new_phase=phase)

    def reset(self):
        """Clear all trace data."""
        self.events.clear()
        self.buf_meta.clear()
        self.buf_last_use.clear()
        self._buf_sizes.clear()
        self.live_bytes = 0
        self.peak_bytes = 0
        self.total_alloc_bytes = 0
        self.total_free_bytes = 0
        self.pool_hits = 0
        self.pool_misses = 0
        self._phase_bytes.clear()
        self._phase_alloc.clear()
        self._phase_free.clear()
        self._data_travel = {"cpu_to_gpu": 0, "gpu_to_cpu": 0, "gpu_to_gpu": 0}
        self._shader_bw.clear()
        self._t0 = time.perf_counter()
        self.phase = "init"


# ═══════════════════════════════════════════════════════════════════════════════
#  REPORT GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_bytes(b):
    """Human-readable byte size."""
    if b < 1024: return f"{b} B"
    if b < 1024**2: return f"{b/1024:.1f} KB"
    if b < 1024**3: return f"{b/1024**2:.2f} MB"
    return f"{b/1024**3:.2f} GB"

def _fmt_ms(ms):
    if ms < 0.001: return f"{ms*1000:.1f} us"
    if ms < 1: return f"{ms:.3f} ms"
    return f"{ms:.1f} ms"


def print_full_trace(tracer, max_events=None):
    """Print every single GPU operation in chronological order."""
    events = tracer.events
    if max_events:
        events = events[:max_events]

    print(f"\n{'='*120}")
    print(f"  FULL GPU PIPELINE TRACE — {len(tracer.events)} operations")
    print(f"{'='*120}")
    print(f"  {'#':>4} {'Time':>8} {'Phase':<10} {'Operation':<25} {'Details':<60} {'Mem':>10}")
    print(f"  {'-'*4} {'-'*8} {'-'*10} {'-'*25} {'-'*60} {'-'*10}")

    for e in events:
        idx = e["idx"]
        ts = f"{e['ts_ms']:.2f}"
        phase = e["phase"][:10]
        mem = _fmt_bytes(e["live_bytes"])
        op = e["op"]

        if op == "Dispatch":
            shader = e["shader"]
            thr = e["threads"]
            groups = e["total_groups"]
            dt = e.get("time_ms", 0)
            r = _fmt_bytes(e.get("read_bytes", 0))
            w = _fmt_bytes(e.get("write_bytes", 0))
            cb = e.get("cb_params", {})
            cb_str = " ".join(f"{k}={v}" for k, v in cb.items()) if cb else ""
            detail = f"[{thr[0]}x{thr[1]}x{thr[2]}={groups}grp] R:{r} W:{w} {_fmt_ms(dt)}"
            if cb_str:
                detail = detail[:48] + " " + cb_str[:11]
            print(f"  {idx:>4} {ts:>8} {phase:<10} {'  >> ' + shader:<25} {detail:<60} {mem:>10}")

        elif op == "CreateBuffer":
            sz = _fmt_bytes(e["size_bytes"])
            src = "CPU→GPU" if e["has_data"] else "empty"
            pool = "pool-hit" if e["pool_hit"] else "new-alloc"
            dt = e.get("time_ms", 0)
            detail = f"{e['count']} floats ({sz}) [{src}] [{pool}] {_fmt_ms(dt)}"
            print(f"  {idx:>4} {ts:>8} {phase:<10} {'  + CreateBuffer':<25} {detail:<60} {mem:>10}")

        elif op == "ReleaseBuffer":
            sz = _fmt_bytes(e["size_bytes"])
            detail = f"freed {sz}"
            print(f"  {idx:>4} {ts:>8} {phase:<10} {'  - ReleaseBuffer':<25} {detail:<60} {mem:>10}")

        elif op == "ReleaseBufferBatch":
            sz = _fmt_bytes(e["total_bytes"])
            detail = f"{e['count']} buffers, total {sz}"
            print(f"  {idx:>4} {ts:>8} {phase:<10} {'  - ReleaseBatch':<25} {detail:<60} {mem:>10}")

        elif op == "ReadBuffer":
            sz = _fmt_bytes(e["size_bytes"])
            dt = e.get("time_ms", 0)
            bw = e.get("bandwidth_gbps", 0)
            detail = f"GPU→CPU {sz} {_fmt_ms(dt)}" + (f" ({bw:.1f} GB/s)" if bw > 0 else "")
            print(f"  {idx:>4} {ts:>8} {phase:<10} {'  < ReadBuffer':<25} {detail:<60} {mem:>10}")

        elif op == "UpdateBuffer":
            sz = _fmt_bytes(e["size_bytes"])
            dt = e.get("time_ms", 0)
            detail = f"CPU→GPU {sz} {_fmt_ms(dt)}"
            print(f"  {idx:>4} {ts:>8} {phase:<10} {'  > UpdateBuffer':<25} {detail:<60} {mem:>10}")

        elif op == "ClearBuffer":
            sz = _fmt_bytes(e["size_bytes"])
            detail = f"zero-fill {sz}"
            print(f"  {idx:>4} {ts:>8} {phase:<10} {'  0 ClearBuffer':<25} {detail:<60} {mem:>10}")

        elif op == "FlushGPU":
            dt = e.get("time_ms", 0)
            detail = f"submit command queue {_fmt_ms(dt)}"
            print(f"  {idx:>4} {ts:>8} {phase:<10} {'  ~ FlushGPU':<25} {detail:<60} {mem:>10}")

        elif op == "SGDBatch":
            n = e["num_params"]
            sz = _fmt_bytes(e["total_param_bytes"])
            dt = e.get("time_ms", 0)
            detail = f"{n} params ({sz}) lr={e['lr']:.4f} clip={e['clip']:.1f} {_fmt_ms(dt)}"
            print(f"  {idx:>4} {ts:>8} {phase:<10} {'  SGDBatch':<25} {detail:<60} {mem:>10}")

        elif op == "AddRefBuffer":
            sz = _fmt_bytes(e["size_bytes"])
            detail = f"refcount++ ({sz})"
            print(f"  {idx:>4} {ts:>8} {phase:<10} {'  ^ AddRefBuffer':<25} {detail:<60} {mem:>10}")

        elif op == "PhaseChange":
            detail = f"─── {e['new_phase'].upper()} ───"
            print(f"  {idx:>4} {ts:>8} {'':10} {'':25} {detail:<60} {mem:>10}")

    print(f"  {'-'*4} {'-'*8} {'-'*10} {'-'*25} {'-'*60} {'-'*10}")
    if max_events and len(tracer.events) > max_events:
        print(f"  ... ({len(tracer.events) - max_events} more events omitted)")
    print()


def print_memory_report(tracer):
    """Memory allocation analysis."""
    print(f"\n{'='*90}")
    print(f"  MEMORY ANALYSIS")
    print(f"{'='*90}")

    print(f"\n  Peak VRAM usage:     {_fmt_bytes(tracer.peak_bytes)}")
    print(f"  Final live VRAM:     {_fmt_bytes(tracer.live_bytes)}")
    print(f"  Total allocated:     {_fmt_bytes(tracer.total_alloc_bytes)}")
    print(f"  Total freed:         {_fmt_bytes(tracer.total_free_bytes)}")
    unfreed = tracer.total_alloc_bytes - tracer.total_free_bytes
    print(f"  Unfreed (leak?):     {_fmt_bytes(abs(unfreed))} {'(OK — params/metrics)' if unfreed > 0 else ''}")
    print(f"  Buffer pool hits:    {tracer.pool_hits}")
    print(f"  Buffer pool misses:  {tracer.pool_misses}")
    hit_rate = tracer.pool_hits / max(tracer.pool_hits + tracer.pool_misses, 1) * 100
    print(f"  Pool hit rate:       {hit_rate:.1f}%")

    # Per-phase memory
    print(f"\n  {'Phase':<15s} {'Peak VRAM':>12s} {'Allocated':>12s} {'Freed':>12s} {'Net':>12s}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for phase in ["forward", "backward", "optimizer", "metrics", "cleanup"]:
        peak = tracer._phase_bytes.get(phase, 0)
        alloc = tracer._phase_alloc.get(phase, 0)
        freed = tracer._phase_free.get(phase, 0)
        net = alloc - freed
        if alloc > 0 or freed > 0:
            print(f"  {phase:<15s} {_fmt_bytes(peak):>12s} {_fmt_bytes(alloc):>12s} {_fmt_bytes(freed):>12s} {'+' if net > 0 else ''}{_fmt_bytes(abs(net)):>11s}")

    # Buffer size histogram
    sizes = defaultdict(int)
    for h, meta in tracer.buf_meta.items():
        bucket = meta["size"]
        sizes[bucket] += 1

    if sizes:
        print(f"\n  Buffer Size Distribution:")
        print(f"  {'Size':>12s} {'Count':>8s} {'Total':>12s}")
        print(f"  {'-'*12} {'-'*8} {'-'*12}")
        for sz in sorted(sizes.keys()):
            cnt = sizes[sz]
            print(f"  {_fmt_bytes(sz):>12s} {cnt:>8d} {_fmt_bytes(sz*cnt):>12s}")

    print(f"{'='*90}\n")


def print_data_travel(tracer):
    """Data movement analysis: CPU↔GPU, GPU↔GPU."""
    travel = tracer._data_travel
    total = sum(travel.values())

    print(f"\n{'='*90}")
    print(f"  DATA TRAVEL ANALYSIS")
    print(f"{'='*90}")
    print(f"  CPU → GPU uploads:   {_fmt_bytes(travel['cpu_to_gpu']):>12s}  "
          f"({travel['cpu_to_gpu']/max(total,1)*100:.1f}%)")
    print(f"  GPU → CPU readbacks: {_fmt_bytes(travel['gpu_to_cpu']):>12s}  "
          f"({travel['gpu_to_cpu']/max(total,1)*100:.1f}%)")
    print(f"  GPU → GPU (shaders): {_fmt_bytes(travel['gpu_to_gpu']):>12s}  "
          f"({travel['gpu_to_gpu']/max(total,1)*100:.1f}%)")
    print(f"  {'─'*40}")
    print(f"  Total data moved:    {_fmt_bytes(total):>12s}")

    # ReadBuffer breakdown
    reads = [e for e in tracer.events if e["op"] == "ReadBuffer"]
    if reads:
        print(f"\n  GPU → CPU Readbacks ({len(reads)} calls):")
        print(f"  {'#':>4} {'Time':>8} {'Phase':<10} {'Size':>10} {'Latency':>10}")
        print(f"  {'-'*4} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
        for e in reads:
            print(f"  {e['idx']:>4} {e['ts_ms']:>8.2f} {e['phase']:<10} "
                  f"{_fmt_bytes(e['size_bytes']):>10} {_fmt_ms(e.get('time_ms',0)):>10}")

    print(f"{'='*90}\n")


def print_shader_report(tracer):
    """Per-shader analysis: call count, total time, bandwidth."""
    print(f"\n{'='*100}")
    print(f"  SHADER EXECUTION ANALYSIS")
    print(f"{'='*100}")

    # Aggregate shader stats
    shader_stats = defaultdict(lambda: {"calls": 0, "total_ms": 0, "groups": [], "read_bytes": 0, "write_bytes": 0})
    for e in tracer.events:
        if e["op"] == "Dispatch":
            s = e["shader"]
            shader_stats[s]["calls"] += 1
            shader_stats[s]["total_ms"] += e.get("time_ms", 0)
            shader_stats[s]["groups"].append(e["total_groups"])
            shader_stats[s]["read_bytes"] += e.get("read_bytes", 0)
            shader_stats[s]["write_bytes"] += e.get("write_bytes", 0)

    total_ms = sum(s["total_ms"] for s in shader_stats.values())
    total_calls = sum(s["calls"] for s in shader_stats.values())

    print(f"\n  {'Shader':<28s} {'Calls':>5} {'Total ms':>9} {'Avg ms':>8} {'%Time':>6} "
          f"{'Avg Grps':>9} {'Read':>10} {'Write':>10}")
    print(f"  {'-'*28} {'-'*5} {'-'*9} {'-'*8} {'-'*6} {'-'*9} {'-'*10} {'-'*10}")

    for name in sorted(shader_stats, key=lambda n: shader_stats[n]["total_ms"], reverse=True):
        s = shader_stats[name]
        avg_ms = s["total_ms"] / s["calls"]
        pct = (s["total_ms"] / total_ms * 100) if total_ms else 0
        avg_groups = np.mean(s["groups"]) if s["groups"] else 0
        print(f"  {name:<28s} {s['calls']:>5} {s['total_ms']:>9.3f} {avg_ms:>8.3f} {pct:>5.1f}% "
              f"{avg_groups:>9.0f} {_fmt_bytes(s['read_bytes']):>10} {_fmt_bytes(s['write_bytes']):>10}")

    print(f"  {'-'*28} {'-'*5} {'-'*9}")
    print(f"  {'TOTAL':<28s} {total_calls:>5} {total_ms:>9.3f}")

    # SGDBatch
    sgd_events = [e for e in tracer.events if e["op"] == "SGDBatch"]
    if sgd_events:
        print(f"\n  SGDBatch: {len(sgd_events)} calls, "
              f"total {sum(e.get('time_ms',0) for e in sgd_events):.3f} ms, "
              f"{sgd_events[0]['num_params']} params each")

    print(f"{'='*100}\n")


def print_backward_trace(tracer):
    """Detailed backward pass dissection — show every op in the backward phase."""
    bwd_events = [e for e in tracer.events if e["phase"] == "backward"]
    if not bwd_events:
        print("\n  (no backward events recorded)")
        return

    print(f"\n{'='*110}")
    print(f"  BACKWARD PASS DISSECTION — {len(bwd_events)} operations")
    print(f"{'='*110}")

    # Group consecutive dispatches
    dispatch_count = 0
    alloc_bytes = 0
    free_bytes = 0

    print(f"  {'#':>4} {'dt ms':>7} {'Operation':<30} {'Details':<55} {'Live Mem':>10}")
    print(f"  {'-'*4} {'-'*7} {'-'*30} {'-'*55} {'-'*10}")

    t_start = bwd_events[0]["ts_ms"] if bwd_events else 0

    for e in bwd_events:
        rel_t = e["ts_ms"] - t_start
        mem = _fmt_bytes(e["live_bytes"])
        op = e["op"]

        if op == "Dispatch":
            dispatch_count += 1
            shader = e["shader"]
            thr = e["threads"]
            r = _fmt_bytes(e.get("read_bytes", 0))
            w = _fmt_bytes(e.get("write_bytes", 0))
            dt = e.get("time_ms", 0)
            cb = e.get("cb_params", {})
            # Annotate shader purpose
            purpose = _annotate_backward_shader(shader, cb)
            detail = f"[{thr[0]}x{thr[1]}x{thr[2]}] R:{r} W:{w} {_fmt_ms(dt)}"
            print(f"  {e['idx']:>4} {rel_t:>7.2f} {'  >> ' + shader:<30} {detail:<55} {mem:>10}")
            if purpose:
                print(f"  {'':>4} {'':>7} {'':30} {'     ' + purpose:<55}")

        elif op == "CreateBuffer":
            alloc_bytes += e["size_bytes"]
            sz = _fmt_bytes(e["size_bytes"])
            src = "CPU→GPU" if e["has_data"] else "empty"
            pool = "pool" if e["pool_hit"] else "new"
            detail = f"{e['count']} floats ({sz}) [{src}] [{pool}]"
            print(f"  {e['idx']:>4} {rel_t:>7.2f} {'  + Alloc':<30} {detail:<55} {mem:>10}")

        elif op == "ReleaseBuffer":
            free_bytes += e["size_bytes"]
            sz = _fmt_bytes(e["size_bytes"])
            print(f"  {e['idx']:>4} {rel_t:>7.2f} {'  - Free':<30} {sz:<55} {mem:>10}")

        elif op == "ClearBuffer":
            detail = f"zero-fill {_fmt_bytes(e['size_bytes'])}"
            print(f"  {e['idx']:>4} {rel_t:>7.2f} {'  0 Clear':<30} {detail:<55} {mem:>10}")

        elif op == "AddRefBuffer":
            detail = f"refcount++ ({_fmt_bytes(e['size_bytes'])})"
            print(f"  {e['idx']:>4} {rel_t:>7.2f} {'  ^ AddRef':<30} {detail:<55} {mem:>10}")

    bwd_ms = bwd_events[-1]["ts_ms"] - t_start if bwd_events else 0
    print(f"\n  Backward summary: {dispatch_count} dispatches, {_fmt_ms(bwd_ms)} wall time")
    print(f"  Allocated: {_fmt_bytes(alloc_bytes)}, Freed: {_fmt_bytes(free_bytes)}, "
          f"Net: {'+' if alloc_bytes > free_bytes else ''}{_fmt_bytes(abs(alloc_bytes - free_bytes))}")
    print(f"{'='*110}\n")


def _annotate_backward_shader(shader, cb):
    """Guess semantic meaning of a backward shader dispatch."""
    annotations = {
        "softmax_ce_grad":       "dL/d(logits) = softmax - one_hot",
        "bias_grad":             "dL/d(bias) = sum(grad, axis=0)",
        "relu_grad":             "dL/d(x) = grad * (x > 0)",
        "grad_accum":            "grad += incoming_grad",
        "conv_grad_reshape":     "reshape grad for conv backward",
        "conv_grad_reshape_relu": "fused reshape + ReLU grad mask",
        "col2im":                "scatter columns back to image",
        "im2col":                "gather image patches to columns",
        "maxpool_backward":      "scatter grad to max positions",
    }
    if shader in annotations:
        return annotations[shader]

    if "matmul" in shader:
        u4 = cb.get("u4", 0) if cb else 0
        if u4 == 1:
            return "A^T @ B: gradient w.r.t. weights"
        elif u4 == 2:
            return "A @ B^T: gradient w.r.t. input"
        else:
            return "matmul forward (in backward context)"
    return ""


def print_memory_watermark(tracer):
    """Show memory watermark at key points throughout the pipeline."""
    print(f"\n{'='*80}")
    print(f"  MEMORY WATERMARK TIMELINE")
    print(f"{'='*80}")

    # Sample memory at every phase boundary and peak
    points = []
    for e in tracer.events:
        if e["op"] == "PhaseChange":
            points.append((e["ts_ms"], e["new_phase"].upper(), e["live_bytes"]))

    # Find peak within each phase
    phase_events = defaultdict(list)
    for e in tracer.events:
        phase_events[e["phase"]].append(e["live_bytes"])

    print(f"\n  {'Phase':<12s} {'Start':>10s} {'Peak':>10s} {'End':>10s} {'Buffers Alloc':>14s} {'Buffers Freed':>14s}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*14} {'-'*14}")

    for phase in ["forward", "backward", "optimizer", "metrics", "cleanup"]:
        evts = phase_events.get(phase, [])
        if not evts:
            continue
        start = _fmt_bytes(evts[0])
        peak = _fmt_bytes(max(evts))
        end = _fmt_bytes(evts[-1])
        allocs = sum(1 for e in tracer.events if e["phase"] == phase and e["op"] == "CreateBuffer")
        frees_single = sum(1 for e in tracer.events if e["phase"] == phase and e["op"] == "ReleaseBuffer")
        frees_batch = sum(e.get("count", 0) for e in tracer.events if e["phase"] == phase and e["op"] == "ReleaseBufferBatch")
        frees = frees_single + frees_batch
        print(f"  {phase:<12s} {start:>10s} {peak:>10s} {end:>10s} {allocs:>14d} {frees:>14d}")

    # ASCII watermark graph
    print(f"\n  Memory Timeline (scaled to peak = {_fmt_bytes(tracer.peak_bytes)}):")
    BAR_WIDTH = 60
    samples = min(len(tracer.events), 200)
    step = max(1, len(tracer.events) // samples)
    peak = max(tracer.peak_bytes, 1)

    last_phase = ""
    for i in range(0, len(tracer.events), step):
        e = tracer.events[i]
        level = int(e["live_bytes"] / peak * BAR_WIDTH)
        bar = "█" * level + "░" * (BAR_WIDTH - level)
        phase_mark = ""
        if e["phase"] != last_phase:
            phase_mark = f" ← {e['phase']}"
            last_phase = e["phase"]
        print(f"  {e['ts_ms']:>7.1f}ms |{bar}| {_fmt_bytes(e['live_bytes']):>8s}{phase_mark}")

    print(f"{'='*80}\n")


def print_leak_check(tracer):
    """Check for potential memory leaks — buffers allocated but never freed."""
    print(f"\n{'='*80}")
    print(f"  MEMORY LEAK DETECTION")
    print(f"{'='*80}")

    # Find buffers that were allocated but not explicitly released
    alloc_events = {e["handle_raw"]: e for e in tracer.events
                    if e["op"] == "CreateBuffer" and "handle_raw" in e}
    free_handles = set()
    for e in tracer.events:
        if e["op"] == "ReleaseBuffer" and "handle" in e:
            free_handles.add(e.get("handle"))
        if e["op"] == "ReleaseBufferBatch":
            # Can't track individual handles easily from batch release
            pass

    # Count expected leaked buffers (persistent params + metrics)
    leaked_bytes = tracer.live_bytes
    if leaked_bytes > 0:
        print(f"  Live buffers at trace end: {_fmt_bytes(leaked_bytes)}")
        print(f"  (Expected: parameter weights + optimizer state + metrics buffers)")
    else:
        print(f"  No leaked buffers detected — all memory freed properly")

    # Check for unexpected growth patterns
    phase_mem = []
    for e in tracer.events:
        if e["op"] == "PhaseChange" and e["new_phase"] == "cleanup":
            phase_mem.append(e["live_bytes"])

    if len(phase_mem) >= 2:
        growth = phase_mem[-1] - phase_mem[0]
        if growth > 1024:  # > 1KB growth between cleanups
            print(f"  [!] Memory grew {_fmt_bytes(growth)} between batch cleanups — possible leak!")
        else:
            print(f"  Batch-to-batch memory: stable (no growth)")

    print(f"{'='*80}\n")


def print_bottleneck_analysis(tracer):
    """Identify performance bottlenecks."""
    print(f"\n{'='*90}")
    print(f"  BOTTLENECK ANALYSIS")
    print(f"{'='*90}")

    # 1. Slowest individual operations
    timed = [(e, e.get("time_ms", 0)) for e in tracer.events if e.get("time_ms", 0) > 0]
    timed.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Top 10 Slowest Operations:")
    print(f"  {'#':>4} {'Time':>8} {'Phase':<10} {'Operation':<55}")
    print(f"  {'-'*4} {'-'*8} {'-'*10} {'-'*55}")
    for e, dt in timed[:10]:
        op = e["op"]
        if op == "Dispatch":
            desc = f"Dispatch {e['shader']} [{e['threads'][0]}x{e['threads'][1]}x{e['threads'][2]}]"
        elif op == "SGDBatch":
            desc = f"SGDBatch ({e['num_params']} params)"
        elif op == "ReadBuffer":
            desc = f"ReadBuffer ({_fmt_bytes(e['size_bytes'])})"
        else:
            desc = op
        print(f"  {e['idx']:>4} {_fmt_ms(dt):>8} {e['phase']:<10} {desc:<55}")

    # 2. Time breakdown by phase
    phase_time = defaultdict(float)
    for e in tracer.events:
        if "time_ms" in e:
            phase_time[e["phase"]] += e["time_ms"]
    total_time = sum(phase_time.values())

    print(f"\n  Time Breakdown by Phase:")
    print(f"  {'Phase':<15s} {'Time':>10s} {'%':>6s}")
    print(f"  {'-'*15} {'-'*10} {'-'*6}")
    for phase in sorted(phase_time, key=phase_time.get, reverse=True):
        ms = phase_time[phase]
        pct = ms / total_time * 100 if total_time else 0
        print(f"  {phase:<15s} {_fmt_ms(ms):>10s} {pct:>5.1f}%")

    # 3. Optimization opportunities
    print(f"\n  Optimization Opportunities:")

    # Check for excessive small buffer allocations
    small_allocs = [e for e in tracer.events if e["op"] == "CreateBuffer" and e["size_bytes"] < 256]
    if len(small_allocs) > 5:
        print(f"  [!] {len(small_allocs)} small buffer allocations (<256 bytes) — consider merging")

    # Check for GPU→CPU readbacks during training
    train_reads = [e for e in tracer.events if e["op"] == "ReadBuffer"
                   and e["phase"] in ("forward", "backward")]
    if train_reads:
        total_read_ms = sum(e.get("time_ms", 0) for e in train_reads)
        print(f"  [!] {len(train_reads)} GPU→CPU readbacks during forward/backward ({_fmt_ms(total_read_ms)})")
        print(f"       Consider keeping data on GPU until epoch end")

    # Pool efficiency
    if tracer.pool_hits + tracer.pool_misses > 0:
        rate = tracer.pool_hits / (tracer.pool_hits + tracer.pool_misses) * 100
        if rate < 50:
            print(f"  [!] Buffer pool hit rate only {rate:.0f}% — increase pool size?")
        else:
            print(f"  [OK] Buffer pool hit rate: {rate:.0f}%")

    # Backward/forward ratio
    fwd_ms = phase_time.get("forward", 0)
    bwd_ms = phase_time.get("backward", 0)
    if fwd_ms > 0:
        ratio = bwd_ms / fwd_ms
        print(f"  Backward/Forward time ratio: {ratio:.2f}x")
        if ratio > 3:
            print(f"       (ratio > 3x is unusual — check backward shader efficiency)")

    print(f"{'='*90}\n")


def print_summary(tracer):
    """Compact one-page summary."""
    total_events = len(tracer.events)
    dispatches = sum(1 for e in tracer.events if e["op"] == "Dispatch")
    allocs = sum(1 for e in tracer.events if e["op"] == "CreateBuffer")
    frees = sum(1 for e in tracer.events if e["op"] in ("ReleaseBuffer", "ReleaseBufferBatch"))
    timed_ops = [e.get("time_ms", 0) for e in tracer.events if e.get("time_ms", 0) > 0]
    total_ms = sum(timed_ops)
    travel = tracer._data_travel

    print(f"\n{'='*70}")
    print(f"  GPU PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  GPU:              {_GPU_NAME}")
    print(f"  Type:             {'Integrated (UMA)' if _IS_IGPU else 'Discrete'}")
    print(f"  VRAM:             {_VRAM_MB}MB dedicated, {_SHARED_MB}MB shared")
    print(f"  ")
    print(f"  Total GPU ops:    {total_events}")
    print(f"  Shader dispatches:{dispatches}")
    print(f"  Buffer allocs:    {allocs} ({_fmt_bytes(tracer.total_alloc_bytes)})")
    print(f"  Buffer frees:     {frees} ({_fmt_bytes(tracer.total_free_bytes)})")
    print(f"  Pool hit rate:    {tracer.pool_hits}/{tracer.pool_hits+tracer.pool_misses} "
          f"({tracer.pool_hits/max(tracer.pool_hits+tracer.pool_misses,1)*100:.0f}%)")
    print(f"  Peak VRAM:        {_fmt_bytes(tracer.peak_bytes)}")
    print(f"  ")
    print(f"  CPU → GPU:        {_fmt_bytes(travel['cpu_to_gpu'])}")
    print(f"  GPU → CPU:        {_fmt_bytes(travel['gpu_to_cpu'])}")
    print(f"  GPU ↔ GPU:        {_fmt_bytes(travel['gpu_to_gpu'])}")
    print(f"  Total data moved: {_fmt_bytes(sum(travel.values()))}")
    print(f"  ")
    print(f"  Wall time:        {_fmt_ms(total_ms)}")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL BUILDER HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_lenet():
    c1 = ConvLayer(1, 6, 5)
    c2 = ConvLayer(6, 16, 5)
    l1 = Linear(16*4*4, 120)
    l2 = Linear(120, 84)
    l3 = Linear(84, 10)
    params = [c1.filters, c1.bias, c2.filters, c2.bias,
              l1.w, l1.b, l2.w, l2.b, l3.w, l3.b]
    names = ["c1.filters", "c1.bias", "c2.filters", "c2.bias",
             "l1.w", "l1.b", "l2.w", "l2.b", "l3.w", "l3.b"]

    def forward(xb):
        x = c1(xb, relu=True)
        x = maxpool2d(x)
        x = c2(x, relu=True)
        x = maxpool2d(x)
        x = flatten(x)
        x = l1(x, relu=True)
        x = l2(x, relu=True)
        return l3(x)

    return forward, params, names


def build_alexnet():
    c1 = ConvLayer(3, 64, 11, stride=4, padding=2)
    c2 = ConvLayer(64, 192, 5, padding=2)
    c3 = ConvLayer(192, 384, 3, padding=1)
    c4 = ConvLayer(384, 256, 3, padding=1)
    c5 = ConvLayer(256, 256, 3, padding=1)
    l1 = Linear(256*6*6, 512)
    l2 = Linear(512, 512)
    l3 = Linear(512, 2)
    params = [c1.filters, c1.bias, c2.filters, c2.bias, c3.filters, c3.bias,
              c4.filters, c4.bias, c5.filters, c5.bias,
              l1.w, l1.b, l2.w, l2.b, l3.w, l3.b]
    names = ["c1.filters", "c1.bias", "c2.filters", "c2.bias",
             "c3.filters", "c3.bias", "c4.filters", "c4.bias",
             "c5.filters", "c5.bias",
             "l1.w", "l1.b", "l2.w", "l2.b", "l3.w", "l3.b"]

    def forward(xb):
        x = c1(xb, relu=True)
        x = maxpool2d(x, pool_size=3, stride=2)
        x = c2(x, relu=True)
        x = maxpool2d(x, pool_size=3, stride=2)
        x = c3(x, relu=True)
        x = c4(x, relu=True)
        x = c5(x, relu=True)
        x = maxpool2d(x, pool_size=3, stride=2)
        x = flatten(x)
        x = l1(x, relu=True)
        x = l2(x, relu=True)
        return l3(x)

    return forward, params, names


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — run pipeline debug
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Deep GPU pipeline debugger")
    parser.add_argument("--model", choices=["lenet", "alexnet"], default="lenet",
                        help="Model to debug (default: lenet)")
    parser.add_argument("--batches", type=int, default=1,
                        help="Number of training batches to trace (default: 1)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--summary", action="store_true",
                        help="Print compact summary only")
    parser.add_argument("--no-trace", action="store_true",
                        help="Skip full trace output (just reports)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Max events to show in full trace")
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"  GPU PIPELINE DEBUGGER — DirectCompute NN Engine")
    print(f"  Model: {args.model.upper()}, Batch size: {args.batch_size}, Batches: {args.batches}")
    print(f"{'#'*70}")

    # ── Build model ──
    if args.model == "lenet":
        forward, params, names = build_lenet()
        # Generate synthetic data (MNIST-like)
        x_data = np.random.randn(args.batch_size * args.batches, 1, 28, 28).astype(np.float32) * 0.3
        y_data = np.random.randint(0, 10, size=args.batch_size * args.batches).astype(np.float32)
        num_classes = 10
    else:
        forward, params, names = build_alexnet()
        # Generate synthetic data (ImageNet-like)
        x_data = np.random.randn(args.batch_size * args.batches, 3, 224, 224).astype(np.float32) * 0.1
        y_data = np.random.randint(0, 2, size=args.batch_size * args.batches).astype(np.float32)
        num_classes = 2

    optimizer = SGD(params, lr=0.01)

    # ── Install tracer ──
    tracer = PipelineTracer()
    tracer.install()

    print(f"\n  Tracing {args.batches} batch(es)...\n")

    for batch_idx in range(args.batches):
        start = batch_idx * args.batch_size
        end = start + args.batch_size

        # Zero grad
        tracer.set_phase("optimizer")
        optimizer.zero_grad()

        # Forward pass
        tracer.set_phase("forward")
        xb = Tensor(x_data[start:end], track=True)
        yb = Tensor(y_data[start:end], track=True)
        logits = forward(xb)
        loss = softmax_ce(logits, yb)

        # Backward pass
        tracer.set_phase("backward")
        loss.backward()

        # Optimizer step
        tracer.set_phase("optimizer")
        optimizer.step(clip=1.0)

        # Cleanup
        tracer.set_phase("cleanup")
        end_batch()

    # ── Uninstall and report ──
    tracer.uninstall()

    if args.summary:
        print_summary(tracer)
    else:
        # Full reports
        print_summary(tracer)

        if not args.no_trace:
            print_full_trace(tracer, max_events=args.max_events)

        print_memory_report(tracer)
        print_data_travel(tracer)
        print_shader_report(tracer)
        print_backward_trace(tracer)
        print_memory_watermark(tracer)
        print_leak_check(tracer)
        print_bottleneck_analysis(tracer)

    # ── Param stats ──
    print(f"\n{'='*70}")
    print(f"  MODEL PARAMETERS")
    print(f"{'='*70}")
    total_p = sum(p.size for p in params)
    print(f"  {'Name':<20s} {'Shape':<20s} {'Size':>10s} {'Bytes':>10s}")
    print(f"  {'-'*20} {'-'*20} {'-'*10} {'-'*10}")
    for p, n in zip(params, names):
        print(f"  {n:<20s} {str(p.shape):<20s} {p.size:>10,} {_fmt_bytes(p.size*4):>10s}")
    print(f"  {'-'*20}")
    print(f"  Total: {total_p:,} parameters ({_fmt_bytes(total_p * 4)})")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
