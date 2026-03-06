"""
nn_verify.py — Gradient, numerical, and diagnostic verification toolkit
for the DirectCompute neural network engine.

Usage:
    from nn_verify import (
        grad_check, grad_check_batched, grad_check_layer,
        verify_matmul, verify_conv2d, verify_relu, verify_softmax_ce,
        GradientMonitor, ActivationMonitor,
        check_health, param_stats, compare_pytorch,
    )

    # ── 1. Numerical gradient check (finite differences vs analytical) ──
    ok = grad_check(lambda: softmax_ce(matmul(x, w), y), w)

    # ── 2. Batched gradient check across all LeNet layers ──
    report = grad_check_batched(forward_fn, params, x_batch, y_batch)

    # ── 3. Forward‑pass verification (GPU vs NumPy reference) ──
    verify_matmul()
    verify_conv2d()

    # ── 4. Training health monitor ──
    mon = GradientMonitor(params, names)
    mon.snapshot()          # call after loss.backward()
    mon.report()            # print gradient statistics
"""

import numpy as np
import ctypes
import time
import sys

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── Import engine primitives ──
from nn_engine import (
    lib, Tensor, MatMul, Conv2D, MaxPool2D, Flatten, AddBias, BiasReLUFunc,
    ReLUFunc, SoftmaxCEFunc,
    matmul, add_bias, bias_relu, relu, softmax_ce, conv2d, maxpool2d, flatten,
    Linear, ConvLayer, SGD, Metrics, end_batch, release_all_buffers,
    _all_tensors, _run_mm, _dispatch,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. NUMERICAL GRADIENT CHECKER (finite differences)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_loss_scalar(loss_fn):
    """Run loss_fn(), sync the scalar result, clean up intermediates."""
    loss = loss_fn()
    loss.sync()
    val = float(loss.data.flat[0])
    release_all_buffers()
    lib.FlushGPU()
    return val


def grad_check(loss_fn, param, eps=1e-3, rtol=1e-2, atol=1e-5, max_elems=None, verbose=True):
    """
    Numerical gradient check for a single parameter tensor.

    Args:
        loss_fn:   callable() -> Tensor(scalar)  — must recreate the forward graph each call
        param:     Tensor with requires_grad=True  — the parameter to check
        eps:       finite-difference step size
        rtol:      relative tolerance
        atol:      absolute tolerance
        max_elems: if set, randomly sample this many elements to check (faster for large params)
        verbose:   print per-element comparison

    Returns:
        (passed: bool, max_relerr: float, report: str)
    """
    # 1. Analytical gradient: forward + backward
    param.grad = None
    loss = loss_fn()
    loss.backward()
    analytical = param.grad.sync().copy().flatten()
    release_all_buffers()
    lib.FlushGPU()

    # 2. Numerical gradient via central differences
    param.sync()
    w = param.data.copy().flatten()
    n = w.size

    indices = np.arange(n)
    if max_elems is not None and max_elems < n:
        indices = np.random.choice(n, max_elems, replace=False)
        indices.sort()

    numerical = np.zeros(len(indices), dtype=np.float32)
    for ii, idx in enumerate(indices):
        # f(w + eps)
        w_plus = w.copy()
        w_plus[idx] += eps
        param.upload(w_plus.reshape(param.shape))
        l_plus = _compute_loss_scalar(loss_fn)

        # f(w - eps)
        w_minus = w.copy()
        w_minus[idx] -= eps
        param.upload(w_minus.reshape(param.shape))
        l_minus = _compute_loss_scalar(loss_fn)

        numerical[ii] = (l_plus - l_minus) / (2 * eps)

    # Restore original weights
    param.upload(w.reshape(param.shape))

    # 3. Compare (np.allclose-style: |a - n| < atol + rtol * max(|a|, |n|))
    ana_sub = analytical[indices]
    diff = np.abs(ana_sub - numerical)
    denom = np.maximum(np.abs(ana_sub), np.abs(numerical))
    threshold = atol + rtol * denom
    rel_err = diff / (denom + 1e-8)

    passed = np.all(diff < threshold)
    max_rel = float(rel_err.max()) if len(rel_err) > 0 else 0.0
    max_abs = float(diff.max()) if len(diff) > 0 else 0.0

    lines = []
    if verbose:
        lines.append(f"  Grad Check: {len(indices)} elements, eps={eps}")
        lines.append(f"  Max rel err: {max_rel:.6e}  Max abs err: {max_abs:.6e}")
        # Show worst 5
        worst = np.argsort(rel_err)[-5:][::-1]
        for w_idx in worst:
            i = indices[w_idx]
            lines.append(f"    [{i:5d}] analytical={ana_sub[w_idx]:+.6e}  numerical={numerical[w_idx]:+.6e}  rel={rel_err[w_idx]:.2e}")
        lines.append(f"  {'PASSED ✓' if passed else 'FAILED ✗'}")

    report = "\n".join(lines)
    if verbose:
        print(report)
    return passed, max_rel, report


def grad_check_layer(layer_name, forward_fn, param, x_data, y_data, eps=1e-3, rtol=5e-2, atol=1e-4, max_elems=50):
    """
    Gradient check for a specific layer parameter.

    Args:
        layer_name: string label (e.g., "Linear1.w")
        forward_fn: callable(xb) -> logits
        param:      the Tensor parameter to verify
        x_data:     numpy input batch
        y_data:     numpy label batch
        eps, rtol, atol: tolerances
        max_elems:  max elements to check

    Returns:
        (passed, max_relerr, report)
    """
    def loss_fn():
        xb = Tensor(x_data, track=True)
        yb = Tensor(y_data, track=True)
        logits = forward_fn(xb)
        return softmax_ce(logits, yb)

    print(f"\n── Grad check: {layer_name} (shape={param.shape}, size={param.size}) ──")
    return grad_check(loss_fn, param, eps=eps, rtol=rtol, atol=atol, max_elems=max_elems)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. BATCHED GRADIENT CHECKER
# ═══════════════════════════════════════════════════════════════════════════════

def grad_check_batched(forward_fn, params, param_names, x_data, y_data,
                       eps=1e-3, rtol=5e-2, atol=1e-4, max_elems=50):
    """
    Run gradient checks on ALL parameters using a batched forward pass.

    Args:
        forward_fn:   callable(xb) -> logits
        params:       list of Tensor parameters
        param_names:  list of string names (same order as params)
        x_data:       numpy input (small batch, e.g., 4-8 samples)
        y_data:       numpy labels
        eps, rtol, atol: tolerances
        max_elems:    max elements per param

    Returns:
        dict: { name: (passed, max_relerr) }
    """
    results = {}
    total = len(params)
    n_passed = 0

    print(f"\n{'='*70}")
    print(f"  BATCHED GRADIENT CHECK — {total} parameters")
    print(f"  Input shape: {x_data.shape}, Labels shape: {y_data.shape}")
    print(f"  eps={eps}, rtol={rtol}, atol={atol}, max_elems={max_elems}")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    for i, (p, name) in enumerate(zip(params, param_names)):
        ok, max_rel, _ = grad_check_layer(name, forward_fn, p, x_data, y_data,
                                           eps=eps, rtol=rtol, atol=atol, max_elems=max_elems)
        results[name] = (ok, max_rel)
        if ok:
            n_passed += 1

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*70}")
    print(f"  RESULTS: {n_passed}/{total} passed  ({elapsed:.1f}s)")
    for name, (ok, rel) in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] {name:<30s} max_rel={rel:.4e}")
    print(f"{'='*70}\n")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  3. FORWARD PASS VERIFICATION (GPU vs NumPy reference)
# ═══════════════════════════════════════════════════════════════════════════════

def _gpu_result(tensor):
    """Sync a Tensor and return numpy array."""
    return tensor.sync().copy()


def verify_matmul(M=16, K=32, N=24, rtol=1e-4, verbose=True):
    """Verify GPU matmul against NumPy reference."""
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    expected = A @ B

    ta = Tensor(A, track=True)
    tb = Tensor(B, track=True)
    tc = matmul(ta, tb)
    got = _gpu_result(tc)
    release_all_buffers()
    lib.FlushGPU()

    err = np.max(np.abs(expected - got))
    rel = err / (np.max(np.abs(expected)) + 1e-8)
    ok = rel < rtol

    if verbose:
        print(f"verify_matmul({M}x{K} @ {K}x{N}): max_abs={err:.6e} rel={rel:.6e} {'PASS' if ok else 'FAIL'}")
    return ok, rel


def verify_matmul_transpose(M=16, K=32, N=24, rtol=1e-4, verbose=True):
    """Verify GPU matmul with transpose flags against NumPy."""
    results = []

    # flags=1 → A^T @ B: actual A is (K,M), compute (M,K) @ (K,N) = (M,N)
    A_stored = np.random.randn(K, M).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    expected = A_stored.T @ B  # (M,N)

    out_handle = lib.CreateBuffer(None, M * N)
    ta = Tensor(A_stored, track=True)
    tb = Tensor(B, track=True)
    _run_mm(ta.gpu_buf, tb.gpu_buf, out_handle, M, K, N, flags=1)
    got = np.empty((M, N), dtype=np.float32)
    lib.ReadBuffer(out_handle, got.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    lib.ReleaseBuffer(out_handle)
    release_all_buffers()
    lib.FlushGPU()

    err = np.max(np.abs(expected - got))
    rel = err / (np.max(np.abs(expected)) + 1e-8)
    ok1 = rel < rtol
    if verbose:
        print(f"verify_matmul_transpose(flags=1, A^T@B): max_abs={err:.6e} rel={rel:.6e} {'PASS' if ok1 else 'FAIL'}")

    # flags=2 → A @ B^T: actual B is (N,K), compute (M,K) @ (K,N) = (M,N)
    A = np.random.randn(M, K).astype(np.float32)
    B_stored = np.random.randn(N, K).astype(np.float32)
    expected = A @ B_stored.T  # (M,N)

    out_handle = lib.CreateBuffer(None, M * N)
    ta = Tensor(A, track=True)
    tb = Tensor(B_stored, track=True)
    _run_mm(ta.gpu_buf, tb.gpu_buf, out_handle, M, K, N, flags=2)
    got = np.empty((M, N), dtype=np.float32)
    lib.ReadBuffer(out_handle, got.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    lib.ReleaseBuffer(out_handle)
    release_all_buffers()
    lib.FlushGPU()

    err = np.max(np.abs(expected - got))
    rel = err / (np.max(np.abs(expected)) + 1e-8)
    ok2 = rel < rtol
    if verbose:
        print(f"verify_matmul_transpose(flags=2, A@B^T): max_abs={err:.6e} rel={rel:.6e} {'PASS' if ok2 else 'FAIL'}")

    return ok1 and ok2


def verify_relu(N=1024, verbose=True):
    """Verify GPU ReLU against NumPy reference."""
    x = np.random.randn(N).astype(np.float32)
    expected = np.maximum(x, 0)

    tx = Tensor(x.reshape(1, N), track=True)
    ty = relu(tx)
    got = _gpu_result(ty).flatten()
    release_all_buffers()
    lib.FlushGPU()

    err = np.max(np.abs(expected - got))
    ok = err < 1e-6
    if verbose:
        print(f"verify_relu({N}): max_abs={err:.6e} {'PASS' if ok else 'FAIL'}")
    return ok, err


def verify_softmax(batch=4, classes=10, verbose=True):
    """Verify GPU softmax against NumPy reference."""
    x = np.random.randn(batch, classes).astype(np.float32)
    # NumPy reference: stable softmax
    x_shifted = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    expected = exp_x / exp_x.sum(axis=1, keepdims=True)

    tx = Tensor(x, track=True)
    # Use SoftmaxCE's internal softmax (we need to access it)
    softmax_handle = lib.CreateBuffer(None, tx.size)
    _dispatch(b"softmax", (tx.gpu_buf,), (softmax_handle,),
              (batch + 15) // 16, 1, 1, u1=batch, u2=classes)
    got = np.empty_like(x)
    lib.ReadBuffer(softmax_handle, got.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    lib.ReleaseBuffer(softmax_handle)
    release_all_buffers()
    lib.FlushGPU()

    err = np.max(np.abs(expected - got))
    rel = err / (np.max(np.abs(expected)) + 1e-8)
    ok = rel < 1e-4
    if verbose:
        print(f"verify_softmax({batch}x{classes}): max_abs={err:.6e} rel={rel:.6e} {'PASS' if ok else 'FAIL'}")
    return ok, rel


def verify_conv2d(batch=2, inC=1, outC=4, inH=8, inW=8, kH=3, stride=1, padding=1, rtol=1e-3, verbose=True):
    """Verify GPU conv2d against NumPy reference (im2col + matmul)."""
    x = np.random.randn(batch, inC, inH, inW).astype(np.float32) * 0.1
    filters = np.random.randn(outC, inC, kH, kH).astype(np.float32) * 0.1
    bias = np.random.randn(outC).astype(np.float32) * 0.01

    # NumPy reference: im2col + matmul
    outH = (inH + 2 * padding - kH) // stride + 1
    outW = (inW + 2 * padding - kH) // stride + 1

    # Pad input
    if padding > 0:
        x_pad = np.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)))
    else:
        x_pad = x

    # im2col
    cols = []
    for b in range(batch):
        for oh in range(outH):
            for ow in range(outW):
                patch = x_pad[b, :, oh*stride:oh*stride+kH, ow*stride:ow*stride+kH]
                cols.append(patch.flatten())
    col_matrix = np.array(cols).T  # (inC*kH*kW, batch*outH*outW)

    # matmul + bias + reshape
    f_reshaped = filters.reshape(outC, -1)  # (outC, inC*kH*kW)
    mm_result = f_reshaped @ col_matrix  # (outC, batch*outH*outW)
    expected = mm_result.reshape(outC, batch, outH, outW).transpose(1, 0, 2, 3)
    expected += bias.reshape(1, outC, 1, 1)

    # GPU
    tx = Tensor(x, track=True)
    tf = Tensor(filters, requires_grad=True, track=False)
    tb = Tensor(bias, requires_grad=True, track=False)
    out = conv2d(tx, tf, tb, stride=stride, padding=padding)
    got = _gpu_result(out)
    release_all_buffers()
    lib.FlushGPU()

    err = np.max(np.abs(expected - got))
    rel = err / (np.max(np.abs(expected)) + 1e-8)
    ok = rel < rtol

    if verbose:
        print(f"verify_conv2d(B={batch},C={inC}->{outC},H={inH},K={kH},S={stride},P={padding}): "
              f"max_abs={err:.6e} rel={rel:.6e} {'PASS' if ok else 'FAIL'}")
    return ok, rel


def verify_maxpool(batch=2, C=4, inH=8, inW=8, pool=2, stride=2, verbose=True):
    """Verify GPU maxpool against NumPy reference."""
    x = np.random.randn(batch, C, inH, inW).astype(np.float32)
    outH = (inH - pool) // stride + 1
    outW = (inW - pool) // stride + 1

    # NumPy reference
    expected = np.zeros((batch, C, outH, outW), dtype=np.float32)
    for b in range(batch):
        for c in range(C):
            for oh in range(outH):
                for ow in range(outW):
                    patch = x[b, c, oh*stride:oh*stride+pool, ow*stride:ow*stride+pool]
                    expected[b, c, oh, ow] = patch.max()

    # GPU
    tx = Tensor(x, track=True)
    out = maxpool2d(tx, pool_size=pool, stride=stride)
    got = _gpu_result(out)
    release_all_buffers()
    lib.FlushGPU()

    err = np.max(np.abs(expected - got))
    ok = err < 1e-6
    if verbose:
        print(f"verify_maxpool(B={batch},C={C},H={inH},pool={pool}): max_abs={err:.6e} {'PASS' if ok else 'FAIL'}")
    return ok, err


def verify_linear(in_f=16, out_f=8, batch=4, verbose=True):
    """Verify GPU Linear layer against NumPy reference (matmul + bias)."""
    x = np.random.randn(batch, in_f).astype(np.float32)
    layer = Linear(in_f, out_f)

    # NumPy reference
    w = layer.w.sync().copy()
    b = layer.b.sync().copy()
    expected = x @ w + b

    # GPU
    tx = Tensor(x, track=True)
    out = layer(tx)
    got = _gpu_result(out)
    release_all_buffers()
    lib.FlushGPU()

    err = np.max(np.abs(expected - got))
    rel = err / (np.max(np.abs(expected)) + 1e-8)
    ok = rel < 1e-4
    if verbose:
        print(f"verify_linear({batch}x{in_f} -> {out_f}): max_abs={err:.6e} rel={rel:.6e} {'PASS' if ok else 'FAIL'}")
    return ok, rel


def verify_all(verbose=True):
    """Run all forward-pass verification tests."""
    print(f"\n{'='*70}")
    print(f"  FORWARD PASS VERIFICATION (GPU vs NumPy)")
    print(f"{'='*70}")

    results = {}
    tests = [
        ("MatMul (small)",       lambda: verify_matmul(8, 16, 12, verbose=verbose)),
        ("MatMul (medium)",      lambda: verify_matmul(64, 128, 64, verbose=verbose)),
        ("MatMul (large)",       lambda: verify_matmul(128, 256, 128, verbose=verbose)),
        ("MatMul transpose",     lambda: verify_matmul_transpose(16, 32, 24, verbose=verbose)),
        ("ReLU",                 lambda: verify_relu(1024, verbose=verbose)),
        ("Softmax",              lambda: verify_softmax(8, 10, verbose=verbose)),
        ("Conv2D (1->4, k=3)",   lambda: verify_conv2d(2, 1, 4, 8, 8, 3, 1, 1, verbose=verbose)),
        ("Conv2D (3->16, k=5)",  lambda: verify_conv2d(2, 3, 16, 16, 16, 5, 1, 2, verbose=verbose)),
        ("MaxPool (2x2)",        lambda: verify_maxpool(2, 4, 8, 8, verbose=verbose)),
        ("MaxPool (3x3,s=2)",    lambda: verify_maxpool(2, 4, 8, 8, 3, 2, verbose=verbose)),
        ("Linear (16->8)",       lambda: verify_linear(16, 8, 4, verbose=verbose)),
        ("Linear (120->84)",     lambda: verify_linear(120, 84, 32, verbose=verbose)),
    ]

    n_pass = 0
    for name, fn in tests:
        try:
            ok_result = fn()
            ok = ok_result[0] if isinstance(ok_result, tuple) else ok_result
            results[name] = ok
            if ok:
                n_pass += 1
        except Exception as e:
            results[name] = False
            if verbose:
                print(f"  {name}: ERROR — {e}")

    print(f"\n{'='*70}")
    print(f"  FORWARD: {n_pass}/{len(tests)} passed")
    for name, ok in results.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"{'='*70}\n")

    return all(results.values())


# ═══════════════════════════════════════════════════════════════════════════════
#  4. GRADIENT FLOW MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class GradientMonitor:
    """
    Track gradient statistics across training steps to detect:
    - Vanishing gradients (norm → 0)
    - Exploding gradients (norm → infinity)
    - Dead parameters (grad ≡ 0)
    - NaN / Inf values

    Usage:
        mon = GradientMonitor(params, ["conv1.w", "conv1.b", "fc1.w", ...])
        # In training loop, after loss.backward():
        mon.snapshot()
        # End of epoch:
        mon.report()
        mon.reset()
    """
    def __init__(self, params, names=None):
        self.params = params
        self.names = names or [f"param_{i}" for i in range(len(params))]
        self._history = {n: [] for n in self.names}  # name -> list of {norm, mean, std, min, max, frac_zero, has_nan}
        self._step = 0

    def snapshot(self):
        """Record gradient stats for all parameters at current step."""
        self._step += 1
        for p, name in zip(self.params, self.names):
            if p.grad is None:
                self._history[name].append({
                    "step": self._step, "norm": 0.0, "mean": 0.0, "std": 0.0,
                    "min": 0.0, "max": 0.0, "frac_zero": 1.0,
                    "has_nan": False, "has_inf": False,
                })
                continue

            g = p.grad.sync().flatten()
            norm = float(np.linalg.norm(g))
            self._history[name].append({
                "step": self._step,
                "norm": norm,
                "mean": float(np.mean(g)),
                "std": float(np.std(g)),
                "min": float(np.min(g)),
                "max": float(np.max(g)),
                "frac_zero": float(np.mean(np.abs(g) < 1e-8)),
                "has_nan": bool(np.any(np.isnan(g))),
                "has_inf": bool(np.any(np.isinf(g))),
            })

    def reset(self):
        """Clear history for a new epoch."""
        for name in self.names:
            self._history[name] = []
        self._step = 0

    def report(self, last_n=None):
        """Print gradient statistics summary."""
        print(f"\n{'='*90}")
        print(f"  GRADIENT MONITOR — {self._step} steps recorded")
        print(f"{'='*90}")
        print(f"  {'Name':<25s} {'Avg Norm':>10s} {'Avg |Mean|':>12s} {'Avg Std':>10s} "
              f"{'%Zero':>7s} {'NaN':>4s} {'Inf':>4s} {'Status':<12s}")
        print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10} {'-'*7} {'-'*4} {'-'*4} {'-'*12}")

        for name in self.names:
            hist = self._history[name]
            if last_n is not None:
                hist = hist[-last_n:]
            if not hist:
                print(f"  {name:<25s} {'(no data)':>10s}")
                continue

            avg_norm = np.mean([h["norm"] for h in hist])
            avg_mean = np.mean([abs(h["mean"]) for h in hist])
            avg_std = np.mean([h["std"] for h in hist])
            avg_zero = np.mean([h["frac_zero"] for h in hist]) * 100
            any_nan = any(h["has_nan"] for h in hist)
            any_inf = any(h["has_inf"] for h in hist)

            # Diagnosis
            status = "OK"
            if any_nan:
                status = "NaN!"
            elif any_inf:
                status = "EXPLODING!"
            elif avg_norm < 1e-7:
                status = "VANISHING"
            elif avg_zero > 90:
                status = "DEAD"
            elif avg_norm > 1e4:
                status = "LARGE"

            print(f"  {name:<25s} {avg_norm:>10.4e} {avg_mean:>12.4e} {avg_std:>10.4e} "
                  f"{avg_zero:>6.1f}% {'Y' if any_nan else '.':>4s} {'Y' if any_inf else '.':>4s} {status:<12s}")

        print(f"{'='*90}\n")

    def check_anomalies(self):
        """Return list of (name, issue) pairs for any detected anomalies."""
        anomalies = []
        for name in self.names:
            hist = self._history[name]
            if not hist:
                continue
            last = hist[-1]
            if last["has_nan"]:
                anomalies.append((name, "NaN in gradients"))
            if last["has_inf"]:
                anomalies.append((name, "Inf in gradients"))
            if len(hist) >= 3:
                norms = [h["norm"] for h in hist[-3:]]
                if all(n < 1e-8 for n in norms):
                    anomalies.append((name, "Vanishing gradients (norm < 1e-8 for last 3 steps)"))
                if all(n > 1e6 for n in norms):
                    anomalies.append((name, "Exploding gradients (norm > 1e6 for last 3 steps)"))
        return anomalies

    def get_norms(self, name):
        """Get gradient norm history for a specific parameter."""
        return [h["norm"] for h in self._history.get(name, [])]


# ═══════════════════════════════════════════════════════════════════════════════
#  5. ACTIVATION MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class ActivationMonitor:
    """
    Monitor intermediate activation statistics to detect:
    - Dead ReLU neurons (output ≡ 0)
    - Saturated activations
    - Distribution shift

    Usage:
        act_mon = ActivationMonitor()
        # During forward pass, after each activation:
        act_mon.record("relu1", tensor)
        # After forward pass:
        act_mon.report()
    """
    def __init__(self):
        self._stats = {}  # name -> list of {mean, std, frac_zero, min, max}
        self._step = 0

    def record(self, name, tensor):
        """Record statistics of a tensor (syncs GPU -> CPU)."""
        data = tensor.sync().flatten()
        if name not in self._stats:
            self._stats[name] = []
        self._stats[name].append({
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "frac_zero": float(np.mean(np.abs(data) < 1e-8)),
            "frac_negative": float(np.mean(data < 0)),
            "has_nan": bool(np.any(np.isnan(data))),
            "shape": tensor.shape,
        })
        self._step += 1

    def reset(self):
        self._stats.clear()
        self._step = 0

    def report(self):
        """Print activation statistics."""
        print(f"\n{'='*90}")
        print(f"  ACTIVATION MONITOR")
        print(f"{'='*90}")
        print(f"  {'Name':<20s} {'Shape':<18s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s} "
              f"{'%Zero':>7s} {'%Neg':>6s} {'Status':<10s}")
        print(f"  {'-'*20} {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*7} {'-'*6} {'-'*10}")

        for name, entries in self._stats.items():
            if not entries:
                continue
            last = entries[-1]
            status = "OK"
            if last["has_nan"]:
                status = "NaN!"
            elif last["frac_zero"] > 0.9:
                status = "DEAD"
            elif last["std"] < 1e-6:
                status = "COLLAPSED"
            elif last["max"] > 1e6:
                status = "EXPLODING"

            shape_str = str(last["shape"])
            print(f"  {name:<20s} {shape_str:<18s} {last['mean']:>10.4f} {last['std']:>10.4f} "
                  f"{last['min']:>10.4f} {last['max']:>10.4f} "
                  f"{last['frac_zero']*100:>6.1f}% {last['frac_negative']*100:>5.1f}% {status:<10s}")

        print(f"{'='*90}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  6. HEALTH CHECK — NaN / Inf / dead parameter detection
# ═══════════════════════════════════════════════════════════════════════════════

def check_health(params, names=None, check_grads=True, verbose=True):
    """
    Scan all parameters (and optionally gradients) for NaN, Inf, and anomalies.

    Args:
        params:      list of Tensor parameters
        names:       list of string labels
        check_grads: also check .grad tensors
        verbose:     print results

    Returns:
        list of (name, issue_string) — empty if all healthy
    """
    if names is None:
        names = [f"param_{i}" for i in range(len(params))]

    issues = []
    for p, name in zip(params, names):
        # Check weights
        w = p.sync().flatten()
        if np.any(np.isnan(w)):
            issues.append((name, f"NaN in weights ({np.sum(np.isnan(w))} elements)"))
        if np.any(np.isinf(w)):
            issues.append((name, f"Inf in weights ({np.sum(np.isinf(w))} elements)"))
        if np.all(np.abs(w) < 1e-10):
            issues.append((name, "All weights near zero"))

        # Check gradients
        if check_grads and p.grad is not None:
            g = p.grad.sync().flatten()
            if np.any(np.isnan(g)):
                issues.append((name + ".grad", f"NaN in gradients ({np.sum(np.isnan(g))} elements)"))
            if np.any(np.isinf(g)):
                issues.append((name + ".grad", f"Inf in gradients ({np.sum(np.isinf(g))} elements)"))

    if verbose:
        if issues:
            print(f"\n  HEALTH CHECK: {len(issues)} issues found!")
            for name, issue in issues:
                print(f"    [!] {name}: {issue}")
        else:
            print(f"  HEALTH CHECK: All {len(params)} parameters healthy")

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
#  7. PARAMETER STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def param_stats(params, names=None, include_grads=True):
    """
    Print weight and gradient statistics for all parameters.

    Usage:
        param_stats(params, ["conv1.w", "conv1.b", "fc1.w", ...])
    """
    if names is None:
        names = [f"param_{i}" for i in range(len(params))]

    print(f"\n{'='*100}")
    print(f"  PARAMETER STATISTICS")
    print(f"{'='*100}")
    print(f"  {'Name':<22s} {'Shape':<18s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s} {'L2 Norm':>10s}")
    print(f"  {'-'*22} {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    total_params = 0
    for p, name in zip(params, names):
        w = p.sync().flatten()
        total_params += w.size
        print(f"  {name:<22s} {str(p.shape):<18s} {np.mean(w):>10.4e} {np.std(w):>10.4e} "
              f"{np.min(w):>10.4e} {np.max(w):>10.4e} {np.linalg.norm(w):>10.4e}")

    print(f"  {'-'*22}")
    print(f"  Total parameters: {total_params:,}")

    if include_grads:
        has_grads = any(p.grad is not None for p in params)
        if has_grads:
            print(f"\n  {'Name':<22s} {'Grad Mean':>10s} {'Grad Std':>10s} {'Grad Min':>10s} "
                  f"{'Grad Max':>10s} {'Grad Norm':>10s} {'Ratio':>10s}")
            print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
            for p, name in zip(params, names):
                if p.grad is None:
                    print(f"  {name:<22s} {'(no grad)':>10s}")
                    continue
                g = p.grad.sync().flatten()
                w = p.sync().flatten()
                g_norm = np.linalg.norm(g)
                w_norm = np.linalg.norm(w)
                ratio = g_norm / (w_norm + 1e-8)
                print(f"  {name:<22s} {np.mean(g):>10.4e} {np.std(g):>10.4e} {np.min(g):>10.4e} "
                      f"{np.max(g):>10.4e} {g_norm:>10.4e} {ratio:>10.4e}")

    print(f"{'='*100}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  8. PYTORCH COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def compare_pytorch(forward_fn_gpu, params_gpu, param_names, x_np, y_np,
                    model_builder_fn, rtol=1e-2, verbose=True):
    """
    Compare forward + backward pass results against PyTorch.
    Requires torch to be installed.

    Args:
        forward_fn_gpu:  callable(Tensor) -> Tensor logits (GPU engine)
        params_gpu:      list of GPU Tensor params
        param_names:     list of string names
        x_np:            input numpy array
        y_np:            label numpy array
        model_builder_fn: callable(params_dict) -> torch.nn.Module
                          receives {name: numpy_array} for weight initialization
        rtol:            tolerance for comparison

    Returns:
        dict of comparison results
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        print("  PyTorch not installed — skipping comparison")
        return {}

    # Get GPU engine weights
    gpu_weights = {}
    for p, name in zip(params_gpu, param_names):
        gpu_weights[name] = p.sync().copy()

    # Build PyTorch model with same weights
    pt_model = model_builder_fn(gpu_weights)
    pt_model.train()

    # Forward pass: GPU engine
    xb = Tensor(x_np, track=True)
    yb = Tensor(y_np, track=True)
    logits_gpu = forward_fn_gpu(xb)
    loss_gpu = softmax_ce(logits_gpu, yb)
    loss_gpu.backward()
    logits_np = logits_gpu.sync().copy()
    loss_np = loss_gpu.sync().copy()

    # Collect GPU gradients
    gpu_grads = {}
    for p, name in zip(params_gpu, param_names):
        if p.grad is not None:
            gpu_grads[name] = p.grad.sync().copy()

    release_all_buffers()
    lib.FlushGPU()

    # Forward pass: PyTorch
    x_pt = torch.tensor(x_np, dtype=torch.float32)
    y_pt = torch.tensor(y_np, dtype=torch.long)
    logits_pt = pt_model(x_pt)
    loss_pt = F.cross_entropy(logits_pt, y_pt)
    loss_pt.backward()

    results = {}

    # Compare logits
    logits_diff = np.max(np.abs(logits_np - logits_pt.detach().numpy()))
    logits_rel = logits_diff / (np.max(np.abs(logits_pt.detach().numpy())) + 1e-8)
    results["logits"] = {"abs_err": logits_diff, "rel_err": logits_rel}

    # Compare loss
    loss_diff = abs(float(loss_np) - float(loss_pt.item()))
    results["loss"] = {"abs_err": loss_diff}

    # Compare gradients
    for name, param_pt in pt_model.named_parameters():
        if name in gpu_grads and param_pt.grad is not None:
            g_pt = param_pt.grad.detach().numpy().flatten()
            g_gpu = gpu_grads[name].flatten()
            if g_pt.shape != g_gpu.shape:
                results[f"grad_{name}"] = {"error": f"shape mismatch: PT={g_pt.shape} GPU={g_gpu.shape}"}
                continue
            diff = np.max(np.abs(g_pt - g_gpu))
            rel = diff / (np.max(np.abs(g_pt)) + 1e-8)
            results[f"grad_{name}"] = {"abs_err": diff, "rel_err": rel}

    if verbose:
        print(f"\n{'='*70}")
        print(f"  PYTORCH COMPARISON")
        print(f"{'='*70}")
        for key, vals in results.items():
            if "error" in vals:
                print(f"  {key:<30s}: {vals['error']}")
            else:
                abs_e = vals.get("abs_err", 0)
                rel_e = vals.get("rel_err", 0)
                ok = rel_e < rtol if "rel_err" in vals else abs_e < rtol
                print(f"  {key:<30s}: abs={abs_e:.6e}  rel={rel_e:.6e}  {'PASS' if ok else 'FAIL'}")
        print(f"{'='*70}\n")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  9. WEIGHT INITIALIZATION CHECKER
# ═══════════════════════════════════════════════════════════════════════════════

def check_init(params, names=None):
    """
    Verify weight initialization follows Xavier/Glorot uniform distribution.
    Checks that std ≈ sqrt(2 / (fan_in + fan_out)) for each parameter.
    """
    if names is None:
        names = [f"param_{i}" for i in range(len(params))]

    print(f"\n{'='*80}")
    print(f"  WEIGHT INITIALIZATION CHECK")
    print(f"{'='*80}")
    print(f"  {'Name':<22s} {'Shape':<18s} {'Actual Std':>12s} {'Expected':>12s} {'Ratio':>8s} {'Status':<8s}")
    print(f"  {'-'*22} {'-'*18} {'-'*12} {'-'*12} {'-'*8} {'-'*8}")

    for p, name in zip(params, names):
        w = p.sync().flatten()
        actual_std = float(np.std(w))

        # Estimate fan_in/fan_out from shape
        shape = p.shape
        if len(shape) == 2:
            fan_in, fan_out = shape
        elif len(shape) == 4:
            # Conv: (outC, inC, kH, kW)
            fan_in = shape[1] * shape[2] * shape[3]
            fan_out = shape[0] * shape[2] * shape[3]
        elif len(shape) == 1:
            # Bias — should be near zero
            expected_std = 0.0
            status = "OK" if actual_std < 0.01 else "WARN"
            print(f"  {name:<22s} {str(shape):<18s} {actual_std:>12.6f} {'~0 (bias)':>12s} {'':>8s} {status:<8s}")
            continue
        else:
            print(f"  {name:<22s} {str(shape):<18s} {actual_std:>12.6f} {'(unknown)':>12s}")
            continue

        # Xavier uniform: std = sqrt(2 / (fan_in + fan_out)) * sqrt(3) ≈ limit / sqrt(3)
        # where limit = sqrt(6 / (fan_in + fan_out))
        expected_std = np.sqrt(2.0 / (fan_in + fan_out))
        ratio = actual_std / (expected_std + 1e-8)
        status = "OK" if 0.5 < ratio < 2.0 else "WARN"

        print(f"  {name:<22s} {str(shape):<18s} {actual_std:>12.6f} {expected_std:>12.6f} {ratio:>8.3f} {status:<8s}")

    print(f"{'='*80}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  10. LOSS CURVE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class LossCurveTracker:
    """
    Track and analyze loss during training.
    Detects: divergence, plateaus, oscillation, NaN/Inf.

    Usage:
        tracker = LossCurveTracker()
        for epoch in range(epochs):
            for batch in batches:
                ...
                tracker.add(float_loss)
            tracker.epoch_summary(epoch)
        tracker.report()
    """
    def __init__(self, window=50):
        self.losses = []
        self.epoch_avg = []
        self.window = window

    def add(self, loss_val):
        """Add a single loss value (per-batch)."""
        self.losses.append(float(loss_val))

    def epoch_summary(self, epoch):
        """Compute and store epoch average. Call at end of each epoch."""
        if not self.losses:
            return
        # Use only losses since last epoch_summary
        start = sum(len(self.epoch_avg) > 0 for _ in []) if not self.epoch_avg else 0
        recent = self.losses[-(len(self.losses) - len(self.epoch_avg) * self.window):]
        avg = np.mean(self.losses[-self.window:]) if len(self.losses) >= self.window else np.mean(self.losses)
        self.epoch_avg.append(float(avg))

    def report(self):
        """Analyze and print loss curve diagnostics."""
        print(f"\n{'='*70}")
        print(f"  LOSS CURVE ANALYSIS — {len(self.losses)} steps, {len(self.epoch_avg)} epochs")
        print(f"{'='*70}")

        if not self.losses:
            print("  (no data)")
            return

        # NaN / Inf check
        nan_count = sum(1 for l in self.losses if np.isnan(l) or np.isinf(l))
        if nan_count > 0:
            print(f"  [!] NaN/Inf detected in {nan_count} steps!")

        # Overall trend
        if len(self.losses) >= 10:
            first_10 = np.mean(self.losses[:10])
            last_10 = np.mean(self.losses[-10:])
            change = (last_10 - first_10) / (abs(first_10) + 1e-8) * 100
            trend = "DECREASING" if change < -5 else "INCREASING" if change > 5 else "FLAT"
            print(f"  Trend: {trend} ({change:+.1f}%)")
            print(f"  First 10 avg: {first_10:.6f}")
            print(f"  Last  10 avg: {last_10:.6f}")
            print(f"  Min loss:     {min(self.losses):.6f} (step {np.argmin(self.losses)})")

        # Oscillation detection
        if len(self.losses) >= 20:
            recent = np.array(self.losses[-20:])
            diffs = np.diff(recent)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            if sign_changes > 14:
                print(f"  [!] High oscillation detected ({sign_changes} sign changes in last 20 steps)")
                print(f"      Consider reducing learning rate")

        # Divergence detection
        if len(self.epoch_avg) >= 3:
            if all(self.epoch_avg[i] > self.epoch_avg[i-1] for i in range(-2, 0)):
                print(f"  [!] Loss increasing for last 2 epochs — possible divergence")
                print(f"      Consider reducing learning rate")

        print(f"{'='*70}\n")

    def get_smoothed(self, window=None):
        """Get smoothed loss curve for visualization."""
        w = window or self.window
        if len(self.losses) < w:
            return self.losses
        return [np.mean(self.losses[max(0, i-w):i+1]) for i in range(len(self.losses))]


# ═══════════════════════════════════════════════════════════════════════════════
#  11. ALL-IN-ONE DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════

def run_diagnostics(forward_fn, params, param_names, x_sample, y_sample,
                    do_grad_check=True, do_forward_verify=True,
                    do_health_check=True, do_init_check=True,
                    grad_check_elems=30, eps=1e-3):
    """
    Run a comprehensive diagnostic suite on the model.

    Args:
        forward_fn:    callable(Tensor) -> Tensor logits
        params:        list of Tensor parameters
        param_names:   list of string names
        x_sample:      small numpy input (4-8 samples)
        y_sample:      numpy labels
        do_grad_check: run numerical gradient verification
        do_forward_verify: run forward-pass GPU-vs-NumPy checks
        do_health_check: check for NaN/Inf
        do_init_check: check weight initialization
        grad_check_elems: elements per param for gradient check
        eps:           finite-difference epsilon

    Returns:
        dict with results
    """
    print(f"\n{'#'*70}")
    print(f"  COMPREHENSIVE MODEL DIAGNOSTICS")
    print(f"{'#'*70}")

    results = {}

    if do_init_check:
        print("\n[1/4] Weight Initialization...")
        check_init(params, param_names)

    if do_health_check:
        print("\n[2/4] Parameter Health...")
        issues = check_health(params, param_names, check_grads=False)
        results["health"] = len(issues) == 0

    if do_forward_verify:
        print("\n[3/4] Forward Pass Verification...")
        results["forward"] = verify_all(verbose=True)

    if do_grad_check:
        print("\n[4/4] Numerical Gradient Check...")
        grad_results = grad_check_batched(
            forward_fn, params, param_names, x_sample, y_sample,
            eps=eps, max_elems=grad_check_elems,
        )
        results["gradients"] = grad_results
        results["all_grads_ok"] = all(ok for ok, _ in grad_results.values())

    print(f"\n{'#'*70}")
    print(f"  DIAGNOSTIC SUMMARY")
    print(f"{'#'*70}")
    for key, val in results.items():
        if isinstance(val, bool):
            print(f"  {key:<25s}: {'PASS' if val else 'FAIL'}")
        elif isinstance(val, dict):
            n_pass = sum(1 for ok, _ in val.values() if ok)
            print(f"  {key:<25s}: {n_pass}/{len(val)} passed")
    print(f"{'#'*70}\n")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — run as standalone script
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("nn_verify.py — DirectCompute Neural Network Verification Toolkit")
    print("=" * 60)

    # Quick self-test: forward pass verification
    print("\n[1] Forward pass verification...")
    all_fwd_ok = verify_all(verbose=True)

    # Quick self-test: gradient check on a small Linear layer
    print("\n[2] Gradient check (small Linear model)...")
    np.random.seed(42)

    l1 = Linear(16, 10)
    x_test = np.random.randn(4, 16).astype(np.float32) * 0.5
    y_test = np.random.randint(0, 10, size=4).astype(np.float32)

    def test_forward(xb):
        return l1(xb)

    grad_results = grad_check_batched(
        test_forward, [l1.w, l1.b], ["fc.w", "fc.b"],
        x_test, y_test, max_elems=20,
    )

    all_grad_ok = all(ok for ok, _ in grad_results.values())

    print(f"\n{'='*60}")
    print(f"  SELF-TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Forward verification: {'PASS' if all_fwd_ok else 'FAIL'}")
    print(f"  Gradient check:       {'PASS' if all_grad_ok else 'FAIL'}")
    print(f"{'='*60}")
