"""Microbenchmark: D @ u (and its VJP) for the operator variants.

Variants: dense | sparse (segment_sum) | conv (stencil) | bcoo (SpMM)
Case:     2D full grid 201x101, delta=0.02, batched over Nt time steps.

Usage:  python benchmarks/bench_operator.py
"""

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp

from nonlocal_pinn.operators import build_D

jax.config.update("jax_default_matmul_precision", "highest")


def _time(fn, *args, warmup=3, iters=30):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def bench_2d(Nx=201, Ny=101, delta=0.02, Nt=50):
    print(f"\n=== 2D full-grid matvec: {Nx}x{Ny}={Nx*Ny} points, "
          f"delta={delta}, Nt={Nt} ===")
    xs = jnp.linspace(0.0, 1.0, Nx)
    ys = jnp.linspace(0.0, 0.5, Ny)
    Xv, Yv = jnp.meshgrid(xs, ys, indexing='xy')
    x = jnp.stack([Xv.ravel(), Yv.ravel()], axis=-1)
    N = len(x)

    key = jax.random.PRNGKey(0)
    u = jax.random.normal(key, (N, Nt), dtype=jnp.float32)

    D_sp = build_D(x, delta, method="sparse")
    D_conv = build_D(x, delta, method="conv")
    nnz = len(D_sp._vals)
    print(f"nnz={nnz} (k_avg={nnz/N:.1f}, k_max={D_sp._k_max})")

    # BCOO comparison point (portable jax.experimental.sparse)
    from jax.experimental import sparse as jsparse
    rows = np.concatenate([D_sp._row_idx_np, np.arange(N, dtype=np.int32)])
    cols = np.concatenate([D_sp._col_idx_np, np.arange(N, dtype=np.int32)])
    vals = np.concatenate([D_sp._vals_np, np.array(D_sp._diag, dtype=np.float32)])
    bcoo = jsparse.BCOO((jnp.array(vals),
                         jnp.array(np.stack([rows, cols], axis=1))),
                        shape=(N, N))

    variants = {
        "sparse": jax.jit(lambda v: D_sp @ v),
        "conv":   jax.jit(lambda v: D_conv @ v),
        "bcoo":   jax.jit(lambda v: bcoo @ v),
    }
    try:
        D_dense = build_D(x, delta, method="dense")
        variants["dense"] = jax.jit(lambda v: D_dense @ v)
    except Exception as e:  # noqa: BLE001 — dense build may OOM at large N
        print(f"dense skipped: {e}")

    y_ref = variants["sparse"](u)
    cond = float(jnp.max(D_sp._diag[:, None] * jnp.abs(u))) + 1e-30

    print(f"{'variant':10s} {'fwd ms':>9s} {'vjp ms':>9s} {'max dev/cond':>13s}")
    for name, f in variants.items():
        fwd_ms = _time(f, u)
        vjp_fn = jax.jit(lambda v: jax.vjp(f, v)[1](y_ref)[0])
        vjp_ms = _time(vjp_fn, u)
        dev = float(jnp.max(jnp.abs(f(u) - y_ref))) / cond
        print(f"{name:10s} {fwd_ms:9.3f} {vjp_ms:9.3f} {dev:13.2e}")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    print(f"jax {jax.__version__}, device: {jax.devices()[0].device_kind}")
    bench_2d()
