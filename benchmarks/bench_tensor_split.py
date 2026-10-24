"""Tensor-product first-layer split: equivalence check + timing.

The losses evaluate the net on a tensor product vmap(x) x vmap(t).  With
tensor_split=True the first MLP-trunk layer is computed as
W_x@ex (x-only) + W_t@et (t-only), which vmap hoists out of the opposing
batch dimension — same math, but the (N_x*N_t, 256, 64) first-layer matmul
disappears from the tensor-product evaluation.

Measures forward+JVP (as in the losses' _compute_all) and full
value_and_grad of a mean-residual-style loss, mlp trunk with split off/on.

Usage:  python benchmarks/bench_tensor_split.py
"""

import time

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from nonlocal_pinn.nn.arch import make_pinn

jax.config.update("jax_default_matmul_precision", "highest")


def _compute_all(net, x_points, t_batch):
    def field_at_t(t):
        return jax.vmap(net, in_axes=(0, None))(x_points, t)
    def field_at_times(ts):
        return jax.vmap(field_at_t)(ts)
    return jax.jvp(field_at_times, (t_batch,), (jnp.ones_like(t_batch),))


def _loss(net, x, t):
    out, out_t = _compute_all(net, x, t)
    return jnp.mean(jnp.abs(out_t[..., 0] + out[..., 0] ** 3 - out[..., 1]))


def _time(fn, *args, warmup=3, iters=20):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters * 1e3


def run_case(label, spatial_dim, N_x, N_t, key):
    net = make_pinn(spatial_dim=spatial_dim, out_dim=2, hidden_dim=64,
                    num_layers=4, trunk_type="mlp", activation='gelu',
                    spatial_emb_dim=64, spatial_scale=2.0,
                    temporal_emb_dim=64, temporal_scale=1.0,
                    tensor_split=False, key=key)
    net_split = make_pinn(spatial_dim=spatial_dim, out_dim=2, hidden_dim=64,
                          num_layers=4, trunk_type="mlp", activation='gelu',
                          spatial_emb_dim=64, spatial_scale=2.0,
                          temporal_emb_dim=64, temporal_scale=1.0,
                          tensor_split=True, key=key)

    x = jr.uniform(jr.fold_in(key, 1), (N_x, spatial_dim))
    t = jr.uniform(jr.fold_in(key, 2), (N_t,), minval=0.001, maxval=1.0)

    # ---- equivalence: forward, JVP, loss, gradient ----
    o_a, ot_a = _compute_all(net, x, t)
    o_b, ot_b = _compute_all(net_split, x, t)
    dev_o = float(jnp.max(jnp.abs(o_a - o_b))) / (float(jnp.max(jnp.abs(o_a))) + 1e-30)
    dev_t = float(jnp.max(jnp.abs(ot_a - ot_b))) / (float(jnp.max(jnp.abs(ot_a))) + 1e-30)

    la, ga = eqx.filter_value_and_grad(_loss)(net, x, t)
    lb, gb = eqx.filter_value_and_grad(_loss)(net_split, x, t)
    dev_l = abs(float(la - lb)) / (abs(float(la)) + 1e-30)
    leaves_a = jax.tree.leaves(eqx.filter(ga, eqx.is_array))
    leaves_b = jax.tree.leaves(eqx.filter(gb, eqx.is_array))
    dev_g = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(leaves_a, leaves_b))
    ref_g = max(float(jnp.max(jnp.abs(a))) for a in leaves_a) + 1e-30
    ok = max(dev_o, dev_t, dev_l, dev_g / ref_g) < 2e-5
    print(f"[{'OK' if ok else 'FAIL'}] {label} equivalence: out {dev_o:.2e}  "
          f"d/dt {dev_t:.2e}  loss {dev_l:.2e}  grad {dev_g/ref_g:.2e}")

    # ---- timing ----
    print(f"{'net':12s} {'fwd+jvp ms':>11s} {'val+grad ms':>12s}")
    times = {}
    for name, m in [("mlp", net), ("mlp+split", net_split)]:
        f = jax.jit(lambda xx, tt, mm=m: _compute_all(mm, xx, tt))
        g = eqx.filter_jit(lambda mm, xx, tt:
                           eqx.filter_value_and_grad(_loss)(mm, xx, tt))
        ms_f = _time(f, x, t)
        ms_g = _time(lambda xx, tt, mm=m: g(mm, xx, tt), x, t)
        times[name] = (ms_f, ms_g)
        print(f"{name:12s} {ms_f:11.2f} {ms_g:12.2f}")
    sf = times["mlp"][0] / times["mlp+split"][0]
    sg = times["mlp"][1] / times["mlp+split"][1]
    print(f"speedup: fwd+jvp {sf:.2f}x, val+grad {sg:.2f}x")
    return ok


if __name__ == "__main__":
    print(f"jax {jax.__version__}, device: {jax.devices()[0].device_kind}")
    ok = True
    print("\n=== 2D shape: N_x=20301, N_t=50 (mlp trunk) ===")
    ok &= run_case("2D", 2, 20301, 50, jr.PRNGKey(0))
    print("\n=== 3D subgraph shape: N_x=13879, N_t=64 (mlp trunk) ===")
    ok &= run_case("3D", 3, 13879, 64, jr.PRNGKey(1))
    raise SystemExit(0 if ok else 1)
