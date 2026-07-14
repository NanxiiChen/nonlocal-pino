"""Numerical-equivalence gate for the performance reformulations.

Verifies, against the reference implementation, that each optimization
produces identical loss values and gradients to FP32 round-off:

  1. ConvD operator vs SparseD   (2D full-grid loss + full grad pytree,
                                  regular and GradNorm-update steps)
  2. tensor_split net vs concat  (outputs, time-derivative JVP, loss, grads)

Run:  python benchmarks/verify_equivalence.py
"""

import importlib.util
import os

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

jax.config.update("jax_default_matmul_precision", "highest")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _grad_dev(g1, g2):
    """Max abs deviation and max ref magnitude across a grad pytree."""
    l1 = jax.tree.leaves(eqx.filter(g1, eqx.is_array))
    l2 = jax.tree.leaves(eqx.filter(g2, eqx.is_array))
    dev = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(l1, l2))
    ref = max(float(jnp.max(jnp.abs(b))) for b in l2)
    return dev, ref + 1e-30


def main():
    from pinn.operators import build_D
    from pinn.arch import make_film_pinn, make_pinn
    from pinn.arch.pinn import _PINN
    from pinn.arch.hard_constraint import HardConstraint

    losses = _load('examples/corrosion2d/losses.py', 'c2d_losses')

    # Small but non-trivial 2D setup
    Nx, Ny, Nt, delta = 61, 31, 8, 0.05
    xs = jnp.linspace(0.0, 1.0, Nx)
    ys = jnp.linspace(0.0, 0.5, Ny)
    Xv, Yv = jnp.meshgrid(xs, ys, indexing='xy')
    x_base = jnp.stack([Xv.ravel(), Yv.ravel()], axis=-1)

    rng = np.random.RandomState(0)
    x_bc_pit = jnp.array(rng.uniform(size=(40, 2)) * np.array([1.0, 0.0]))
    x_bc_solid = jnp.array(rng.uniform(size=(30, 2)) * np.array([1.0, 0.5]))
    x_ic = jnp.array(rng.uniform(size=(200, 2)) * np.array([1.0, 0.5]))

    class _Phys:  # matches config.physical fields used by ic_values
        omega_phi, alpha_phi, c_se = 1.76e7, 1.03e-4, 1.0
    phi0, c0 = losses.ic_values(x_ic, _Phys, 100e-6, ((0.5, 0.0, 0.1),))
    nd = (0.425, 1.07e10, 1.76e9, 1.03e-2, 1.0 - 5100.0 / 1.43e5,
          5100.0 / 1.43e5)

    net = make_film_pinn(spatial_dim=2, out_dim=2, hidden_dim=32, num_layers=3,
                         film_hidden_dim=16, film_num_layers=2,
                         activation='gelu', spatial_emb_dim=16,
                         spatial_scale=2.0, temporal_emb_dim=16,
                         temporal_scale=1.0,
                         hard_constraint=HardConstraint(1.0, 5100.0 / 1.43e5),
                         key=jr.PRNGKey(3))
    t_batch = jr.uniform(jr.PRNGKey(7), (Nt,), minval=0.001, maxval=1.0)

    D_sparse = build_D(x_base, delta, method="sparse")
    D_conv = build_D(x_base, delta, method="conv")

    def full_loss_and_grad(D, update_w):
        lfn = losses.make_loss_fn(x_base, D, x_bc_pit, x_bc_solid, nd,
                                  x_ic=x_ic, ac_scale=1e-7, ch_scale=1.0)
        f = eqx.filter_jit(lambda n: lfn(
            n, t_batch, phi0, c0, jnp.array(1.0), jnp.array(True),
            jnp.array(True), jnp.ones(5), jnp.ones(5), update_w,
            jnp.array(0.5), None, x_points=x_base, sub_data=None,
            core_mask=None))
        return f(net)

    print("jax", jax.__version__, "device:", jax.devices()[0].device_kind)
    tol = 2e-5   # fp32 round-off from summation-order changes
    ok = True

    # ---- 1: ConvD vs SparseD through the full loss ----
    for update_w, label in [(False, "regular step"), (True, "update step ")]:
        ref = full_loss_and_grad(D_sparse, update_w)
        got = full_loss_and_grad(D_conv, update_w)
        v_dev = abs(float(got[0] - ref[0])) / (abs(float(ref[0])) + 1e-30)
        g_dev, g_ref = _grad_dev(got[2], ref[2])
        rel = g_dev / g_ref
        w_dev = float(jnp.max(jnp.abs(got[3] - ref[3])
                              / (jnp.abs(ref[3]) + 1e-30)))
        status = "OK" if (v_dev < tol and rel < tol and w_dev < tol) else "FAIL"
        ok &= status == "OK"
        print(f"[{status}] ConvD vs SparseD ({label}) loss rel dev {v_dev:.2e}, "
              f"grad rel dev {rel:.2e}, weights rel dev {w_dev:.2e}")

    # ---- 2: tensor_split net vs concat net ----
    def _compute_all(n, xp, tb):
        def field_at_t(t):
            return jax.vmap(n, in_axes=(0, None))(xp, t)
        return jax.jvp(jax.vmap(field_at_t), (tb,), (jnp.ones_like(tb),))

    for dim, label in [(2, "2D"), (3, "3D")]:
        m = make_pinn(spatial_dim=dim, out_dim=2, hidden_dim=32, num_layers=3,
                      trunk_type="mlp", activation='gelu',
                      spatial_emb_dim=16, spatial_scale=2.0,
                      temporal_emb_dim=16, temporal_scale=1.0,
                      key=jr.PRNGKey(4 + dim))
        m_split = _PINN(m.spatial_emb, m.temporal_emb, m.trunk,
                        m.hard_constraint, m.dirichlet_bc, True)
        xp = jr.uniform(jr.PRNGKey(11 + dim), (500, dim))

        o_a, ot_a = _compute_all(m, xp, t_batch)
        o_b, ot_b = _compute_all(m_split, xp, t_batch)
        dev_o = float(jnp.max(jnp.abs(o_a - o_b))) \
            / (float(jnp.max(jnp.abs(o_a))) + 1e-30)
        dev_t = float(jnp.max(jnp.abs(ot_a - ot_b))) \
            / (float(jnp.max(jnp.abs(ot_a))) + 1e-30)

        def loss(n):
            out, out_t = _compute_all(n, xp, t_batch)
            return jnp.mean(jnp.abs(out_t[..., 0] + out[..., 0] ** 3
                                    - out[..., 1]))

        la, ga = eqx.filter_value_and_grad(loss)(m)
        lb, gb = eqx.filter_value_and_grad(loss)(m_split)
        dev_l = abs(float(la - lb)) / (abs(float(la)) + 1e-30)
        g_dev, g_ref = _grad_dev(gb, ga)
        status = "OK" if max(dev_o, dev_t, dev_l, g_dev / g_ref) < tol else "FAIL"
        ok &= status == "OK"
        print(f"[{status}] tensor_split vs concat ({label}) out {dev_o:.2e}, "
              f"d/dt {dev_t:.2e}, loss {dev_l:.2e}, grad {g_dev/g_ref:.2e}")

    print("\nALL EQUIVALENCE CHECKS PASSED" if ok else "\nFAILURES DETECTED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
