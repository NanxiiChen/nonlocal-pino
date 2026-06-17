"""Loss functions for 1D KKS corrosion PINN — split-PINN version.

Uses the split architecture's batched spatial/temporal API:
  Fx = net.spatial_batch(x_all)   — once per iteration
  Ft = net.temporal(t)             — per time point
  dFt = net.temporal_derivative(t) — per time point
  net.project(Fx, vt) → (N_x, 2)  — dot-product fusion

GradNorm balances PDE and IC loss weights.
"""

import jax.numpy as jnp
import jax
import equinox as eqx
import numpy as np
from jax.flatten_util import ravel_pytree


def _h(phi):
    return -2.0 * phi**3 + 3.0 * phi**2

def _hp(phi):
    return -6.0 * phi**2 + 6.0 * phi

def _gp(phi):
    return 4.0 * phi**3 - 6.0 * phi**2 + 2.0 * phi


# ── IC ──────────────────────────────────────────────────────────────────────

def ic_values(x, phys, L_c, center=0.5):
    k = np.sqrt(phys.omega_phi / (2.0 * phys.alpha_phi)) * L_c
    dist = np.array(x)[:, 0] - center
    phi0 = 0.5 * (1.0 - np.tanh(k * dist))
    c0 = _h(phi0) * phys.c_se
    return jnp.array(phi0), jnp.array(c0)


# ── All-at-once forward ─────────────────────────────────────────────────────

def _compute_all(net, x_all, t_batch):
    """Compute (phi, c, dphi_dt, dc_dt) for all (x, t) in one shot."""
    Fx = net.spatial_batch(x_all)                    # (N_x, 2p)
    Ft = net.temporal_batch(t_batch)                 # (N_t, 2p)
    dFt = jax.vmap(net.temporal_derivative)(t_batch) # (N_t, 2p)
    out = net.combine(Fx, Ft, t_batch)                 # (N_x, N_t, 2)  + bias
    d_dt = net.combine(Fx, dFt)                         # (N_x, N_t, 2)
    db_dt = net.combine_bias_derivative(t_batch)        # (N_t, 2)
    dphi = d_dt[..., 0] + db_dt[None, :, 0]
    dc   = d_dt[..., 1] + db_dt[None, :, 1]
    return out[..., 0], out[..., 1], dphi, dc


# ── BC enforcement (standalone, for IC/eval use) ────────────────────────────

def apply_bc(u, mask_free, bc_values):
    """Set constrained positions to BC values."""
    m = mask_free[:, None] if u.ndim == 2 else mask_free
    v = bc_values[:, None] if u.ndim == 2 else bc_values
    return jnp.where(m, u, v)


# ── Masked MSE ──────────────────────────────────────────────────────────────

def _masked_mse(r, mask):
    """MSE over masked points.  mask: (N,) bool, True = include."""
    m = mask[:, None] if r.ndim == 2 else mask
    return jnp.sum(jnp.where(m, r**2, 0.0)) / jnp.sum(m)


# ── Per-loss functions ──────────────────────────────────────────────────────

def loss_ac(net, x_all, t_batch, D, nondim_params, ac_scale=None):
    """Allen-Cahn residual MSE — all points (horizon truncation is correct physics)."""
    _, N_AC1, N_AC2, N_AC3, dc0, c_Le = nondim_params
    phi, c, dphi_dt, _ = _compute_all(net, x_all, t_batch)
    nl_phi = D @ phi
    r = (dphi_dt + N_AC3 * nl_phi + N_AC2 * _gp(phi)
         - N_AC1 * (c - _h(phi) * dc0 - c_Le) * dc0 * _hp(phi))
    r = ac_scale * r if ac_scale is not None else r
    return jnp.mean(r**2)


def loss_ch(net, x_all, t_batch, D, nondim_params, ch_scale=None):
    """Cahn-Hilliard residual MSE — all points."""
    N_CH, _, _, _, dc0, c_Le = nondim_params
    phi, c, _, dc_dt = _compute_all(net, x_all, t_batch)
    mu = c - _h(phi) * dc0
    nl_mu = D @ mu
    r = dc_dt + N_CH * nl_mu
    r = ch_scale * r if ch_scale is not None else r
    return jnp.mean(r**2)


def loss_ic(net, x_all, phi0, c0):
    """IC loss: MSE at t=0."""
    Fx = net.spatial_batch(x_all)
    field = net.project(Fx, net.temporal(jnp.array(0.0)))
    phi_pred, c_pred = field[:, 0], field[:, 1]
    return jnp.mean((phi_pred - phi0)**2 + (c_pred - c0)**2)


def loss_bc(net, x_all, t_batch, bc_values, constr_idx):
    """Soft BC penalty: MSE at constrained points over all times."""
    phi, c, _, _ = _compute_all(net, x_all, t_batch)
    err_phi = phi[constr_idx, :] - bc_values[constr_idx, None]
    err_c   = c[constr_idx, :] - bc_values[constr_idx, None]
    return jnp.mean(err_phi**2 + err_c**2)


# ── GradNorm weighting ──────────────────────────────────────────────────────

def grad_norm_weights(grads, eps=1e-6, cap=1000):
    norms = jnp.array([jnp.linalg.norm(ravel_pytree(g)[0]) for g in grads])
    norms = jnp.clip(norms, eps, 1.0 / eps)
    weights = jnp.mean(norms) / norms
    weights = jnp.clip(weights, eps, float(cap))
    return jax.lax.stop_gradient(weights)


# ── Combined loss function ──────────────────────────────────────────────────

def make_loss_fn(x_all, D, mask_free, bc_values, nondim_params,
                 alpha=1.0, ac_scale=None, ch_scale=None,
                 gradnorm_cap=1000, weight_mode="gradnorm",
                 fixed_weights=(1.0, 1.0, 1.0, 1.0)):
    """Factory: create total loss (AC, CH, BC, IC) with GradNorm or fixed weights."""
    constr_idx = ~mask_free

    def loss_fn(net, t_batch, phi0, c0, w_ema):
        (ac_val, ac_grad) = eqx.filter_value_and_grad(loss_ac)(
            net, x_all, t_batch, D, nondim_params, ac_scale)
        (ch_val, ch_grad) = eqx.filter_value_and_grad(loss_ch)(
            net, x_all, t_batch, D, nondim_params, ch_scale)
        (bc_val, bc_grad) = eqx.filter_value_and_grad(loss_bc)(
            net, x_all, t_batch, bc_values, constr_idx)
        (ic_val, ic_grad) = eqx.filter_value_and_grad(loss_ic)(
            net, x_all, phi0, c0)

        if weight_mode == "fixed":
            w = jnp.array(fixed_weights, dtype=jnp.float32)
            w = jax.lax.stop_gradient(w)
            w_ac, w_ch, w_bc, w_ic = w
        else:
            w_new = grad_norm_weights([ac_grad, ch_grad, bc_grad, ic_grad],
                                       cap=gradnorm_cap)
            w_ac = alpha * w_new[0] + (1.0 - alpha) * w_ema[0]
            w_ch = alpha * w_new[1] + (1.0 - alpha) * w_ema[1]
            w_bc = alpha * w_new[2] + (1.0 - alpha) * w_ema[2]
            w_ic = alpha * w_new[3] + (1.0 - alpha) * w_ema[3]

        total = w_ac * ac_val + w_ch * ch_val + w_bc * bc_val + w_ic * ic_val
        total_grad = jax.tree.map(
            lambda g1, g2, g3, g4: w_ac * g1 + w_ch * g2 + w_bc * g3 + w_ic * g4,
            ac_grad, ch_grad, bc_grad, ic_grad)

        aux = {
            'loss_ac': ac_val, 'loss_ch': ch_val, 'loss_bc': bc_val, 'loss_ic': ic_val,
            'weight_ac': w_ac, 'weight_ch': w_ch, 'weight_bc': w_bc, 'weight_ic': w_ic,
        }
        return total, aux, total_grad, jnp.array([w_ac, w_ch, w_bc, w_ic])

    return loss_fn


# ── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import jax.random as jr
    from pinn.operators import build_D, condense_dense
    from pinn.arch import make_split_pinn

    key = jr.PRNGKey(0)
    delta = 0.1
    Nx, Nt, rank_p = 101, 20, 10

    x_all = jnp.linspace(0.0, 1.0, Nx)[:, None]
    D = build_D(x_all, delta)
    D_free = condense_dense(D, jnp.ones(Nx, dtype=bool))
    net = make_split_pinn(spatial_dim=1, rank_p=rank_p, hidden_dim=32,
                          num_layers=3, key=key)

    # Nondim params
    D_phys, A, L = 8.5e-10, 5.35e7, 1e-11
    alpha_phi, omega_phi = 1.03e-4, 1.76e7
    c_se, c_le = 1.0, 5100.0 / 1.43e5
    L_c, T_c = 100e-6, 100000.0
    nd = (D_phys * T_c / L_c**2,
          2.0 * A * L * T_c,
          L * omega_phi * T_c,
          L * alpha_phi * T_c / L_c**2,
          c_se - c_le, c_le)

    t_batch = jr.uniform(key, (Nt,), minval=0.0, maxval=1.0)

    # All-at-once forward check
    phi, c, dphi_dt, dc_dt = _compute_all(net, x_all, t_batch)
    assert phi.shape == (Nx, Nt) and dphi_dt.shape == (Nx, Nt)
    print(f"_compute_all: phi {phi.shape}, dphi_dt {dphi_dt.shape}  ✓")

    mask_free = jnp.ones(Nx, bool).at[0].set(False).at[-1].set(False)
    bc_values = jnp.zeros(Nx).at[0].set(1.0)

    ac_raw = loss_ac(net, x_all, t_batch, D_free, nd)
    ch_raw = loss_ch(net, x_all, t_batch, D_free, nd)
    print(f"unscaled: loss_ac={ac_raw:.2e}, loss_ch={ch_raw:.2e}  (CH >> AC)")

    ch_scale = 1.0 / nd[0]
    ac_val = loss_ac(net, x_all, t_batch, D_free, nd)
    ch_val = loss_ch(net, x_all, t_batch, D_free, nd, ch_scale)
    print(f"scaled:   loss_ac={ac_val:.4f}, loss_ch={ch_val:.4f} (ch_scale={ch_scale:.2e})  ✓")

    class Phys: pass
    phys = Phys()
    phys.omega_phi = omega_phi; phys.alpha_phi = alpha_phi
    phys.c_se = c_se; phys.c_le = c_le
    phi0, c0 = ic_values(x_all, phys, L_c)
    ic_val = loss_ic(net, x_all, phi0, c0)
    print(f"loss_ic: {ic_val:.6f}  ✓")

    lfn = make_loss_fn(x_all, D_free, mask_free, bc_values, nd, ch_scale=ch_scale)
    w_ema = jnp.array([1.0, 1.0, 1.0, 1.0])
    w_ema = jnp.array([1.0, 1.0, 1.0])
    total, aux, grad, new_w = lfn(net, t_batch, phi0, c0, w_ema)
    print(f"total: {total:.6f}, "
          f"ac={aux['loss_ac']:.2e} ch={aux['loss_ch']:.1e} ic={aux['loss_ic']:.4f}, "
          f"w=[{aux['weight_ac']:.1f},{aux['weight_ch']:.1f},{aux['weight_ic']:.1f}], "
          f"grad leaves={sum(1 for _ in jax.tree.leaves(grad))}  ✓")

    print("All tests passed.")
