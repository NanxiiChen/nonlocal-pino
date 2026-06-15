"""Loss functions for 1D KKS phase-field corrosion PINN (non-local, GradNorm).

Single-time-slice residual design: each function operates on one t, with
spatial batching handled internally by vmap over x.  Time batching via
an outer vmap over t.  GradNorm balances PDE and IC losses automatically.
"""

import jax.numpy as jnp
import jax
import equinox as eqx
import numpy as np
from jax.flatten_util import ravel_pytree


# ── helpers ─────────────────────────────────────────────────────────────────

def _h(phi):
    return -2.0 * phi**3 + 3.0 * phi**2


def _hp(phi):
    return -6.0 * phi**2 + 6.0 * phi


def _gp(phi):
    return 4.0 * phi**3 - 6.0 * phi**2 + 2.0 * phi


# ── IC ──────────────────────────────────────────────────────────────────────

def ic_values(x, phys, L_c):
    """Analytical tanh IC for the 1D KKS corrosion problem.

    φ₀(x*) = 0.5·(1 − tanh(k·(x* − 0.5)))
    c₀(x*) = h(φ₀)·c_Se + (1 − h(φ₀))·c_Le
    """
    k = np.sqrt(phys.omega_phi / (2.0 * phys.alpha_phi)) * L_c
    dist = np.array(x)[:, 0] - 0.5
    phi0 = 0.5 * (1.0 - np.tanh(k * dist))
    c0 = _h(phi0) * phys.c_se + (1.0 - _h(phi0)) * phys.c_le
    return jnp.array(phi0), jnp.array(c0)


# ── PDE residual ────────────────────────────────────────────────────────────

def residual_at_time(net, t, x_all, D, nondim_params):
    """Non-local KKS PDE residuals at all spatial points for a single time t.

    Args:
        net: (x, t) → (φ, c), single-sample eqx.Module.
        t: scalar time.
        x_all: (N, dim) spatial points.
        D: L_delta operator (N, N).
        nondim_params: (N_CH, N_AC1, N_AC2, N_AC3, dc0, c_Le).

    Returns:
        r_phi, r_c: (N,) raw residuals at each spatial point.
    """
    N_CH, N_AC1, N_AC2, N_AC3, dc0, c_Le = nondim_params

    # Forward over all x at this t (vmap over x)
    out = jax.vmap(net, in_axes=(0, None))(x_all, t)   # (N, 2)
    phi, c = out[:, 0], out[:, 1]                       # (N,), (N,)

    # Non-local operators (matvec on full spatial slice)
    nl_phi = D @ phi
    mu = c - _h(phi) * dc0 - c_Le
    nl_mu = D @ mu

    # Time derivatives at each x (vmap over x, grad w.r.t. t)
    dphi_dt = jax.vmap(
        jax.grad(lambda x_, t_: net(x_, t_)[0], argnums=1),
        in_axes=(0, None))(x_all, t)
    dc_dt = jax.vmap(
        jax.grad(lambda x_, t_: net(x_, t_)[1], argnums=1),
        in_axes=(0, None))(x_all, t)

    # Allen-Cahn residual
    r_phi = (dphi_dt
             + N_AC3 * nl_phi
             + N_AC2 * _gp(phi)
             - N_AC1 * (c - _h(phi) * dc0 - c_Le) * dc0 * _hp(phi))

    # Cahn-Hilliard residual
    r_c = dc_dt + N_CH * nl_mu

    return r_phi, r_c


# ── Per-loss functions (for separate gradient computation) ──────────────────

def loss_pde(net, x_all, t_batch, D, nondim_params):
    """PDE loss: mean squared residual over all (x, t)."""
    def mse_at_t(t):
        r_phi, r_c = residual_at_time(net, t, x_all, D, nondim_params)
        return jnp.mean(r_phi**2 + r_c**2)
    return jnp.mean(jax.vmap(mse_at_t)(t_batch))


def loss_ic(net, x_free, phi0, c0):
    """IC loss: MSE of network output vs initial condition at t=0."""
    out = jax.vmap(net, in_axes=(0, None))(x_free, jnp.array(0.0))  # (N, 2)
    phi_pred, c_pred = out[:, 0], out[:, 1]
    return jnp.mean((phi_pred - phi0)**2 + (c_pred - c0)**2)


# ── GradNorm weighting ──────────────────────────────────────────────────────

def grad_norm_weights(grads, eps=1e-6):
    """Compute GradNorm weights from list of gradient PyTrees.

    weight_i = mean(||g_j||) / ||g_i||  (inverse balancing).
    """
    norms = jnp.array([jnp.linalg.norm(ravel_pytree(g)[0]) for g in grads])
    norms = jnp.clip(norms, eps, 1.0 / eps)
    weights = jnp.mean(norms) / norms
    weights = jnp.clip(weights, eps, 1.0 / eps)
    return jax.lax.stop_gradient(weights)


# ── Combined loss function ──────────────────────────────────────────────────

def make_loss_fn(x_all, D, mask_free, nondim_params, alpha=1.0):
    """Factory: create the total loss function with GradNorm weighting.

    Returns loss_fn(net, t_batch, phi0, c0, w_ema) →
        (total_loss, aux, total_grad, new_w_ema).

    Use in a training loop by passing the returned total_grad to optax.

    Args:
        x_all: (N, dim) all spatial points (free + constrained).
        D: L_delta operator.
        mask_free: bool array of length N.
        nondim_params: (N_CH, N_AC1, N_AC2, N_AC3, dc0, c_Le).
        alpha: EMA factor for GradNorm weights (1.0 = no EMA).

    Returns:
        loss_fn(net, t_batch, phi0, c0, w_ema) →
            (total_loss, aux, total_grad, new_w_ema)
    """
    x_free = x_all[mask_free]

    def loss_fn(net, t_batch, phi0, c0, w_ema):
        # PDE loss + gradient
        (pde_val, pde_grad) = eqx.filter_value_and_grad(loss_pde)(
            net, x_all, t_batch, D, nondim_params)

        # IC loss + gradient
        (ic_val, ic_grad) = eqx.filter_value_and_grad(loss_ic)(
            net, x_free, phi0, c0)

        # GradNorm weights
        w_new = grad_norm_weights([pde_grad, ic_grad])
        w_pde = alpha * w_new[0] + (1.0 - alpha) * w_ema[0]
        w_ic  = alpha * w_new[1] + (1.0 - alpha) * w_ema[1]

        # Weighted total
        total = w_pde * pde_val + w_ic * ic_val

        # Weighted total gradient
        total_grad = jax.tree.map(
            lambda g0, g1: w_pde * g0 + w_ic * g1,
            pde_grad, ic_grad)

        aux = {
            'loss_pde': pde_val,
            'loss_ic': ic_val,
            'weight_pde': w_pde,
            'weight_ic': w_ic,
        }
        return total, aux, total_grad, jnp.array([w_pde, w_ic])

    return loss_fn


if __name__ == "__main__":
    # Quick smoke test: residual_at_time with a dummy net
    import jax.random as jr
    from pinn.operators import build_D, condense_dense
    from pinn.arch import make_pinn

    key = jr.PRNGKey(0)
    delta = 0.1
    Nx = 101
    Nt = 20

    x_all = jnp.linspace(0.0, 1.0, Nx)[:, None]
    D = build_D(x_all, delta)
    D_free = condense_dense(D, jnp.ones(Nx, dtype=bool))

    net = make_pinn(spatial_dim=1, out_dim=2, hidden_dim=32, num_layers=3, key=key)

    # Nondim params from FEM config
    phys = type('Phys', (), {})()
    phys.D = 8.5e-10; phys.A = 5.35e7; phys.L = 1e-11
    phys.alpha_phi = 1.03e-4; phys.omega_phi = 1.76e7
    phys.c_se = 1.0; phys.c_le = 5100.0 / 1.43e5
    L_c = 100e-6; T_c = 100000.0
    nd = (phys.D * T_c / L_c**2,
          2.0 * phys.A * phys.L * T_c,
          phys.L * phys.omega_phi * T_c,
          phys.L * phys.alpha_phi * T_c / L_c**2,
          phys.c_se - phys.c_le,
          phys.c_le)

    t_batch = jr.uniform(key, (Nt,), minval=0.0, maxval=1.0)

    # Test residual_at_time
    r_phi, r_c = residual_at_time(net, jnp.array(0.5), x_all, D_free, nd)
    assert r_phi.shape == (Nx,) and r_c.shape == (Nx,)
    print(f"residual_at_time: r_phi {r_phi.shape}, r_c {r_c.shape}  ✓")

    # Test loss_pde
    pde_val = loss_pde(net, x_all, t_batch, D_free, nd)
    print(f"loss_pde: {pde_val:.6f}  ✓")

    # Test loss_ic
    phi0, c0 = ic_values(x_all, phys, L_c)
    ic_val = loss_ic(net, x_all, phi0, c0)
    print(f"loss_ic: {ic_val:.6f}  ✓")

    # Test make_loss_fn
    lfn = make_loss_fn(x_all, D_free, jnp.ones(Nx, dtype=bool), nd)
    w_ema = jnp.array([1.0, 1.0])
    total, aux, grad, new_w = lfn(net, t_batch, phi0, c0, w_ema)
    print(f"total_loss: {total:.6f}, grad has {sum(1 for _ in jax.tree.leaves(grad))} leaves  ✓")
    print("All tests passed.")
