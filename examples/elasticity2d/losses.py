"""Loss functions for 2D linear elasticity PINN — plate with hole.

PDE residual (fully nonlocal):
  (E_δ u)_i = Σ_j w_ij · K(r̂_ij) · (u_i - u_j)

where K(r̂) = μ·I + (λ+μ)·(r̂ ⊗ r̂).

Hard constraint handles BCs (left clamped, right pulled).
Free edges auto-handled by E_δ horizon truncation (same mechanism as D_δ).
→ Only PDE loss, no BC loss terms, no AD in residual.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jax.flatten_util import ravel_pytree


# ── Network evaluation (steady-state) ────────────────────────────────────────

def _net_u(net, x):
    """Evaluate net(x, t=0) → (N, 2) displacement."""
    return jax.vmap(net, in_axes=(0, None))(x, jnp.array(0.0))


# ── PDE residual ─────────────────────────────────────────────────────────────

def pde_residual(net, x_points, D, lam, mu):
    """Fully nonlocal Navier-Cauchy residual via E_δ.

    No AD — the nonlocal operator handles the entire PDE.

    Args:
        net: callable (x, t) -> (2,)
        x_points: (N, 2) spatial coordinates
        D: ElasticD operator
        lam, mu: non-dimensional Lamé constants

    Returns:
        r_x, r_y: (N,) residual components (→ 0 at equilibrium)
    """
    u = _net_u(net, x_points)          # (N, 2)
    r = D.elastic_matvec(u, lam, mu)   # (N, 2) — the residual itself
    return r[:, 0], r[:, 1]


# ── Loss term ────────────────────────────────────────────────────────────────

def loss_pde(net, x_all, D, lam, mu, batch_idx=None):
    """PDE residual MAE (L1) over spatial points.

    Residual is always computed on ALL points (the operator needs the
    full neighbour graph).  Loss is evaluated on *batch_idx* subset if
    provided.
    """
    r_x, r_y = pde_residual(net, x_all, D, lam, mu)
    if batch_idx is not None:
        r_x = r_x[batch_idx]
        r_y = r_y[batch_idx]
    return jnp.mean(jnp.abs(r_x) + jnp.abs(r_y))


# ── GradNorm ─────────────────────────────────────────────────────────────────

def grad_norm_weights(grads, eps=1e-6, cap=1000.0):
    """GradNorm: w_i = mean(||g_i||) / ||g_i||, clipped to [0, cap]."""
    norms = jnp.array([jnp.sum(jnp.abs(ravel_pytree(g)[0])) for g in grads])
    norms = jnp.clip(norms, eps, 1.0 / eps)
    w = jnp.mean(norms) / norms
    return jax.lax.stop_gradient(jnp.clip(w, 0.0, cap))


# ── Loss factory ─────────────────────────────────────────────────────────────

def make_loss_fn(x_all, D, lam, mu,
                 gradnorm_cap=1000.0):
    """Factory returning a loss_fn for the training loop.

    Single loss term (PDE only).  GradNorm trivial but kept for API
    consistency with other examples.

    Returns:
        loss_fn(net, w_ema, bias, update_w, x_points, sub_data, core_mask)
            -> (total, aux, total_grad, new_w)
    """
    N_LOSS = 1  # PDE only

    def loss_fn(net, w_ema, bias, update_w,
                x_points=None, batch_idx=None):
        (val, _), grad = eqx.filter_value_and_grad(
            lambda n: (loss_pde(n, x_all, D, lam, mu,
                                batch_idx=batch_idx), None),
            has_aux=True,
        )(net)

        # Single loss: weight = 1.0 (GradNorm has no effect on 1 term)
        w = jnp.ones(1)
        total = val
        total_grad = grad

        aux = {'loss/pde': val, 'weight/pde': w[0]}
        return total, aux, total_grad, w

    return loss_fn
