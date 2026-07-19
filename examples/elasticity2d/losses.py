"""Loss functions for 2D linear elasticity PINN — plate with hole.

Two modes (selected by D is not None / D is None):
  nonlocal: E_δ operator → PDE only  (N_LOSS=1, Neumann auto-handled)
  local:    all-AD Navier-Cauchy → PDE + Neumann BCs  (N_LOSS=4)

Hard constraint handles Dirichlet BCs (left clamped, right pulled).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jax.flatten_util import ravel_pytree


# ── Network evaluation (steady-state) ────────────────────────────────────────

def _net_u(net, x):
    """Evaluate net(x, t=0) → (N, 2) displacement."""
    return jax.vmap(net, in_axes=(0, None))(x, jnp.array(0.0))


# ═══════════════════════════════════════════════════════════════════════════════
# PDE residual
# ═══════════════════════════════════════════════════════════════════════════════

def _pde_nonlocal(net, x_points, D, lam, mu):
    """E_δ matvec — the residual itself."""
    u = _net_u(net, x_points)          # (N, 2)
    r = D.elastic_matvec(u, lam, mu)   # (N, 2)
    return r[:, 0], r[:, 1]


def _pde_local(net, x_points, lam, mu):
    """Navier-Cauchy: μ∇²u + (λ+μ)∇(∇·u) via per-point Hessian AD."""
    def _navier_single(x):
        hess_ux = jax.hessian(lambda xx: net(xx, 0.0)[0])(x)
        hess_uy = jax.hessian(lambda xx: net(xx, 0.0)[1])(x)
        lap_ux = jnp.trace(hess_ux)
        lap_uy = jnp.trace(hess_uy)
        gd_x = hess_ux[0, 0] + hess_uy[0, 1]
        gd_y = hess_ux[1, 0] + hess_uy[1, 1]
        r_x = mu * lap_ux + (lam + mu) * gd_x
        r_y = mu * lap_uy + (lam + mu) * gd_y
        return jnp.array([r_x, r_y])
    r = jax.vmap(_navier_single)(x_points)
    return r[:, 0], r[:, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# Loss terms (match electropolish2d API: net first, then data)
# ═══════════════════════════════════════════════════════════════════════════════

def loss_pde(net, x_points, D, lam, mu):
    """PDE residual MAE (L1)."""
    if D is not None:
        r_x, r_y = _pde_nonlocal(net, x_points, D, lam, mu)
    else:
        r_x, r_y = _pde_local(net, x_points, lam, mu)
    return jnp.mean(jnp.abs(r_x) + jnp.abs(r_y))


def _traction_mse(net, x_bc, normals, lam, mu):
    """||σ·n||² averaged over BC points."""
    def _single(x, n):
        J = jax.jacfwd(lambda xx: net(xx, 0.0))(x)
        eps = 0.5 * (J + J.T)
        tr = eps[0, 0] + eps[1, 1]
        sig = lam * tr * jnp.eye(2) + 2.0 * mu * eps
        t = sig @ n
        return jnp.sum(t ** 2)
    return jnp.mean(jax.vmap(_single)(x_bc, normals))


def loss_bc_top(net, x_bc, lam, mu):
    """Top edge (free, y=Ly): σ·ê_y = 0."""
    n = jnp.tile(jnp.array([0.0, 1.0])[None, :], (len(x_bc), 1))
    return _traction_mse(net, x_bc, n, lam, mu)


def loss_bc_bottom(net, x_bc, lam, mu):
    """Bottom edge (free, y=0): -σ·ê_y = 0."""
    n = jnp.tile(jnp.array([0.0, -1.0])[None, :], (len(x_bc), 1))
    return _traction_mse(net, x_bc, n, lam, mu)


def loss_bc_hole(net, x_bc, hole_center, lam, mu):
    """Hole boundary (free): σ·n = 0  (n outward = toward centre)."""
    dx = x_bc - jnp.array(hole_center)[None, :]
    n = -dx / (jnp.linalg.norm(dx, axis=1, keepdims=True) + 1e-10)
    return _traction_mse(net, x_bc, n, lam, mu)


# ═══════════════════════════════════════════════════════════════════════════════
# GradNorm
# ═══════════════════════════════════════════════════════════════════════════════

def grad_norm_weights(grads, eps=1e-6, cap=1000.0):
    """w_i = mean(||g_i||) / ||g_i||, clipped."""
    norms = jnp.array([jnp.sum(jnp.abs(ravel_pytree(g)[0])) for g in grads])
    norms = jnp.clip(norms, eps, 1.0 / eps)
    w = jnp.mean(norms) / norms
    return jax.lax.stop_gradient(jnp.clip(w, 0.0, cap))


# ═══════════════════════════════════════════════════════════════════════════════
# Loss factory (electropolish2d pattern)
# ═══════════════════════════════════════════════════════════════════════════════

def make_loss_fn(x_all, D, lam, mu,
                 x_bc_top=None, x_bc_bottom=None, x_bc_hole=None,
                 hole_center=None,
                 alpha=0.9, gradnorm_cap=1000.0):
    """Factory returning a loss_fn closure.

    Nonlocal (D is not None):  N_LOSS = 1  (PDE only)
    Local    (D is None):      N_LOSS = 4  (PDE + top + bottom + hole BCs)
    """
    local_mode = (D is None)

    # ── Nonlocal path (fast, single-term) ─────────────────────────────────
    if not local_mode:
        def _pde_fn(n):
            return loss_pde(n, x_all, D, lam, mu)

        def loss_fn(net, w_ema, bias, update_w,
                    x_points=None):
            (pde_v, pde_grad) = eqx.filter_value_and_grad(_pde_fn)(net)
            w = jnp.ones(1)
            return pde_v, {'loss/pde': pde_v, 'weight/pde': w[0]}, pde_grad, w
        return loss_fn

    # ── Local path (4-term with GradNorm) ────────────────────────────────
    N_LOSS = 4
    _bc_t = lambda n: loss_bc_top(n, x_bc_top, lam, mu)
    _bc_b = lambda n: loss_bc_bottom(n, x_bc_bottom, lam, mu)
    _bc_h = lambda n: loss_bc_hole(n, x_bc_hole, hole_center, lam, mu)

    def loss_fn(net, w_ema, bias, update_w,
                x_points=None):
        _pde = lambda n: loss_pde(n, x_points, D, lam, mu)

        (pde_v, pde_g) = eqx.filter_value_and_grad(_pde)(net)
        (bc_t_v, bc_t_g) = eqx.filter_value_and_grad(_bc_t)(net)
        (bc_b_v, bc_b_g) = eqx.filter_value_and_grad(_bc_b)(net)
        (bc_h_v, bc_h_g) = eqx.filter_value_and_grad(_bc_h)(net)

        vals = [pde_v, bc_t_v, bc_b_v, bc_h_v]
        grads = [pde_g, bc_t_g, bc_b_g, bc_h_g]

        if update_w:
            w_new = grad_norm_weights(grads, cap=gradnorm_cap)
            w = [alpha * w_new[i] + (1.0 - alpha) * w_ema[i]
                 for i in range(N_LOSS)]
            w = [w[i] * bias[i] for i in range(N_LOSS)]
        else:
            w = [w_ema[i] * bias[i] for i in range(N_LOSS)]

        total = sum(w[i] * vals[i] for i in range(N_LOSS))
        total_grad = jax.tree.map(
            lambda *gs: sum(w[i] * gs[i] for i in range(N_LOSS)), *grads)
        new_w = jnp.array(w if update_w else w_ema)

        aux = {
            'loss/pde': vals[0], 'weight/pde': w[0],
            'loss/bc_top': vals[1], 'weight/bc_top': w[1],
            'loss/bc_bottom': vals[2], 'weight/bc_bottom': w[2],
            'loss/bc_hole': vals[3], 'weight/bc_hole': w[3],
        }
        return total, aux, total_grad, new_w

    return loss_fn
