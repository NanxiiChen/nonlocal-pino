"""Loss functions for 3D ice melting PINN — single-field Allen-Cahn.

PDE residual:
  r = ∂φ/∂t* + N1·D_δ φ + N2·(φ³ - φ) + N3·(1 - φ²)

Only two loss terms: PDE (MAE) + IC (MSE). Natural BCs only — no BC losses.
Always uses sub-graph for PDE (3D grid too large for full D in JIT).
"""

import jax
import jax.numpy as jnp
import equinox as eqx


# ── Analytical solution ────────────────────────────────────────────────────

def analytic_solution(x, t, R0_star, lam_star, xc, denom):
    """φ = tanh((R0* - λ*·t - |x - xc|) / denom)."""
    r = jnp.sqrt(jnp.sum((x - jnp.array(xc))**2, axis=-1))
    return jnp.tanh((R0_star - lam_star * t - r) / denom)


def ic_values(x_all, R0_star, xc, denom):
    """IC at t=0: tanh profile (lam_star irrelevant since t=0)."""
    return analytic_solution(x_all, 0.0, R0_star, 0.0, xc, denom)


# ── Network evaluation ─────────────────────────────────────────────────────

def _compute_all(net, x_points, t_batch):
    """Evaluate net on points for all t, compute time derivatives via JVP.

    Returns:
        phi:     (N_x, N_t)  phase field
        dphi_dt: (N_x, N_t)  time derivative
    """
    def field_at_t(t):
        return jax.vmap(net, in_axes=(0, None))(x_points, t)  # (N_x, 1)

    def field_at_times(ts):
        return jax.vmap(field_at_t)(ts)  # (N_t, N_x, 1)

    out, out_t = jax.jvp(field_at_times, (t_batch,), (jnp.ones_like(t_batch),))
    phi = jnp.swapaxes(out[..., 0], 0, 1)       # (N_x, N_t)
    dphi_dt = jnp.swapaxes(out_t[..., 0], 0, 1)  # (N_x, N_t)
    return phi, dphi_dt


# ── PDE residual ───────────────────────────────────────────────────────────

def _compute_pde(net, x_points, t_batch, sub_data, N1, N2, N3):
    """PDE residual using sub-graph segment_sum D operator."""
    sub_row, sub_col, sub_vals, sub_diag = sub_data
    phi, dphi_dt = _compute_all(net, x_points, t_batch)

    def _D(u):
        contrib = jax.ops.segment_sum(
            sub_vals[:, None] * u[sub_col, :],
            sub_row, num_segments=u.shape[0])
        return sub_diag[:, None] * u + contrib

    D_phi = _D(phi)
    r = dphi_dt + N1 * D_phi + N2 * (phi**3 - phi) + N3 * (1.0 - phi**2)
    return r, phi


# ── Interface weight ───────────────────────────────────────────────────────

def _iface_weight(phi, gamma):
    """Interface indicator for φ∈(-1,1): w = 1 + γ·(1-φ²). Peaks at φ=0."""
    phi_sg = jax.lax.stop_gradient(phi)
    return 1.0 + gamma * (1.0 - phi_sg**2)


# ── Masked mean ────────────────────────────────────────────────────────────

def _masked_mean(r, core_mask):
    """Mean of |r| over core points × all time steps."""
    mask_f = core_mask.astype(r.dtype)[:, None]
    n_valid = jnp.sum(mask_f) * r.shape[1]
    return jnp.sum(jnp.abs(r) * mask_f) / jnp.maximum(n_valid, 1.0)


# ── Loss terms ─────────────────────────────────────────────────────────────

def loss_pde(net, x_points, t_batch, sub_data, core_mask, N1, N2, N3,
             iface_gamma=0.0, causal=None, eps=None):
    """MAE of PDE residual, masked to core points.  causal triggers per-t weighting."""
    r, phi = _compute_pde(net, x_points, t_batch, sub_data, N1, N2, N3)
    iw = _iface_weight(phi, iface_gamma)
    if causal is not None and eps is not None:
        mask_f = core_mask.astype(r.dtype)[:, None]
        val, w, chunks = causal.weighted_loss(r * iw * mask_f, t_batch, eps)
        return val, (w, chunks)
    return _masked_mean(r * iw, core_mask), (None, None)


def loss_ic(net, x_points, phi0):
    """MSE of predicted vs analytical IC at t=0."""
    phi_pred = jax.vmap(net, in_axes=(0, None))(x_points, jnp.array(0.0))
    return jnp.mean((phi_pred[:, 0] - phi0)**2)


# ── GradNorm ───────────────────────────────────────────────────────────────

def grad_norm_weights(grads, cap=1000.0):
    """GradNorm: w_i = mean(||g_i||) / ||g_i||, clipped to [0, cap]."""
    norms = jnp.array([jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(g)))
                       for g in grads])
    mean_norm = jnp.mean(norms)
    w = jnp.where(norms > 1e-12, mean_norm / norms, 1.0)
    return jnp.clip(w, 0.0, cap)


# ── Loss factory ───────────────────────────────────────────────────────────

def make_loss_fn(x_grid, D, N1, N2, N3, x_ic=None,
                 alpha=1.0, gradnorm_cap=1000.0, causal=None):
    """Factory: returns a loss_fn closure for the training loop.

    Only 2 loss terms: PDE + IC.  Always uses sub-graph (3D grid).
    """
    N_LOSS = 2
    _x_ic = x_ic if x_ic is not None else x_grid

    def loss_fn(net, t_batch, phi0, w_ema, bias,
                update_w, iface_gamma, eps, x_points, sub_data, core_mask):
        _pde = lambda n: loss_pde(
            n, x_points, t_batch, sub_data, core_mask, N1, N2, N3,
            iface_gamma=iface_gamma, causal=causal, eps=eps)

        if update_w:
            (pde_v, c_aux), pde_grad = eqx.filter_value_and_grad(
                _pde, has_aux=True)(net)
            (ic_v, ic_grad) = eqx.filter_value_and_grad(loss_ic)(
                net, _x_ic, phi0)
            grads = [pde_grad, ic_grad]
            vals = [pde_v, ic_v]
            w_new = grad_norm_weights(grads, cap=gradnorm_cap)
            w = [alpha * w_new[i] + (1.0 - alpha) * w_ema[i]
                 for i in range(N_LOSS)]
            w = [w[i] * bias[i] for i in range(N_LOSS)]
            total = w[0] * vals[0] + w[1] * vals[1]
            total_grad = jax.tree.map(
                lambda *gs: w[0] * gs[0] + w[1] * gs[1], *grads)
            new_w = jnp.array(w)
        else:
            w = [w_ema[i] * bias[i] for i in range(N_LOSS)]

            def total_fn(n):
                pde_v, c_aux = _pde(n)
                ic_v = loss_ic(n, _x_ic, phi0)
                return w[0] * pde_v + w[1] * ic_v, (pde_v, ic_v, c_aux)

            (total, (pde_v, ic_v, c_aux)), total_grad = \
                eqx.filter_value_and_grad(total_fn, has_aux=True)(net)
            new_w = jnp.array(w)

        if c_aux is not None:
            causal_w, causal_chunks = c_aux
        else:
            causal_w = causal_chunks = None

        aux = {'loss_pde': pde_v, 'loss_ic': ic_v,
               'weight_pde': new_w[0], 'weight_ic': new_w[1],
               'causal_w': causal_w, 'causal_chunks': causal_chunks}
        return total, aux, total_grad, new_w

    return loss_fn
