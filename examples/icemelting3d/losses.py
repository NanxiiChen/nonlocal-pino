"""Loss functions for 3D ice melting PINN — single-field Allen-Cahn.

PDE residual (nonlocal):
  r = ∂φ/∂t* + N1·D_δ φ + N2·(φ³ - φ) + N3·(1 - φ²)

PDE residual (local, autograd Laplacian):
  r = ∂φ/∂t* - N1·∇²φ + N2·(φ³ - φ) + N3·(1 - φ²)

Raw mode: PDE (MAE) + IC (MSE). Hard-IC modes: PDE only, no redundant IC loss.
The ic_latent mode uses the exact tanh-transformed residual (same PDE zeros).
Natural BCs only — no BC losses.
Nonlocal mode uses sub-graph for PDE (3D grid too large for full D in JIT).
Local mode uses AD Hessian + LHS random sampling — no grid or D operator needed.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from examples.icemelting3d.phase import FORMULATIONS, InitialPhase


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


def _compute_all_local(net, x_points, t_batch):
    """Evaluate net on spatial points for all t, with AD Laplacian.

    Same tensor-product structure as _compute_all, but ∇²φ is computed
    via jax.hessian instead of the D_δ operator.

    Args:
        x_points: (N_x, 3)  spatial sample points
        t_batch:  (N_t,)    time points
    Returns:
        phi:       (N_x, N_t)  phase field
        dphi_dt:   (N_x, N_t)  time derivative (JVP)
        laplacian: (N_x, N_t)  ∇²φ via Hessian AD
    """
    # φ and ∂φ/∂t (same tensor-product pattern as _compute_all)
    def field_at_t(t):
        return jax.vmap(net, in_axes=(0, None))(x_points, t)  # (N_x, 1)

    def field_at_times(ts):
        return jax.vmap(field_at_t)(ts)  # (N_t, N_x, 1)

    out, out_t = jax.jvp(field_at_times, (t_batch,), (jnp.ones_like(t_batch),))
    phi = jnp.swapaxes(out[..., 0], 0, 1)
    dphi_dt = jnp.swapaxes(out_t[..., 0], 0, 1)

    # Spatial Laplacian per (x, t): ∇²φ = tr(Hessian_x(net(x, t)))
    def _lap_single(x, t):
        f = lambda xx: net(xx, t)[0]
        return jnp.trace(jax.hessian(f)(x))

    # vmap over x (inner), vmap over t (outer)
    lap_xt = jax.vmap(
        jax.vmap(_lap_single, in_axes=(0, None)),
        in_axes=(None, 0))(x_points, t_batch)  # (N_t, N_x)
    laplacian = jnp.swapaxes(lap_xt, 0, 1)       # (N_x, N_t)

    return phi, dphi_dt, laplacian


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


def _compute_pde_local(net, x_points, t_batch, N1, N2, N3):
    """PDE residual using autograd Laplacian (no D operator)."""
    phi, dphi_dt, laplacian = _compute_all_local(net, x_points, t_batch)
    # D_δ ≈ -∇², so N1·D_δ φ ≈ -N1·∇²φ
    r = dphi_dt - N1 * laplacian + N2 * (phi**3 - phi) + N3 * (1.0 - phi**2)
    return r, phi


def _latent_all(net, x_points, t_batch):
    def values(ts):
        return jax.vmap(lambda t: jax.vmap(net.latent, in_axes=(0, None))(x_points, t))(ts)
    u, u_t = jax.jvp(values, (t_batch,), (jnp.ones_like(t_batch),))
    return u[..., 0].T, u_t[..., 0].T


def _tanh_difference_ratio(ui, uj):
    """(tanh(ui)-tanh(uj))/sech(ui)^2, without saturated subtraction/division."""
    log_ratio = jnp.logaddexp(ui, -ui) - jnp.logaddexp(uj, -uj)
    return jnp.sinh(ui - uj) * jnp.exp(log_ratio)


def _compute_pde_latent(net, x_points, t_batch, sub_data, N1, N2, N3):
    u, u_t = _latent_all(net, x_points, t_batch)
    phi = jnp.tanh(u)
    if sub_data is not None:
        row, col, vals, _ = sub_data
        # D annihilates constants: D(phi)_i=sum_j (-D_ij)*(phi_i-phi_j).
        ratio = _tanh_difference_ratio(u[row], u[col])
        diffusion = jax.ops.segment_sum(-vals[:, None] * ratio, row, num_segments=len(x_points))
    else:
        # Delta(tanh u)/sech(u)^2 = Delta u - 2*tanh(u)*|grad u|^2.
        def spatial(x, t):
            f = lambda xx: net.latent(xx, t)[0]
            grad = jax.grad(f)(x)
            return jnp.trace(jax.hessian(f)(x)), jnp.sum(grad * grad)
        lap, grad2 = jax.vmap(jax.vmap(spatial, in_axes=(0, None)), in_axes=(None, 0))(x_points, t_batch)
        diffusion = -lap.T + 2 * phi * grad2.T
    return u_t + N1 * diffusion - N2 * phi + N3, phi


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
             iface_gamma=0.0, causal=None, eps=None, *, formulation="raw_phase"):
    """MAE of PDE residual.

    sub_data is not None → nonlocal D_δ operator.
    sub_data is None     → local autograd Laplacian.
    causal triggers per-t weighting.
    """
    if formulation not in FORMULATIONS:
        raise ValueError(f"Invalid formulation: {formulation!r}")
    if formulation == "ic_latent":
        r, phi = _compute_pde_latent(net, x_points, t_batch, sub_data, N1, N2, N3)
    elif sub_data is not None:
        r, phi = _compute_pde(net, x_points, t_batch, sub_data, N1, N2, N3)
    else:
        r, phi = _compute_pde_local(net, x_points, t_batch, N1, N2, N3)
    iw = _iface_weight(phi, iface_gamma)
    if causal is not None and eps is not None:
        if core_mask is not None:
            mask_f = core_mask.astype(r.dtype)[:, None]
        else:
            mask_f = jnp.ones_like(r)
        val, w, chunks = causal.weighted_loss(r * iw * mask_f, t_batch, eps)
        return val, (w, chunks)
    if core_mask is not None:
        return _masked_mean(r * iw, core_mask), (None, None)
    else:
        return jnp.mean(jnp.abs(r * iw)), (None, None)


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
                 alpha=1.0, gradnorm_cap=1000.0, causal=None, *, formulation="raw_phase"):
    """Factory: returns a loss_fn closure for the training loop.

    Raw mode has PDE + soft IC; IC-only modes use a single unweighted PDE loss.
    Nonlocal mode: always uses sub-graph (3D grid too large for full D in JIT).
    Local mode: sub_data=None triggers autograd Laplacian path.
    """
    if formulation not in FORMULATIONS:
        raise ValueError(f"Invalid formulation: {formulation!r}")
    if formulation == "ic_latent" and D is not None:
        row = jax.device_get(D._row_idx)
        vals = jax.device_get(D._vals)
        diag = jax.device_get(D._diag)
        import numpy as np
        sums = np.bincount(row, weights=vals, minlength=len(diag))
        if np.any(vals > 0) or not np.allclose(diag, -sums, rtol=2e-6, atol=1e-6):
            raise ValueError("Transformed graph residual requires nonpositive off-diagonal weights and zero row sums")
    N_LOSS = 2
    _x_ic = x_ic if x_ic is not None else x_grid

    def loss_fn(net, t_batch, phi0, w_ema, bias,
                update_w, iface_gamma, eps, x_points, sub_data, core_mask):
        if formulation != "raw_phase" and not isinstance(net, InitialPhase):
            raise ValueError("Hard-IC losses require an InitialPhase model")
        _pde = lambda n: loss_pde(
            n, x_points, t_batch, sub_data, core_mask, N1, N2, N3,
            iface_gamma=iface_gamma, causal=causal, eps=eps, formulation=formulation)

        # Separate per-loss gradients: XLA compiles smaller backward graphs
        # for PDE and IC independently, ~1.5× faster with tensor_split
        # (benchmarks/TENSOR_SPLIT_RTX_PRO_6000.md).
        (pde_v, c_aux), pde_grad = eqx.filter_value_and_grad(
            _pde, has_aux=True)(net)
        if formulation != "raw_phase":
            zero = jnp.zeros_like(pde_v)
            weights = jnp.asarray([1., 0.], dtype=pde_v.dtype)
            aux = {'loss/pde': pde_v, 'loss/ic': zero,
                   'weight/pde': weights[0], 'weight/ic': weights[1],
                   'causal/w': c_aux[0], 'causal/chunks': c_aux[1]}
            return pde_v, aux, pde_grad, weights
        (ic_v, ic_grad) = eqx.filter_value_and_grad(loss_ic)(
            net, _x_ic, phi0)
        grads = [pde_grad, ic_grad]
        vals = [pde_v, ic_v]

        if update_w:
            w_new = grad_norm_weights(grads, cap=gradnorm_cap)
            w = [alpha * w_new[i] + (1.0 - alpha) * w_ema[i]
                 for i in range(N_LOSS)]
            w = [w[i] * bias[i] for i in range(N_LOSS)]
        else:
            w = [w_ema[i] * bias[i] for i in range(N_LOSS)]

        total = w[0] * vals[0] + w[1] * vals[1]
        total_grad = jax.tree.map(
            lambda *gs: w[0] * gs[0] + w[1] * gs[1], *grads)
        new_w = jnp.array(w) if update_w else jnp.array(w)

        if c_aux is not None:
            causal_w, causal_chunks = c_aux
        else:
            causal_w = causal_chunks = None

        aux = {'loss/pde': pde_v, 'loss/ic': ic_v,
               'weight/pde': new_w[0], 'weight/ic': new_w[1],
               'causal/w': causal_w, 'causal/chunks': causal_chunks}
        return total, aux, total_grad, new_w

    return loss_fn
