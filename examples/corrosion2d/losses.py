"""Loss functions for full MLP PINN on 2D KKS phase-field corrosion."""

import jax.numpy as jnp
import jax
import equinox as eqx
import numpy as np
from jax.flatten_util import ravel_pytree


def _h(phi): return -2.0 * phi**3 + 3.0 * phi**2
def _hp(phi): return -6.0 * phi**2 + 6.0 * phi
def _gp(phi): return 4.0 * phi**3 - 6.0 * phi**2 + 2.0 * phi


def ic_values(x_all, phys, L_c, pits):
    """2D pit IC: φ=0 inside pit, φ=1 in solid, c = h(φ)·c_se.

    Interface width uses physical k = sqrt(ω_φ / (2·α_φ)) · L_c.
    """
    k = np.sqrt(phys.omega_phi / (2.0 * phys.alpha_phi)) * L_c
    x_np = np.array(x_all)
    phi0 = np.ones(len(x_all))
    for cx, cy, R in pits:
        dist = np.sqrt((x_np[:, 0] - cx)**2 + (x_np[:, 1] - cy)**2)
        phi_pit = 0.5 * (1.0 + np.tanh(k * (dist - R)))
        phi0 = np.minimum(phi0, phi_pit)
    c0 = _h(phi0) * phys.c_se
    return jnp.array(phi0), jnp.array(c0)


def _compute_all(net, x_points, t_batch):
    def field_at_t(t):
        return jax.vmap(net, in_axes=(0, None))(x_points, t)  # (N_x, 2)
    def field_at_times(ts):
        return jax.vmap(field_at_t)(ts)  # (N_t, N_x, 2)
    out, out_t = jax.jvp(field_at_times, (t_batch,), (jnp.ones_like(t_batch),))
    out = jnp.swapaxes(out, 0, 1)      # (N_x, N_t, 2)
    out_t = jnp.swapaxes(out_t, 0, 1)  # (N_x, N_t, 2)
    phi, c = out[..., 0], out[..., 1]
    dphi_dt, dc_dt = out_t[..., 0], out_t[..., 1]
    return phi, c, dphi_dt, dc_dt


def _compute_pde(net, x_points, t_batch, D, nd, *, ac_scale=None, ch_scale=None,
                 sub_data=None):
    """AC+CH residuals from a single _compute_all call.

    sub_data=None  → full-grid,  D @ u  matvec.
    sub_data=(row, col, vals, diag)  → sub-graph,  segment_sum matvec.
    """
    N_CH, N_AC1, N_AC2, N_AC3, dc0, c_Le = nd

    if sub_data is not None:
        sub_row, sub_col, sub_vals, sub_diag = sub_data
        phi, c, dphi_dt, dc_dt = _compute_all(net, x_points, t_batch)

        def _D(u):
            contrib = jax.ops.segment_sum(sub_vals[:, None] * u[sub_col, :],
                                           sub_row, num_segments=u.shape[0])
            return sub_diag[:, None] * u + contrib
    else:
        phi, c, dphi_dt, dc_dt = _compute_all(net, x_points, t_batch)

        def _D(u):
            return D @ u

    D_phi = _D(phi)
    r_ac = (dphi_dt + N_AC3 * D_phi + N_AC2 * _gp(phi)
            - N_AC1 * (c - _h(phi) * dc0 - c_Le) * dc0 * _hp(phi))
    if ac_scale is not None:
        r_ac = ac_scale * r_ac

    mu = c - _h(phi) * dc0 - c_Le
    r_ch = dc_dt + N_CH * _D(mu)
    if ch_scale is not None:
        r_ch = ch_scale * r_ch

    return r_ac, r_ch, phi


def _masked_mean(r, mask):
    """mean(|r|) over masked entries. mask: (K',), r: (K', N_t)."""
    mask_f = mask.astype(r.dtype)[:, None]
    n_valid = jnp.sum(mask_f) * r.shape[1]
    return jnp.sum(jnp.abs(r) * mask_f) / jnp.maximum(n_valid, 1.0)


def _iface_weight(phi, gamma):
    phi_sg = jax.lax.stop_gradient(phi)
    phi_c = jnp.clip(phi_sg, 0.0, 1.0)
    return 1.0 + gamma * 4.0 * phi_c * (1.0 - phi_c)


# ── Unified loss functions ─────────────────────────────────────────────────

def loss_ac(net, x_points, t_batch, D, nd, ac_scale=None, ch_scale=None,
            causal=None, eps=None, t_weights=None, iface_gamma=0.0,
            sub_data=None, core_mask=None):
    """AC scalar loss.  core_mask triggers sub-graph masking."""
    r_ac, _, phi = _compute_pde(
        net, x_points, t_batch, D, nd,
        ac_scale=ac_scale, ch_scale=None, sub_data=sub_data)
    iw = _iface_weight(phi, iface_gamma)
    r_w = r_ac * iw
    if t_weights is not None:
        r_w = r_w * t_weights[None, :]
    if core_mask is not None:
        if causal is not None and eps is not None:
            mf = core_mask.astype(r_w.dtype)[:, None]
            val, w, chunks = causal.weighted_loss(r_w * mf, t_batch, eps)
            return val, (w, chunks)
        return _masked_mean(r_w, core_mask), (None, None)
    if causal is not None and eps is not None:
        val, w, chunks = causal.weighted_loss(r_w, t_batch, eps)
        return val, (w, chunks)
    return jnp.mean(jnp.abs(r_w)), (None, None)


def loss_ch(net, x_points, t_batch, D, nd, ac_scale=None, ch_scale=None,
            causal=None, eps=None, t_weights=None, iface_gamma=0.0,
            sub_data=None, core_mask=None):
    """CH scalar loss.  core_mask triggers sub-graph masking."""
    _, r_ch, phi = _compute_pde(
        net, x_points, t_batch, D, nd,
        ac_scale=None, ch_scale=ch_scale, sub_data=sub_data)
    iw = _iface_weight(phi, iface_gamma)
    r_w = r_ch * iw
    if t_weights is not None:
        r_w = r_w * t_weights[None, :]
    if core_mask is not None:
        if causal is not None and eps is not None:
            mf = core_mask.astype(r_w.dtype)[:, None]
            val, w, chunks = causal.weighted_loss(r_w * mf, t_batch, eps)
            return val, (w, chunks)
        return _masked_mean(r_w, core_mask), (None, None)
    if causal is not None and eps is not None:
        val, w, chunks = causal.weighted_loss(r_w, t_batch, eps)
        return val, (w, chunks)
    return jnp.mean(jnp.abs(r_w)), (None, None)


def loss_pde(net, x_points, t_batch, D, nd, *,
             ac_scale=None, ch_scale=None, causal=None, eps=None,
             t_weights=None, iface_gamma=0.0,
             sub_data=None, core_mask=None):
    """Fused PDE loss.  core_mask triggers sub-graph masking."""
    r_ac, r_ch, phi = _compute_pde(
        net, x_points, t_batch, D, nd,
        ac_scale=ac_scale, ch_scale=ch_scale, sub_data=sub_data)
    iw = _iface_weight(phi, iface_gamma)
    r_ac_w, r_ch_w = r_ac * iw, r_ch * iw
    if t_weights is not None:
        r_ac_w = r_ac_w * t_weights[None, :]
        r_ch_w = r_ch_w * t_weights[None, :]
    if core_mask is not None:
        if causal is not None and eps is not None:
            mf = core_mask.astype(r_ac_w.dtype)[:, None]
            v_ac, w_ac, chunks_ac = causal.weighted_loss(r_ac_w * mf, t_batch, eps)
            v_ch, w_ch, chunks_ch = causal.weighted_loss(r_ch_w * mf, t_batch, eps)
            return (v_ac, v_ch), (w_ac, w_ch, chunks_ac, chunks_ch)
        return (_masked_mean(r_ac_w, core_mask), _masked_mean(r_ch_w, core_mask)), None
    if causal is not None and eps is not None:
        v_ac, w_ac, chunks_ac = causal.weighted_loss(r_ac_w, t_batch, eps)
        v_ch, w_ch, chunks_ch = causal.weighted_loss(r_ch_w, t_batch, eps)
        return (v_ac, v_ch), (w_ac, w_ch, chunks_ac, chunks_ch)
    return (jnp.mean(jnp.abs(r_ac_w)), jnp.mean(jnp.abs(r_ch_w))), None


def loss_bc_solid(net, x_bc, t_batch):
    """Dirichlet BC: φ=1, c=1 on solid boundaries (left/right/top edges)."""
    f = jax.vmap(lambda x: jax.vmap(lambda t: net(x, t))(t_batch))(x_bc)
    phi_bc, c_bc = f[..., 0], f[..., 1]
    return jnp.mean((phi_bc - 1.0)**2 + (c_bc - 1.0)**2)


def loss_ic(net, x_points, phi0, c0):
    out = jax.vmap(net, in_axes=(0, None))(x_points, jnp.array(0.0))
    phi_pred, c_pred = out[:, 0], out[:, 1]
    return jnp.mean((phi_pred - phi0)**2 + (c_pred - c0)**2)


def loss_bc_pit(net, x_bc, t_batch):
    """Dirichlet BC: φ=0, c=0 at pit-mouth points (vmap over boundary only)."""
    f = jax.vmap(lambda x: jax.vmap(lambda t: net(x, t))(t_batch))(x_bc)
    phi_bc, c_bc = f[..., 0], f[..., 1]  # (N_bc, N_t)
    return jnp.mean(phi_bc**2 + c_bc**2)


def grad_norm_weights(grads, eps=1e-6, cap=1000):
    norms = jnp.array([jnp.sum(jnp.abs(ravel_pytree(g)[0])) for g in grads])
    norms = jnp.clip(norms, eps, 1.0 / eps)
    weights = jnp.mean(norms) / norms
    weights = jnp.clip(weights, eps, float(cap))
    return jax.lax.stop_gradient(weights)


def make_loss_fn(x_grid, D, x_bc_pit, x_bc_solid, nondim_params,
                 x_ic=None,
                 alpha=1.0, ac_scale=None, ch_scale=None,
                 gradnorm_cap=1000, causal=None):
    """Return losses/gradients and unmasked, unbiased adaptive-weight state.

    aux['weight/*'] contains the effective weights for this update only.
    """
    N_LOSS = 5  # ac, ch, bc_pit, bc_solid, ic
    _x_ic = x_ic if x_ic is not None else x_grid

    def loss_fn(net, t_batch, phi0, c0, eps, train_ac, train_ch, w_ema, bias,
                update_w, iface_gamma, t_weights=None,
                x_points=None, sub_data=None, core_mask=None):
        _kw = dict(ac_scale=ac_scale, ch_scale=ch_scale,
                   causal=causal, eps=eps, t_weights=t_weights,
                   iface_gamma=iface_gamma,
                   sub_data=sub_data, core_mask=core_mask)

        if update_w:
            ((ac_v, (w_ac_c, chunks_ac)), ac_grad) = eqx.filter_value_and_grad(
                lambda n: loss_ac(n, x_points, t_batch, D, nondim_params, **_kw),
                has_aux=True)(net)
            ((ch_v, (w_ch_c, chunks_ch)), ch_grad) = eqx.filter_value_and_grad(
                lambda n: loss_ch(n, x_points, t_batch, D, nondim_params, **_kw),
                has_aux=True)(net)
            (bc_pit_v, bc_pit_grad) = eqx.filter_value_and_grad(loss_bc_pit)(
                net, x_bc_pit, t_batch)
            (bc_solid_v, bc_solid_grad) = eqx.filter_value_and_grad(loss_bc_solid)(
                net, x_bc_solid, t_batch)
            (ic_v, ic_grad) = eqx.filter_value_and_grad(loss_ic)(
                net, _x_ic, phi0, c0)

            grads = [ac_grad, ch_grad, bc_pit_grad, bc_solid_grad, ic_grad]
            vals = [ac_v, ch_v, bc_pit_v, bc_solid_v, ic_v]

            w_new = grad_norm_weights(grads, cap=gradnorm_cap)
            adaptive_w = [alpha * w_new[i] + (1.0 - alpha) * w_ema[i] for i in range(N_LOSS)]
            w = [adaptive_w[i] * bias[i] for i in range(N_LOSS)]
            w[0] = jnp.where(train_ac, w[0], 0.0)
            w[1] = jnp.where(train_ch, w[1], 0.0)

            total = sum(w[i] * vals[i] for i in range(N_LOSS))
            total_grad = jax.tree.map(
                lambda *gs: sum(w[i] * gs[i] for i in range(N_LOSS)), *grads)
            causal_aux = (w_ac_c, w_ch_c, chunks_ac, chunks_ch)
        else:
            adaptive_w = w_ema
            w = [adaptive_w[i] * bias[i] for i in range(N_LOSS)]
            w[0] = jnp.where(train_ac, w[0], 0.0)
            w[1] = jnp.where(train_ch, w[1], 0.0)

            def total_fn(n):
                (a_v, b_v), c_aux = loss_pde(
                    n, x_points, t_batch, D, nondim_params, **_kw)
                c_v = loss_bc_pit(n, x_bc_pit, t_batch)
                d_v = loss_bc_solid(n, x_bc_solid, t_batch)
                e_v = loss_ic(n, _x_ic, phi0, c0)
                t = w[0]*a_v + w[1]*b_v + w[2]*c_v + w[3]*d_v + w[4]*e_v
                return t, (a_v, b_v, c_v, d_v, e_v, c_aux)

            (total, (ac_v, ch_v, bc_pit_v, bc_solid_v, ic_v, c_aux)), total_grad = \
                eqx.filter_value_and_grad(total_fn, has_aux=True)(net)
            causal_aux = c_aux

        if causal_aux is None:
            causal_aux = (None, None, None, None)

        w_ac_c, w_ch_c, chunks_ac, chunks_ch = causal_aux
        aux = {
            'loss/ac': ac_v, 'loss/ch': ch_v,
            'loss/bc_pit': bc_pit_v, 'loss/bc_solid': bc_solid_v,
            'loss/ic': ic_v,
            'weight/ac': w[0], 'weight/ch': w[1],
            'weight/bc_pit': w[2], 'weight/bc_solid': w[3], 'weight/ic': w[4],
            'adaptive_weight/ac': adaptive_w[0], 'adaptive_weight/ch': adaptive_w[1],
            'causal/w_ac': w_ac_c, 'causal/w_ch': w_ch_c,
            'causal/chunks_ac': chunks_ac, 'causal/chunks_ch': chunks_ch,
        }
        return total, aux, total_grad, jnp.array(adaptive_w)

    return loss_fn
