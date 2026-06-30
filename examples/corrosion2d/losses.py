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


def _compute_all(net, x_all, t_batch):
    def field_at_t(t):
        return jax.vmap(net, in_axes=(0, None))(x_all, t)  # (N_x, 2)
    def field_at_times(ts):
        return jax.vmap(field_at_t)(ts)  # (N_t, N_x, 2)
    out, out_t = jax.jvp(field_at_times, (t_batch,), (jnp.ones_like(t_batch),))
    out = jnp.swapaxes(out, 0, 1)      # (N_x, N_t, 2)
    out_t = jnp.swapaxes(out_t, 0, 1)  # (N_x, N_t, 2)
    phi, c = out[..., 0], out[..., 1]
    dphi_dt, dc_dt = out_t[..., 0], out_t[..., 1]
    return phi, c, dphi_dt, dc_dt


def _compute_ac_residual(net, x_all, t_batch, D, nd, ac_scale=None):
    _, N_AC1, N_AC2, N_AC3, dc0, c_Le = nd
    phi, c, dphi_dt, _ = _compute_all(net, x_all, t_batch)
    nl_phi = D @ phi
    r = (dphi_dt + N_AC3 * nl_phi + N_AC2 * _gp(phi)
         - N_AC1 * (c - _h(phi) * dc0 - c_Le) * dc0 * _hp(phi))
    r = ac_scale * r if ac_scale is not None else r
    return r, phi


def _compute_ch_residual(net, x_all, t_batch, D, nd, ch_scale=None):
    N_CH, _, _, _, dc0, c_Le = nd
    phi, c, _, dc_dt = _compute_all(net, x_all, t_batch)
    mu = c - _h(phi) * dc0 - c_Le
    nl_mu = D @ mu
    r = dc_dt + N_CH * nl_mu
    r = ch_scale * r if ch_scale is not None else r
    return r, phi


def _compute_pde_both(net, x_all, t_batch, D, nd, *, ac_scale=None, ch_scale=None):
    """Compute AC and CH residuals from a single _compute_all call."""
    N_CH, N_AC1, N_AC2, N_AC3, dc0, c_Le = nd
    phi, c, dphi_dt, dc_dt = _compute_all(net, x_all, t_batch)
    # AC
    r_ac = (dphi_dt + N_AC3 * (D @ phi) + N_AC2 * _gp(phi)
            - N_AC1 * (c - _h(phi) * dc0 - c_Le) * dc0 * _hp(phi))
    r_ac = ac_scale * r_ac if ac_scale is not None else r_ac
    # CH
    mu = c - _h(phi) * dc0 - c_Le
    r_ch = dc_dt + N_CH * (D @ mu)
    r_ch = ch_scale * r_ch if ch_scale is not None else r_ch
    return r_ac, r_ch, phi


def _iface_weight(phi, gamma):
    """Interface indicator weight: w = 1 + γ·4φ(1-φ), peaked at φ=0.5.

    4φ(1-φ) is the standard phase-field interface indicator (not an area).
    It equals 1 at φ=0.5 (interface center), 0 at φ=0,1 (bulk).
    """
    phi_sg = jax.lax.stop_gradient(phi)
    phi_c = jnp.clip(phi_sg, 0.0, 1.0)
    return 1.0 + gamma * 4.0 * phi_c * (1.0 - phi_c)


def loss_ac(net, x_all, t_batch, D, nondim_params, ac_scale=None, causal=None,
            eps=None, iface_gamma=0.0):
    r, phi = _compute_ac_residual(net, x_all, t_batch, D, nondim_params, ac_scale)
    iw = _iface_weight(phi, iface_gamma)
    if causal is not None and eps is not None:
        val, w, chunks = causal.weighted_loss(r * iw, t_batch, eps)
        return val, (w, chunks)
    return jnp.mean(jnp.abs(r) * iw), (None, None)


def loss_ch(net, x_all, t_batch, D, nondim_params, ch_scale=None, causal=None, eps=None,
            iface_gamma=0.0):
    r, phi = _compute_ch_residual(net, x_all, t_batch, D, nondim_params, ch_scale)
    iw = _iface_weight(phi, iface_gamma)
    if causal is not None and eps is not None:
        val, w, chunks = causal.weighted_loss(r * iw, t_batch, eps)
        return val, (w, chunks)
    return jnp.mean(jnp.abs(r) * iw), (None, None)


def loss_pde(net, x_all, t_batch, D, nondim_params, *,
             ac_scale=None, ch_scale=None, causal=None, eps=None, iface_gamma=0.0):
    """Fused PDE loss: AC + CH from one _compute_all call."""
    r_ac, r_ch, phi = _compute_pde_both(net, x_all, t_batch, D, nondim_params,
                                         ac_scale=ac_scale, ch_scale=ch_scale)
    iw = _iface_weight(phi, iface_gamma)
    if causal is not None and eps is not None:
        v_ac, w_ac, chunks_ac = causal.weighted_loss(r_ac * iw, t_batch, eps)
        v_ch, w_ch, chunks_ch = causal.weighted_loss(r_ch * iw, t_batch, eps)
        return (v_ac, v_ch), (w_ac, w_ch, chunks_ac, chunks_ch)
    return (jnp.mean(jnp.abs(r_ac) * iw), jnp.mean(jnp.abs(r_ch) * iw)), None


def loss_bc_solid(net, x_bc, t_batch):
    """Dirichlet BC: φ=1, c=1 on solid boundaries (left/right/top edges)."""
    f = jax.vmap(lambda x: jax.vmap(lambda t: net(x, t))(t_batch))(x_bc)
    phi_bc, c_bc = f[..., 0], f[..., 1]
    return jnp.mean((phi_bc - 1.0)**2 + (c_bc - 1.0)**2)


def loss_ic(net, x_all, phi0, c0):
    out = jax.vmap(net, in_axes=(0, None))(x_all, jnp.array(0.0))
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


def make_loss_fn(x_all, D, x_bc_pit, x_bc_solid, nondim_params,
                 x_ic=None,
                 alpha=1.0, ac_scale=None, ch_scale=None,
                 gradnorm_cap=1000, causal=None):
    N_LOSS = 5  # ac, ch, bc_pit, bc_solid, ic
    _x_ic = x_ic if x_ic is not None else x_all

    def loss_fn(net, t_batch, phi0, c0, eps, train_ac, train_ch, w_ema, bias, update_w, iface_gamma):
        if update_w:
            ((ac_v, (w_ac_c, chunks_ac)), ac_grad) = eqx.filter_value_and_grad(
                loss_ac, has_aux=True)(
                net, x_all, t_batch, D, nondim_params, ac_scale, causal, eps, iface_gamma=iface_gamma)
            ((ch_v, (w_ch_c, chunks_ch)), ch_grad) = eqx.filter_value_and_grad(
                loss_ch, has_aux=True)(
                net, x_all, t_batch, D, nondim_params, ch_scale, causal, eps, iface_gamma=iface_gamma)
            (bc_pit_v, bc_pit_grad) = eqx.filter_value_and_grad(loss_bc_pit)(
                net, x_bc_pit, t_batch)
            (bc_solid_v, bc_solid_grad) = eqx.filter_value_and_grad(loss_bc_solid)(
                net, x_bc_solid, t_batch)
            (ic_v, ic_grad) = eqx.filter_value_and_grad(loss_ic)(
                net, _x_ic, phi0, c0)

            grads = [ac_grad, ch_grad, bc_pit_grad, bc_solid_grad, ic_grad]
            vals = [ac_v, ch_v, bc_pit_v, bc_solid_v, ic_v]

            w_new = grad_norm_weights(grads, cap=gradnorm_cap)
            w = [alpha * w_new[i] + (1.0 - alpha) * w_ema[i] for i in range(N_LOSS)]
            w[0] = jnp.where(train_ac, w[0], 0.0)
            w[1] = jnp.where(train_ch, w[1], 0.0)
            for i in range(N_LOSS):
                w[i] = w[i] * bias[i]

            total = sum(w[i] * vals[i] for i in range(N_LOSS))
            total_grad = jax.tree.map(
                lambda *gs: sum(w[i] * gs[i] for i in range(N_LOSS)), *grads)
            causal_aux = (w_ac_c, w_ch_c, chunks_ac, chunks_ch)
        else:
            w = [w_ema[i] * bias[i] for i in range(N_LOSS)]
            w[0] = jnp.where(train_ac, w[0], 0.0)
            w[1] = jnp.where(train_ch, w[1], 0.0)

            def total_fn(n):
                (a_v, b_v), c_aux = loss_pde(
                    n, x_all, t_batch, D, nondim_params,
                    ac_scale=ac_scale, ch_scale=ch_scale,
                    causal=causal, eps=eps, iface_gamma=iface_gamma)
                c_v = loss_bc_pit(n, x_bc_pit, t_batch)
                d_v = loss_bc_solid(n, x_bc_solid, t_batch)
                e_v = loss_ic(n, _x_ic, phi0, c0)
                t = w[0]*a_v + w[1]*b_v + w[2]*c_v + w[3]*d_v + w[4]*e_v
                return t, (a_v, b_v, c_v, d_v, e_v, c_aux)

            (total, (ac_v, ch_v, bc_pit_v, bc_solid_v, ic_v, causal_aux)), total_grad = \
                eqx.filter_value_and_grad(total_fn, has_aux=True)(net)
            if causal_aux is None:
                causal_aux = (None, None, None, None)

        w_ac_c, w_ch_c, chunks_ac, chunks_ch = causal_aux
        aux = {
            'loss_ac': ac_v, 'loss_ch': ch_v,
            'loss_bc_pit': bc_pit_v, 'loss_bc_solid': bc_solid_v,
            'loss_ic': ic_v,
            'weight_ac': w[0], 'weight_ch': w[1],
            'weight_bc_pit': w[2], 'weight_bc_solid': w[3], 'weight_ic': w[4],
            'causal_w_ac': w_ac_c, 'causal_w_ch': w_ch_c,
            'causal_chunks_ac': chunks_ac, 'causal_chunks_ch': chunks_ch,
        }
        return total, aux, total_grad, jnp.array(w)

    return loss_fn
