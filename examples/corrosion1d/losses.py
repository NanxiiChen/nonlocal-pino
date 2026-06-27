"""Loss functions for 1D KKS corrosion PINN — full MLP and split architectures.

Architecture-specific code is injected via compute_all and eval_t0 callables;
everything else is shared between all architectures.

Usage:
  from corrosion1d.losses import make_loss_fn, ic_values, _make_full_compute
  compute_all, eval_t0 = _make_full_compute()
  lfn = make_loss_fn(..., compute_all=compute_all, eval_t0=eval_t0)
"""

import jax.numpy as jnp
import jax
import equinox as eqx
import numpy as np
from jax.flatten_util import ravel_pytree


# ── Phase-field helpers ─────────────────────────────────────────────────────

def _h(phi): return -2.0 * phi**3 + 3.0 * phi**2
def _hp(phi): return -6.0 * phi**2 + 6.0 * phi
def _gp(phi): return 4.0 * phi**3 - 6.0 * phi**2 + 2.0 * phi


def _iface_weight(phi, gamma):
    """Interface indicator weight: w = 1 + γ·4φ(1-φ), peaked at φ=0.5.

    4φ(1-φ) is the standard phase-field interface indicator (not an area).
    It equals 1 at φ=0.5 (interface center), 0 at φ=0,1 (bulk).
    """
    phi_sg = jax.lax.stop_gradient(phi)
    phi_c = jnp.clip(phi_sg, 0.0, 1.0)
    return 1.0 + gamma * 4.0 * phi_c * (1.0 - phi_c)


# ── IC ──────────────────────────────────────────────────────────────────────

def ic_values(x, phys, L_c, center=0.5):
    k = np.sqrt(phys.omega_phi / (2.0 * phys.alpha_phi)) * L_c
    dist = np.array(x)[:, 0] - center
    phi0 = 0.5 * (1.0 - np.tanh(k * dist))
    c0 = _h(phi0) * phys.c_se
    return jnp.array(phi0), jnp.array(c0)


# ── Compute factories (architecture-specific) ───────────────────────────────

def _make_full_compute():
    """Return (compute_all, eval_t0) for full MLP / modified-MLP architecture."""

    def compute_all(net, x_all, t_batch):
        fwd = jax.vmap(jax.vmap(net, in_axes=(None, 0)), in_axes=(0, None))
        out = fwd(x_all, t_batch)                           # (N_x, N_t, 2)
        phi, c = out[..., 0], out[..., 1]
        jac = jax.vmap(jax.vmap(jax.jacrev(net, argnums=1), in_axes=(None, 0)),
                       in_axes=(0, None))(x_all, t_batch)    # (N_x, N_t, 2)
        dphi_dt, dc_dt = jac[..., 0], jac[..., 1]
        return phi, c, dphi_dt, dc_dt

    def eval_t0(net, x_all):
        out = jax.vmap(net, in_axes=(0, None))(x_all, jnp.array(0.0))
        return out[:, 0], out[:, 1]

    return compute_all, eval_t0


def _make_split_compute():
    """Return (compute_all, eval_t0) for split-PINN architecture."""

    def compute_all(net, x_all, t_batch):
        Fx = net.spatial_batch(x_all)                    # (N_x, 2p)
        Ft = net.temporal_batch(t_batch)                 # (N_t, 2p)
        dFt = jax.vmap(net.temporal_derivative)(t_batch) # (N_t, 2p)
        out_raw = net.combine(Fx, Ft, t_batch)             # (N_x, N_t, 2)
        d_raw = net.combine(Fx, dFt)                       # (N_x, N_t, 2)
        db_dt = net.combine_bias_derivative(t_batch)       # (N_t, 2)

        if net.bc is not None:
            mask = net.bc.mask(x_all)      # (N_x, 2)
            base = net.bc.baseline(x_all)  # (N_x, 2)
            out = base[:, None, :] + mask[:, None, :] * out_raw
            dphi = mask[:, None, 0] * d_raw[..., 0] + db_dt[None, :, 0]
            dc   = mask[:, None, 1] * d_raw[..., 1] + db_dt[None, :, 1]
        else:
            out = out_raw
            dphi = d_raw[..., 0] + db_dt[None, :, 0]
            dc   = d_raw[..., 1] + db_dt[None, :, 1]

        return out[..., 0], out[..., 1], dphi, dc

    def eval_t0(net, x_all):
        Fx = net.spatial_batch(x_all)
        field = net.project(Fx, net.temporal(jnp.array(0.0)))
        return field[:, 0], field[:, 1]

    return compute_all, eval_t0


# ── PDE residuals ───────────────────────────────────────────────────────────

def _compute_ac_residual(net, x_all, t_batch, D, nd, compute_all, ac_scale=None):
    _, N_AC1, N_AC2, N_AC3, dc0, c_Le = nd
    phi, c, dphi_dt, _ = compute_all(net, x_all, t_batch)
    nl_phi = D @ phi
    r = (dphi_dt + N_AC3 * nl_phi + N_AC2 * _gp(phi)
         - N_AC1 * (c - _h(phi) * dc0 - c_Le) * dc0 * _hp(phi))
    r = ac_scale * r if ac_scale is not None else r
    return r, phi


def _compute_ch_residual(net, x_all, t_batch, D, nd, compute_all, ch_scale=None):
    N_CH, _, _, _, dc0, c_Le = nd
    phi, c, _, dc_dt = compute_all(net, x_all, t_batch)
    mu = c - _h(phi) * dc0 - c_Le
    nl_mu = D @ mu
    r = dc_dt + N_CH * nl_mu
    r = ch_scale * r if ch_scale is not None else r
    return r, phi


# ── Per-loss functions ──────────────────────────────────────────────────────

def loss_ac(net, x_all, t_batch, D, nondim_params, compute_all,
            ac_scale=None, causal=None, eps=None, iface_gamma=0.0):
    r, phi = _compute_ac_residual(net, x_all, t_batch, D, nondim_params,
                                   compute_all, ac_scale)
    iw = _iface_weight(phi, iface_gamma)
    if causal is not None and eps is not None:
        val, w, chunks = causal.weighted_loss(r * iw, t_batch, eps)
        return val, (w, chunks)
    return jnp.mean(jnp.abs(r) * iw), None


def loss_ch(net, x_all, t_batch, D, nondim_params, compute_all,
            ch_scale=None, causal=None, eps=None, iface_gamma=0.0):
    r, phi = _compute_ch_residual(net, x_all, t_batch, D, nondim_params,
                                   compute_all, ch_scale)
    iw = _iface_weight(phi, iface_gamma)
    if causal is not None and eps is not None:
        val, w, chunks = causal.weighted_loss(r * iw, t_batch, eps)
        return val, (w, chunks)
    return jnp.mean(jnp.abs(r) * iw), None


def loss_ic(net, x_all, phi0, c0, eval_t0):
    phi_pred, c_pred = eval_t0(net, x_all)
    return jnp.mean((phi_pred - phi0)**2 + (c_pred - c0)**2)


def loss_bc(net, x_all, t_batch, bc_values, constr_idx, compute_all):
    phi, c, _, _ = compute_all(net, x_all, t_batch)
    err_phi = phi[constr_idx, :] - bc_values[constr_idx, None]
    err_c   = c[constr_idx, :] - bc_values[constr_idx, None]
    return jnp.mean(err_phi**2 + err_c**2)


# ── GradNorm weighting ──────────────────────────────────────────────────────

def grad_norm_weights(grads, eps=1e-6, cap=1000):
    norms = jnp.array([jnp.sum(jnp.abs(ravel_pytree(g)[0])) for g in grads])
    norms = jnp.clip(norms, eps, 1.0 / eps)
    weights = jnp.mean(norms) / norms
    weights = jnp.clip(weights, eps, float(cap))
    return jax.lax.stop_gradient(weights)


# ── Combined loss function ──────────────────────────────────────────────────

def make_loss_fn(x_all, D, mask_free, bc_values, nondim_params,
                 compute_all, eval_t0,
                 alpha=1.0, ac_scale=None, ch_scale=None,
                 gradnorm_cap=1000, causal=None,
                 iface_gamma=0.0):
    constr_idx = ~mask_free

    def loss_fn(net, t_batch, phi0, c0, eps, train_ac, train_ch, w_ema, bias):
        ((ac_val, ac_aux), ac_grad) = eqx.filter_value_and_grad(
            loss_ac, has_aux=True)(
            net, x_all, t_batch, D, nondim_params, compute_all,
            ac_scale, causal, eps, iface_gamma=iface_gamma)
        ((ch_val, ch_aux), ch_grad) = eqx.filter_value_and_grad(
            loss_ch, has_aux=True)(
            net, x_all, t_batch, D, nondim_params, compute_all,
            ch_scale, causal, eps, iface_gamma=iface_gamma)
        w_ac_c, chunks_ac = ac_aux if ac_aux is not None else (None, None)
        w_ch_c, chunks_ch = ch_aux if ch_aux is not None else (None, None)

        (ic_val, ic_grad) = eqx.filter_value_and_grad(loss_ic)(
            net, x_all, phi0, c0, eval_t0)

        (bc_val, bc_grad) = eqx.filter_value_and_grad(loss_bc)(
            net, x_all, t_batch, bc_values, constr_idx, compute_all)

        w_new = grad_norm_weights([ac_grad, ch_grad, bc_grad, ic_grad],
                                   cap=gradnorm_cap)
        w_ac = alpha * w_new[0] + (1.0 - alpha) * w_ema[0]
        w_ch = alpha * w_new[1] + (1.0 - alpha) * w_ema[1]
        w_bc = alpha * w_new[2] + (1.0 - alpha) * w_ema[2]
        w_ic = alpha * w_new[3] + (1.0 - alpha) * w_ema[3]

        w_ac = jnp.where(train_ac, w_ac, 0.0)
        w_ch = jnp.where(train_ch, w_ch, 0.0)

        w_ac *= bias[0]
        w_ch *= bias[1]
        w_bc *= bias[2]
        w_ic *= bias[3]

        total = w_ac * ac_val + w_ch * ch_val + w_bc * bc_val + w_ic * ic_val
        total_grad = jax.tree.map(
            lambda g1, g2, g3, g4: w_ac * g1 + w_ch * g2 + w_bc * g3 + w_ic * g4,
            ac_grad, ch_grad, bc_grad, ic_grad)

        aux = {
            'loss_ac': ac_val, 'loss_ch': ch_val, 'loss_bc': bc_val, 'loss_ic': ic_val,
            'weight_ac': w_ac, 'weight_ch': w_ch, 'weight_bc': w_bc, 'weight_ic': w_ic,
            'causal_w_ac': w_ac_c, 'causal_w_ch': w_ch_c,
            'causal_chunks_ac': chunks_ac, 'causal_chunks_ch': chunks_ch,
        }
        return total, aux, total_grad, jnp.array([w_ac, w_ch, w_bc, w_ic])

    return loss_fn
