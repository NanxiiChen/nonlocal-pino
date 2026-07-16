"""Loss functions for 2D KKS electropolishing PINN — rough interface."""

import jax.numpy as jnp
import jax
import equinox as eqx
import numpy as np
from jax.flatten_util import ravel_pytree


def _h(phi): return -2.0 * phi**3 + 3.0 * phi**2
def _hp(phi): return -6.0 * phi**2 + 6.0 * phi
def _hpp(phi): return -12.0 * phi + 6.0     # h''(φ)
def _gp(phi): return 4.0 * phi**3 - 6.0 * phi**2 + 2.0 * phi


# ── Rough surface builder ──────────────────────────────────────────────────

def _rough_surface(x, y0, modes):
    """Evaluate rough surface height at x coordinates."""
    y = np.full_like(x, y0)
    for amp, freq, phase in modes:
        y += amp * (np.sin(2.0 * np.pi * freq * x + phase)
                    + np.cos(2.0 * np.pi * freq * x + phase + 0.3))
    return y


def _sdf_to_curve(xq, yq, x_curve, y_curve, y_surf_at_xq, window=16):
    """Signed distance to piecewise-linear profile y=f(x).

    Vectorised over query-point columns — each column checks only segments
    within ±window of the nearest x-index.  Positive = below curve (solid).
    """
    s0x = x_curve[:-1]; s0y = y_curve[:-1]
    dx = x_curve[1:] - s0x; dy = y_curve[1:] - s0y
    len2 = dx*dx + dy*dy + 1e-20
    n_seg, n_q = len(s0x), len(xq)

    # For each query point, find the nearest x-index in the curve
    xi_idx = np.searchsorted(x_curve, xq)  # (n_q,)

    # Accumulate minimum distance over window offsets
    dist2 = np.full(n_q, np.inf)
    for offset in range(-window, window + 1):
        seg = np.clip(xi_idx + offset, 0, n_seg - 1)  # (n_q,) — segment per query
        sx, sdx = s0x[seg], dx[seg]
        sy, sdy = s0y[seg], dy[seg]
        sl2 = len2[seg]
        rx, ry = xq - sx, yq - sy
        t = np.clip((rx*sdx + ry*sdy) / sl2, 0.0, 1.0)
        px, py = sx + t*sdx, sy + t*sdy
        d2 = (xq - px)**2 + (yq - py)**2
        dist2 = np.minimum(dist2, d2)

    sign = np.where(yq < y_surf_at_xq, 1.0, -1.0)
    return sign * np.sqrt(dist2)


def ic_values(x_all, phys, L_c, surface_cfg):
    """Rough interface IC: φ = tanh(k * sdf(x)), c = h(φ)·c_se."""
    k = np.sqrt(phys.omega_phi / (2.0 * phys.alpha_phi)) * L_c
    x_np = np.array(x_all)
    rng = np.random.RandomState(surface_cfg.seed)
    modes = []
    for _ in range(surface_cfg.n_modes):
        amp = rng.uniform(0.3, 1.0) * surface_cfg.amplitude
        freq = rng.uniform(surface_cfg.freq_lo, surface_cfg.freq_hi)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        modes.append((amp, freq, phase))
    x_curve = np.linspace(0, 1, 2000)
    y_curve = _rough_surface(x_curve, surface_cfg.y0, modes)
    y_surf = _rough_surface(x_np[:, 0], surface_cfg.y0, modes)
    sdf = _sdf_to_curve(x_np[:, 0], x_np[:, 1], x_curve, y_curve, y_surf)
    phi0 = 0.5 * (1.0 + np.tanh(k * sdf))
    c0 = _h(phi0) * phys.c_se
    return jnp.array(phi0), jnp.array(c0)


# ── Network evaluation ─────────────────────────────────────────────────────

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


def _compute_all_local(net, x_points, t_batch):
    """Tensor-product eval with AD spatial derivatives (2D, two-field).

    For each (x, t): computes φ, c, ∂φ/∂t, ∂c/∂t, ∇²φ, ∇²c, |∇φ|²
    needed for the local (non-D_δ) PDE residual.

    Returns:
        phi:        (N_x, N_t)
        c:          (N_x, N_t)
        dphi_dt:    (N_x, N_t)
        dc_dt:      (N_x, N_t)
        lap_phi:    (N_x, N_t)   ∇²φ
        lap_c:      (N_x, N_t)   ∇²c
        grad_phi_sq:(N_x, N_t)   |∇φ|²
    """
    # φ, c, dφ/dt, dc/dt (same tensor-product as _compute_all)
    def field_at_t(t):
        return jax.vmap(net, in_axes=(0, None))(x_points, t)  # (N_x, 2)
    def field_at_times(ts):
        return jax.vmap(field_at_t)(ts)  # (N_t, N_x, 2)
    out, out_t = jax.jvp(field_at_times, (t_batch,), (jnp.ones_like(t_batch),))
    out = jnp.swapaxes(out, 0, 1)
    out_t = jnp.swapaxes(out_t, 0, 1)
    phi, c = out[..., 0], out[..., 1]
    dphi_dt, dc_dt = out_t[..., 0], out_t[..., 1]

    # Spatial derivatives per (x, t)
    def _spatial(x, t):
        # Hessian of both outputs: (2, 2, 2) for 2D
        hess = jax.hessian(lambda xx: net(xx, t))(x)
        lap_phi = jnp.trace(hess[0])   # ∇²φ
        lap_c = jnp.trace(hess[1])     # ∇²c
        grad_phi = jax.grad(lambda xx: net(xx, t)[0])(x)
        grad_phi_sq = jnp.sum(grad_phi**2)
        return jnp.stack([lap_phi, lap_c, grad_phi_sq])

    # vmap over x (inner) and t (outer)
    derivs = jax.vmap(
        jax.vmap(_spatial, in_axes=(0, None)),
        in_axes=(None, 0))(x_points, t_batch)  # (N_t, N_x, 3)
    derivs = jnp.swapaxes(derivs, 0, 1)         # (N_x, N_t, 3)
    lap_phi = derivs[..., 0]
    lap_c = derivs[..., 1]
    grad_phi_sq = derivs[..., 2]

    return phi, c, dphi_dt, dc_dt, lap_phi, lap_c, grad_phi_sq


# ── PDE residual ───────────────────────────────────────────────────────────

def _compute_pde(net, x_points, t_batch, D, nd, *, ac_scale=None, ch_scale=None,
                 sub_data=None):
    N_CH, N_AC1, N_AC2, N_AC3, dc0, c_Le = nd

    # Local autograd path: D is None → use AD Hessian instead of D_δ
    if D is None:
        phi, c, dphi_dt, dc_dt, lap_phi, lap_c, grad_phi_sq = \
            _compute_all_local(net, x_points, t_batch)
        # D_δ ≈ -∇², so +N_AC3·D_δ φ → -N_AC3·∇²φ
        r_ac = (dphi_dt - N_AC3 * lap_phi + N_AC2 * _gp(phi)
                - N_AC1 * (c - _h(phi) * dc0 - c_Le) * dc0 * _hp(phi))
        # CH: ∇²μ = ∇²c - Δc₀[h''(φ)|∇φ|² + h'(φ)∇²φ]
        lap_mu = lap_c - dc0 * (_hpp(phi) * grad_phi_sq + _hp(phi) * lap_phi)
        r_ch = dc_dt - N_CH * lap_mu
    else:
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

        mu = c - _h(phi) * dc0 - c_Le
        r_ch = dc_dt + N_CH * _D(mu)

    if ac_scale is not None:
        r_ac = ac_scale * r_ac
    if ch_scale is not None:
        r_ch = ch_scale * r_ch

    return r_ac, r_ch, phi


def _masked_mean(r, mask):
    mask_f = mask.astype(r.dtype)[:, None]
    n_valid = jnp.sum(mask_f) * r.shape[1]
    return jnp.sum(jnp.abs(r) * mask_f) / jnp.maximum(n_valid, 1.0)


def _iface_weight(phi, gamma):
    phi_sg = jax.lax.stop_gradient(phi)
    phi_c = jnp.clip(phi_sg, 0.0, 1.0)
    return 1.0 + gamma * 4.0 * phi_c * (1.0 - phi_c)


# ── Loss terms ─────────────────────────────────────────────────────────────

def loss_ac(net, x_points, t_batch, D, nd, ac_scale=None, ch_scale=None,
            causal=None, eps=None, t_weights=None, iface_gamma=0.0,
            sub_data=None, core_mask=None):
    r_ac, _, phi = _compute_pde(net, x_points, t_batch, D, nd,
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
    _, r_ch, phi = _compute_pde(net, x_points, t_batch, D, nd,
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
    r_ac, r_ch, phi = _compute_pde(net, x_points, t_batch, D, nd,
                                   ac_scale=ac_scale, ch_scale=ch_scale,
                                   sub_data=sub_data)
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


def loss_bc_top(net, x_bc, t_batch):
    """Top BC (y=0.5): liquid — φ=0, c=0."""
    f = jax.vmap(lambda x: jax.vmap(lambda t: net(x, t))(t_batch))(x_bc)
    phi_bc, c_bc = f[..., 0], f[..., 1]
    return jnp.mean(phi_bc**2 + c_bc**2)


def loss_bc_bottom(net, x_bc, t_batch):
    """Bottom BC (y=0): solid — φ=1, c=1."""
    f = jax.vmap(lambda x: jax.vmap(lambda t: net(x, t))(t_batch))(x_bc)
    phi_bc, c_bc = f[..., 0], f[..., 1]
    return jnp.mean((phi_bc - 1.0)**2 + (c_bc - 1.0)**2)


def loss_ic(net, x_points, phi0, c0):
    out = jax.vmap(net, in_axes=(0, None))(x_points, jnp.array(0.0))
    phi_pred, c_pred = out[:, 0], out[:, 1]
    return jnp.mean((phi_pred - phi0)**2 + (c_pred - c0)**2)


def grad_norm_weights(grads, eps=1e-6, cap=1000):
    norms = jnp.array([jnp.sum(jnp.abs(ravel_pytree(g)[0])) for g in grads])
    norms = jnp.clip(norms, eps, 1.0 / eps)
    weights = jnp.mean(norms) / norms
    weights = jnp.clip(weights, eps, float(cap))
    return jax.lax.stop_gradient(weights)


# ── Loss factory ───────────────────────────────────────────────────────────

def make_loss_fn(x_grid, D, x_bc_top, x_bc_bottom, nondim_params,
                 x_ic=None,
                 alpha=1.0, ac_scale=None, ch_scale=None,
                 gradnorm_cap=1000, causal=None):
    N_LOSS = 5  # ac, ch, bc_top, bc_bottom, ic
    _x_ic = x_ic if x_ic is not None else x_grid

    def loss_fn(net, t_batch, phi0, c0, eps, train_ac, train_ch, w_ema, bias,
                update_w, iface_gamma, t_weights=None,
                x_points=None, sub_data=None, core_mask=None):
        _kw = dict(ac_scale=ac_scale, ch_scale=ch_scale,
                   causal=causal, eps=eps, t_weights=t_weights,
                   iface_gamma=iface_gamma,
                   sub_data=sub_data, core_mask=core_mask)

        # BC and IC: same computation regardless of update_w
        (bc_top_v, bc_top_grad) = eqx.filter_value_and_grad(loss_bc_top)(
            net, x_bc_top, t_batch)
        (bc_bot_v, bc_bot_grad) = eqx.filter_value_and_grad(loss_bc_bottom)(
            net, x_bc_bottom, t_batch)
        (ic_v, ic_grad) = eqx.filter_value_and_grad(loss_ic)(
            net, _x_ic, phi0, c0)

        if update_w:
            # GradNorm: need separate AC/CH gradients for per-term weights.
            # AC and CH each call _compute_pde → _compute_all, duplicating
            # the expensive forward+JVP.  This is intentional — the
            # alternative (shared-forward/per-term-VJP) is future work
            # See benchmarks/TENSOR_SPLIT_RTX_PRO_6000.md, "electropolish2d".
            ((ac_v, (w_ac_c, chunks_ac)), ac_grad) = eqx.filter_value_and_grad(
                lambda n: loss_ac(n, x_points, t_batch, D, nondim_params, **_kw),
                has_aux=True)(net)
            ((ch_v, (w_ch_c, chunks_ch)), ch_grad) = eqx.filter_value_and_grad(
                lambda n: loss_ch(n, x_points, t_batch, D, nondim_params, **_kw),
                has_aux=True)(net)
            grads = [ac_grad, ch_grad, bc_top_grad, bc_bot_grad, ic_grad]
            vals = [ac_v, ch_v, bc_top_v, bc_bot_v, ic_v]

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
            new_w = jnp.array(w_new)
        else:
            # Regular: AC+CH grouped — one _compute_all, separate weights.
            # grad(w[0]*ac + w[1]*ch) = w[0]*grad(ac) + w[1]*grad(ch).
            # benchmarks/TENSOR_SPLIT_RTX_PRO_6000.md §3.
            w = [w_ema[i] * bias[i] for i in range(N_LOSS)]
            w[0] = jnp.where(train_ac, w[0], 0.0)
            w[1] = jnp.where(train_ch, w[1], 0.0)

            def _pde_group(n):
                (ac_v, ch_v), c_aux = loss_pde(
                    n, x_points, t_batch, D, nondim_params, **_kw)
                return w[0]*ac_v + w[1]*ch_v, (ac_v, ch_v, c_aux)

            (_, (ac_v, ch_v, c_aux)), pde_grad = \
                eqx.filter_value_and_grad(_pde_group, has_aux=True)(net)

            total = (w[0]*ac_v + w[1]*ch_v + w[2]*bc_top_v
                     + w[3]*bc_bot_v + w[4]*ic_v)
            total_grad = jax.tree.map(
                lambda g_pde, g_bt, g_bb, g_ic:
                    g_pde + w[2] * g_bt + w[3] * g_bb + w[4] * g_ic,
                pde_grad, bc_top_grad, bc_bot_grad, ic_grad)
            causal_aux = c_aux
            new_w = w_ema

        if causal_aux is None:
            causal_aux = (None, None, None, None)

        w_ac_c, w_ch_c, chunks_ac, chunks_ch = causal_aux
        aux = {
            'loss/ac': ac_v, 'loss/ch': ch_v,
            'loss/bc_top': bc_top_v, 'loss/bc_bot': bc_bot_v,
            'loss/ic': ic_v,
            'weight/ac': w[0], 'weight/ch': w[1],
            'weight/bc_top': w[2], 'weight/bc_bot': w[3], 'weight/ic': w[4],
            'causal/w_ac': w_ac_c, 'causal/w_ch': w_ch_c,
            'causal/chunks_ac': chunks_ac, 'causal/chunks_ch': chunks_ch,
        }
        return total, aux, total_grad, jnp.array(w)

    return loss_fn
