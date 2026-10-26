"""Experimental KKS residuals, without hard initial conditions.

The latent objective is r_phi/[4*phi*(1-phi)] = r_z/2, where
phi=(1+tanh(z))/2. Its interface-centre scale agrees with r_phi.
"""

import jax
import jax.numpy as jnp
import numpy as np


def bounded_phase_weight(phi, eta):
    """Explicit positive inverse-Jacobian weight, centre 1 and cap 1+1/eta."""
    if eta is None or not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"bounded_eta must be finite and positive, got {eta}")
    return (1.0 + eta) / (4.0 * phi * (1.0 - phi) + eta)


def phase_only_constraint(raw):
    """Bound phase for its latent residual; predict concentration directly."""
    return jnp.stack([0.5 * (1.0 + jnp.tanh(raw[..., 0])), raw[..., 1]], axis=-1)


def tanh_edge_ratio(zi, zj):
    """(tanh(zi)-tanh(zj))/sech(zi)^2 without saturated subtraction."""
    log_ratio = jnp.logaddexp(zi, -zi) - jnp.logaddexp(zj, -zj)
    return jnp.sinh(zi - zj) * jnp.exp(log_ratio)


def _raw_fields_and_derivatives(net, x, times):
    """One raw-network JVP, then the selected output map and its JVP."""
    def raw_at_times(ts):
        return jax.vmap(lambda t: jax.vmap(net.net, in_axes=(0, None))(x, t))(ts)

    raw, raw_t = jax.jvp(raw_at_times, (times,), (jnp.ones_like(times),))
    fields, fields_t = jax.jvp(
        net.constraint, (raw,), (raw_t,)
    )
    return raw, raw_t, fields.transpose(1, 0, 2), fields_t.transpose(1, 0, 2)


def fields_and_derivatives(net, x, times):
    raw, raw_t, fields, fields_t = _raw_fields_and_derivatives(net, x, times)
    return raw[..., 0].T, raw_t[..., 0].T, fields, fields_t


def graph_action(values, sub_data):
    row, col, vals, diag = sub_data
    off = jax.ops.segment_sum(
        vals[:, None] * values[col], row, num_segments=values.shape[0]
    )
    return diag[:, None] * values + off


def graph_edge_action(values, sub_data):
    """Same positive graph Laplacian; constant fields give exact floating zero."""
    row, col, vals, _ = sub_data
    return jax.ops.segment_sum(
        -vals[:, None] * (values[row] - values[col]), row,
        num_segments=values.shape[0],
    )


def compute_pde(net, x, times, sub_data, nd, formulation, normalization, *, bounded_eta=None):
    """Legacy physical-difference evaluation, retained for archived callers."""
    return compute_transport_pde(net, x, times, sub_data, nd, formulation, normalization,
        transport_evaluation="physical_difference", bounded_eta=bounded_eta)


def compute_transport_pde(net, x, times, sub_data, nd, formulation, normalization,
                          *, transport_evaluation, bounded_eta=None):
    """Return the same PDE with an explicit CH chemical-potential evaluation."""
    n_ch, n1, n2, n3, dc, c_le = nd
    raw, raw_t, f, f_t = _raw_fields_and_derivatives(net, x, times)
    z, z_t = raw[..., 0].T, raw_t[..., 0].T
    phi, c = f[..., 0], f[..., 1]
    h = -2.0 * phi**3 + 3.0 * phi**2
    q = c - h * dc - c_le
    if transport_evaluation == "physical_difference":
        dq = graph_action(q, sub_data)
    elif transport_evaluation == "kks_raw_edges":
        # c=cle*(1+tanh(v))/2 + dc*h(phi), hence q=cle*(tanh(v)-1)/2.
        # Factory validation requires the corresponding KKS map.
        raw_q = .5 * c_le * (jnp.tanh(raw[..., 1].T) - 1.0)
        dq = graph_edge_action(raw_q, sub_data)
    else:
        raise ValueError(f"Unknown transport evaluation: {transport_evaluation}")
    r_ch = f_t[..., 1] + n_ch * dq
    if formulation in ("phase", "bounded"):
        gp = 4.0 * phi**3 - 6.0 * phi**2 + 2.0 * phi
        hp = -6.0 * phi**2 + 6.0 * phi
        r_ac = f_t[..., 0] + n3 * graph_action(phi, sub_data) + n2 * gp - n1 * q * dc * hp
        if formulation == "bounded":
            r_ac = bounded_phase_weight(phi, bounded_eta) * r_ac
    elif formulation == "latent":
        row, col, vals, _ = sub_data
        edge = -vals[:, None] * tanh_edge_ratio(z[row], z[col])
        diffusion = jax.ops.segment_sum(edge, row, num_segments=x.shape[0])
        r_ac = normalization * (z_t + n3 * diffusion + n2 * (1.0 - 2.0 * phi) - 3.0 * n1 * q * dc)
    else:
        raise ValueError(f"Unknown phase residual: {formulation}")
    return r_ac, r_ch


def make_loss_vector(config, nd, x_ic, ic, x_bc_zero, x_bc_one):
    """Legacy physical-data objective; compatible with archived config objects."""
    return _make_data_loss_vector(config, nd, x_ic, ic, x_bc_zero, x_bc_one,
        normalize_bc=False, normalize_ic=False, transport_evaluation="physical_difference",
        ch_norm="l1", rms_floor=1e-12)


def validate_normalized_data(config):
    if not config.arch.use_hard_constraint:
        raise ValueError("Normalized concentration data requires the KKS output map")
    if config.physical.c_se != 1.0 or not np.isfinite(config.physical.c_le) or not 0 < config.physical.c_le < 1:
        raise ValueError("Normalized s data requires c_se=1 and finite 0<c_le<1")


def normalized_data_fields(net, points, times):
    """Return (phi, s=c_L/c_le) directly from KKS raw outputs, avoiding cancellation."""
    raw = jax.vmap(lambda t: jax.vmap(net.net, in_axes=(0, None))(points, t))(times)
    fields = net.constraint(raw)
    return jnp.stack([fields[..., 0], .5 * (1 + jnp.tanh(raw[..., 1]))], axis=-1)


def _validate_normalized_initial(config, x_ic, ic, normalize_initial):
    validate_normalized_data(config)
    if not isinstance(normalize_initial, bool):
        raise ValueError("normalize_initial must be an explicit boolean")
    if normalize_initial:
        target = np.asarray(ic)
        if target.shape != (len(x_ic), 2) or not np.all(np.isfinite(target)) or np.any((target[:, 0] < 0) | (target[:, 0] > 1)):
            raise ValueError("Normalized IC requires finite (N,2) data with phi in [0,1]")
        expected = 3 * target[:, 0]**2 - 2 * target[:, 0]**3
        if not np.allclose(target[:, 1], expected, rtol=0, atol=3e-7):
            raise ValueError("Normalized IC requires declared c_0=c_se*h(phi_0), including Dirichlet overrides")


def make_normalized_data_loss_vector(config, nd, x_ic, ic, x_bc_zero, x_bc_one, *, normalize_initial):
    """Normalize concentration BC and optionally IC; both PDEs are unchanged."""
    _validate_normalized_initial(config, x_ic, ic, normalize_initial)
    return _make_data_loss_vector(config, nd, x_ic, ic, x_bc_zero, x_bc_one,
        normalize_bc=True, normalize_ic=normalize_initial,
        transport_evaluation="physical_difference", ch_norm="l1", rms_floor=1e-12)


def validate_transport_options(config):
    e = config.experiment
    if e.transport_evaluation not in ("physical_difference", "kks_raw_edges"):
        raise ValueError(f"Unknown transport evaluation: {e.transport_evaluation}")
    if e.transport_evaluation == "kks_raw_edges":
        validate_normalized_data(config)
    if e.ch_norm not in ("l1", "rms"):
        raise ValueError(f"Unknown CH norm: {e.ch_norm}")
    if not np.isfinite(e.ch_rms_floor) or e.ch_rms_floor <= 0:
        raise ValueError("ch_rms_floor must be finite and positive")


def make_transport_loss_vector(config, nd, x_ic, ic, x_bc_zero, x_bc_one):
    """Explicit transport-evaluation/norm controls; legacy factories stay fixed."""
    validate_transport_options(config)
    mode = config.experiment.data_loss_mode
    if mode not in ("physical", "s_bc", "s_bc_ic"):
        raise ValueError(f"Unknown data loss mode: {mode}")
    if mode != "physical":
        _validate_normalized_initial(config, x_ic, ic, mode == "s_bc_ic")
    return _make_data_loss_vector(config, nd, x_ic, ic, x_bc_zero, x_bc_one,
        normalize_bc=mode != "physical", normalize_ic=mode == "s_bc_ic",
        transport_evaluation=config.experiment.transport_evaluation,
        ch_norm=config.experiment.ch_norm, rms_floor=config.experiment.ch_rms_floor)


def masked_ch_loss(ch, mask, count, *, norm, rms_floor):
    """RMS retains residual units and has a finite, zero gradient at zero."""
    if norm == "l1":
        return jnp.sum(jnp.abs(ch) * mask) / count
    if norm == "rms":
        return jnp.sqrt(jnp.sum(ch**2 * mask) / count + rms_floor**2) - rms_floor
    raise ValueError(f"Unknown CH norm: {norm}")


def _make_data_loss_vector(config, nd, x_ic, ic, x_bc_zero, x_bc_one, *,
                          normalize_bc, normalize_ic, transport_evaluation, ch_norm, rms_floor):
    formulation = config.experiment.formulation
    normalization = config.experiment.latent_normalization
    eta = config.experiment.bounded_eta if formulation == "bounded" else None
    if formulation == "bounded":
        bounded_phase_weight(0.5, eta)
    ac_scale, ch_scale = config.loss.ac_scale, config.loss.ch_scale

    def batch_fields(net, points, times):
        return jax.vmap(lambda t: jax.vmap(net, in_axes=(0, None))(points, t))(times)

    def loss_vector(net, times, x_points, sub_data, core_mask):
        ac, ch = compute_transport_pde(net, x_points, times, sub_data, nd, formulation, normalization,
            transport_evaluation=transport_evaluation, bounded_eta=eta)
        mask = core_mask.astype(ac.dtype)[:, None]
        count = jnp.sum(mask) * times.shape[0]
        ac_loss = ac_scale * jnp.sum(jnp.abs(ac) * mask) / count
        ch_loss = ch_scale * masked_ch_loss(ch, mask, count, norm=ch_norm, rms_floor=rms_floor)
        data_fields = normalized_data_fields if normalize_bc else batch_fields
        zero = data_fields(net, x_bc_zero, times)
        one = data_fields(net, x_bc_one, times)
        bc_zero = jnp.mean(jnp.sum(zero**2, axis=-1))
        bc_one = jnp.mean(jnp.sum((one - 1.0)**2, axis=-1))
        if normalize_ic:
            initial = normalized_data_fields(net, x_ic, jnp.zeros(1, dtype=times.dtype))[0]
            p0 = ic[..., 0]
            target = jnp.stack([p0, 3 * p0**2 - 2 * p0**3], axis=-1)
        else:
            initial = jax.vmap(net, in_axes=(0, None))(x_ic, jnp.array(0.0, dtype=times.dtype))
            target = ic
        ic_loss = jnp.mean(jnp.sum((initial - target)**2, axis=-1))
        return jnp.stack([ac_loss, ch_loss, bc_zero, bc_one, ic_loss])

    return loss_vector
