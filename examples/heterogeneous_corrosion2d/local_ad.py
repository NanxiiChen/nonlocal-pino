"""Local spatial AD control, with explicit reflecting-wall boundary losses.

The fixed material encoder is shared across all coordinate derivatives, while
remaining inside parameter differentiation. Independent query rows permit
coordinate-direction JVPs without constructing a dense spatial Jacobian.
"""
import jax
import jax.numpy as jnp
import numpy as np

from examples.heterogeneous_corrosion2d.network import physical_fields
from examples.heterogeneous_corrosion2d.problem import nondim


LOSS_NAMES = ('ac', 'ch', 'mouth_phi', 'mouth_c_over_cle', 'initial',
              'wall_phi_normal', 'wall_q_normal_over_cle')


def encoded_raw(net, encoded, points, times, c):
    """Only coordinates enter decoder queries; times remain physically scaled."""
    local = jnp.empty((len(points), 0), dtype=points.dtype)
    prepared = net.prepare_encoded(encoded, points, local)
    return net.decode(prepared, times / c.sample.t_max)


def phase_driving_fields(raw, c):
    """Physical phase and q from the stable KKS liquid channel, not subtraction."""
    phi = .5 * (1 + jnp.tanh(raw[..., 0]))
    q = c.physical.c_le * (raw[..., 1] - 1)
    return jnp.stack((phi, q), axis=-1)


def spatial_laplacians(net, encoded, points, times, c):
    """Return (time,space,2) Laplacians of phi and q by directional JVPs.

    ConditionalFieldOperator queries never attend to other query rows. A common
    x or y tangent therefore differentiates each independent spatial coordinate,
    instead of summing interactions between points. Encoder tokens stay fixed
    only with respect to coordinates, never with respect to network parameters.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('Local derivative points must have shape (N,2)')

    def fields(x):
        return phase_driving_fields(encoded_raw(net, encoded, x, times, c), c)

    result = jnp.zeros((len(times), len(points), 2), dtype=points.dtype)
    for axis in range(2):
        tangent = jnp.broadcast_to(jax.nn.one_hot(axis, 2, dtype=points.dtype), points.shape)

        def first(x):
            return jax.jvp(fields, (x,), (tangent,))[1]

        result = result + jax.jvp(first, (points,), (tangent,))[1]
    return result


def normal_derivatives(net, encoded, points, normals, times, c):
    """Return L_ref*(phi_n, q_n/c_le) at independently supplied wall points."""
    if points.ndim != 2 or points.shape[1] != 2 or normals.shape != points.shape:
        raise ValueError('Boundary points and normals must share shape (N,2)')

    def normalized(x):
        raw = encoded_raw(net, encoded, x, times, c)
        phi = .5 * (1 + jnp.tanh(raw[..., 0]))
        return c.geometry.length * jnp.stack((phi, raw[..., 1] - 1), axis=-1)

    return jax.jvp(normalized, (points,), (normals,))[1]


def local_residual(net, encoded, points, times, kappa, c):
    """Original AC/CH equations with constant diffusion and pointwise mobility."""
    if kappa.shape != (len(points),):
        raise ValueError('Local mobility must align with the core point block')
    raw, raw_t = jax.jvp(lambda ts: encoded_raw(net, encoded, points, ts, c),
                         (times,), (jnp.ones_like(times),))
    fields, fields_t = jax.jvp(lambda values: physical_fields(values, c), (raw,), (raw_t,))
    phi = fields[..., 0]
    q = c.physical.c_le * (raw[..., 1] - 1)
    laplacian = spatial_laplacians(net, encoded, points, times, c)
    nch, n1, n2, n3, dc, _ = nondim(c)
    hp = 6 * phi * (1 - phi)
    gp = 2 * phi * (1 - phi) * (1 - 2 * phi)
    ac = fields_t[..., 0] + kappa[None, :] * (-n3 * laplacian[..., 0] + n2 * gp - n1 * q * dc * hp)
    ch = fields_t[..., 1] - nch * laplacian[..., 1]
    return ac, ch


def make_loss_vector(c, n_core, n_ic, n_bc, n_neumann, normals, boundary_weights):
    """Five matching physical losses plus two length-weighted Neumann losses.

    Query order: padded PDE core, fixed IC nodes, reservoir mouth, wall points.
    The root trainer provides normals and line-quadrature weights separately;
    no normal is guessed from coordinates, and reservoir points must be absent
    from that wall rule. Payload contains images, core kappa, queries and IC.
    """
    sizes = (n_core, n_ic, n_bc, n_neumann)
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in sizes):
        raise ValueError('Core/initial/mouth/Neumann counts must be positive integers')
    normals_host = np.asarray(normals)
    weights_host = np.asarray(boundary_weights)
    if (normals_host.shape != (n_neumann, 2) or not np.isfinite(normals_host).all()
            or not np.allclose(np.linalg.norm(normals_host, axis=-1), 1, rtol=0, atol=1e-6)):
        raise ValueError('Neumann normals must be finite unit vectors with shape (Nwall,2)')
    if (weights_host.shape != (n_neumann,) or not np.isfinite(weights_host).all()
            or np.any(weights_host <= 0)):
        raise ValueError('Neumann quadrature weights must be positive finite lengths')
    scales = np.asarray(c.loss.neumann_scales)
    if scales.shape != (2,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('loss.neumann_scales must contain two positive finite values')
    if not np.isfinite(c.geometry.length) or c.geometry.length <= 0:
        raise ValueError('Positive geometry.length is required for Neumann normalization')
    cle = c.physical.c_le

    def losses(net, times, payload, sub_data, core_mask):
        if sub_data is not None:
            raise ValueError('Local AD loss must receive sub_data=None')
        images, kappa, queries, initial_ic = payload
        if (queries.shape != (sum(sizes), 2) or initial_ic.shape != (n_ic, 2)
                or kappa.shape != (len(images), n_core)
                or core_mask.shape != (n_core,) or core_mask.dtype != jnp.bool_
                or times.ndim != 1 or len(times) < 1):
            raise ValueError('Local loss query, mobility, IC, core mask or time shape mismatch')
        normal_array = jnp.asarray(normals_host, dtype=queries.dtype)
        line_weights = jnp.asarray(weights_host, dtype=queries.dtype)

        def one(image, mobility):
            encoded = net.encode(image)
            ac, ch = local_residual(net, encoded, queries[:n_core], times, mobility, c)
            mask = core_mask[None, :].astype(ac.dtype)
            count = jnp.sum(mask) * len(times)
            fixed = queries[n_core:n_core+n_ic+n_bc]
            cached = net.prepare_encoded(encoded, fixed, jnp.empty((len(fixed), 0), dtype=queries.dtype))
            initial = physical_fields(net.decode(cached[:n_ic], jnp.zeros(1, dtype=times.dtype))[0], c)
            mouth = physical_fields(net.decode(cached[n_ic:], times / c.sample.t_max), c)
            initial_error = (initial-initial_ic) / jnp.asarray((1., cle), dtype=initial.dtype)
            normal = normal_derivatives(net, encoded, queries[-n_neumann:], normal_array, times, c)
            normal_mse = jnp.sum(normal**2 * line_weights[None, :, None], axis=(0, 1)) / (len(times)*jnp.sum(line_weights))
            return jnp.stack((
                c.loss.ac_scale*jnp.sum(jnp.abs(ac)*mask)/count,
                c.loss.ch_scale*jnp.sum(jnp.abs(ch)*mask)/count,
                jnp.mean(mouth[..., 0]**2),
                jnp.mean((mouth[..., 1]/cle)**2),
                jnp.mean(jnp.sum(initial_error**2, axis=-1)),
                c.loss.neumann_scales[0]*normal_mse[0],
                c.loss.neumann_scales[1]*normal_mse[1]))

        return jax.vmap(one)(images, kappa).mean(axis=0)

    return losses
