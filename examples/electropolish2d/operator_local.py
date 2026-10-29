"""Local AD control for the original rough-interface conditional operator.

The D_delta -> -Laplacian limit of the existing physical AC/CH equations is
used. Side walls require phi_n=q_n=0 at positive times. Initial fields are
unchanged and are supervised separately; compatibility at t0 is not assumed.
"""
import jax
import jax.numpy as jnp

from examples.electropolish2d.operator_training import physical_fields


def encoded_fields(net, encoded, points, times, c):
    """Encoder output is shared inside parameter AD, outside coordinate AD."""
    local = jnp.zeros((len(points), 2), dtype=points.dtype)
    prepared = net.prepare_encoded(encoded, points, local)
    return physical_fields(net.decode(prepared, times), c)


def phase_driving_fields(fields, c):
    phi, conc = fields[..., 0], fields[..., 1]
    dc = c.physical.c_se-c.physical.c_le
    q = conc-dc*(3*phi**2-2*phi**3)-c.physical.c_le
    return jnp.stack([phi, q], -1)


def spatial_laplacians(net, encoded, points, times, c):
    """Trace of the two coordinate Hessians, with no dense query Jacobian.

    Query rows are independent: attention keys/values are encoded image tokens,
    not other query rows. A common coordinate-direction JVP therefore gives
    each point's partial derivative; it does not sum cross-query derivatives.
    """
    def fn(x):
        return phase_driving_fields(encoded_fields(net, encoded, x, times, c), c)

    laplacian = jnp.zeros((len(times), len(points), 2), dtype=points.dtype)
    for axis in range(2):
        tangent = jnp.broadcast_to(jax.nn.one_hot(axis, 2, dtype=points.dtype), points.shape)
        def first(x):
            return jax.jvp(fn, (x,), (tangent,))[1]
        laplacian = laplacian+jax.jvp(first, (points,), (tangent,))[1]
    return laplacian


def normal_derivatives(net, encoded, points, normals, times, c):
    """Return (phi_n, q_n/c_le); normalization matches prior KKS local controls."""
    def normalized(x):
        fields = phase_driving_fields(encoded_fields(net, encoded, x, times, c), c)
        return fields/jnp.asarray([1., c.physical.c_le], dtype=fields.dtype)
    return jax.jvp(normalized, (points,), (normals,))[1]


def local_residual(net, encoded, points, times, nd, c):
    nch, n1, n2, n3, dc, cle = nd
    fields, dt = jax.jvp(lambda ts: encoded_fields(net, encoded, points, ts, c),
                         (times,), (jnp.ones_like(times),))
    phi, conc = fields[..., 0], fields[..., 1]
    q = conc-dc*(3*phi**2-2*phi**3)-cle
    lap = spatial_laplacians(net, encoded, points, times, c)
    ac = dt[..., 0]-n3*lap[..., 0]+n2*(4*phi**3-6*phi**2+2*phi)-n1*q*dc*(6*phi-6*phi**2)
    ch = dt[..., 1]-nch*lap[..., 1]
    return ac, ch


def make_loss_vector(c, nd, n_ic, n_bc, n_neumann):
    """Original five losses plus positive-time side-wall no-flux MSE.

    Payload ordering: padded core, IC, top, bottom, two open side walls.
    IC supervision uses prescribed known fields, independent of query features.
    No positive-time reference field enters this function.
    """
    def losses(net, times, payload, sub_data, core_mask):
        images, known, x = payload
        n_core = len(core_mask)

        def per_condition(image, initial):
            encoded = net.encode(image)
            ac, ch = local_residual(net, encoded, x[:n_core], times, nd, c)
            mask = core_mask[None, :].astype(ac.dtype)
            count = jnp.sum(mask)*len(times)
            prepared = net.prepare_encoded(encoded, x[n_core:], jnp.zeros_like(initial[n_core:]))
            ic = physical_fields(net.decode(prepared[:n_ic], jnp.zeros(1, dtype=times.dtype))[0], c)
            boundary = physical_fields(net.decode(prepared[n_ic:n_ic+2*n_bc], times), c)
            wall = x[-n_neumann:]
            normals = jnp.zeros_like(wall).at[:, 0].set(jnp.where(wall[:, 0] == 0., -1., 1.))
            derivative = normal_derivatives(net, encoded, wall, normals, times, c)
            return jnp.stack([
                c.loss.ac_scale*jnp.sum(jnp.abs(ac)*mask)/count,
                c.loss.ch_scale*jnp.sum(jnp.abs(ch)*mask)/count,
                jnp.mean(jnp.sum(boundary[:, :n_bc]**2, -1)),
                jnp.mean(jnp.sum((boundary[:, n_bc:]-1)**2, -1)),
                jnp.mean(jnp.sum((ic-initial[n_core:n_core+n_ic])**2, -1)),
                c.loss.neumann_scale*jnp.mean(jnp.sum(derivative**2, -1))])

        return jax.vmap(per_condition)(images, known).mean(0)
    return losses
