"""Local spatial-AD control on identical particle geometries and PDE core points."""
import jax
import jax.numpy as jnp

from examples.particle_corrosion2d.network import decode, encode, physical_fields, time_jets
from examples.particle_corrosion2d.losses import weighted_mean


LOSS_NAMES = ('ac', 'ch', 'mouth_phi', 'mouth_c_over_cle', 'initial',
              'wall_phi_normal', 'wall_q_normal_over_cle')


def encoded_raw(net, encoded, points, times, c):
    return decode(net, net.prepare_encoded(encoded, points), times, c)


def spatial_laplacians(net, encoded, points, times, c):
    """Independent-query coordinate JVPs; no dense spatial-query Jacobian."""
    def fields(x):
        raw = encoded_raw(net, encoded, x, times, c)
        phi = .5*(1+jnp.tanh(raw[..., 0]))
        # Dropping the constant does not change any spatial derivative.
        q = c.physical.c_le*raw[..., 1]
        return jnp.stack((phi, q), axis=-1)

    laplacian = jnp.zeros((len(times), len(points), 2), dtype=points.dtype)
    for axis in range(2):
        tangent = jnp.broadcast_to(jax.nn.one_hot(axis, 2, dtype=points.dtype), points.shape)

        def first(x):
            return jax.jvp(fields, (x,), (tangent,))[1]

        laplacian = laplacian+jax.jvp(first, (points,), (tangent,))[1]
    return laplacian


def normal_derivatives(net, encoded, points, normals, times, c):
    """L_ref*(phi_n,q_n/c_le), with outward normals of the perforated domain."""
    if points.ndim != 2 or points.shape[1] != 2 or normals.shape != points.shape:
        raise ValueError('Wall points and normals must have the same (N,2) shape')

    def normalized(x):
        raw = encoded_raw(net, encoded, x, times, c)
        return c.geometry.length*jnp.stack((.5*(1+jnp.tanh(raw[..., 0])), raw[..., 1]), axis=-1)

    return jax.jvp(normalized, (points,), (normals,))[1]


def local_residual(net, encoded, points, times, c):
    raw, fields, fields_t = time_jets(net, net.prepare_encoded(encoded, points), times, c)
    lap = spatial_laplacians(net, encoded, points, times, c)
    p = c.physical
    phi, q = fields[..., 0], p.c_le*(raw[..., 1]-1)
    hp, gp = 6*phi*(1-phi), 2*phi*(1-phi)*(1-2*phi)
    ac = fields_t[..., 0]+p.kappa*(-p.n_ac3*lap[..., 0]+p.n_ac2*gp-p.n_ac1*q*(p.c_se-p.c_le)*hp)
    ch = fields_t[..., 1]-p.n_ch*lap[..., 1]
    return ac, ch


def condition_losses(net, times, data, c):
    encoded = encode(net, data)
    cores = data['graph_points'][data['core_indices']]
    ac, ch = local_residual(net, encoded, cores, times, c)
    n_ic = len(data['ic_points'])
    fixed = jnp.concatenate((data['ic_points'], data['mouth_points']), axis=0)
    cached = net.prepare_encoded(encoded, fixed)
    initial = physical_fields(decode(net, cached[:n_ic], jnp.zeros(1, dtype=times.dtype), c)[0], c)
    mouth = physical_fields(decode(net, cached[n_ic:], times, c), c)
    initial_error = (initial-data['initial_ic'])/jnp.asarray((1., c.physical.c_le), dtype=initial.dtype)
    normal = normal_derivatives(net, encoded, data['wall_points'], data['wall_normals'], times, c)
    return jnp.stack((
        c.loss.ac_scale*weighted_mean(jnp.abs(ac), data['pde_weights']),
        c.loss.ch_scale*weighted_mean(jnp.abs(ch), data['pde_weights']),
        weighted_mean(mouth[..., 0]**2, data['mouth_weights']),
        weighted_mean((mouth[..., 1]/c.physical.c_le)**2, data['mouth_weights']),
        weighted_mean(jnp.sum(initial_error**2, axis=-1), data['ic_weights']),
        c.loss.neumann_scales[0]*weighted_mean(normal[..., 0]**2, data['wall_weights']),
        c.loss.neumann_scales[1]*weighted_mean(normal[..., 1]**2, data['wall_weights'])))


def make_loss_vector(c):
    def loss(net, times, payload):
        return jax.vmap(lambda data: condition_losses(net, times, data, c))(payload).mean(axis=0)
    return loss
