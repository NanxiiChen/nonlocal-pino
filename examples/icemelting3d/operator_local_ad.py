"""Local spatial AD of the learned conditional field with analytic known IC jets."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.icemelting3d.ellipsoids import signed_distance
from examples.icemelting3d.ellipsoid_jets import signed_distance_jets
from examples.icemelting3d.operator_data import physics


def validate_local_ad(c):
    s = c.operator.local_ad
    if set(s) != {'boundary_points', 'boundary_weight', 'boundary_norm', 'initial_derivatives'}:
        raise ValueError('Local AD requires explicit boundary points/weight/norm and IC derivative rule')
    if (type(s.boundary_points) is not int or s.boundary_points < 6 or s.boundary_points % 6
            or not np.isfinite(s.boundary_weight) or s.boundary_weight <= 0
            or s.boundary_norm != 'physical_normal_mse'
            or s.initial_derivatives != 'analytic_unique_projection'
            or c.phase.formulation != 'ic_latent'):
        raise ValueError('Require balanced six-face physical Neumann MSE and analytic IC-latent geometry')


def sample_boundary(c, step):
    rng = np.random.default_rng(np.random.SeedSequence([c.train.seed, 12671, step]))
    points = rng.random((c.operator.local_ad.boundary_points, 3)).astype(np.float32)
    face = np.arange(len(points)) % 6
    axis, side = face//2, face % 2
    points[np.arange(len(points)), axis] = side
    normals = (np.eye(3)[axis]*(2*side[:, None]-1)).astype(np.float32)
    return points, normals


def make_initial_jets(c):
    """Prepare known input jets outside parameter differentiation."""
    width = physics(c)[3]

    @eqx.filter_jit
    def prepare(points, boundary, parameters):
        def one(p):
            gradient, laplacian, valid = signed_distance_jets(points, p, c.initial.sdf_iterations)
            bg, _, bv = signed_distance_jets(boundary, p, c.initial.sdf_iterations)
            b0 = signed_distance(boundary, p, c.initial.sdf_iterations)/width
            return (gradient/width, laplacian/width, b0, bg/width,
                    jnp.all(valid) & jnp.all(bv))
        return jax.vmap(one)(parameters)
    return prepare


def correction(net, encoded, points, times):
    prepared = net.prepare_encoded(encoded, points, jnp.zeros((len(points), 0), dtype=points.dtype))
    return net.decode(prepared, times)[..., 0]


def local_fields(net, encoded, points, times, u0, g0, l0):
    """Batched directional JVPs are pointwise: queries do not attend each other."""
    at = lambda xx, tt: correction(net, encoded, xx, tt)
    f, ft = jax.jvp(lambda tt: at(points, tt), (times,), (jnp.ones_like(times),))
    axes = jnp.eye(3, dtype=points.dtype)

    def spatial(direction):
        tangent = jnp.broadcast_to(direction, points.shape)
        first = lambda xx: jax.jvp(lambda yy: at(yy, times), (xx,), (tangent,))[1]
        return jax.jvp(first, (points,), (tangent,))

    gradient, diagonal = jax.vmap(spatial)(axes)
    u = u0[None, :]+times[:, None]*f
    ut = f+times[:, None]*ft
    gu = g0[None, :, :]+times[:, None, None]*jnp.moveaxis(gradient, 0, -1)
    lu = l0[None, :]+times[:, None]*diagonal.sum(axis=0)
    return u, ut, gu, lu


def stable_tanh_jacobian(u):
    # Algebraically 1-tanh(u)^2 without loss to FP32 saturation.
    e = jnp.exp(-2*jnp.abs(u))
    return 4*e/(1+e)**2


def make_loss_fn(c):
    validate_local_ad(c)
    n1, n2, n3, _ = physics(c)

    def loss(net, times, images, u0, points, jets, boundary, normals):
        g0, l0, b0, bg0, _ = jets

        def one(image, initial, gradient0, laplacian0, boundary0, boundary_gradient0):
            encoded = net.encode(image)
            u, ut, gradient, laplacian = local_fields(
                net, encoded, points, times, initial, gradient0, laplacian0)
            phi = jnp.tanh(u)
            residual = ut-n1*(laplacian-2*phi*jnp.sum(gradient**2, axis=-1))-n2*phi+n3
            fb, fbn = jax.jvp(lambda xx: correction(net, encoded, xx, times),
                              (boundary,), (normals,))
            ub = boundary0[None, :]+times[:, None]*fb
            un = jnp.sum(boundary_gradient0*normals, axis=-1)[None, :]+times[:, None]*fbn
            flux = stable_tanh_jacobian(ub)*un
            return jnp.stack((jnp.mean(jnp.abs(residual)), jnp.mean(flux**2)))

        parts = jax.vmap(one)(images, u0, g0, l0, b0, bg0).mean(axis=0)
        return parts[0]+c.operator.local_ad.boundary_weight*parts[1], parts
    return loss


def make_step(c, optimizer):
    loss = make_loss_fn(c)

    @eqx.filter_jit
    def step(net, state, times, images, u0, points, jets, boundary, normals):
        (value, parts), grad = eqx.filter_value_and_grad(loss, has_aux=True)(
            net, times, images, u0, points, jets, boundary, normals)
        finite = jnp.isfinite(value) & jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(grad) if eqx.is_array(a)]))
        updates, state = optimizer.update(grad, state, net)
        return eqx.apply_updates(net, updates), state, value, finite, parts
    return step
