"""Prescribed ellipsoid ICs, using Euclidean nearest-surface distances.

Lagrange-root construction independently derived from the nearest-point normal
condition; see Eberly, DistancePointEllipseEllipsoid.pdf (Geometric Tools).
Parameters are center XYZ, semiaxes XYZ, and quaternion WXYZ. Volume order is
CZYX; point coordinates always XYZ. Positive signed distance means solid.
"""
import numpy as np


def validate_parameters(parameters):
    p = np.asarray(parameters)
    if p.shape[-1:] != (10,) or not np.isfinite(p).all():
        raise ValueError('Ellipsoid parameters must be finite (...,10) arrays')
    if np.any(p[..., 3:6] <= 0) or not np.allclose(np.linalg.norm(p[..., 6:], axis=-1), 1., rtol=0, atol=2e-6):
        raise ValueError('Ellipsoid semiaxes must be positive and quaternion unit length')
    radius = p[..., 3:6].max(axis=-1, keepdims=True)
    if np.any(p[..., :3]-radius <= 0) or np.any(p[..., :3]+radius >= 1):
        raise ValueError('Ellipsoid bounding sphere must be strictly inside the unit cube')


def rotation(quaternion, xp):
    w, x, y, z = quaternion/xp.linalg.norm(quaternion)
    return xp.stack([xp.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)]),
                     xp.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)]),
                     xp.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)])])


def _root_setup(points, parameters, xp):
    a = parameters[3:6]
    q = xp.abs((points-parameters[:3]) @ rotation(parameters[6:], xp))
    a2 = a*a
    amin2 = xp.min(a2)
    b = a2-amin2
    minimum = b == 0
    safe_b = xp.where(minimum, 1., b)
    reduced = xp.sum(xp.where(minimum, 0., (a*q/safe_b)**2), axis=-1)
    singular = xp.all(xp.where(minimum, q == 0, True), axis=-1) & (reduced <= 1)
    upper = amin2+xp.sqrt(xp.sum((a*q)**2, axis=-1))
    return a, q, a2, amin2, b, minimum, safe_b, reduced, singular, upper


def _finish(a, q, a2, amin2, b, minimum, safe_b, reduced, singular, root, xp):
    nearest = a2*q/(b+root[..., None])
    ordinary = xp.sum((nearest-q)**2, axis=-1)
    # Near the smallest-axis singularity, root can be below bisection's
    # absolute resolution. The surface constraint gives the norm of the
    # smallest-axis projection without dividing tiny input coordinates by it.
    large_norm2 = xp.sum(xp.where(minimum, 0., (nearest/a)**2), axis=-1)
    small_input_norm = xp.sqrt(xp.sum(xp.where(minimum, q*q, 0.), axis=-1))
    small_surface_norm = xp.sqrt(amin2*xp.maximum(0., 1-large_norm2))
    stable = xp.sum(xp.where(minimum, 0., (nearest-q)**2), axis=-1)+(small_surface_norm-small_input_norm)**2
    ordinary = xp.where(root < amin2/2, stable, ordinary)
    special_nearest = xp.where(minimum, 0., a2*q/safe_b)
    # On this branch reduced<=1 mathematically; max removes roundoff only.
    special = xp.sum((special_nearest-q)**2, axis=-1)+amin2*xp.maximum(0., 1-reduced)
    distance = xp.sqrt(xp.where(singular, special, ordinary))
    return xp.where(xp.sum((q/a)**2, axis=-1) <= 1, distance, -distance)


def signed_distance(points, parameters, iterations):
    """Fixed-shape bisection; no differentiation through prescribed geometry."""
    import jax
    import jax.numpy as jnp
    if points.ndim != 2 or points.shape[1] != 3 or parameters.shape != (10,) or iterations < 1:
        raise ValueError('Distance needs (N,3) points, (10,) parameters and positive iterations')
    args = _root_setup(points, parameters, jnp)
    a, q, _, _, b, _, _, _, _, upper = args
    def step(_, bracket):
        lo, hi = bracket
        mid = (lo+hi)/2
        value = jnp.sum((a*q/(b+mid[:, None]))**2, axis=-1)
        return jnp.where(value > 1, mid, lo), jnp.where(value > 1, hi, mid)
    lo, hi = jax.lax.fori_loop(0, iterations, step, (jnp.zeros_like(upper), upper))
    return _finish(*args[:-1], (lo+hi)/2, jnp)


def signed_distance_numpy(points, parameters, iterations):
    """Host reference IC with float64 bisection; independently tested geometry."""
    points, parameters = np.asarray(points, dtype=float), np.asarray(parameters, dtype=float)
    validate_parameters(parameters)
    args = _root_setup(points, parameters, np)
    a, q, _, _, b, _, _, _, _, hi = args
    lo = np.zeros_like(hi)
    for _ in range(iterations):
        mid = (lo+hi)/2
        value = np.sum((a*q/(b+mid[:, None]))**2, axis=-1)
        lo, hi = np.where(value > 1, mid, lo), np.where(value > 1, hi, mid)
    return _finish(*args[:-1], (lo+hi)/2, np)


def sample_parameters(key, count, initial):
    import jax
    import jax.numpy as jnp
    kc, ka, kr = jax.random.split(key, 3)
    center = jax.random.uniform(kc, (count, 3), minval=initial.center_min, maxval=initial.center_max)
    axes = jax.random.uniform(ka, (count, 3), minval=initial.axis_min, maxval=initial.axis_max)
    quaternion = jax.random.normal(kr, (count, 4))
    quaternion = quaternion/jnp.linalg.norm(quaternion, axis=-1, keepdims=True)
    return jnp.concatenate([center, axes, quaternion], -1)


def image_points(shape):
    if len(shape) != 4 or shape[0] != 1 or min(shape) < 1:
        raise ValueError('Initial phase volume shape must be (1,Nz,Ny,Nx)')
    zz, yy, xx = np.meshgrid(*(np.linspace(0, 1, n) for n in shape[1:]), indexing='ij')
    return np.stack([xx, yy, zz], -1).reshape(-1, 3)
