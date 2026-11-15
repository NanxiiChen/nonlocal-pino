"""Known initial-geometry derivatives, not AD through a bisection program.

For the unique closest surface point y, outward normal n and outward distance
d=-s, let P=I-nn^T, A=diag(a^-2), and S=PAP/|Ay|. The normal-coordinate map
x=y+d*n gives H_s=-S(I+d*S)^-1 and grad(s)=-n. These formulas apply off the
medial axis. ``valid`` exposes singular projections instead of repairing them.
The original signed-distance forward evaluation remains in ellipsoids.py.
"""
import jax
import jax.numpy as jnp

from examples.icemelting3d.ellipsoids import _root_setup, rotation


def signed_distance_jets(points, parameters, iterations):
    """Return gradient, Laplacian, validity for a prescribed ellipsoid SDF."""
    if points.ndim != 2 or points.shape[1] != 3 or parameters.shape != (10,) or iterations < 1:
        raise ValueError('Geometry jets require (N,3) points, ten parameters and positive iterations')
    a, q, a2, amin2, b, minimum, _, _, singular, upper = _root_setup(points, parameters, jnp)

    def bisect(_, bracket):
        lo, hi = bracket
        mid = (lo+hi)/2
        value = jnp.sum((a*q/(b+mid[:, None]))**2, axis=-1)
        return jnp.where(value > 1, mid, lo), jnp.where(value > 1, hi, mid)

    lo, hi = jax.lax.fori_loop(0, iterations, bisect, (jnp.zeros_like(upper), upper))
    root = (lo+hi)/2
    y = a2*q/(b+root[:, None])
    # Same surface-constraint reconstruction as the existing forward SDF.
    # It resolves the smallest-axis component without dividing by a tiny root.
    large_norm2 = jnp.sum(jnp.where(minimum, 0., (y/a)**2), axis=-1)
    small_input = jnp.sqrt(jnp.sum(jnp.where(minimum, q*q, 0.), axis=-1))
    small_surface = jnp.sqrt(amin2*jnp.maximum(0., 1-large_norm2))
    small_direction = jnp.where(minimum, q, 0.)/jnp.where(small_input > 0, small_input, 1.)[:, None]
    stable_y = jnp.where(minimum, small_surface[:, None]*small_direction, y)
    y = jnp.where((root < amin2/2)[:, None], stable_y, y)
    r = rotation(parameters[6:], jnp)
    z = (points-parameters[:3]) @ r
    y = y*jnp.sign(z)
    ay = y/a2
    g = jnp.linalg.norm(ay, axis=-1)
    normal = ay/g[:, None]
    p = jnp.eye(3, dtype=points.dtype)-normal[:, :, None]*normal[:, None, :]
    shape = (p/a2[None, None, :]) @ p/g[:, None, None]
    distance = jnp.sum((z-y)*normal, axis=-1)
    offset = jnp.eye(3, dtype=points.dtype)+distance[:, None, None]*shape
    hessian = -jnp.linalg.solve(offset, shape)
    gradient = -normal @ r.T
    laplacian = jnp.trace(hessian, axis1=-2, axis2=-1)
    valid = (~singular & (g > 0) & (jnp.linalg.det(offset) > 0)
             & jnp.isfinite(laplacian) & jnp.all(jnp.isfinite(gradient), axis=-1))
    return gradient, laplacian, valid
