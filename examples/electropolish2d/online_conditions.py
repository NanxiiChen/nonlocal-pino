"""Online continuous rough-interface samples, with the existing discrete SDF.

Random amplitude/frequency/phase triples are generated afresh each update.
Height validity conditions the sampling distribution; there is no finite input
bank or evolved-field training data. Geometry arrays retain fixed JIT shapes.
"""
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def rough_surface(x, coefficients, y0):
    amplitude, frequency, phase = coefficients.T
    angle = 2*jnp.pi*x[:, None]*frequency + phase
    return y0 + jnp.sum(amplitude*(jnp.sin(angle)+jnp.cos(angle+.3)), axis=-1)


def initial_fields(coefficients, points, curve_x, nearest, *, y0, k, c_se, window):
    """Windowed signed Euclidean segment distance; positive below profile.

    The 2000-vertex/±16-segment search matches the original IC, but is not
    guaranteed to find the globally closest segment of the entire curve.
    """
    curve_y = rough_surface(curve_x, coefficients, y0)
    dx, dy = jnp.diff(curve_x), jnp.diff(curve_y)
    length_squared = dx*dx + dy*dy + 1e-20

    def segment_distance(offset, distance_squared):
        segment = jnp.clip(nearest + offset - window, 0, len(curve_x)-2)
        rx = points[:, 0] - curve_x[segment]
        ry = points[:, 1] - curve_y[segment]
        projection = jnp.clip((rx*dx[segment]+ry*dy[segment])/length_squared[segment], 0., 1.)
        squared = (rx-projection*dx[segment])**2 + (ry-projection*dy[segment])**2
        return jnp.minimum(distance_squared, squared)

    distance_squared = jax.lax.fori_loop(0, 2*window+1, segment_distance,
                                       jnp.full(len(points), jnp.inf, dtype=points.dtype))
    sign = jnp.where(points[:, 1] < rough_surface(points[:, 0], coefficients, y0), 1., -1.)
    phi = .5*(1+jnp.tanh(k*sign*jnp.sqrt(distance_squared)))
    phi = jnp.where(jnp.isclose(points[:, 1], .5), 0., phi)
    phi = jnp.where(jnp.isclose(points[:, 1], 0.), 1., phi)
    return jnp.stack([phi, c_se*(3*phi**2-2*phi**3)], -1)


def validate_online_config(c):
    o = c.operator.online
    if 'training_indices' in c.operator:
        raise ValueError('Online sampling cannot also select a finite training subset')
    if o.curve_points != 2000 or o.sdf_window != 16:
        raise ValueError('Online pilot must preserve the existing2000-vertex/±16-segment SDF')
    if isinstance(o.candidate_factor, bool) or not isinstance(o.candidate_factor, int) or o.candidate_factor < 1:
        raise ValueError('online.candidate_factor must be a positive integer')
    if not (np.isfinite(o.height_min) and np.isfinite(o.height_max)
            and 0 < o.height_min < c.surface.y0 < o.height_max < .5):
        raise ValueError('Online profile height acceptance must stay strictly inside the domain')
    if (c.surface.n_modes < 1 or not 0 < c.surface.amplitude
            or not 0 < c.surface.freq_lo <= c.surface.freq_hi):
        raise ValueError('Invalid continuous rough-interface distribution')


def make_sampler(c):
    """Return (sample, geometry); sample has fixed shapes and no parameter AD.

    Each call oversamples a fixed candidate count, keeps the first B profiles
    within the declared height bounds, and reports accepted_count. The caller
    MUST reject the whole update when accepted_count<B; padded selections are
    never a silent replacement for missing valid profiles.
    """
    validate_online_config(c)
    o = c.operator.online
    _, height, width = c.operator.image_shape
    sx, sy = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, .5, height), indexing='xy')
    sensors = np.column_stack([sx.ravel(), sy.ravel()]).astype(np.float32)
    curve_x = jnp.asarray(np.linspace(0, 1, o.curve_points).astype(np.float32))
    sensor_nearest = jnp.asarray(np.searchsorted(np.asarray(curve_x), sensors[:, 0]), dtype=jnp.int32)
    batch, candidates = c.operator.condition_batch, o.candidate_factor*c.operator.condition_batch
    physical_k = np.sqrt(c.physical.omega_phi/(2*c.physical.alpha_phi))*c.nondim.L_c
    fields = partial(initial_fields, curve_x=curve_x, y0=c.surface.y0,
                     k=physical_k, c_se=c.physical.c_se, window=o.sdf_window)

    def geometry(points):
        """Precompute nearest curve indices for a fixed query set once."""
        return jnp.asarray(np.searchsorted(np.asarray(curve_x), np.asarray(points)[:, 0]), dtype=jnp.int32)

    @eqx.filter_jit
    def sample(key, points, nearest):
        uniform = jax.random.uniform(key, (candidates, c.surface.n_modes, 3), dtype=jnp.float32)
        amplitudes = (.3+.7*uniform[..., 0])*c.surface.amplitude
        frequencies = c.surface.freq_lo+(c.surface.freq_hi-c.surface.freq_lo)*uniform[..., 1]
        phases = 2*jnp.pi*uniform[..., 2]
        all_coefficients = jnp.stack([amplitudes, frequencies, phases], -1)
        curves = jax.vmap(lambda a: rough_surface(curve_x, a, c.surface.y0))(all_coefficients)
        valid = (curves.min(-1) > o.height_min) & (curves.max(-1) < o.height_max)
        count = jnp.sum(valid)
        chosen = jnp.nonzero(valid, size=batch, fill_value=0)[0]
        coefficients = all_coefficients[chosen]
        # condition images and all local IC values use the SAME profile triples.
        image_fields = jax.vmap(lambda a:fields(a, jnp.asarray(sensors), nearest=sensor_nearest))(coefficients)
        local = jax.vmap(lambda a:fields(a, points, nearest=nearest))(coefficients)
        images = image_fields[..., 0].reshape(batch, 1, height, width)
        return images, local, coefficients, count

    return sample, geometry
