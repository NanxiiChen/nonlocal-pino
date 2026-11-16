"""Finite prescribed initial profiles; no evolved fields or reference labels."""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def load_library(c):
    spec = c.operator.condition_library
    if (set(spec) != {'path', 'count', 'sampling'}
            or spec.sampling != 'uniform_with_replacement'
            or isinstance(spec.count, bool) or not isinstance(spec.count, int)
            or spec.count < 1):
        raise ValueError('Condition library requires explicit path/count/uniform_with_replacement')
    path = Path(spec.path).resolve(strict=True)
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest['schema'] != 'electropolish_initial_library_v1' or manifest['evolved_fields_generated']:
        raise ValueError('Only prescribed initial-profile libraries are allowed')
    for name in ('surface', 'physical', 'nondim'):
        if manifest['config'][name] != json.loads(c.to_json())[name]:
            raise ValueError('Initial library physical prior differs: '+name)
    for name in ('curve_points', 'sdf_window', 'height_min', 'height_max'):
        if manifest['config']['operator']['online'][name] != c.operator.online[name]:
            raise ValueError('Initial library acceptance differs: '+name)
    with np.load(path/'coefficients.npz', allow_pickle=False) as data:
        if data.files != ['coefficients']:
            raise ValueError('Library permits only coefficient triples')
        coefficients = data['coefficients']
    if (coefficients.shape != (manifest['count'], c.surface.n_modes, 3)
            or coefficients.dtype != np.float32 or not np.isfinite(coefficients).all()
            or spec.count > len(coefficients)):
        raise ValueError('Invalid fixed coefficient library shape/dtype/count')
    a, f, p = np.moveaxis(coefficients, -1, 0)
    if (np.any(a < np.float32(.3*c.surface.amplitude)) or np.any(a > c.surface.amplitude)
            or np.any(f < c.surface.freq_lo) or np.any(f > c.surface.freq_hi)
            or np.any(p < 0) or np.any(p >= np.float32(2*np.pi))):
        raise ValueError('Initial coefficients lie outside the original prior')
    if len(np.unique(coefficients.reshape(len(coefficients), -1), axis=0)) != len(coefficients):
        raise ValueError('Fixed initial profiles must be distinct')
    from .online_conditions import rough_surface
    xs = jnp.asarray(np.linspace(0, 1, c.operator.online.curve_points).astype(np.float32))
    for profile in coefficients:
        ys = np.asarray(rough_surface(xs, jnp.asarray(profile), c.surface.y0))
        if not (ys.min() > c.operator.online.height_min and ys.max() < c.operator.online.height_max):
            raise ValueError('Fixed profile violates the online height acceptance')
    return coefficients[:spec.count].copy(), manifest
