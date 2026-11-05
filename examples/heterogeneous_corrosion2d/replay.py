"""Strict cold-run input replay for the declared optimizer/derivative controls.

Only known input artifacts are loaded here. Initial model/optimizer loading is
owned by train.py; no evolved reference values enter this module.
"""
import json
from pathlib import Path

import numpy as np

from examples.heterogeneous_corrosion2d.problem import (
    boundary_mask, material_field, parameter_count, validate,
)


_SPATIAL_KEYS = ('core_ids', 'core_mask', 'ic_ids', 'mouth_ids', 'actual_halo_nodes')


def _leaves(value, prefix=''):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(_leaves(item, f'{prefix}.{key}' if prefix else key))
        return result
    return {prefix: value}


def _check_config(source, target, allow_learning_rate_change=False):
    if type(allow_learning_rate_change) is not bool:
        raise ValueError('allow_learning_rate_change must be an explicit Boolean control')
    old, new = _leaves(source), _leaves(target)
    for key in sorted(set(old) | set(new)):
        if key in old and key in new and old[key] == new[key]:
            continue
        if key == 'train.optimizer' and key in old and key in new:
            allowed = (old[key], new[key]) == ('adam', 'soap')
        elif key == 'nl.method' and key in old and key in new:
            allowed = (old[key], new[key]) == ('sparse', 'local_ad')
        elif key == 'train.iterations' and key in old and key in new:
            allowed = (type(old[key]) is int and type(new[key]) is int
                       and 0 < new[key] <= old[key])
        elif key == 'train.lr_init' and key in old and key in new:
            rates = (old[key], new[key])
            allowed = (allow_learning_rate_change
                       and all(type(rate) in (int, float) for rate in rates)
                       and np.isfinite(rates).all() and min(rates) > 0)
        elif key == 'loss.gradnorm_bias' and key in old and key in new:
            allowed = (source['nl']['method'] == 'sparse'
                       and target['nl']['method'] == 'local_ad'
                       and len(old[key]) == 5 and len(new[key]) == 7
                       and new[key][:5] == old[key]
                       and np.isfinite(new[key]).all() and min(new[key]) > 0)
        elif key == 'loss.neumann_scales' and key not in old and key in new:
            allowed = (target['nl']['method'] == 'local_ad'
                       and np.shape(new[key]) == (2,)
                       and np.isfinite(new[key]).all() and min(new[key]) > 0)
        else:
            allowed = False
        if not allowed:
            raise ValueError(f'Paired config differs outside the declared control: {key}')


def _integer_ids(name, values, shape, n_points, unique=False):
    if (values.shape != shape or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0) or np.any(values >= n_points)):
        raise ValueError(f'Paired {name} must be valid integer indices with shape {shape}')
    if unique and len(np.unique(values)) != values.size:
        raise ValueError(f'Paired {name} contains repeated indices')


def _equal(name, actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
        raise ValueError(f'Paired {name} differs from the exact saved array or dtype')


def check_spatial(actual, expected):
    """Check reconstructed fixed-pool identities, including every padding slot."""
    for name in _SPATIAL_KEYS:
        if name not in actual or name not in expected:
            raise ValueError(f'Paired spatial trace is missing {name}')
        _equal(name, actual[name], expected[name])


def load_control(path, c, points, graph, allow_learning_rate_change=False):
    """Return seed, IC IDs, material/time traces and exact spatial provenance.

    A shorter explicit probe uses the source trace prefix. The canonical trainer
    still draws fresh inputs from the recovered RNG and compares every update
    to this trace, preserving the ordinary sampling work in measured timings.
    The optional learning-rate control admits only positive finite lr_init
    changes; decay, floor and every other configuration guard remain unchanged.
    """
    validate(c)
    root = Path(path).resolve(strict=True)
    source = json.loads((root/'config.json').read_text())
    target = json.loads(c.to_json())
    _check_config(source, target, allow_learning_rate_change)
    runtime = json.loads((root/'runtime.json').read_text())
    seed = runtime['seed']
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Paired runtime seed must be an unsigned 32-bit integer')
    if runtime['training_uses_reference_targets'] is not False:
        raise ValueError('Paired source must declare physics-only training')
    for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
        if not (root/name).is_file():
            raise ValueError(f'Paired run is missing {name}')
    points = np.asarray(points)
    n_points = c.sample.Nx*c.sample.Ny
    if points.shape != (n_points, 2) or not np.isfinite(points).all():
        raise ValueError('Paired coordinates must be the finite complete spatial grid')
    with np.load(root/'graph.npz', allow_pickle=False) as archive:
        required = {'points', 'graph_row', 'graph_col', 'graph_weight', *_SPATIAL_KEYS}
        if not required.issubset(archive.files):
            raise ValueError(f'Paired graph is missing {sorted(required-set(archive.files))}')
        saved = {name: archive[name] for name in required}
    _equal('points', points, saved['points'])
    if len(graph) != 3:
        raise ValueError('Paired graph requires row, column and positive edge weight arrays')
    for name, actual in zip(('graph_row', 'graph_col', 'graph_weight'), graph, strict=True):
        _equal(name, actual, saved[name])
    rows, cols, weights = (saved[name] for name in ('graph_row', 'graph_col', 'graph_weight'))
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError('Paired graph must have a nonempty one-dimensional edge list')
    _integer_ids('graph_row', rows, rows.shape, n_points)
    _integer_ids('graph_col', cols, rows.shape, n_points)
    if weights.shape != rows.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Paired graph edge weights must be finite and positive')
    shape = (c.sample.K_pool, c.sample.K_core)
    ids, mask = saved['core_ids'], saved['core_mask']
    _integer_ids('core_ids', ids, shape, n_points)
    if mask.shape != shape or mask.dtype != np.dtype(bool) or np.any(mask.sum(axis=1) == 0):
        raise ValueError('Paired core_mask must contain nonempty Boolean core rows')
    mouth = np.flatnonzero(boundary_mask(c, points))
    _integer_ids('mouth_ids', saved['mouth_ids'], mouth.shape, n_points, unique=True)
    if not np.array_equal(saved['mouth_ids'], mouth):
        raise ValueError('Paired mouth indices differ from the prescribed reservoir opening')
    _integer_ids('ic_ids', saved['ic_ids'], (c.operator.ic_points,), n_points, unique=True)
    for index, (row_ids, row_mask) in enumerate(zip(ids, mask, strict=True)):
        count = int(row_mask.sum())
        if (not np.array_equal(row_mask, np.arange(c.sample.K_core) < count)
                or len(np.unique(row_ids[row_mask])) != count
                or np.isin(row_ids[row_mask], mouth).any()
                or np.any(row_ids[~row_mask] != row_ids[count-1])):
            raise ValueError(f'Paired core pool row {index} has invalid cores or padding')
    counts = saved['actual_halo_nodes']
    if (counts.shape != (c.sample.K_pool,) or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < c.sample.K_core) or np.any(counts > n_points)):
        raise ValueError('Paired actual_halo_nodes must be valid full core-plus-halo counts')
    with np.load(root/'sampling.npz', allow_pickle=False) as archive:
        if not {'parameters', 'times'}.issubset(archive.files):
            raise ValueError('Paired sampling requires parameters and times')
        parameters, times = archive['parameters'], archive['times']
    steps = source['train']['iterations']
    expected_parameters = (steps, c.operator.condition_batch, parameter_count(c))
    if (parameters.shape != expected_parameters or parameters.dtype != np.dtype('float32')
            or not np.isfinite(parameters).all()):
        raise ValueError(f'Paired material trace must be finite float32 {expected_parameters}')
    if (times.shape != (steps, c.sample.Nt) or times.dtype != np.dtype('float32')
            or not np.isfinite(times).all() or np.any(times < 0) or np.any(times > c.sample.t_max)):
        raise ValueError('Paired time trace has invalid shape, dtype or physical range')
    scales = np.asarray(c.material.coefficient_scale)
    if scales.ndim == 1 and np.any(parameters[..., scales == 0] != 0):
        raise ValueError('Paired material trace violates fixed zero-variance coefficients')
    # The prescribed logistic map is analytically bounded at every coordinate.
    # Check its implementation on all sampled inputs at domain-wide probe nodes,
    # without constructing a large trajectory-by-grid tensor during startup.
    probes = points[np.unique(np.linspace(0, n_points-1, 17).astype(int))]
    flat = parameters.reshape(-1, parameter_count(c))
    for start in range(0, len(flat), 1024):
        values = material_field(c, flat[start:start+1024], probes, np)
        if (not np.isfinite(values).all() or np.any(values < c.material.minimum)
                or np.any(values > c.material.maximum)):
            raise ValueError('Paired known material field violates its finite positive bounds')
    expected = {name: saved[name] for name in _SPATIAL_KEYS}
    return seed, saved['ic_ids'], parameters[:c.train.iterations], times[:c.train.iterations], expected
