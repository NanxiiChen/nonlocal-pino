"""Strict rolling spatial-method pairing and actual PDE-query audit, CPU only.

Both methods retain the same known-input graph producer to isolate sampling.
The AD residual evaluates cores only. Graph construction/storage therefore
remains declared comparison overhead, not a required cost of local AD.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ml_collections import ConfigDict

from .compare import _canonical_config, _same_file
from .online_validation import SHARED_SOURCES


SOURCES = tuple(dict.fromkeys((*SHARED_SOURCES,
    'examples/particle_corrosion2d/local_ad.py',
    'examples/particle_corrosion2d/train.py',
    'examples/particle_corrosion2d/online.py',
    'examples/particle_corrosion2d/online_validation.py',
    'examples/particle_corrosion2d/method_validation.py',
    'examples/particle_corrosion2d/compare.py',
    'examples/particle_corrosion2d/sampling.py',
    'examples/particle_corrosion2d/strategies.py',
    'examples/particle_corrosion2d/dimensional.py',
    'src/nonlocal_pinn/training/runtime.py')))


def sources(c):
    """The new optional module is required only by the declared FEM study."""
    return SOURCES + (('examples/particle_corrosion2d/fem_weak.py',)
                      if c.operator.method_comparison == 'rolling_fem_v1' else ())


def _read(path):
    return json.loads(Path(path).read_text())


def enabled(c):
    return 'method_comparison' in c.operator


def validate_config(c):
    """An explicit original-residual, rolling core/halo comparison only."""
    if (not enabled(c) or c.operator.method_comparison not in ('rolling_spatial_v1','rolling_fem_v1')
            or 'online' not in c.operator or 'full_grid' in c.sample
            or 'phase' in c or 'boundary' in c):
        raise ValueError('Method comparison requires rolling_spatial_v1 or rolling_fem_v1 and original rolling core/halo fields')
    _canonical_config(json.loads(c.to_json()))
    if c.train.compute_dtype != 'float32':
        raise ValueError('The explicit rolling spatial comparison requires FP32')
    if c.nl.method == 'fem_weak':
        from .fem_weak import validate_fem
        validate_fem(c)


def compare_banks(first, second, context, different_operators):
    """FEM changes stencils/padding, never known conditioning or actual cores."""
    if not different_operators:
        return _compare_arrays(first, second, context)
    from .online import FIXED_KEYS, GRAPH_KEYS
    if set(first) != set(FIXED_KEYS+GRAPH_KEYS) or set(second) != set(first):
        raise ValueError(f'{context}: operator bank schema differs')
    _compare_arrays({k:first[k] for k in FIXED_KEYS}, {k:second[k] for k in FIXED_KEYS}, context)
    def cores(bank):
        return dict(points=np.take_along_axis(bank['graph_points'], bank['core_indices'][...,None], axis=2),
                    weights=bank['pde_weights'])
    _compare_arrays(cores(first), cores(second), context+' actual core points/weights')


def _compare_arrays(first, second, context):
    if set(first) != set(second):
        raise ValueError(f'{context}: array schemas differ')
    for name in first:
        a, b = np.asarray(first[name]), np.asarray(second[name])
        if (a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all()
                or not np.isfinite(b).all() or not np.array_equal(a, b)):
            raise ValueError(f'{context}: actual array differs: {name}')


def _compare_npz(first, second, context):
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        _compare_arrays(a, b, context)


def _completed(run, c):
    summary, runtime = _read(run/'summary.json'), _read(run/'runtime.json')
    if (summary['steps'] != c.train.iterations or not runtime['cold_initialization']
            or not runtime['rolling_inputs'] or runtime['training_uses_reference_targets']
            or not (run/'model_final.eqx').is_file()
            or not (run/'optimizer_final.eqx').is_file()):
        raise ValueError('Method control must be a completed cold physics-only rolling run')
    receipt = _read(run/'online_validation.json')
    if (receipt['status'] != 'passed' or Path(receipt['run']).resolve() != run
            or receipt['updates'] != c.train.iterations or receipt['evaluation_overlap'] != 0
            or receipt['future_training_fields'] != 0
            or receipt['exact_final_slots_reconstructed']+receipt['unchanged_initial_slots']
            != c.operator.geometry_count):
        raise ValueError('Method control needs its completed independent online input audit')
    core = _read(run/'core_trace.json')
    if (core['status'] != 'passed' or core['updates'] != c.train.iterations
            or core['shape'] != list(core_shape(c)) or core['dtype'] != 'float32'
            or not (run/'core_trace.npz').is_file()):
        raise ValueError('Method control needs the complete actual-core trace')
    return runtime


def _sampling(run, c, seed):
    """Replay the complete canonical RNG stream, independently of peer files."""
    from .sampling import sample_times
    rng = np.random.default_rng(seed)
    shape = (c.train.iterations, c.operator.condition_batch)
    expected = dict(condition_ids=rng.integers(c.operator.geometry_count, size=shape, dtype=np.int32),
                    pool_ids=rng.integers(c.sample.K_pool, size=shape, dtype=np.int32),
                    times=sample_times(rng, c))
    with np.load(run/'sampling.npz', allow_pickle=False) as actual:
        _compare_arrays(actual, expected, 'Canonical sampling replay')
    with np.load(run/'online_sampling.npz', allow_pickle=False) as actual:
        if set(actual.files) != {'geometry_draw_ids', 'spatial_versions'}:
            raise ValueError('Unexpected consumed geometry/spatial identity schema')
        for name in actual.files:
            values = actual[name]
            if values.shape != shape or values.dtype != np.int64:
                raise ValueError(f'Incomplete consumed geometry/spatial identities: {name}')
        if np.any(actual['spatial_versions'] < 0):
            raise ValueError('Consumed spatial versions must be nonnegative')


def validate_control(c, control, inputs, reference, bank, exclusions):
    """Reject non-method changes before allocating the cold model or training."""
    validate_config(c)
    control = Path(control).resolve(strict=True)
    prior = ConfigDict(_read(control/'config.json'))
    validate_config(prior)
    old, old_method = _canonical_config(json.loads(prior.to_json()))
    new, new_method = _canonical_config(json.loads(c.to_json()))
    fem_pair = 'fem_weak' in (old_method,new_method)
    permitted = ({old_method,new_method} == {'nonlocal','local_ad'} or
                 (c.operator.method_comparison == 'rolling_fem_v1' and fem_pair and old_method != new_method))
    if old != new or not permitted:
        raise ValueError('Rolling method control permits only method and two unit Neumann biases')
    if not fem_pair and c.operator.online != prior.operator.online:
        raise ValueError('Nonlocal/AD controls require identical rolling padding')
    runtime = _completed(control, prior)
    _sampling(control, prior, runtime['seed'])
    if (runtime['inputs'] != str(Path(inputs).resolve(strict=True))
            or runtime['reference'] != str(Path(reference).resolve(strict=True))):
        raise ValueError('Rolling method control requires identical bank/reference paths')
    if _read(control/'excluded_geometries.json') != exclusions:
        raise ValueError('Rolling method control requires identical excluded geometries')
    root = Path(__file__).resolve().parents[2]
    for source in sources(c):
        if not _same_file(control/'source'/source, root/source):
            raise ValueError(f'Rolling method source changed: {source}')
    with np.load(control/'online_initial_inputs.npz', allow_pickle=False) as expected:
        compare_banks(bank, expected, 'Initial rolling bank', fem_pair)
    return runtime


def core_shape(c):
    return (c.train.iterations, c.operator.condition_batch, c.sample.K_core, 3)


class CoreTrace:
    """Record actual gathered x,y and PDE weights, not reconstructed draw IDs.

    The temporary .npy is a bounded disk mapping (234.4 MiB for 20k/B4/K256).
    It is compressed only after every update is recorded, then removed. Failed
    runs retain their partial trace and cannot pass the completion gate.
    """
    def __init__(self, run, c):
        validate_config(c)
        self.run, self.shape = Path(run), core_shape(c)
        self.path = self.run/'core_trace_incomplete.npy'
        if self.path.exists() or (self.run/'core_trace.npz').exists():
            raise FileExistsError('Core trace must start in a fresh run')
        self.array = np.lib.format.open_memmap(self.path, mode='w+', dtype=np.float32, shape=self.shape)
        self.count = 0

    def record(self, iteration, points, weights):
        points, weights = np.asarray(points), np.asarray(weights)
        if (iteration != self.count or iteration >= self.shape[0]
                or points.shape != self.shape[1:3]+(2,) or weights.shape != self.shape[1:3]
                or points.dtype != np.float32 or weights.dtype != np.float32
                or not np.isfinite(points).all() or not np.isfinite(weights).all()
                or np.any(weights <= 0)):
            raise ValueError('Actual core trace requires ordered, finite FP32 cores and positive weights')
        self.array[iteration, ..., :2] = points
        self.array[iteration, ..., 2] = weights
        self.count += 1

    def finish(self):
        if self.count != self.shape[0]:
            raise ValueError('Cannot publish an incomplete actual-core trace')
        self.array.flush()
        np.savez_compressed(self.run/'core_trace.npz', cores=self.array)
        del self.array
        self.path.unlink()
        record = dict(status='passed', updates=self.count, shape=list(self.shape), dtype='float32',
            columns=['x', 'y', 'pde_weight'], uncompressed_bytes=int(np.prod(self.shape))*4,
            compressed_bytes=(self.run/'core_trace.npz').stat().st_size,
            scope='Actual post-gather PDE core coordinates and weights at every update; no halo inference in AD')
        (self.run/'core_trace.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        return record


def _journal_physics(run, *, different_operators=False):
    result = []
    for line in (run/'online_refresh.jsonl').read_text().splitlines():
        row = json.loads(line)
        # Worker scheduling/cache/timing are execution properties, never inputs.
        row.pop('wait_seconds')
        for item in row['records']:
            item.pop('worker_seconds')
            item.pop('cache_hit')
            if different_operators:
                item.pop('actual_halo')
                item.pop('actual_neighbors')
        result.append(row)
    return result


def audit_pair(run, control):
    """Check both complete runs, source/cold bytes and every consumed input."""
    run, control = Path(run).resolve(strict=True), Path(control).resolve(strict=True)
    if run == control:
        raise ValueError('Rolling method comparison needs two distinct runs')
    c = ConfigDict(_read(run/'config.json'))
    validate_config(c)
    runtime = _completed(run, c)
    _sampling(run, c, runtime['seed'])
    with np.load(run/'online_initial_inputs.npz', allow_pickle=False) as bank:
        previous = validate_control(c, control, runtime['inputs'], runtime['reference'],
                                    bank, _read(run/'excluded_geometries.json'))
    if (runtime['seed'] != previous['seed']
            or runtime['rolling_method_control'] != str(control)
            or runtime['excluded_reference_paths'] != previous['excluded_reference_paths']
            or runtime['paired_optimizer_initialization'] is not True):
        raise ValueError('Rolling method seed/control/exclusion/cold optimizer provenance differs')
    root = Path(__file__).resolve().parents[2]
    for source in sources(c):
        if (not _same_file(run/'source'/source, control/'source'/source)
                or not _same_file(run/'source'/source, root/source)):
            raise ValueError(f'Actual rolling method source differs: {source}')
    for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
        if not _same_file(run/name, control/name):
            raise ValueError(f'Cold rolling method state differs: {name}')
    for name in ('sampling.npz', 'online_sampling.npz', 'core_trace.npz'):
        _compare_npz(run/name, control/name, name)
    prior = _read(control/'config.json')
    fem_pair = 'fem_weak' in (c.nl.method,prior['nl']['method'])
    for name in ('online_initial_inputs.npz','online_final_inputs.npz'):
        with np.load(run/name,allow_pickle=False) as a, np.load(control/name,allow_pickle=False) as b:
            compare_banks(a,b,name,fem_pair)
    with np.load(run/'core_trace.npz', allow_pickle=False) as data:
        if set(data.files) != {'cores'} or data['cores'].shape != core_shape(c) or data['cores'].dtype != np.float32:
            raise ValueError('Actual core trace has wrong schema/shape/dtype')
        if np.any(data['cores'][..., 2] <= 0):
            raise ValueError('Actual core trace has nonpositive PDE weights')
    if _journal_physics(run,different_operators=fem_pair) != _journal_physics(control,different_operators=fem_pair):
        raise ValueError('Actual installed geometry/spatial journal differs')
    state_name = f'online_pool_{c.train.iterations:06d}.json'
    if _read(run/state_name) != _read(control/state_name):
        raise ValueError('Actual final rolling pool identities differ')
    return dict(status='passed', run=str(run), control=str(control), updates=c.train.iterations,
        seed=runtime['seed'], identical_initial_model=True, identical_initial_optimizer=True,
        identical_all_geometry_spatial_core_time_draws=True,
        identical_initial_final_known_inputs=True, identical_spatial_stencils=not fem_pair,
        identical_actual_installed_geometries=True,
        actual_core_queries=c.train.iterations*c.operator.condition_batch*c.sample.K_core,
        shared_sources=list(sources(c)), training_uses_reference_targets=False,
        allowed_config_changes=(['nl.method','loss.gradnorm_bias[5:]','fem','operator.online.halo','operator.online.neighbors']
                                if fem_pair else ['nl.method','loss.gradnorm_bias[5:]']),
        cost_scope=('FEM uses signed P1 element support; nonlocal uses visible horizons. Same actual core/IC/BC/geometry/time draws.'
                    if fem_pair else 'AD evaluates no halo; both retain the common graph producer/storage for exact input pairing. Graph preparation is declared unused AD comparison overhead.'),
        limitation='Different native closure versus explicit local Neumann terms; five versus seven balanced losses. No equal-accuracy speed claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    args = parser.parse_args()
    result = audit_pair(args.run, args.control)
    (args.run/'rolling_method_validation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
