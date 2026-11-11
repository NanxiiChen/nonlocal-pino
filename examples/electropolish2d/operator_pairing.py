"""Direct-path receipts and strict audits for the rolling derivative comparison.

This is a research control, not a new production sampling/default policy.
The local arm reuses graph construction to identify the identical core stream;
only its core/IC/Dirichlet/side-wall queries enter the neural network.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np


CONTROL = 'rolling_derivative_pair'
BASE_LOSSES = ('AC', 'CH', 'top_Dirichlet', 'bottom_Dirichlet', 'initial')
WITNESS_ATOL = 3e-7  # FP32 IC evaluation can be fused differently by query shape.


def validate_control(c):
    if (c.operator.method_control != CONTROL or 'spatial' not in c.sample
            or c.sample.spatial.mode != 'rolling_pool' or c.arch.local_input != 'zero'
            or c.train.optimizer != 'adam' or c.operator.reference_action != 'load'
            or c.loss.use_causal or 'online' not in c.operator):
        raise ValueError('Method control requires the rolling online image-only Adam/load recipe')


def witness_steps(iterations, pool_capacity):
    """Observe the initial draw, first replacement, and last actual update."""
    if iterations < 1 or pool_capacity < 1:
        raise ValueError('Need a positive update budget and pool capacity')
    return sorted({0, min(pool_capacity, iterations-1), iterations-1})


def input_witness(images, known, points, n_core, n_ic, n_bc):
    """Keep only shared prescribed inputs, never a positive-time label."""
    initial = slice(n_core, n_core+n_ic)
    boundary = slice(n_core+n_ic, n_core+n_ic+2*n_bc)
    return dict(images=np.asarray(images), initial=np.asarray(known[:, initial]),
                dirichlet=np.asarray(known[:, boundary]),
                ic_points=np.asarray(points[initial]),
                dirichlet_points=np.asarray(points[boundary]))


def control_record(c, run):
    validate_control(c)
    run = Path(run)
    local = c.nl.method == 'local_ad'
    return dict(kind=CONTROL, method=c.nl.method, config=str(run/'config.json'),
        source_manifest=str(run/'source/files.json'), dataset=c.operator.dataset_path,
        sampling=str(run/'sampling.npz'), coefficients=str(run/'online_coefficients.npy'),
        accepted_counts=str(run/'online_accepted.npy'), inputs=str(run/'paired_inputs.npz'),
        losses=list(BASE_LOSSES)+(['side_Neumann_phi_q_over_cle'] if local else []),
        witness_updates=witness_steps(c.train.iterations, c.sample.K_pool), input_witness_atol=WITNESS_ATOL,
        query_initial_features='zero; IC images only enter encoder; IC targets remain physical',
        time_derivative='shared-time JVP; independent uniform positive-time draws every update',
        spatial_derivative=('coordinate JVP-of-JVP Laplacians; explicit side normal JVP'
                            if local else 'complete sparse nonlocal graph rows on sampled cores'),
        boundary=('positive-time phi_n=q_n/c_le=0; top/bottom corners excluded'
                  if local else 'truncated natural graph boundary; top/bottom Dirichlet losses'),
        initial_boundary_compatibility='original rough profile retained; no t=0 Neumann penalty',
        local_shared_preparation_overhead=local,
        timing_scope='Shared graph/subgraph construction remains in setup/producer timing for input parity; '
                     'local queries omit all graph halos. Do not call this optimized local end-to-end timing.',
        training_uses_positive_time_labels=False)


def validate_config_pair(nonlocal_config, local_config):
    """Reject every undeclared scheme difference, including physical/BC scales."""
    a, b = copy.deepcopy(nonlocal_config), copy.deepcopy(local_config)
    if a['nl']['method'] != 'sparse' or b['nl']['method'] != 'local_ad':
        raise ValueError('Expected sparse versus local_ad, in that order')
    for c in (a, b):
        if c['operator']['method_control'] != CONTROL:
            raise ValueError('Missing explicit rolling derivative control')
    if ('neumann_scale' in a['loss'] or b['loss']['neumann_scale'] != 1.
            or tuple(b['loss']['gradnorm_bias']) != tuple(a['loss']['gradnorm_bias'])+(1.,)):
        raise ValueError('Only the declared sixth Neumann loss/bias may be added')
    b['nl']['method'] = a['nl']['method']
    b['loss']['gradnorm_bias'] = a['loss']['gradnorm_bias']
    del b['loss']['neumann_scale']
    for c in (a, b):
        c['log'].pop('run_name')
    if a != b:
        raise ValueError('Paired configurations differ beyond method/Neumann/run_name')


def _read(path):
    def reject(value):
        raise ValueError(f'Nonfinite JSON at {path}: {value}')
    return json.loads(Path(path).read_text(), parse_constant=reject)


def _assert_same_arrays(first, second, name, *, atol):
    if first.shape != second.shape or first.dtype != second.dtype:
        raise ValueError(f'Paired {name} shape/dtype mismatch')
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError(f'Paired {name} contains nonfinite values')
    if not np.allclose(first, second, rtol=0, atol=atol):
        raise ValueError(f'Paired {name} values differ')


def validate_pair(nonlocal_run, local_run):
    """Audit completed arms with actual arrays and direct file bytes, no hashes."""
    runs = tuple(Path(p) for p in (nonlocal_run, local_run))
    configs = [_read(run/'config.json') for run in runs]
    validate_config_pair(*configs)
    expected_steps = configs[0]['train']['iterations']
    sample, operator = configs[0]['sample'], configs[0]['operator']
    batch, n_ic = operator['condition_batch'], operator['ic_points']
    n_bc, n_sides = 2*sample['Nx'], 2*(sample['Ny']-2)
    for run, c in zip(runs, configs, strict=True):
        record = _read(run/'method_control.json')
        if (record['kind'] != CONTROL or record['method'] != c['nl']['method']
                or record['training_uses_positive_time_labels'] is not False):
            raise ValueError(f'Invalid method receipt at {run}')
        summary = _read(run/'summary.json')
        if (summary['iterations'] != expected_steps or summary['final']['step'] != expected_steps
                or summary['training_uses_positive_time_labels'] is not False
                or summary['spatial_operator'] != c['nl']['method']):
            raise ValueError(f'Expected complete fixed-budget physics-only run at {run}')
    for filename in ('model_initial.eqx', 'optimizer_initial.eqx'):
        if (runs[0]/filename).read_bytes() != (runs[1]/filename).read_bytes():
            raise ValueError(f'Paired cold-state mismatch: {filename}')
    manifests = [_read(run/'source/files.json') for run in runs]
    if manifests[0] != manifests[1] or len(set(manifests[0])) != len(manifests[0]):
        raise ValueError('Paired source manifests differ or contain duplicates')
    for filename in manifests[0]:
        if (runs[0]/'source'/filename).read_bytes() != (runs[1]/'source'/filename).read_bytes():
            raise ValueError(f'Paired source bytes differ: {filename}')
    sampling_keys = ('times', 'draw_ids', 'entry_draw_ids', 'core_ids', 'core_mask',
                     'nonlocal_halo_counts', 'ic_ids', 'dirichlet_ids', 'side_ids')
    with np.load(runs[0]/'sampling.npz') as a, np.load(runs[1]/'sampling.npz') as b:
        if set(a.files) != set(sampling_keys) or set(b.files) != set(sampling_keys):
            raise ValueError('Unexpected rolling sampling receipt schema')
        for key in sampling_keys:
            _assert_same_arrays(a[key], b[key], 'sampling/'+key, atol=0)
        np.testing.assert_array_equal(a['draw_ids'], np.arange(expected_steps))
        np.testing.assert_array_equal(a['entry_draw_ids'], np.arange(expected_steps))
        if a['times'].shape != (expected_steps, configs[0]['sample']['Nt']):
            raise ValueError('Incorrect actual time-draw shape')
        expected = dict(core_ids=(expected_steps, sample['K_core']),
            core_mask=(expected_steps, sample['K_core']), nonlocal_halo_counts=(expected_steps,),
            ic_ids=(n_ic,), dirichlet_ids=(n_bc,), side_ids=(n_sides,))
        if any(a[key].shape != shape for key, shape in expected.items()):
            raise ValueError('Incorrect actual spatial-draw shape')
        if (a['core_mask'].dtype != np.bool_ or np.any(a['core_mask'].sum(1) == 0)
                or np.any(a['core_ids'] < 0) or np.any(a['core_ids'] >= sample['Nx']*sample['Ny'])):
            raise ValueError('Invalid actual core identities/masks')
        if np.any(a['times'] < configs[0]['sample']['t_min']) or np.any(a['times'] >= 1.):
            raise ValueError('Time draws outside the declared positive interval')
    for filename in ('online_coefficients.npy', 'online_accepted.npy'):
        a, b = [np.load(run/filename) for run in runs]
        _assert_same_arrays(a, b, filename, atol=0)
        if len(a) != expected_steps:
            raise ValueError(f'Incomplete actual condition receipt: {filename}')
        expected_shape = ((expected_steps, batch, configs[0]['surface']['n_modes'], 3)
                          if filename == 'online_coefficients.npy' else (expected_steps,))
        if a.shape != expected_shape:
            raise ValueError(f'Incorrect actual condition receipt shape: {filename}')
        if filename == 'online_accepted.npy' and np.any(a < configs[0]['operator']['condition_batch']):
            raise ValueError('A recorded condition draw had too few accepted candidates')
    witness_keys = ('steps', 'images', 'initial', 'dirichlet', 'ic_points', 'dirichlet_points')
    max_gap = 0.
    with np.load(runs[0]/'paired_inputs.npz') as a, np.load(runs[1]/'paired_inputs.npz') as b:
        if set(a.files) != set(witness_keys) or set(b.files) != set(witness_keys):
            raise ValueError('Unexpected actual input witness schema')
        np.testing.assert_array_equal(a['steps'], witness_steps(expected_steps, configs[0]['sample']['K_pool']))
        nw = len(a['steps'])
        expected = dict(images=(nw, batch, *operator['image_shape']), initial=(nw, batch, n_ic, 2),
            dirichlet=(nw, batch, n_bc, 2), ic_points=(nw, n_ic, 2), dirichlet_points=(nw, n_bc, 2))
        if any(a[key].shape != shape for key, shape in expected.items()):
            raise ValueError('Incorrect actual input witness shape')
        for key in witness_keys:
            tolerance = WITNESS_ATOL if key in ('initial', 'dirichlet') else 0.
            _assert_same_arrays(a[key], b[key], 'witness/'+key, atol=tolerance)
            max_gap = max(max_gap, float(np.max(np.abs(a[key].astype(float)-b[key]))))
    return dict(status='passed', nonlocal_run=str(runs[0]), local_run=str(runs[1]),
        updates=expected_steps, paired_seed=configs[0]['train']['seed'],
        same_cold_model_and_optimizer=True, same_source_bytes=True,
        same_actual_time_condition_core_ic_dirichlet_draws=True,
        actual_input_witness_updates=witness_steps(expected_steps, configs[0]['sample']['K_pool']), witness_max_gap=max_gap,
        local_added_loss='positive-time side Neumann, scaled by c_le for q',
        timing_caveat='Local setup retains shared graph construction for pairing; compare synchronized '
                      'updates and separately report this avoidable preparation overhead',
        scope='Matched inputs and initial states do not guarantee bitwise optimizer trajectory reproducibility')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nonlocal-run', type=Path, required=True)
    parser.add_argument('--local-run', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_pair(args.nonlocal_run, args.local_run), indent=2, allow_nan=False))
