"""Narrow cold10k -> cold20k/LR control; no physical or shape changes."""
import json
from pathlib import Path

import numpy as np


def _same_recipe(now, before):
    value = json.loads(json.dumps(now))
    for key in ('iterations', 'lr_decay_rate'):
        value['train'][key] = before['train'][key]
    value['log']['run_name'] = before['log']['run_name']
    if value != before:
        raise ValueError('Cold20k may change only iterations, LR decay rate and run name')


def comparison_config(c, control):
    path = Path(control)
    if not path.is_absolute():
        raise ValueError('Budget control requires an absolute completed run')
    before = json.loads((path/'config.json').read_text())
    now = json.loads(c.to_json())
    if (json.loads((path/'summary.json').read_text())['status'] != 'completed'
            or json.loads((path/'validation.json').read_text())['status'] != 'replay_and_hard_constraints_passed'
            or json.loads((path/'online-validation.json').read_text())['status'] != 'actual_rolling_inputs_reconstructed'):
        raise ValueError('Budget control requires completed replay and online-input audits')
    if (before['train']['iterations'] != 10000 or now['train']['iterations'] != 20000
            or before['train']['lr_decay_rate'] != .7
            or now['train']['lr_decay_rate'] != 10.**(-.2)
            or now['train']['lr_init'] != .001 or now['train']['lr_min'] != 1e-5
            or now['train']['lr_decay_every'] != 2000
            or now['train']['microbatch'] != dict(kind='time', size=20)
            or now['arch']['width'] != 192 or now['operator']['condition_batch'] != 8
            or now['nl']['method'] != 'integral_moment'):
        raise ValueError('Explicit192/B8/time20 cold10k ->20k exponential-LR control required')
    _same_recipe(now, before)
    return before


def resource_evidence(c, control, probe, gpu, fraction):
    """Reuse same-shape evidence explicitly; not a new LR accuracy/resource run."""
    comparison_config(c, control)
    path = Path(probe)
    if not path.is_absolute():
        raise ValueError('Absolute qualified resource-probe path required')
    evidence = json.loads((path/'result.json').read_text())
    _same_recipe(json.loads(c.to_json()), evidence['config'])
    device = evidence['hardware_trial']
    limit = min(.95*device['total_memory_bytes'], .97*evidence['memory']['bytes_limit'])
    if (evidence['status'] != 'finite_production_probe_passed' or evidence['updates'] < 7
            or evidence['fitted_probe_weights_exported']
            or not evidence['capacity_trial']['full_geometry_pool_on_device']
            or not evidence['capacity_trial']['finite_nominal_evaluation']
            or fraction != .97 or device['allocator_fraction'] != fraction
            or gpu['uuid'] != device['gpu_uuid'] or gpu['name'] != device['gpu_name']
            or gpu['total_memory_bytes'] != device['total_memory_bytes']
            or evidence['memory']['peak_bytes_in_use'] > limit):
        raise ValueError('Historical same-shape resource evidence does not cover this GPU/execution')
    for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
        if (path/name).read_bytes() != (Path(control)/name).read_bytes():
            raise ValueError(f'Qualified unfitted resource state differs from control: {name}')
    return dict(status='historical_same_shape_resource_evidence_reused', probe=str(path),
                gpu=gpu, allocator_fraction=fraction,
                historical_peak_bytes=evidence['memory']['peak_bytes_in_use'],
                historical_warm_update_seconds=evidence['warm_update_seconds'],
                new_resource_updates=0, fitted_state_imported=False,
                scope='Only budget/LR decay differ; actual cold state and inputs are checked separately')


def audit_cold(run, control):
    from ml_collections import ConfigDict
    run, control = Path(run), Path(control)
    comparison_config(ConfigDict(json.loads((run/'config.json').read_text())), control)
    for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
        if (run/name).read_bytes() != (control/name).read_bytes():
            raise ValueError(f'Actual cold state differs from the10k initialization: {name}')
    for name in ('known_inputs.npz', 'evaluation_inputs.npz', 'reserved_geometries.npz'):
        with np.load(run/name, allow_pickle=False) as a, np.load(control/name, allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise ValueError(f'Cold input schema differs: {name}')
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key], err_msg=name+':'+key)
    return dict(status='cold_model_complete_optimizer_and_input_bank_exact',
                model_and_full_optimizer_exact=True, initial_geometry_bank_exact=True,
                evaluation_and_reserved_inputs_exact=True, fitted_state_imported=False)


def audit_completed(run, control):
    run, control = Path(run), Path(control)
    cold = audit_cold(run, control)
    if json.loads((run/'summary.json').read_text())['iterations'] != 20000:
        raise ValueError('The actual prescribed20k endpoint is missing')
    if json.loads((run/'online-validation.json').read_text())['status'] != 'actual_rolling_inputs_reconstructed':
        raise ValueError('Complete rolling stream must be independently audited')
    with np.load(run/'sampling.npz', allow_pickle=False) as a, np.load(control/'sampling.npz', allow_pickle=False) as b:
        if set(a.files) != set(b.files) or a['indices'].shape != (20000, 8):
            raise ValueError('Actual20k input/time sampling schema differs')
        for name in a.files:
            np.testing.assert_array_equal(a[name][:len(b[name])], b[name], err_msg=name)
    return dict(status='cold20k_budget_LR_comparison_passed', cold=cold,
                first10000_geometry_slots_draws_amplitudes_and_times_exact=True,
                scope='Budget and LR schedule change together; one initialization, not cross-seed evidence')
