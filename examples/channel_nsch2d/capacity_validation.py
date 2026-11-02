"""Explicit combined width/batch/budget trial against the completed random walls."""
import json
from pathlib import Path

import numpy as np


def comparison_config(c, control):
    """Permit only128/B2/5k/timeblock8 ->192/B8/10k/timeblock2."""
    from examples.channel_nsch2d.geometry import random_walls

    path = Path(control)
    if not path.is_absolute():
        raise ValueError('Capacity control must be an absolute completed run')
    before = json.loads((path/'config.json').read_text())
    now = json.loads(c.to_json())
    if (not random_walls(c) or before['geometry']['kind'] != 'random_spline'
            or before['nl']['method'] != 'integral_moment'
            or json.loads((path/'summary.json').read_text())['status'] != 'completed'
            or json.loads((path/'validation.json').read_text())['status']
                != 'replay_and_hard_constraints_passed'):
        raise ValueError('Capacity trial requires the audited random-wall integral control')
    old = (before['arch']['width'], before['operator']['condition_batch'],
           before['train']['iterations'])
    new = (now['arch']['width'], now['operator']['condition_batch'],
           now['train']['iterations'])
    if (old != (128, 2, 5000) or new != (192, 8, 10000)
            or before['train']['microbatch'] != dict(kind='time',size=8)
            or now['train']['microbatch'] != dict(kind='time',size=2)):
        raise ValueError('Capacity trial requires128/B2/5k/timeblock8 ->192/B8/10k/timeblock2')
    now['arch']['width'] = before['arch']['width']
    now['operator']['condition_batch'] = before['operator']['condition_batch']
    now['train']['iterations'] = before['train']['iterations']
    now['train']['microbatch'] = before['train']['microbatch']
    now['log']['run_name'] = before['log']['run_name']
    if now != before:
        raise ValueError('Capacity trial may change only width, geometry batch, budget and run name')
    return before


def execution_config(c, control):
    """Qualify explicit40-time divisors on the unchanged192/B8/10k problem."""
    from ml_collections import ConfigDict

    settings=c.train.microbatch.to_dict()
    if (set(settings)!={'kind','size'} or settings['kind']!='time'
            or isinstance(settings['size'],bool) or not isinstance(settings['size'],int)
            or settings['size'] not in (2,4,5,8,10,20,40)):
        raise ValueError('Hardware comparison requires an explicit40-time divisor in(2,4,5,8,10,20,40)')
    canonical=ConfigDict(c.to_dict())
    canonical.train.microbatch.size=2
    return comparison_config(canonical, control)


def audit_probe_state(run, probe):
    """Compare the actual cold model/fullSOAP with the probe's unfitted state."""
    run, probe = Path(run), Path(probe)
    before = json.loads((probe/'config.json').read_text())
    now = json.loads((run/'config.json').read_text())
    now['log']['run_name'] = before['log']['run_name']
    if now != before:
        raise ValueError('Actual capacity run configuration differs from qualified probe')
    for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
        if (run/name).read_bytes() != (probe/name).read_bytes():
            raise ValueError(f'Capacity cold state differs from unfitted resource probe: {name}')
    return dict(status='fresh_capacity_model_and_complete_optimizer_exact',
                cold=True, imported_fitted_probe_state=False,
                compared_to='Resource probe state saved BEFORE any update; architecture differs from old control')


def audit_comparison(run, control, probe, hardware_trial=False):
    """Check unchanged actual inputs/time prefix; B8 condition draws differ."""
    from ml_collections import ConfigDict

    run, control = Path(run), Path(control)
    c = ConfigDict(json.loads((run/'config.json').read_text()))
    if hardware_trial:execution_config(c, control)
    else:comparison_config(c, control)
    cold = audit_probe_state(run, probe)
    with np.load(run/'known_inputs.npz', allow_pickle=False) as a, \
            np.load(control/'known_inputs.npz', allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError('Capacity trial initial input schema differs')
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    with np.load(run/'sampling.npz', allow_pickle=False) as a, \
            np.load(control/'sampling.npz', allow_pickle=False) as b:
        np.testing.assert_array_equal(a['times'][:len(b['times'])], b['times'])
        if a['indices'].shape != (c.train.iterations, c.operator.condition_batch):
            raise ValueError('Actual geometry batch does not match the capacity trial')
    return dict(status='capacity_comparison_passed', cold=cold,
                initial_geometry_bank_exact=True, shared_time_prefix_exact=True,
                condition_draws='Different batch shape; independently replayed by validate_run',
                scope='Combined width, batch and budget trial; cannot isolate their individual effects')
