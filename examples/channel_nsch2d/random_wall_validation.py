"""Strict geometry-family extension audit against the completed integral control."""
import json
from pathlib import Path
import numpy as np


def comparison_config(c,control):
    from examples.channel_nsch2d.geometry import random_walls
    path=Path(control)
    if not path.is_absolute():raise ValueError('Random-wall control must be an absolute completed run')
    before=json.loads((path/'config.json').read_text());now=json.loads(c.to_json())
    if (not random_walls(c) or 'kind' in before['geometry'] or before['nl']['method']!='integral_moment'
            or now['nl']!=before['nl'] or json.loads((path/'summary.json').read_text())['status']!='completed'
            or json.loads((path/'validation.json').read_text())['status']!='replay_and_hard_constraints_passed'):
        raise ValueError('Random-wall pilot requires a completed audited integral single-bump control')
    now['geometry']=before['geometry'];now['reference']['root']=before['reference']['root']
    if now['reference']['coarse_grid']!=[64,24] or now['reference']['fine_grid']!=[96,36]:
        raise ValueError('Random walls require the explicitly qualified64x24/96x36 reference pair')
    for key in ('coarse_grid','fine_grid'):now['reference'][key]=before['reference'][key]
    now['log']['run_name']=before['log']['run_name']
    now['operator']['validation_manifest']=before['operator']['validation_manifest']
    if now!=before:raise ValueError('Random-wall pilot may change only geometry and declared reference/manifest/run paths')
    return before


def audit_comparison(run,control):
    from ml_collections import ConfigDict
    run,control=Path(run),Path(control)
    c=ConfigDict(json.loads((run/'config.json').read_text()));comparison_config(c,control)
    for name in ('model_initial.eqx','optimizer_initial.eqx'):
        if (run/name).read_bytes()!=(control/name).read_bytes():raise ValueError(f'Random-wall cold state differs: {name}')
    with np.load(run/'sampling.npz',allow_pickle=False) as a,np.load(control/'sampling.npz',allow_pickle=False) as b:
        if set(a.files)!=set(b.files):raise ValueError('Random-wall sampling schemas differ')
        for key in set(a.files)-{'amplitudes'}:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
    return dict(status='matched_cold_state_and_actual_time_slot_draws_passed',
        initial_model_complete_optimizer_exact=True,all_actual_time_slot_and_online_ids_exact=True,
        geometry_stream_intentionally_changed=True,
        scope='Geometry-family extension; independently reconstructed inputs, not same physical problem or error target')
