"""Strict host-only comparison of integral and GMLS derivative controls."""
import json
from pathlib import Path

import numpy as np


OPERATOR_ARRAYS={'neighbors','laplacian','gradient_x','gradient_y',
    'boundary_neighbors','boundary_gradient_x','boundary_gradient_y'}


def comparison_config(c,control):
    control=Path(control)
    if not control.is_absolute():raise ValueError('Integral control must be an absolute completed run')
    for name in ('config.json','summary.json','validation.json','sampling.npz',
                 'known_inputs.npz','known_inputs_final.npz','model_initial.eqx','optimizer_initial.eqx'):
        if not (control/name).is_file():raise FileNotFoundError(control/name)
    before=json.loads((control/'config.json').read_text());now=json.loads(c.to_json())
    summary=json.loads((control/'summary.json').read_text())
    if (summary['status']!='completed' or summary['iterations']!=c.train.iterations
            or json.loads((control/'validation.json').read_text())['status']!='replay_and_hard_constraints_passed'):
        raise ValueError('Integral comparison requires a matching completed/audited control')
    if (before['nl']['method']!='gmls' or now['nl']['method']!='integral_moment'
            or before['derivatives']['kind']!='gmls_first'
            or now['derivatives']['kind']!='integral_first'
            or before['phase_transport']['kind']!='gmls_conservative'
            or now['phase_transport']['kind']!='integral_conservative'):
        raise ValueError('Integral comparison requires explicit GMLS/integral derivative selectors')
    for key in ('nl','derivatives'):now[key]=before[key]
    now['phase_transport']['kind']=before['phase_transport']['kind']
    now['log']['run_name']=before['log']['run_name']
    if now!=before:
        raise ValueError('Integral comparison may change only nl, derivative/flux selectors and run name')
    return before


def audit_cold_inputs(run,control):
    run,control=Path(run),Path(control)
    for name in ('model_initial.eqx','optimizer_initial.eqx'):
        if (run/name).read_bytes()!=(control/name).read_bytes():
            raise ValueError(f'Integral control cold state differs: {name}')
    with np.load(run/'known_inputs.npz',allow_pickle=False)as a,np.load(control/'known_inputs.npz',allow_pickle=False)as b:
        if set(a.files)!=set(b.files):raise ValueError('Integral initial payload schema differs')
        for key in set(a.files)-OPERATOR_ARRAYS:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
    return dict(status='passed',initial_model_complete_optimizer_exact=True,
        initial_geometry_encoder_and_quadrature_arrays_exact=True,
        operator_arrays_intentionally_changed=sorted(OPERATOR_ARRAYS))


def audit_comparison(run,control):
    from ml_collections import ConfigDict
    run,control=Path(run),Path(control)
    c=ConfigDict(json.loads((run/'config.json').read_text()));comparison_config(c,control)
    cold=audit_cold_inputs(run,control)
    for name in ('known_inputs_final.npz','evaluation_inputs.npz','sampling.npz'):
        with np.load(run/name,allow_pickle=False)as a,np.load(control/name,allow_pickle=False)as b:
            if set(a.files)!=set(b.files):raise ValueError(f'Integral input schema differs: {name}')
            for key in set(a.files)-(OPERATOR_ARRAYS if name!='sampling.npz'else set()):
                np.testing.assert_array_equal(a[key],b[key],err_msg=f'{name}:{key}')
    with np.load(run/'prediction.npz',allow_pickle=False)as a,np.load(control/'prediction.npz',allow_pickle=False)as b:
        for key in ('reference','points','time','amplitudes'):np.testing.assert_array_equal(a[key],b[key])
    results=[]
    for label,path in (('GMLS',control),('moment_corrected_integral',run)):
        saved=json.loads((path/'summary.json').read_text())
        with np.load(path/'prediction.npz',allow_pickle=False)as f:v,r=f['fields'],f['reference']
        relative=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
        metrics=dict(phi=relative(v[...,0],r[...,0]),terminal_phi=relative(v[-1,:,0],r[-1,:,0]),
            velocity=relative(v[1:,:,2:4],r[1:,:,2:4]),pressure=relative(v[1:,:,4],r[1:,:,4]),
            chemical=relative(v[1:,:,1],r[1:,:,1]),
            dispersed_phase=relative((1-v[...,0])/2,(1-r[...,0])/2))
        for key,value in metrics.items():np.testing.assert_allclose(value,saved['metrics'][key],atol=2e-6,rtol=0)
        results.append(dict(label=label,run=str(path),metrics=metrics,
            warm_update_seconds=saved['warm_update_seconds'],wall_seconds=saved['wall_seconds']))
    return dict(status='matched_cold_state_geometry_encoder_quadrature_and_all_time_draws_passed',
        cold=cold,results=results,all_actual_geometry_and_time_draws_exact=True,
        scope='One initialization/nominal local-FEM geometry; changed spatial derivative operators, no population or conservation claim')
