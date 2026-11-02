"""Strict fixed-scale/termwise comparison and host-only saved-run audits."""
import json
from pathlib import Path
import numpy as np


def comparison_config(c,control):
    control=Path(control)
    if not control.is_absolute():raise ValueError('Balance control requires an absolute completed run')
    for name in ('config.json','summary.json','validation.json','sampling.npz','prediction.npz',
                 'known_inputs.npz','known_inputs_final.npz','model_initial.eqx','optimizer_initial.eqx'):
        if not (control/name).is_file():raise FileNotFoundError(control/name)
    saved=json.loads((control/'config.json').read_text());current=json.loads(c.to_json())
    summary=json.loads((control/'summary.json').read_text())
    if (summary['status']!='completed' or summary['iterations']!=c.train.iterations
            or json.loads((control/'validation.json').read_text())['status']!='replay_and_hard_constraints_passed'):
        raise ValueError('Balance comparison requires a matching completed/audited budget control')
    expected=dict(kind='gradnorm',norm='l1',grouping='terms',epsilon=1e-6,
        cap=1000.,new_weight_fraction=.9,update_every=100)
    if ('balance' in saved or 'balance' not in current or current['balance']!=expected
            or current['loss']['phase_scale']!=1. or current['loss']['boundary_mu_scale']!=1.
            or saved['loss']['phase_scale']!=.2 or saved['loss']['boundary_mu_scale']!=4.):
        raise ValueError('Balance comparison requires explicit nine-term weights and raw residuals')
    del current['balance']
    for field in ('phase_scale','boundary_mu_scale'):current['loss'][field]=saved['loss'][field]
    current['log']['run_name']=saved['log']['run_name']
    if current!=saved:
        raise ValueError('Balance comparison may change only adaptive weights, phase/boundary divisors and run name')
    return saved


def audit_comparison(run,control):
    from ml_collections import ConfigDict
    run,control=Path(run),Path(control)
    c=ConfigDict(json.loads((run/'config.json').read_text()));comparison_config(c,control)
    for name in ('model_initial.eqx','optimizer_initial.eqx'):
        if (run/name).read_bytes()!=(control/name).read_bytes():raise ValueError(f'Cold state differs: {name}')
    for name in ('known_inputs.npz','known_inputs_final.npz','sampling.npz'):
        with np.load(run/name,allow_pickle=False)as a,np.load(control/name,allow_pickle=False)as b:
            if set(a.files)!=set(b.files):raise ValueError(f'Input schema differs: {name}')
            for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=f'{name}:{key}')
    with np.load(run/'prediction.npz')as a,np.load(control/'prediction.npz')as b:
        for key in ('reference','points','time','amplitudes'):np.testing.assert_array_equal(a[key],b[key])
    summary=json.loads((run/'summary.json').read_text());groups=summary['balance']['groups']
    names=['phase','chemical','momentum_x','momentum_y','wall_phase','wall_mu',
           'outlet_traction','outlet_phase','outlet_mu']
    if groups!=names:raise ValueError('Nine independently balanced terms required')
    refresh=[json.loads(s)for s in (run/'balance.jsonl').read_text().splitlines()]
    if [row['step']for row in refresh]!=list(range(1,c.train.iterations+1,c.balance.update_every)):
        raise ValueError('Incomplete actual balance refresh history')
    previous=np.ones(9);settings=c.balance
    for row in refresh:
        norms=np.asarray(row['gradient_l1_norms']);weights=np.asarray(row['weights'])
        if (norms.shape!=(9,)or not np.isfinite(norms).all()or np.any(norms<0)
                or row['groups']!=groups):raise ValueError('Invalid complete-time norm record')
        safe=np.clip(norms,settings.epsilon,1/settings.epsilon)
        target=np.clip(safe.mean()/safe,settings.epsilon,settings.cap)
        expected=settings.new_weight_fraction*target+(1-settings.new_weight_fraction)*previous
        np.testing.assert_allclose(weights,expected,rtol=3e-6,atol=3e-6)
        previous=weights
    for row in (json.loads(s)for s in (run/'loss.jsonl').read_text().splitlines()):
        latest=refresh[(row['step']-1)//settings.update_every]
        weights=np.asarray([row['weights'][n]for n in groups])
        np.testing.assert_array_equal(weights,latest['weights'])
        parts=np.asarray([row['parts'][n]for n in groups])
        np.testing.assert_allclose(weights@parts,row['total'],rtol=3e-5,atol=3e-5)
    with np.load(run/'balance_final.npz')as file:np.testing.assert_array_equal(file['weights'],previous)
    results=[]
    for label,path in (('fixed_scales',control),('raw_residuals_nine_adaptive_weights',run)):
        with np.load(path/'prediction.npz')as file:
            v,r=file['fields'],file['reference']
        relative=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
        metrics=dict(phi=relative(v[...,0],r[...,0]),terminal_phi=relative(v[-1,:,0],r[-1,:,0]),
            velocity=relative(v[1:,:,2:4],r[1:,:,2:4]),pressure=relative(v[1:,:,4],r[1:,:,4]))
        saved=json.loads((path/'summary.json').read_text())
        for key,value in metrics.items():np.testing.assert_allclose(value,saved['metrics'][key],atol=2e-6,rtol=0)
        results.append(dict(label=label,run=str(path),metrics=metrics,
            warm_update_seconds=saved['warm_update_seconds'],wall_seconds=saved['wall_seconds']))
    return dict(status='matched_cold_state_all_inputs_time_and_nine_weight_history_passed',
        control=str(control),run=str(run),initial_model_complete_optimizer_exact=True,
        all_initial_final_input_and_actual_sampling_arrays_exact=True,refresh_records=len(refresh),
        raw_residual_divisors=True,physical_output_scales_unchanged=True,results=results,
        scope='One initialization/nominal development geometry; not population accuracy or original learned GradNorm')
