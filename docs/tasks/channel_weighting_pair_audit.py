"""Strict postrun proof of the uniform/adaptive timeJVP control pair."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np


def audit(uniform,adaptive,output):
    uniform,adaptive,output=map(Path,(uniform,adaptive,output))
    if any(not p.is_absolute()for p in (uniform,adaptive,output)):raise ValueError('Absolute direct paths required')
    if output.exists():raise FileExistsError(output)
    configs=[json.loads((p/'config.json').read_text())for p in (uniform,adaptive)]
    before=json.loads(json.dumps(configs[1]));del before['causal']
    before['log']['run_name']=configs[0]['log']['run_name']
    if 'causal'in configs[0]or 'adaptation'not in configs[1]['causal']or before!=configs[0]:
        raise ValueError('Weighting pair must differ ONLY by adaptivecausal settings and runname')
    if configs[0]['time_derivative']!={'kind':'jvp','sampling':'continuous_stratified'}:
        raise ValueError('Both controls require continuous physicaltimeJVP')
    summaries=[]
    for path in (uniform,adaptive):
        v=json.loads((path/'validation.json').read_text());s=json.loads((path/'summary.json').read_text())
        if (v['status']!='replay_and_hard_constraints_passed'or not v['actual_draws_reproduced_from_global_keys']
                or not v['actual_time_strata_passed']or s['status']!='completed'
                or s['supervised_evolved_field_samples']!=0):
            raise ValueError('Both actual complete purephysics runs require successful canonicalreplay/key validation')
        summaries.append(s)
    for name in ('model_initial.eqx','optimizer_initial.eqx'):
        if (uniform/name).read_bytes()!=(adaptive/name).read_bytes():raise ValueError(f'Actual cold state differs: {name}')
    checked=[]
    for name in ('known_inputs.npz','known_inputs_final.npz','sampling.npz'):
        with np.load(uniform/name,allow_pickle=False)as a,np.load(adaptive/name,allow_pickle=False)as b:
            if set(a.files)!=set(b.files):raise ValueError(f'Actual input schema differs: {name}')
            for key in a.files:
                if a[key].dtype!=b[key].dtype:raise ValueError(f'Actual input dtype differs: {name}:{key}')
                np.testing.assert_array_equal(a[key],b[key],err_msg=f'{name}:{key}');checked.append(name+':'+key)
    metrics=[];maximum_gap=0.
    for path,s in zip((uniform,adaptive),summaries,strict=True):
        with np.load(path/'prediction.npz',allow_pickle=False)as f:
            value=f['fields'];target=f['reference'];keys={k:f[k]for k in ('points','time','amplitudes','reference')}
        if path==uniform:reference_keys=keys
        else:
            for key in keys:np.testing.assert_array_equal(keys[key],reference_keys[key])
        relative=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
        score=dict(phi=relative(value[...,0],target[...,0]),terminal_phi=relative(value[-1,:,0],target[-1,:,0]),
            velocity=relative(value[1:,:,2:4],target[1:,:,2:4]),pressure=relative(value[1:,:,4],target[1:,:,4]))
        gap=max(abs(score[k]-s['metrics'][k])for k in score);maximum_gap=max(maximum_gap,gap)
        if gap>2e-6:raise ValueError('Physical endpoint metrics do not independently reproduce')
        metrics.append(dict(run=str(path),metrics=score))
    report=dict(status='actual_cold_state_all_continuous_times_condition_and_geometry_inputs_exact',
        uniform=str(uniform),adaptive=str(adaptive),updates=configs[0]['train']['iterations'],
        configurations_differ_only_causal_and_runname=True,complete_initial_model_optimizer_bytes_exact=True,
        actual_array_fields_checked=checked,maximum_metric_recompute_gap=maximum_gap,results=metrics,
        scope='One initialization/reused nominalFEM geometry, not a multiseed/population or stable optimizer-causality claim; postrun audit only, no updates')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x')as f:f.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    source=output.parent/'source/docs/tasks'/Path(__file__).name;source.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(__file__),source)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('uniform','adaptive','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();audit(a.uniform,a.adaptive,a.output)
