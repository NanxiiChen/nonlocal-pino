"""Audit paired final10k weights/draws and existing native/local reference norms."""
import argparse
import json
from pathlib import Path

import numpy as np


def metrics(predicted, target):
    predicted,target=np.asarray(predicted,dtype=np.float64),np.asarray(target,dtype=np.float64)
    if (predicted.shape!=target.shape or predicted.ndim!=2
            or not np.isfinite(predicted).all() or not np.isfinite(target).all()
            or np.any(np.linalg.norm(target,axis=1)==0)):
        raise ValueError('Finite matching predicted/target fields required')
    difference=predicted-target
    return dict(pooled_phi=float(np.linalg.norm(difference)/np.linalg.norm(target)),
                terminal_phi=float(np.linalg.norm(difference[-1])/np.linalg.norm(target[-1])),
                by_time=(np.linalg.norm(difference,axis=1)/np.linalg.norm(target,axis=1)).tolist())


def evaluate(nonlocal_run, fem_run, output):
    runs=dict(nonlocal_pino=Path(nonlocal_run),fem_pino=Path(fem_run))
    configs={name:json.loads((run/'config.json').read_text()) for name,run in runs.items()}
    a,b=configs.values()
    for key in a:
        if key not in ('operator','log') and a[key]!=b[key]:raise ValueError('Unpaired configuration: '+key)
    for key in a['operator']:
        if key!='spatial_discretization' and a['operator'][key]!=b['operator'][key]:raise ValueError(key)
    for name in ('model_initial.eqx','optimizer_initial.eqx'):
        if (runs['nonlocal_pino']/name).read_bytes()!=(runs['fem_pino']/name).read_bytes():
            raise ValueError('Cold state mismatch: '+name)
    with np.load(runs['nonlocal_pino']/'sampling.npz') as first,np.load(runs['fem_pino']/'sampling.npz') as second:
        if set(first.files)!=set(second.files):raise ValueError('Sampling schema mismatch')
        for key in first.files:np.testing.assert_array_equal(first[key],second[key],err_msg=key)
        if len(first['times'])!=10000:raise ValueError('Require complete cold10k controls')
        for draw,cores in zip(first['entry_draw_ids'],first['core_ids'],strict=True):
            expected=np.sort(np.random.default_rng(int(draw)).choice(128**3,256,replace=False))
            np.testing.assert_array_equal(cores,expected)
    validation=Path('docs/tasks/archive/melting-local-fem-validation.json')
    accepted=json.loads(validation.read_text())
    if accepted['status'] not in ('passed','accepted_by_user'):raise ValueError('Local reference is not accepted')
    local=Path(accepted['conditions'][0]['path'])
    reference=Path(a['operator']['dataset_path']);manifest=json.loads((reference/'manifest.json').read_text())
    with np.load(local) as ref,np.load(reference/'condition_005.npz') as native:
        problem=json.loads(str(ref['metadata_json']))['problem']
        for key in ('physical','nondim','initial'):
            if a[key]!=problem[key]:raise ValueError('Local reference physical mismatch: '+key)
        reference_gap=metrics(native['phi'],ref['phi'])
    results={}
    for name,run in runs.items():
        native=[]
        for index in manifest['test_indices']:
            with np.load(run/f'prediction_{index:03d}.npz') as prediction,np.load(reference/f'condition_{index:03d}.npz') as ref:
                for key in ('points','time','parameters'):np.testing.assert_array_equal(prediction[key],ref[key])
                np.testing.assert_array_equal(prediction['target'],ref['phi'])
                native.append(dict(index=index,**metrics(prediction['prediction'],ref['phi'])))
        with np.load(run/'prediction_005.npz') as prediction,np.load(local) as ref:
            for key in ('points','time','parameters'):np.testing.assert_array_equal(prediction[key],ref[key])
            local_metrics=metrics(prediction['prediction'],ref['phi'])
        timing=json.loads((run/'timing.json').read_text());summary=json.loads((run/'summary.json').read_text())
        warm=np.asarray(timing['update_seconds'][1:]);inclusive=warm+np.asarray(timing['sample_seconds'][1:])+np.asarray(timing['spatial_seconds'][1:])
        results[name]=dict(run=str(run),checkpoint=str(run/'model_final.eqx'),
            native_test_count=len(native),native_records=native,
            native_mean=float(np.mean([r['pooled_phi'] for r in native])),
            native_std=float(np.std([r['pooled_phi'] for r in native],ddof=1)),
            local_case005=local_metrics,warm_update_median_seconds=float(np.median(warm)),
            warm_update_mean_seconds=float(warm.mean()),warm_inclusive_median_seconds=float(np.median(inclusive)),
            trainer_seconds=summary['trainer_seconds'],setup_seconds=summary['setup_seconds'],
            peak_active_gib=summary['peak_active_GiB'])
    result=dict(status='passed',steps=10000,seed=42,paired_cold_and_all_inputs=True,
        independent_core_replay=True,metric='Unweighted complete space-time relative L2 per sample; means across samples',
        local_reference=str(local),local_acceptance=accepted['status'],
        strict_local_refinement_passed=accepted['strict_refinement_passed'],
        native_reference=str(reference),native_to_local_case005=reference_gap,results=results,
        limitations=['Native-reference accuracy is not continuum accuracy.',
                     'One reused local-FEM case, eight native cases and one initialization.',
                     'Natural boundary treatment does not remove boundary/discretization error.'])
    Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nonlocal-run',required=True);parser.add_argument('--fem-run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();evaluate(args.nonlocal_run,args.fem_run,args.output)
