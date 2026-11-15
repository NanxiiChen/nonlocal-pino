"""Independent saved-field reduction and strict channel sampling-run pairing.

No model inference, training import, JAX backend or reference solve is used.
Historical unweighted metrics are retained: phase uses all saved times; flow
and chemical potential use positive times; velocity is a joint vector norm.
Area-weighted scores and derivative/flux checks are outside this audit's scope.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def relative(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Aligned finite fields required')
    denominator=np.sum(b*b,dtype=np.float64)
    if denominator<=0:raise ValueError('Positive reference norm required')
    return float(np.sqrt(np.sum((a-b)**2,dtype=np.float64)/denominator))


def norms(prediction,reference):
    if prediction.ndim!=3 or prediction.shape!=reference.shape or prediction.shape[-1]!=5 or prediction.shape[0]<2:
        raise ValueError('Time-by-query-by-five-channel fields required')
    if not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError('All saved channels and times must be finite')
    result={}
    for name,begin,channels in (('phi',0,slice(0,1)),('chemical',1,slice(1,2)),
                               ('velocity',1,slice(2,4)),('pressure',1,slice(4,5))):
        result[name]=relative(prediction[begin:,:,channels],reference[begin:,:,channels])
    result['terminal_phi']=relative(prediction[-1,:,0],reference[-1,:,0])
    result['dispersed_fraction']=relative(1-prediction[:,:,0],1-reference[:,:,0])
    return result


def response(prediction,reference):
    a=prediction-prediction.mean(axis=0)
    b=reference-reference.mean(axis=0)
    aa=float(np.sum(a*a));bb=float(np.sum(b*b))
    if bb<=0:raise ValueError('Nonzero geometry response required')
    return dict(relative_error=float(np.sqrt(np.sum((a-b)**2)/bb)),
                amplitude_ratio=float(np.sqrt(aa/bb)),
                correlation=float(np.sum(a*b)/np.sqrt(aa*bb)) if aa>0 else None)


def compare_numbers(actual,reported):
    for key,value in actual.items():
        if value is None:
            if reported[key] is not None:raise ValueError('Undefined response correlation was reported as a number')
        else:np.testing.assert_allclose(value,reported[key],rtol=3e-12,atol=1e-12,err_msg=key)


def audit_fields(run,evaluation,manifest):
    metadata=read(evaluation/'evaluation.json');spec=read(manifest)
    config=read(run/'config.json');summary=read(run/'summary.json')
    family=dict(formulation='matched-nsch-open-traction-v1',
                **{key:config[key] for key in ('geometry','physical','initial')})
    if family!=spec['physical_family']:
        raise ValueError('Training and reference physical families differ')
    if (summary['status']!='completed' or summary['iterations']!=config['train']['iterations']
            or summary['supervised_evolved_field_samples']!=0 or 'parent_run' in summary):
        raise ValueError('Complete cold pure-physics training required')
    if (metadata['status']!='frozen_unseen_geometry_evaluation_completed'
            or Path(metadata['run']).resolve()!=run.resolve()
            or Path(metadata['manifest']).resolve()!=manifest.resolve()
            or metadata['geometries']!=spec['count'] or spec['count']<2):
        raise ValueError('Evaluation identity or cohort size mismatch')
    indices=[r['index'] for r in spec['cases']]
    if len(set(indices))!=spec['count'] or [r['index'] for r in metadata['per_geometry']]!=indices:
        raise ValueError('Every declared geometry must appear once in declared order')
    if read(run/'online-validation.json')['status']!='actual_rolling_inputs_reconstructed':
        raise ValueError('Actual online-input reconstruction is required')
    all_pred=[];all_ref=[];common_pred=[];common_ref=[];rows=[]
    common_points=None
    for case,reported in zip(spec['cases'],metadata['per_geometry'],strict=True):
        gate=read(Path(case['root'])/'reference-validation.json')
        if gate['status']!='passed' or reported['reference']!=gate['reference']:
            raise ValueError('Independent declared reference is not validated')
        with (np.load(evaluation/f'case_{case["index"]:03d}'/'prediction.npz') as saved,
              np.load(gate['reference']) as original):
            for key in ('points','time','amplitudes','common_points','common_query_identity'):
                np.testing.assert_array_equal(saved[key],original[key])
            np.testing.assert_array_equal(saved['reference'],original['fields'])
            np.testing.assert_array_equal(saved['common_reference'],original['common_fields'])
            np.testing.assert_array_equal(saved['amplitudes'],case['amplitudes'])
            np.testing.assert_array_equal(saved['amplitudes'],reported['amplitudes'])
            time=saved['time']
            if time.ndim!=1 or time[0]!=0 or not np.isfinite(time).all() or np.any(np.diff(time)<=0):
                raise ValueError('Ordered time grid beginning at zero required')
            np.testing.assert_allclose(time,np.linspace(0,config['physical']['terminal_time'],
                config['reference']['snapshots']),rtol=0,atol=1e-14)
            prediction=saved['fields'];target=saved['reference']
            cp=saved['common_fields'];ct=saved['common_reference']
            if (prediction.shape[:2]!=(len(time),len(saved['points']))
                    or len(saved['points'])!=int(np.prod(spec['query_grid']))
                    or cp.shape[:2]!=(len(time),len(saved['common_points']))):
                raise ValueError('Fields do not match saved space-time queries')
            if common_points is None:common_points=saved['common_points'].copy()
            else:np.testing.assert_array_equal(common_points,saved['common_points'])
            values=norms(prediction,target);common_values=norms(cp,ct)
            compare_numbers(values,reported['metrics']);compare_numbers(common_values,reported['common_metrics'])
            rows.append(dict(index=case['index'],metrics=values,common_metrics=common_values))
            all_pred.append(prediction);all_ref.append(target);common_pred.append(cp);common_ref.append(ct)
    aggregate={}
    for name in rows[0]['metrics']:
        values=np.array([row['metrics'][name] for row in rows])
        aggregate[name]=dict(mean=float(values.mean()),p90=float(np.quantile(values,.9)),worst=float(values.max()))
        compare_numbers(aggregate[name],metadata['aggregate'][name])
    for p,q,label in ((all_pred,all_ref,'mapped_coordinate_response'),
                      (common_pred,common_ref,'geometry_response')):
        p,q=np.stack(p),np.stack(q)
        for name,begin,channels in (('phi',0,slice(0,1)),('velocity',1,slice(2,4)),('pressure',1,slice(4,5))):
            compare_numbers(response(p[:,begin:,:,channels],q[:,begin:,:,channels]),metadata[label][name])
    return dict(run=str(run),evaluation=str(evaluation),geometries=spec['count'],
                per_geometry=rows,aggregate=aggregate,
                scope='Unweighted own/common-frame field norms and centered geometry responses; targets match original FEM files')


def equal_arrays(first,second,keys):
    with np.load(first) as a,np.load(second) as b:
        names=a.files if keys is None else keys
        if keys is None and set(a.files)!=set(b.files):raise ValueError('Known-input schemas differ')
        for key in names:np.testing.assert_array_equal(a[key],b[key])


def paired_inputs(runs):
    base=runs[0]
    expected=read(base/'config.json');expected['log'].pop('run_name');expected['sample'].pop('spatial')
    if 'initialization' in read(base/'summary.json'):raise ValueError('Require independently cold arms')
    source={p.relative_to(base/'source'):p.read_bytes() for p in (base/'source').rglob('*.py')}
    if not source:raise ValueError('Frozen training source required')
    summaries=[]
    for run in runs:
        config=read(run/'config.json');config['log'].pop('run_name');spatial=config['sample'].pop('spatial')
        if config!=expected:raise ValueError('Uncontrolled configuration change')
        summary=read(run/'summary.json')
        if 'initialization' in summary:raise ValueError('Require independently cold arms')
        for name in ('model_initial.eqx','optimizer_initial.eqx'):
            if (base/name).read_bytes()!=(run/name).read_bytes():raise ValueError('Cold state differs')
        other={p.relative_to(run/'source'):p.read_bytes() for p in (run/'source').rglob('*.py')}
        if source!=other:raise ValueError('Training source differs')
        equal_arrays(base/'sampling.npz',run/'sampling.npz',('indices','times','draw_ids','amplitudes'))
        for name in ('known_inputs.npz','known_inputs_final.npz','evaluation_inputs.npz','reserved_geometries.npz'):
            equal_arrays(base/name,run/name,None)
        # Actual rolling identity/geometry records are compared without timing
        # fields; the separate canonical reconstruction audits every draw.
        for name,keys in (('online_inputs.jsonl',('draw_id','amplitudes','training_amplitudes','pde_solves','fem_solves')),
                          ('online_refresh.jsonl',('completed_updates','batch','slots','draw_ids'))):
            a=[json.loads(s) for s in (base/name).read_text().splitlines()]
            b=[json.loads(s) for s in (run/name).read_text().splitlines()]
            if len(a)!=len(b):raise ValueError('Rolling record counts differ')
            for x,y in zip(a,b,strict=True):
                if any(x[k]!=y[k] for k in keys):raise ValueError('Actual rolling identities differ')
        if summary['device']!=read(base/'summary.json')['device']:
            raise ValueError('Recorded GPU/backend/JAX version differs')
        with np.load(run/'spatial_timing.npz') as f:
            update=f['update'];prep=f['batch_preparation']
        steps=config['train']['iterations']
        if (update.shape!=(steps,) or prep.shape!=(steps,) or not np.isfinite(update).all()
                or not np.isfinite(prep).all() or np.any(update<=0) or np.any(prep<0) or steps<=10):
            raise ValueError('Complete finite positive timing arrays required')
        median=float(np.median((update+prep)[10:]))
        np.testing.assert_allclose(median,summary['spatial']['warm_step_with_batch_median_seconds'],rtol=1e-12)
        summaries.append(dict(run=str(run),spatial=spatial,warm_median_seconds=median,
            wall_seconds=summary['wall_seconds'],peak_active_bytes=summary['device_memory_bytes']['peak_bytes_in_use']))
    return dict(status='passed',source_files=len(source),runs=summaries,
                scope='Same cold states, configurations except spatial/runname, frozen sources and actual known-input streams; GPU UUID is checked separately by the serial owner')


def audit(runs,evaluations,manifest,output):
    if not runs or len(runs)!=len(evaluations) or len(set(map(str,runs)))!=len(runs):
        raise ValueError('Distinct runs and one explicit evaluation directory per run required')
    results=[audit_fields(r,e,manifest) for r,e in zip(runs,evaluations,strict=True)]
    pairing=paired_inputs(runs) if len(runs)>1 else None
    spatial=[]
    for run in runs:
        sampling=read(run/'config.json')['sample']
        # Historical full runs predate the explicit spatial-study record.
        if 'spatial' in sampling and sampling['spatial']['mode']!='full':
            from docs.tasks.pino_channel_spatial_replay import audit_run
            print(json.dumps(dict(stage='replay_all_spatial_draws',run=str(run))),flush=True)
            spatial.append(audit_run(run))
    result=dict(status='passed',manifest=str(manifest),results=results,pairing=pairing,spatial_replay=spatial,
        metric='Phase over all saved space-time points; velocity jointly over both components, pressure and chemical over positive saved times; average sample norms after normalization',
        training=False,inference=False,pde_solves=0,
        exclusions='Area-weighted scores and derivative/flux constraints are not independently recalculated here; single-run audit proves no sampler comparison')
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    shutil.copyfile(__file__,output/'source.py')
    if spatial:
        shutil.copyfile(Path(__file__).with_name('pino_channel_spatial_replay.py'),output/'spatial_replay_source.py')
    print(json.dumps(dict(status='passed',arms=len(results),pairing_checked=pairing is not None,
        errors=[dict(run=r['run'],means={k:v['mean'] for k,v in r['aggregate'].items()}) for r in results]),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--evaluations',type=Path,nargs='+',required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.runs,args.evaluations,args.manifest,args.output)
