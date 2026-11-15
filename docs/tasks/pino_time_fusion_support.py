"""Independent final-field/input and common-initial-backbone time-fusion audit."""
import argparse
import copy
import io
import json
from pathlib import Path

import numpy as np

from scripts.pino_comparison_data import history, norm, read, spec

ROOT = Path(__file__).resolve().parents[2]


def config_pair(a, b):
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    if 'temporal_query' in a['arch'] or b['arch'].pop('temporal_query') != 'joint_query':
        raise ValueError('Expected original decoder-FiLM versus explicit joint-query')
    for c in (a,b):
        c['log'].pop('run_name')
        if c['train']['iterations'] != 10000 or c['train']['seed'] != 42:
            raise ValueError('Complete cold10k/seed42 controls required')
    if a != b:
        raise ValueError('Unexpected physics/optimizer/sampling difference in time-fusion pair')


def initialized_backbone(case, configs, runs):
    import equinox as eqx
    import jax
    from ml_collections import ConfigDict
    if case == 'electropolish':
        from examples.electropolish2d.operator_training import _build_network
        build = lambda c:_build_network(c.arch, c.operator.image_shape, jax.random.key(0))
    else:
        from examples.icemelting3d.operator_data import build_network
        build = lambda c:build_network(c,jax.random.key(0))
    nets = [eqx.tree_deserialise_leaves(run/'model_initial.eqx', build(ConfigDict(c)))
            for c,run in zip(configs,runs,strict=True)]
    if nets[1].kind != 'joint_query' or nets[1].backbone.time_film is not None:
        raise ValueError('Joint query must remove decoder FiLM')
    shared = [eqx.tree_at(lambda n:n.time_film,nets[0],None),nets[1].backbone]
    buffers = [io.BytesIO(),io.BytesIO()]
    for buffer,net in zip(buffers,shared,strict=True):eqx.tree_serialise_leaves(buffer,net)
    if buffers[0].getvalue() != buffers[1].getvalue():
        raise ValueError('Common initial encoder/attention/decoder leaves differ')
    return dict(shared_initial_backbone_equal=True, complete_initial_trees_equal=False,
        parameter_count_scope='Stored numeric entries; coordinate frequencies are frozen metadata',
        stored_entries=[sum(x.size for x in jax.tree.leaves(eqx.filter(n,eqx.is_array))) for n in nets],
        shared_stored_entries=sum(x.size for x in jax.tree.leaves(eqx.filter(shared[0],eqx.is_array))))


def pair_inputs(case, runs):
    with np.load(runs[0]/'sampling.npz') as a,np.load(runs[1]/'sampling.npz') as b:
        if set(a.files) != set(b.files):raise ValueError('Time-fusion sampling schema mismatch')
        for name in a.files:np.testing.assert_array_equal(a[name],b[name],err_msg=name)
        if len(a['times']) != 10000:raise ValueError('Incomplete actual training draws')
    if case == 'electropolish':
        for filename in ('online_coefficients.npy','online_accepted.npy'):
            np.testing.assert_array_equal(np.load(runs[0]/filename),np.load(runs[1]/filename))
        with np.load(runs[0]/'paired_inputs.npz') as a,np.load(runs[1]/'paired_inputs.npz') as b:
            if set(a.files)!=set(b.files):raise ValueError('Known-input witness schemas differ')
            for name in a.files:np.testing.assert_array_equal(a[name],b[name],err_msg=name)
    for run in runs:
        for name in read(run/'source/files.json'):
            if (run/'source'/name).read_bytes()!=(ROOT/name).read_bytes():
                raise ValueError('Current/frozen numerical source differs: '+name)
    return dict(all_actual_sampling_arrays_equal=True, source_bytes_verified=True,
                optimizer_trees='Different time-module shapes; identical optimizer recipe and fresh initialization')


def timing(case, run):
    data=read(run/'timing.json');memory=read(run/'memory.json')
    inclusive=sum(np.asarray(data[k][1:]) for k in ('update_seconds','sample_seconds','spatial_seconds'))
    key='allocator_stats' if case=='electropolish' else 'allocator'
    peak=max(row[key]['peak_bytes_in_use'] for row in memory)/1024**3
    return dict(warm_inclusive_median_seconds=float(np.median(inclusive)),
        warm_inclusive_mean_seconds=float(np.mean(inclusive)),
        first_update_seconds=data['update_seconds'][0],peak_active_gib=peak,
        source=str(run/'timing.json'), memory=str(run/'memory.json'))


def audit(case, baseline_report, run, baseline_journal, joint_journal):
    from scripts.pino_comparison_data import electro,melting
    original = (electro if case=='electropolish' else melting)(baseline_report)
    baseline=Path(original['arms'][0]['run']);runs=[baseline,Path(run)]
    configs=[read(p/'config.json') for p in runs];config_pair(*configs)
    pairing={**initialized_backbone(case,configs,runs),**pair_inputs(case,runs)}
    histories=[];final=[];field_audit=[];local=[];steps=list(range(0,10001,250))
    fields=original['fields'];ids=original['test_ids']
    if len(ids)!=64 or original['monitor_ids']!=ids:raise ValueError('Complete common64test histories required')
    for path,c in zip(runs,configs,strict=True):
        values=[]
        if case=='electropolish':
            monitor=read(path/'local_monitor.json')
            baseline_monitor=read(baseline/'local_monitor.json')
            if monitor!=baseline_monitor:raise ValueError('Time-fusion local reference/cohort changed')
            histories.append(history(path/'local_metrics.jsonl',steps,ids,('phi','c'),('phi','c')))
            metadata=read(path/'spatial_evaluation64.json')
            if metadata['status']!='passed' or metadata['count']!=64:raise ValueError('Incomplete final64evaluation')
            take=metadata['regular_time_indices']
            with np.load(path/'spatial_evaluation64.npz') as pred:
                np.testing.assert_array_equal(pred['time'][take],np.linspace(0,1,21))
                for index in ids:
                    reference=Path(c['operator']['comparison_monitor']['reference_root'])/f'condition_{index:03d}/local101.npz'
                    with np.load(reference) as target:
                        for key in ('time','X','Y'):np.testing.assert_array_equal(pred[key],target[key])
                        scores=[norm(pred['prediction'][index,take,...,k],target['fields'][take,...,k]) for k in range(2)]
                    values.append([r['error'] for r in scores])
                    field_audit.append(dict(run=str(path),index=index,reference=str(reference),channels=scores))
        else:
            monitor=read(path/'monitoring.json')
            if monitor!=read(baseline/'monitoring.json'):raise ValueError('Time-fusion native reference/cohort changed')
            records=[r for r in map(json.loads,(path/'metrics.jsonl').read_text().splitlines()) if r['split']=='monitor']
            if [r['step'] for r in records]!=steps:raise ValueError('Incomplete native monitoring steps')
            rows=[]
            for r in records:
                if [x['index'] for x in r['conditions']]!=ids:raise ValueError('Incomplete native64monitoring')
                v=[[x['pooled_phi']] for x in r['conditions']]
                np.testing.assert_allclose(np.mean(v),r['pooled_phi'],rtol=2e-12)
                rows.append(v)
            histories.append(rows)
            for index in ids:
                reference=Path(c['operator']['dataset_path'])/f'condition_{index:03d}.npz'
                with np.load(path/f'prediction_{index:03d}.npz') as p,np.load(reference) as r:
                    for key in ('parameters','time','points'):np.testing.assert_array_equal(p[key],r[key])
                    np.testing.assert_array_equal(p['target'],r['phi']);score=norm(p['prediction'],r['phi'])
                values.append([score['error']]);field_audit.append(dict(run=str(path),index=index,reference=str(reference),channels=[score]))
            reference=read(baseline_report)['local_reference']
            with np.load(path/'prediction_005.npz') as p,np.load(reference) as r:
                for key in ('parameters','time','points'):np.testing.assert_array_equal(p[key],r[key])
                local.append(dict(run=str(path),reference=reference,**norm(p['prediction'],r['phi'])))
        final.append(values)
    np.testing.assert_allclose(final[0],original['arms'][0]['final_errors'],rtol=2e-13,atol=1e-14)
    a,b=read(baseline_journal),read(joint_journal)
    if a['gpu']!=b['gpu']:raise ValueError('GPU/driver changed between temporal arms')
    original_stage=next(s for s in a['stages'] if s['name']=='train-nonlocal')
    new_stage=next(s for s in b['stages'] if s['name']=='train-'+case)
    if original_stage['status']!='passed' or new_stage['status']!='passed':raise ValueError('Complete cold processes required')
    arms=[]
    for method,path,curves,values,stage in zip(('decoder_film','joint_query'),runs,histories,final,(original_stage,new_stage),strict=True):
        scores=np.asarray(values)
        arms.append(dict(method=method,run=str(path),configuration=str(path/'config.json'),checkpoint=str(path/'model_final.eqx'),
            monitor=str(path/('local_metrics.jsonl' if case=='electropolish' else 'metrics.jsonl')),
            monitor_errors=curves,final_errors=values,mean_channels=scores.mean(axis=0).tolist(),
            mean_over_channels_and_samples=float(scores.mean()),
            std_of_channel_mean_over_samples=float(scores.mean(axis=1).std(ddof=1)),
            timing=dict(**timing(case,path),cold_process_seconds=stage['seconds'],gpu=a['gpu'])))
    data=spec(original['example'],[(f['key'],f['label']) for f in fields],10000,steps,ids,ids,arms,
        original['reference'],original['convergence_metric'],'whole_trajectory',.22)
    data.update(comparison_kind='time_fusion',saved_field_audit=field_audit,independent_local_single_case=local,
                pairing=pairing,source_reports=[str(baseline_report),str(baseline_journal),str(joint_journal)],
                limitation='One initialization; time-module capacity differs; Melting native64 reference is not continuum qualification')
    return dict(status='passed',case=case,data=data,pairing=pairing)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('electropolish','melting'),required=True)
    p.add_argument('--baseline-report',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--baseline-journal',type=Path,required=True);p.add_argument('--joint-journal',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.case,args.baseline_report,args.run,args.baseline_journal,args.joint_journal)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('TIME_FUSION_AUDIT '+str(args.output),flush=True)
