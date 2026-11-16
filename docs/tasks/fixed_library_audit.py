"""Audit completed finite-library runs and independently recompute final errors."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def norms(prediction, target):
    p, r = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    if p.shape != r.shape or p.shape[-1] != 2 or not np.isfinite(p).all():
        raise ValueError('Finite matching physical two-channel fields required')
    return [float(np.linalg.norm((p[..., k]-r[..., k]).ravel())/np.linalg.norm(r[..., k].ravel()))
            for k in range(2)]


def audit(case, run, first):
    config=read(run/'config.json'); summary=read(run/'summary.json')
    c=config['operator']; count=(c['geometry_count'] if case=='ceramic' else c['condition_library']['count'])
    expected_steps=20000 if case=='ceramic' else 10000
    steps=summary['steps'] if case=='ceramic' else summary['iterations']
    if steps != expected_steps:raise ValueError('Incomplete fixed training budget')
    if first != run:
        for name in ('model_initial.eqx','optimizer_initial.eqx'):
            if (first/name).read_bytes() != (run/name).read_bytes():raise ValueError('Changed cold initial state: '+name)
        previous=read(first/'config.json')
        current=read(run/'config.json')
        if case=='ceramic':current['operator']['geometry_count']=previous['operator']['geometry_count']
        else:
            current['operator']['condition_library']['count']=previous['operator']['condition_library']['count']
            current['log']['run_name']=previous['log']['run_name']
        if current!=previous:raise ValueError('Configuration changed beyond library size/run name')
        with np.load(first/'sampling.npz',allow_pickle=False) as a,np.load(run/'sampling.npz',allow_pickle=False) as b:
            keys=('times','pool_ids') if case=='ceramic' else a.files
            for key in keys:np.testing.assert_array_equal(a[key],b[key])
    values=[]
    if case=='ceramic':
        from examples.particle_corrosion2d.online_validation import audit_online_run
        reconstruction=audit_online_run(run,workers=4)
        if (summary['online']['installed_geometries']!=0
                or summary['online']['installed_spatial_refreshes']<=0):raise ValueError('Geometry freeze or spatial rolling violated')
        runtime=read(run/'runtime.json'); root=Path(runtime['reference']); manifest=read(root/'manifest.json')
        with np.load(run/'sampling.npz') as data,np.load(run/'online_sampling.npz') as online:
            # Field names follow the production sampler receipt, checked explicitly.
            ids=data['condition_ids']; expected=-1-ids.astype(np.int64)
            np.testing.assert_array_equal(online['geometry_draw_ids'],expected)
            sampled=np.unique(ids).tolist()
        for i,record in enumerate(manifest['cases']):
            with np.load(root/record['file']) as r,np.load(run/f'prediction_{i:03d}.npz') as p:
                np.testing.assert_array_equal(r['points'],p['points']);np.testing.assert_array_equal(r['times'],p['times'])
                values.append(norms(p['prediction'],r['fields']))
        rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        if [r['step'] for r in rows]!=list(range(0,20001,500)):raise ValueError('Incomplete64case monitoring schedule')
        logged=[[r['equal_phi'],r['equal_c']] for r in rows[-1]['conditions']]
        evaluation_source=str(root)
    else:
        from examples.electropolish2d.condition_library import load_library
        from ml_collections import ConfigDict
        library,_=load_library(ConfigDict(config))
        lookup={tuple(p.ravel()):i for i,p in enumerate(library)}
        actual=np.load(run/'online_coefficients.npy')
        sampled_ids=np.array([lookup[tuple(p.ravel())] for p in actual.reshape(-1,*library.shape[1:])])
        sampled=np.unique(sampled_ids).tolist()
        if actual.shape[:2]!=(10000,c['condition_batch']):raise ValueError('Incomplete actual condition draws')
        from docs.tasks.electro_spatial_evaluate import LOCAL
        evaluated=read(run/'spatial_evaluation64.json')
        if evaluated['status']!='passed':raise ValueError('Missing independent final inference')
        with np.load(run/'spatial_evaluation64.npz') as data:
            take=evaluated['regular_time_indices']
            for i,predicted in enumerate(data['prediction']):
                with np.load(LOCAL/f'condition_{i:03d}/local101.npz') as r:
                    np.testing.assert_array_equal(data['time'],r['time'])
                    values.append(norms(predicted[take],r['fields'][take]))
        rows=[json.loads(line) for line in (run/'local_metrics.jsonl').read_text().splitlines()]
        if [r['step'] for r in rows]!=list(range(0,10001,250)):raise ValueError('Incomplete64case monitoring schedule')
        logged=[[r['phi'],r['c']] for r in rows[-1]['conditions']]
        reconstruction=dict(actual_draws_restricted_to_library=True,
            spatial_sampling=read(run/'spatial_sampling.json'))
        evaluation_source=str(LOCAL)
    if values and len(values)!=64:raise ValueError('Require all64test samples')
    np.testing.assert_allclose(values,logged,rtol=3e-6,atol=2e-8)
    errors=np.asarray(values);means=errors.mean(axis=1)
    if sampled!=list(range(count)):raise ValueError('Declared library has unsampled conditions')
    result=dict(status='passed',case=case,library_size=count,run=str(run),config=str(run/'config.json'),
        checkpoint=str(run/'model_final.eqx'),steps=steps,reference=evaluation_source,
        channels=['phi','c'],channel_errors=errors.tolist(),sample_errors=means.tolist(),
        mean=float(means.mean()),sample_std=float(means.std(ddof=1)),channel_means=errors.mean(axis=0).tolist(),
        metric='Equal-point full-space-time L2 per channel; arithmetic channel mean per sample',
        sample_count=64,actually_sampled_count=len(sampled),input_audit=reconstruction,
        initialization_and_spatial_time_draws_match_first=True)
    (run/'fixed_library_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','case','library_size','mean','sample_std')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=('ceramic','electropolish'),required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--first',type=Path,required=True)
    a=p.parse_args();audit(a.case,a.run.resolve(strict=True),a.first.resolve(strict=True))
