"""Geometry-only postprocessing of saved ceramic predictions; no neural inference."""
import argparse
import json
from pathlib import Path
import time
import numpy as np


# Independent float64 reductions traverse hundreds of thousands of positive
# terms in different orders. Keep a numerical audit tolerance, not a change
# to the reported error: the64case extended-precision replay gives <2.2e-13
# relative drift. A2e-13gate previously rejected valid saved predictions.
REDUCTION_RTOL=1e-11


def assert_norm_replay(actual, expected):
    np.testing.assert_allclose(actual,expected,rtol=REDUCTION_RTOL,atol=0)


def reduce_regions(prediction, target, masks, weights):
    """Per-channel space-time norms on a complete disjoint spatial partition."""
    p,y=np.asarray(prediction,dtype=np.float64),np.asarray(target,dtype=np.float64)
    w=np.asarray(weights,dtype=np.float64)
    if (p.shape!=y.shape or p.ndim!=3 or w.shape!=p.shape[:2]
            or not all(np.isfinite(a).all() for a in (p,y,w)) or np.any(w<=0)):
        raise ValueError('Finite matching(T,N,C)fields and positive(T,N)weights required')
    if not masks or any(np.asarray(mask).shape!=(p.shape[1],) or np.asarray(mask).dtype!=bool
                        or not np.any(mask) for mask in masks.values()):
        raise ValueError('Nonempty spatial boolean masks required')
    if not np.all(np.sum(np.stack(list(masks.values())),axis=0)==1):
        raise ValueError('Regions must partition every evaluation point exactly once')
    error=w[...,None]*(p-y)**2;energy=w[...,None]*y*y
    global_num=error.sum(axis=(0,1));global_den=energy.sum(axis=(0,1))
    records={}
    for name,mask in dict(all=np.ones(p.shape[1],dtype=bool),**masks).items():
        num=error[:,mask].sum(axis=(0,1));den=energy[:,mask].sum(axis=(0,1))
        if np.any(den<=0):raise ValueError('Regional relative norm undefined: '+name)
        relative=np.sqrt(num/den)
        share=np.divide(num,global_num,out=np.zeros_like(num),where=global_num>0)
        records[name]=dict(numerator=num.tolist(),denominator=den.tolist(),
            channel_errors=relative.tolist(),channel_mean=float(relative.mean()),
            error_energy_fraction=share.tolist(),rmse=np.sqrt(num/w[:,mask].sum()).tolist())
    np.testing.assert_allclose(sum(np.asarray(records[k]['numerator']) for k in masks),global_num,rtol=REDUCTION_RTOL,atol=1e-13)
    np.testing.assert_allclose(sum(np.asarray(records[k]['denominator']) for k in masks),global_den,rtol=REDUCTION_RTOL,atol=1e-13)
    return records


def ceramic(evaluation,output,*,schema):
    from examples.particle_corrosion2d.geometry import boundary
    from examples.particle_corrosion2d.inputs import particle_wall_band
    started=time.perf_counter()
    if schema not in ('unified_final64','historical_step20000'):
        raise ValueError('Explicit supported evaluation schema required')
    filename='summary.json' if schema=='unified_final64' else 'step_020000.json'
    summary=json.loads((evaluation/filename).read_text())
    if summary['status']!='passed' or summary['count']!=64 or Path(summary['checkpoint']).name!='model_final.eqx':
        raise ValueError('Completed64case final-weight evaluation required')
    if schema=='historical_step20000' and summary['step']!=20000:
        raise ValueError('Historical control must use its final20000step')
    run=Path(summary['run']);reference=Path(summary['reference'])
    config=json.loads((run/'config.json').read_text())
    width=config['nl']['delta']
    if not np.isfinite(width) or width<=0:raise ValueError('Positive explicit wall band required')
    manifest=json.loads((reference/'manifest.json').read_text())
    if len(manifest['cases'])!=64:raise ValueError('Reference cohort size differs')
    records=[]
    for i,entry in enumerate(manifest['cases']):
        prediction_path=(evaluation/f'prediction_{i:03d}.npz' if schema=='unified_final64'
                         else Path(summary['saved_prediction_files'][str(i)]))
        with np.load(prediction_path) as saved,np.load(reference/entry['file']) as ref:
            for key in ('points','times'):np.testing.assert_array_equal(saved[key],ref[key])
            p,y=saved['prediction'],ref['fields']
            theta=2*np.pi*np.arange(entry['mesh']['boundary_points'])/entry['mesh']['boundary_points']
            polygons=np.stack([boundary(particle,theta)[0] for particle in entry['geometry']['particles']])
            near=particle_wall_band(ref['points'],polygons,width)
            masks=dict(near_particle_wall=near,bulk=~near)
            times,volumes=ref['times'],ref['volumes']
            temporal=np.empty(len(times));temporal[0]=(times[1]-times[0])/2;temporal[-1]=(times[-1]-times[-2])/2
            temporal[1:-1]=(times[2:]-times[:-2])/2
            euclidean=reduce_regions(p,y,masks,np.ones(p.shape[:2]))
            physical=reduce_regions(p,y,masks,temporal[:,None]*volumes[None,:])
            prior=summary['records'][i]
            if schema=='unified_final64':
                assert_norm_replay(euclidean['all']['channel_errors'],[prior['phi'],prior['concentration']])
                prior=prior['physical_weighted']
            np.testing.assert_allclose(physical['all']['channel_errors'],
                [prior['phi_pooled'],prior['c_pooled']],rtol=2e-7)
            records.append(dict(index=i,points=len(near),near_wall_points=int(near.sum()),
                near_wall_volume_fraction=float(volumes[near].sum()/volumes.sum()),
                euclidean=euclidean,physical=physical))
    statistics={}
    for measure in ('euclidean','physical'):
        statistics[measure]={}
        for region in ('all','near_particle_wall','bulk'):
            values=np.array([r[measure][region]['channel_errors'] for r in records])
            statistics[measure][region]=dict(channel_mean=values.mean(axis=0).tolist(),
                channel_std=values.std(axis=0,ddof=1).tolist(),
                sample_channel_mean=float(values.mean(axis=1).mean()),
                sample_channel_std=float(values.mean(axis=1).std(ddof=1)),
                mean_error_energy_fraction=np.mean([r[measure][region]['error_energy_fraction'] for r in records],axis=0).tolist())
    result=dict(status='passed',evaluation=str(evaluation),schema=schema,reference=str(reference),count=64,wall_band=width,
        regions='Distance to actual reference polygonal ceramic contours <= training delta; bulk is exact complement',
        statistic='First space-time norm per channel and sample; then channel/sample averages',
        masks_depend_on_predictions=False,masks_depend_on_evolved_reference=False,training_or_inference=False,
        norm_replay_rtol=REDUCTION_RTOL,
        statistics=statistics,records=records,seconds=time.perf_counter()-started)
    with output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(regional_output=str(output),statistics=statistics)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--schema',choices=('unified_final64','historical_step20000'),required=True)
    args=parser.parse_args();ceramic(args.evaluation,args.output,schema=args.schema)
