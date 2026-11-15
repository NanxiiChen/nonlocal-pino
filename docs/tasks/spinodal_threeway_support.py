"""Direct-state/input/source and saved-field checks for the matched CH controls."""
import argparse
import json
from pathlib import Path

import numpy as np
from ml_collections import ConfigDict

from docs.tasks.electro_threeway_queue import read,write

ROOT=Path(__file__).resolve().parents[2]
LOGS=Path('/root/autodl-tmp/tf-logs/confined_ch2d')
METHODS=('nonlocal','local_ad','fem')


def normalized(config):
    c=json.loads(json.dumps(config));method=c['operator']['spatial_discretization']
    if method=='local_ad':
        settings=c['operator'].pop('local_ad')
        if settings!=dict(query_chunk=128,boundary='existing_physical_measure_positive_times',
                         phase_flux_scale=1.,chemical_flux_scale=1.,boundary_weight=1.):
            raise ValueError('Unexpected local-AD boundary/implementation settings')
    elif method=='fem_p1_lumped':
        settings=c['operator'].pop('fem')
        if settings!=dict(kind='p1_lumped',max_points=1024,max_neighbors=32,
                         reaction='nodal',evaluation='neural_phase_p1_chemical'):
            raise ValueError('Unexpected FEM assembly/reconstruction settings')
    elif method!='nonlocal':raise ValueError('Unknown derivative method')
    c['operator']['spatial_discretization']='nonlocal';c['log']['run_name']='paired'
    if c['loss']['initial_residual_fraction']!=0.:raise ValueError('Positive-time objective must be matched')
    return c


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def paired(runs,steps):
    import jax
    import jax.numpy as jnp
    from examples.confined_ch2d.operator_training import operator_temporal_samples
    from examples.confined_ch2d.operator_online import sample_geometry
    jax.config.update('jax_enable_x64',True)
    if jax.default_backend()!='cpu':raise RuntimeError('Input and field audit must run on CPU')
    base=runs[0];config=read(base/'config.json');c=ConfigDict(config)
    if c.train.iterations!=steps:raise ValueError('Unexpected run length')
    original=normalized(config)
    with np.load(base/'training_draws.npz') as f:draws={key:f[key] for key in f.files}
    if set(draws)!={'condition_ids','geometry_draw_ids','times'}:raise ValueError('Unexpected input trace schema')
    if draws['condition_ids'].shape!=(steps,c.operator.condition_batch):raise ValueError('Incomplete condition trace')
    keys=jax.random.split(jax.random.key(c.train.seed),3)

    @jax.jit
    def expected(iterations):
        ids=jax.vmap(lambda i:jax.random.randint(jax.random.fold_in(keys[1],i),
            (c.operator.condition_batch,),0,c.operator.train_conditions))(iterations)
        times=jax.vmap(lambda i:operator_temporal_samples(c,i,jax.random.fold_in(keys[2],i))[0])(iterations)
        return ids,times

    for begin in range(0,steps,256):
        ids,times=expected(jnp.arange(begin+1,min(begin+257,steps+1),dtype=jnp.uint32))
        np.testing.assert_array_equal(draws['condition_ids'][begin:begin+256],ids)
        np.testing.assert_array_equal(draws['times'][begin:begin+256],times)
    count=(np.arange(1,steps+1)+c.operator.online.refresh_every-1)//c.operator.online.refresh_every
    produced=count[:,None]*c.operator.online.refresh_size;slots=draws['condition_ids']
    latest=slots+((produced-1-slots)//c.operator.train_conditions)*c.operator.train_conditions
    expected_draw=np.where(produced>slots,latest,-1-slots)
    np.testing.assert_array_equal(draws['geometry_draw_ids'],expected_draw)
    checked_sources=[];online_count=int(count[-1])*c.operator.online.refresh_size
    baseline_online=lines(base/'online_inputs.jsonl')
    if [row['draw_id'] for row in baseline_online]!=list(range(online_count)):
        raise ValueError('Incomplete consumed online geometry stream')
    for row in baseline_online:
        np.testing.assert_array_equal(row['holes'],sample_geometry(c,row['draw_id']))
        if row['pde_solves']!=0 or row['fem_solves']!=0:
            raise ValueError('Training geometry construction must be input-only')
    for run in runs:
        if normalized(read(run/'config.json'))!=original:raise ValueError('Unexpected physics/network/optimizer differences')
        for name in ('model_initial.eqx','optimizer_initial.eqx'):
            if (run/name).read_bytes()!=(base/name).read_bytes():raise ValueError('Cold state mismatch '+str(run/name))
        for name in ('training_draws.npz','known_encoder_inputs.npz','first_online_encoder_inputs.npz'):
            with np.load(run/name) as a,np.load(base/name) as b:
                if set(a.files)!=set(b.files):raise ValueError('Input schema mismatch')
                for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=str(run/name)+':'+key)
        records=lines(run/'online_inputs.jsonl')
        if len(records)!=online_count:raise ValueError('Incomplete online stream')
        for a,b in zip(records,baseline_online,strict=True):
            for key in ('draw_id','holes','nodes','degree','area','pde_solves','fem_solves'):
                if a[key]!=b[key]:raise ValueError('Common geometry construction changed: '+key)
        names=read(run/'source/files.json')
        for name in names:
            if (run/'source'/name).read_bytes()!=(ROOT/name).read_bytes():
                raise ValueError('Current/frozen numerical source mismatch '+name)
        checked_sources.append(len(names))
    # Independently regenerate representative ACTUAL installed encoder inputs.
    from examples.confined_ch2d.operator_online import build_input
    with np.load(base/'first_online_encoder_inputs.npz') as receipt:
        for i,draw in enumerate(receipt['draw_ids']):
            payload,_=build_input(c.to_dict(),int(draw))
            for key in receipt.files:
                if key!='draw_ids':np.testing.assert_array_equal(receipt[key][i],payload[key],err_msg=key)
    return dict(status='passed',steps=steps,paired_initial_states=True,all_actual_draws_equal=True,
        independently_replayed_times_conditions_and_online_ids=True,online_inputs=online_count,
        installed_encoder_inputs_rebuilt=4,source_files_per_run=checked_sources)


def resource_rows(runs):
    result=[]
    for method,run in zip(METHODS,runs,strict=True):
        summary=read(run/'summary.json')
        if summary['new_updates']!=30 or summary['start_step']!=0:raise ValueError('Require cold30step production probes')
        preflight=read(run/'online_preflight.json')
        if preflight['status']!='passed' or preflight['step']!=25:raise ValueError('Incomplete finite-state warmup gate')
        with np.load(run/'update_timing.npz') as f:
            inclusive=np.asarray(f['update_seconds'])+np.asarray(f['batch_preparation_seconds'])
        if inclusive.shape!=(30,) or not np.isfinite(inclusive).all():raise ValueError('Incomplete warm timing')
        memory=summary['device_memory_stats'];median=float(np.median(inclusive[10:]))
        if memory['peak_bytes_in_use']>.95*memory['bytes_limit']:raise ValueError('Insufficient allocator headroom')
        result.append(dict(method=method,run=str(run),warm_inclusive_median_seconds=median,
            projected_50k_update_hours=median*50000/3600,
            peak_active_gib=memory['peak_bytes_in_use']/1024**3,
            process_internal_seconds=summary['wall_seconds'],setup_seconds=summary['setup_seconds'],
            first_update_seconds=summary['first_update_seconds']))
    return result


def reference_checks(config):
    """Recompute every space/time gate from the independent saved solver fields."""
    c=ConfigDict(config);root=Path(c.operator.comparison_reference.path)
    native=Path(c.operator.dataset_path);bank=read(native/'manifest.json')
    manifest=read(root/'manifest.json');rows=[]
    if manifest['status']!='passed' or manifest['indices']!=bank['test_indices']:
        raise ValueError('Common local cohort is incomplete/unqualified')
    for item in manifest['cases']:
        index=item['index'];directory=root/f'case_{index:04d}'
        with np.load(directory/'fine-fields.npz') as fine,np.load(root/item['file']) as target,\
             np.load(native/bank['conditions'][index]['file']) as original:
            for key in ('phi','mu','mass','energy'):np.testing.assert_array_equal(fine[key],target[key])
            for key in ('points','volumes','time','holes'):np.testing.assert_array_equal(target[key],original[key])
            cells=fine['mesh_cells'];edges=np.sort(np.concatenate((cells[:,[0,1]],cells[:,[1,2]],cells[:,[2,0]])),axis=1)
            expected_dofs=2*(len(fine['mesh_points'])+len(np.unique(edges,axis=0)))
            if int(fine['dofs'])!=expected_dofs:raise ValueError('Reference is not the declared mixed P2 space')
            v=target['volumes'];check={}
            for tag,label in (('space-coarse','space'),('time-coarse','time')):
                with np.load(directory/(tag+'-fields.npz')) as other:
                    for name,begin,key,tol in (('phi',0,'phase',c.reference.phase_tolerance),
                            ('mu',1,'chemical_positive',c.reference.chemical_tolerance)):
                        a=np.asarray(other[name][begin:],dtype=float);b=np.asarray(fine[name][begin:],dtype=float)
                        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                            raise ValueError('Invalid refinement fields')
                        relative=np.sqrt(np.einsum('tn,n->t',(a-b)**2,v)/np.einsum('tn,n->t',b*b,v))
                        maximum=float(relative.max());check[label+'_'+key]=maximum
                        if not maximum<tol:raise ValueError('Saved reference does not pass its declared refinement gate')
            check['mass_drift']=float(np.max(abs(fine['mass']-fine['mass'][0])))
            check['energy_increase']=float(max(0.,np.diff(fine['energy']).max()))
            if check['mass_drift']>=1e-8 or check['energy_increase']>=1e-8:raise ValueError('Failed invariant gate')
            for key,value in check.items():np.testing.assert_allclose(value,item['checks'][key],rtol=2e-12,atol=1e-14)
            gaps={}
            for field in ('phi','mu'):
                a=np.asarray(original[field],dtype=float);b=np.asarray(target[field],dtype=float)
                gaps[field]=float(np.linalg.norm(a-b)/np.linalg.norm(b))
            rows.append(dict(index=index,refinement=check,native_to_local_unweighted=gaps))
    return dict(status='passed',reference=str(root),cases=rows,
        initial_chemical_convention=manifest['initial_chemical_convention'])


def final_metrics(runs,steps):
    from examples.confined_ch2d.operator_problem import specification
    result={}
    for method,run in zip(METHODS,runs,strict=True):
        c=ConfigDict(read(run/'config.json'));native=Path(c.operator.dataset_path)
        bank=read(native/'manifest.json');local=Path(c.operator.comparison_reference.path)
        manifest=read(local/'manifest.json')
        if (manifest['status']!='passed' or manifest['specification']!=specification(c)
                or manifest['indices']!=bank['test_indices']):raise ValueError('Unqualified or mismatched common local reference')
        for name in manifest['sources']:
            if (local/'source'/name).read_bytes()!=(ROOT/name).read_bytes():
                raise ValueError('Reference construction source changed '+name)
        monitor=lines(run/'local_metrics.jsonl')
        expected=list(range(0,steps+1,c.log.eval_every))
        if expected[-1]!=steps:expected.append(steps)
        if [row['step'] for row in monitor]!=expected:raise ValueError('Missing actual local evaluation records')
        for row in monitor:
            if [entry['index'] for entry in row['conditions']]!=bank['test_indices']:
                raise ValueError('Convergence cohort changed')
            for channel,key in (('phi','phase_mean'),('chemical','chemical_mean'),('channel_mean','channel_mean')):
                np.testing.assert_allclose(row[key],np.mean([r[channel] for r in row['conditions']]),rtol=2e-14)
        by_index={row['index']:row for row in monitor[-1]['conditions']};rows=[]
        for item in manifest['cases']:
            index=item['index']
            with np.load(run/f'prediction_{index:03d}.npz') as prediction,np.load(local/item['file']) as ref:
                with np.load(native/bank['conditions'][index]['file']) as source:
                    for key in ('points','volumes','time','holes','mesh_points','mesh_cells'):
                        np.testing.assert_array_equal(prediction[key],source[key])
                    for field in ('phi','mu'):
                        np.testing.assert_array_equal(prediction[field+'_reference'],source[field])
                for key in ('points','volumes','time','holes'):np.testing.assert_array_equal(prediction[key],ref[key])
                p=np.stack((prediction['phi'],prediction['mu']),axis=-1).astype(float)
                r=np.stack((ref['phi'],ref['mu']),axis=-1).astype(float)
                if not np.isfinite(p).all() or not np.isfinite(r).all():raise ValueError('Nonfinite final fields')
                norm=np.linalg.norm(r.reshape(-1,2),axis=0)
                if np.any(norm<=0):raise ValueError('Zero channel norm')
                errors=np.linalg.norm((p-r).reshape(-1,2),axis=0)/norm
                later=np.linalg.norm((p[1:]-r[1:]).reshape(-1,2),axis=0)/np.linalg.norm(r[1:].reshape(-1,2),axis=0)
                row=dict(index=index,phi=float(errors[0]),chemical=float(errors[1]),
                    channel_mean=float(errors.mean()),chemical_positive_times=float(later[1]))
                for key in ('phi','chemical','channel_mean','chemical_positive_times'):
                    np.testing.assert_allclose(row[key],by_index[index][key],rtol=2e-13,atol=2e-14)
                rows.append(row)
        result[method]=dict(run=str(run),checkpoint=str(run/'model_final.eqx'),cases=rows,
            means={key:float(np.mean([r[key] for r in rows])) for key in ('phi','chemical','channel_mean','chemical_positive_times')},
            monitor=str(run/'local_metrics.jsonl'),native_monitor=str(run/'metrics.jsonl'))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--prefix',required=True)
    parser.add_argument('--stage',choices=('resources','final'),required=True)
    args=parser.parse_args();probe=args.stage=='resources'
    runs=[LOGS/(args.prefix+('-probe-' if probe else '-')+method) for method in METHODS]
    steps=30 if probe else 50000
    result=dict(status='passed',pairing=paired(runs,steps),reference=reference_checks(read(runs[0]/'config.json')))
    if probe:result['resources']=resource_rows(runs)
    else:result['results']=final_metrics(runs,steps)
    result['scope']='One initialization;8reused reserved geometries; matched16positive-time PDE nodes; hardIC unchanged'
    write(args.bundle/(args.stage+'.json'),result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
