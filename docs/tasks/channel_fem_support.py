"""Explicit input-only screen and matched historical/full-FE validation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

HISTORICAL = Path('/root/autodl-tmp/tf-logs/channel_nsch2d/20261103-pro6000-integral-random-walls-online-width192-b8-time18-nt36-cold20k-lr1e5-04')
LOGS = Path('/root/autodl-tmp/tf-logs/channel_nsch2d')
COMMON = ('points','volumes','interior','amplitudes','boundary','boundary_weights','domain','domain_weights')


def json_write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def configuration():
    from examples.channel_nsch2d.configs.train_operator_fem_weak import get_config
    from nonlocal_pinn.training.runtime import with_run_seed
    return with_run_seed(get_config(),42)


def construct(job):
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.fem_weak import geometry_data
    from examples.channel_nsch2d.online import validate_payload, INTEGER_INPUTS
    config,parameters,label=job;c=ConfigDict(config)
    raw,report=geometry_data(parameters,c)
    cast={k:v.astype('int32' if k in INTEGER_INPUTS else c.train.compute_dtype)for k,v in raw.items()}
    validate_payload(cast,c)
    return dict(label=label,parameters=np.asarray(parameters).tolist(),**report)


def screen(output):
    from examples.channel_nsch2d.configs.train_operator_random_walls_width192_batch8_time18_nt36_cold20k import get_config as base
    from examples.channel_nsch2d.geometry import initial_parameters
    from examples.channel_nsch2d.online import sample_geometry
    from examples.channel_nsch2d.problem import validate_config
    from nonlocal_pinn.training.runtime import with_run_seed
    c=configuration();validate_config(c);start=time.perf_counter()
    current=json.loads(with_run_seed(base(),42).to_json());old=json.loads((HISTORICAL/'config.json').read_text())
    current['log']['run_name']=old['log']['run_name']
    if current!=old:raise ValueError('Historical configuration no longer matches the declared full20k baseline')
    for relative in ('network.py','geometry.py','initialization.py','time_batching.py','temporal.py'):
        a=HISTORICAL/'source/examples/channel_nsch2d'/relative;b=Path('examples/channel_nsch2d')/relative
        if a.read_bytes()!=b.read_bytes():raise ValueError('Historical core source changed: '+relative)
    for relative in ('src/nonlocal_pinn/nn/point_conditional.py','src/nonlocal_pinn/training/optimizers.py','src/nonlocal_pinn/training/runtime.py'):
        if (HISTORICAL/'source'/relative).read_bytes()!=Path(relative).read_bytes():raise ValueError(relative)
    jobs=[(c.to_dict(),a,'initial-'+str(i))for i,a in enumerate(initial_parameters(c))]
    count=((c.train.iterations-1)//c.operator.online.refresh_every+c.operator.online.prefetch_batches)*c.operator.online.refresh_size
    jobs.extend((c.to_dict(),sample_geometry(c,i),'online-'+str(i))for i in range(count))
    with np.load(HISTORICAL/'reserved_geometries.npz')as f:reserved=f['amplitudes']
    for _,a,label in jobs:
        if np.any(np.all(np.asarray(a,dtype=c.train.compute_dtype)==reserved.astype(c.train.compute_dtype),axis=1)):
            raise ValueError('Reserved geometry appears in '+label)
    records=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'))as pool:
        for i,record in enumerate(pool.map(construct,jobs,chunksize=2)):
            records.append(record)
            if (i+1)%100==0:print(json.dumps(dict(constructed=i+1,total=len(jobs),seconds=time.perf_counter()-start)),flush=True)
    result=dict(status='passed',historical=str(HISTORICAL),config=c.to_dict(),
        initial=16,online=count,cases=records,seconds=time.perf_counter()-start,
        minimum_jacobian=min(r['minimum_jacobian']for r in records),
        minimum_mass_diagonal=min(r['minimum_mass_diagonal']for r in records),
        evolved_training_labels=0,pde_solves=0,network_optimizer_source_unchanged=True)
    json_write(output,result);print(json.dumps({k:v for k,v in result.items()if k not in ('cases','config')}),flush=True)


def paired(runs, count):
    for run in runs:
        for name in ('model_initial.eqx','optimizer_initial.eqx'):
            if (run/name).read_bytes()!=(HISTORICAL/name).read_bytes():raise ValueError('Historical cold-state mismatch: '+str(run/name))
        with np.load(run/'sampling.npz')as now,np.load(HISTORICAL/'sampling.npz')as old:
            for key in ('indices','times','amplitudes','draw_ids'):
                np.testing.assert_array_equal(now[key],old[key][:count],err_msg=key)
        with np.load(run/'known_inputs.npz')as now,np.load(HISTORICAL/'known_inputs.npz')as old:
            for key in COMMON:np.testing.assert_array_equal(now[key],old[key],err_msg=key)
        if count==20000:
            with np.load(run/'known_inputs_final.npz')as now,np.load(HISTORICAL/'known_inputs_final.npz')as old:
                for key in COMMON:np.testing.assert_array_equal(now[key],old[key],err_msg=key)
    return dict(cold_states=True,all_condition_time_geometry_draws=True,initial_encoder_fields=True,updates=count)


def qualify(bundle,prefix):
    runs=[LOGS/(prefix+'-probe-'+m)for m in ('nonlocal','fem')]
    pair=paired(runs,60);rows=[]
    for method,run in zip(('nonlocal','fem'),runs,strict=True):
        s=json.loads((run/'summary.json').read_text())
        if s['status']!='completed' or s['iterations']!=60:raise ValueError('Complete production60 probe required')
        memory=s['device_memory_bytes'];peak=memory['peak_bytes_in_use']
        if peak>.95*memory['bytes_limit']:raise ValueError('Probe lacks allocator headroom')
        with np.load(run/'spatial_timing.npz')as timing:
            update=timing['update'][10:];preparation=timing['batch_preparation'][10:]
            first_update=float(timing['update'][0])
            if len(update)!=50 or len(preparation)!=50:raise ValueError('Incomplete warm timings')
        rows.append(dict(method=method,run=str(run),warm_update_median=float(np.median(update)),
            warm_inclusive_median=float(np.median(update+preparation)),peak_active_gib=peak/1024**3,
            first_update_seconds=first_update))
    result=dict(status='passed',paired=pair,resources=rows,
        note='Same current GPU and settings; historical full20k accuracy remains a separate earlier run')
    json_write(Path(bundle)/'resources.json',result);print(json.dumps(result),flush=True)


def finish(bundle,prefix):
    run=LOGS/(prefix+'-fem-cold20k')
    pair=paired([run],20000)
    new=json.loads((run/'unseen3/evaluation.json').read_text())
    old=json.loads((HISTORICAL/'unseen3/evaluation.json').read_text())
    reductions={}
    for root,record in ((HISTORICAL,old),(run,new)):
        cases=[]
        for case in record['per_geometry']:
            path=root/'unseen3'/f'case_{case["index"]:03d}'/'prediction.npz'
            with np.load(path)as fields,np.load(case['reference'])as ref:
                for key in ('points','time','amplitudes'):np.testing.assert_array_equal(fields[key],ref[key])
                np.testing.assert_array_equal(fields['reference'],ref['fields'])
                metrics={}
                for name,index,begin in (('phi',slice(0,1),0),('velocity',slice(2,4),1),
                                         ('pressure',slice(4,5),1),('chemical',slice(1,2),1)):
                    actual=np.asarray(fields['fields'][begin:,...,index],dtype=np.float64)
                    target=np.asarray(ref['fields'][begin:,...,index],dtype=np.float64)
                    error=float(np.linalg.norm(actual-target)/np.linalg.norm(target))
                    if not np.isfinite(error) or abs(error-case['metrics'][name])>1e-12:
                        raise ValueError('Independent saved-field norm mismatch')
                    metrics[name]=error
            cases.append(metrics)
        reductions[str(root)]={name:float(np.mean([r[name]for r in cases]))for name in cases[0]}
        for name,value in reductions[str(root)].items():
            if abs(value-record['aggregate'][name]['mean'])>1e-12:raise ValueError('Cohort mean mismatch')
    files=sorted((run/'source').rglob('*.py'))
    for file in files:
        if file.read_bytes()!=file.relative_to(run/'source').read_bytes():
            raise ValueError('Training/current source drift: '+str(file))
    report=dict(status='passed',historical=str(HISTORICAL),fem_run=str(run),
        paired=pair,resources=json.loads((Path(bundle)/'resources.json').read_text()),
        fem_summary=json.loads((run/'summary.json').read_text()),
        historical_cohort=old,fem_cohort=new,independent_reductions=reductions,
        training_source_files_match_current=len(files),
        limitations=['One initialization and three repeatedly used reserved geometries.',
                     'Historical accuracy and new resource probes have different physical GPU UUIDs.',
                     'FEM uses a mixed FE weak norm and natural boundaries; nonlocal retains nine pointwise losses.'])
    json_write(Path(bundle)/'comparison.json',report)
    print(json.dumps(dict(status='passed',comparison=str(Path(bundle)/'comparison.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('screen','qualify','finish'),required=True)
    parser.add_argument('--bundle',required=True);parser.add_argument('--prefix',required=True)
    args=parser.parse_args()
    if args.action=='screen':screen(Path(args.bundle)/'input-screen.json')
    elif args.action=='qualify':qualify(args.bundle,args.prefix)
    else:finish(args.bundle,args.prefix)
