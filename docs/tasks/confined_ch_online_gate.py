"""CPU preparation gate and independent completed-run audit for rolling inputs."""
import argparse
import importlib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import tempfile
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from examples.confined_ch2d.configs.train_operator_online_rolling import get_config
from examples.confined_ch2d.geometry import quadrature
from examples.confined_ch2d.operator_continuation import continuation_spec,initial_state
from examples.confined_ch2d.operator_online import build_input,_cpu_worker,sample_geometry,pool_identities
from examples.confined_ch2d.operator_problem import validate,case_config
from examples.confined_ch2d.operator_precision import make_compute_optimizer
from examples.confined_ch2d.operator_training import build_network,fields,operator_temporal_samples


def cpu_gate(output,config_module=None):
    if os.environ['JAX_PLATFORMS']!='cpu':raise ValueError('CPU-only gate required')
    if output.exists():raise FileExistsError(output)
    jax.config.update('jax_enable_x64',True)
    c=(get_config() if config_module is None else importlib.import_module(config_module).get_config())
    c.train.seed=42;validate(c);spec=continuation_spec(c)
    keys=jax.random.split(jax.random.key(c.train.seed),3)
    model=build_network(c,keys[0]);opt=make_compute_optimizer(c)
    model,state,start,_=initial_state(model,opt,spec)
    counts=sorted(set(int(a) for a in jax.tree.leaves(state)
        if eqx.is_array(a) and a.shape==() and np.issubdtype(a.dtype,np.integer)))
    if spec is None:
        if start!=0 or counts!=[0] or c.operator.online.origin_step!=0:
            raise ValueError('Fresh online model/SOAP/schedule must start at zero')
        if not all(np.isfinite(a).all() and a.dtype==np.dtype(c.train.compute_dtype)
                for tree in (model,state) for a in jax.tree.leaves(tree) if eqx.is_inexact_array(a)):
            raise ValueError('Fresh model/SOAP must be finite and use compute_dtype')
    else:
        if start!=25000 or 25000 not in counts:raise ValueError('SOAP global schedule counter was not restored')
        with tempfile.TemporaryDirectory(prefix='online-state-gate-') as temporary:
            temporary=Path(temporary)
            for name,tree,source in [('model',model,spec['model']),('optimizer',state,spec['optimizer_state'])]:
                target=temporary/(name+'.eqx');eqx.tree_serialise_leaves(target,tree)
                if target.read_bytes()!=Path(source).read_bytes():raise ValueError(f'Nonexact {name} state restore')
    records=[];samples={};tick=time.perf_counter()
    with ProcessPoolExecutor(max_workers=c.operator.online.workers,
            mp_context=multiprocessing.get_context('spawn'),initializer=_cpu_worker) as executor:
        jobs=[executor.submit(build_input,c.to_dict(),draw) for draw in range(64)]
        for draw,job in enumerate(jobs):
            data,record=job.result()
            if record['draw_id']!=draw:raise ValueError('Producer draw order changed')
            if draw<3:samples[draw]=data
            records.append(record)
    wall=time.perf_counter()-tick
    mesh_comparison=[]
    for draw in range(2):
        holes=sample_geometry(c,draw);cc=case_config(c,holes,draw)
        with tempfile.TemporaryDirectory(prefix='online-mesh-parity-') as temporary:
            temporary=Path(temporary);request=temporary/'config.json';request.write_text(cc.to_json())
            subprocess.run([c.operator.reference_python,'-c',
                'import sys,numpy as np; from pathlib import Path; import json; '
                'from ml_collections import ConfigDict; from examples.confined_ch2d.geometry import create_mesh; '
                'c=ConfigDict(json.loads(Path(sys.argv[1]).read_text())); '
                'v,t=create_mesh(c,c.geometry.mesh_size,Path(sys.argv[2])); '
                'np.savez(sys.argv[3],vertices=v,cells=t)',
                str(request),str(temporary/'old.xdmf'),str(temporary/'old.npz')],
                check=True,capture_output=True,text=True)
            with np.load(temporary/'old.npz',allow_pickle=False) as data:
                points,volumes=quadrature(data['vertices'],data['cells'])
            n=len(points);new=samples[draw]
            np.testing.assert_allclose(new['points'][:n],points.astype(np.float32),rtol=0,atol=1e-7)
            np.testing.assert_allclose(new['volumes'][:n],volumes.astype(np.float32),rtol=1e-6,atol=1e-10)
            mesh_comparison.append(dict(draw=draw,nodes=n,point_max_gap=float(abs(new['points'][:n]-points).max()),
                volume_max_gap=float(abs(new['volumes'][:n]-volumes).max()),topology_and_order='same triangle-centroid order'))
    _cpu_worker();repeat,record=build_input(c.to_dict(),0)
    for name in samples[0]:np.testing.assert_array_equal(samples[0][name],repeat[name])
    manifest=json.loads((Path(c.operator.dataset_path)/'manifest.json').read_text())
    forbidden=np.asarray([manifest['conditions'][index]['holes'] for index in manifest['test_indices']])
    for record in records:
        if np.any(np.all(forbidden==np.asarray(record['holes']),axis=(1,2))):raise ValueError('Reserved geometry generated')
        if record['pde_solves'] or record['fem_solves']:raise ValueError('Input producer integrated a PDE')
    result=dict(status='passed',checked_utc=datetime.now(timezone.utc).isoformat(),
        parent=None if spec is None else spec['parent_run'],checkpoint=None if spec is None else spec['checkpoint'],
        start_step=start,optimizer_integer_counters=counts,
        initialization='fresh' if spec is None else 'paired_checkpoint',
        exact_model_and_optimizer_bytes=None if spec is None else True,producer_inputs=64,workers=c.operator.online.workers,
        producer_wall_seconds=wall,producer_inputs_per_second=64/wall,
        mean_stage_seconds={key:float(np.mean([r['seconds'][key] for r in records])) for key in records[0]['seconds']},
        max_nodes=max(r['nodes'] for r in records),max_degree=max(r['degree'] for r in records),
        constant_mode_error_max=max(r['constant_mode_error'] for r in records),
        mesh_comparison=mesh_comparison,repeated_draw0_all_eight_arrays_exact=True,
        future_labels_in_payload=False,pde_solves=0,fem_solves=0,
        padding=dict(points=c.operator.online.max_points,neighbors=c.operator.online.max_neighbors),
        scope='Production CPU producer incl spawn/Gmsh/graph/stable metric/padding/IPC; three payloads retained for checks, no GPU transfer/refresh or trainer measured. Same continuous valid geometry prior; no evolved labels.')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


def audit(run,output):
    if os.environ['JAX_PLATFORMS']!='cpu':raise ValueError('Independent audit must use CPU')
    jax.config.update('jax_enable_x64',True)
    c=ConfigDict(json.loads((run/'config.json').read_text()));summary=json.loads((run/'summary.json').read_text())
    if 'continuation' in c.train:
        parent=Path(c.train.continuation.parent_run);marker=json.loads(Path(c.train.continuation.checkpoint).read_text())
        for child,key in [('model_initial.eqx','model'),('optimizer_initial.eqx','optimizer')]:
            if (run/child).read_bytes()!=(parent/marker[key]).read_bytes():raise ValueError('Initial state differs from parent checkpoint')
        initialization='Exact paired checkpoint model/SOAP import'
    else:
        receipt=json.loads((run/'initial_restore_audit.json').read_text())
        if receipt['status']!='passed' or receipt['step']!=0 or summary['start_step']!=0:
            raise ValueError('Missing fresh-initialization gate')
        template=build_network(c,jax.random.split(jax.random.key(c.train.seed),3)[0])
        initial=eqx.tree_deserialise_leaves(run/'model_initial.eqx',template)
        fresh=make_compute_optimizer(c).init(eqx.filter(initial,eqx.is_array))
        saved_initial=eqx.tree_deserialise_leaves(run/'optimizer_initial.eqx',fresh)
        for a,b in zip(jax.tree.leaves(fresh),jax.tree.leaves(saved_initial),strict=True):
            if eqx.is_array(a):np.testing.assert_array_equal(a,b)
        initialization='Fresh model provenance and independently reconstructed zero-count SOAP state'
    with np.load(run/'training_draws.npz',allow_pickle=False) as f:
        ids,actual,times=f['condition_ids'],f['geometry_draw_ids'],f['times']
    start=summary['start_step'];end=summary['final_metrics']['step'];steps=np.arange(start+1,end+1)
    keys=jax.random.split(jax.random.key(c.train.seed),3)
    expected=jax.jit(jax.vmap(lambda i:jax.random.randint(jax.random.fold_in(keys[1],i),
        (c.operator.condition_batch,),0,c.operator.train_conditions)))(steps)
    np.testing.assert_array_equal(ids,expected)
    expected_times=jax.jit(jax.vmap(lambda i:operator_temporal_samples(c,i,jax.random.fold_in(keys[2],i))[0]))(steps)
    np.testing.assert_array_equal(times,expected_times)
    counts=(steps-c.operator.online.origin_step+c.operator.online.refresh_every-1)//c.operator.online.refresh_every
    latest=counts[:,None]*c.operator.online.refresh_size-1
    expected_ids=np.where(latest>=ids,ids+((latest-ids)//c.operator.train_conditions)*c.operator.train_conditions,-1-ids)
    np.testing.assert_array_equal(actual,expected_ids)
    records=[json.loads(s) for s in (run/'online_inputs.jsonl').read_text().splitlines()]
    first_count=(start-c.operator.online.origin_step+c.operator.online.refresh_every-1)//c.operator.online.refresh_every
    if [r['draw_id'] for r in records]!=list(range(first_count*c.operator.online.refresh_size,
            int(counts[-1])*c.operator.online.refresh_size)):
        raise ValueError('Online production record has missing or reordered identities')
    for record in records:np.testing.assert_array_equal(record['holes'],sample_geometry(c,record['draw_id']))
    saved=json.loads((run/f'online_pool_{end:06d}.json').read_text())
    np.testing.assert_array_equal(saved['draw_ids'],pool_identities(int(counts[-1]),c.operator.train_conditions,c.operator.online.refresh_size))
    model=build_network(c,keys[0]);model=eqx.tree_deserialise_leaves(run/'model_final.eqx',model)
    state=make_compute_optimizer(c).init(eqx.filter(model,eqx.is_array))
    state=eqx.tree_deserialise_leaves(run/'optimizer_final.eqx',state)
    if not all(np.isfinite(a).all() for tree in (model,state) for a in jax.tree.leaves(tree) if eqx.is_array(a)):
        raise FloatingPointError('Nonfinite final model/SOAP state')
    replay=[]
    for row in summary['final_metrics']['conditions']:
        index=row['index']
        with np.load(run/f'prediction_{index:03d}.npz',allow_pickle=False) as f:
            prediction={name:f[name] for name in f.files}
        bank=Path(c.operator.dataset_path);manifest=json.loads((bank/'manifest.json').read_text())
        with np.load(bank/manifest['conditions'][index]['file'],allow_pickle=False) as f:
            data={name:f[name] for name in ('points','volumes','initial','neighbors','coefficients','metric','condition','condition_weights') if name in f}
        if 'initial' not in data:
            from examples.confined_ch2d.operator_problem import fixed_initial
            data['initial']=fixed_initial(data['points'],c,xp=np)
        output_fields=np.asarray(eqx.filter_jit(lambda net:fields(net,data,prediction['time'],c))(model))
        gaps=[float(abs(output_fields[...,j]-prediction[name]).max()) for j,name in enumerate(['phi','mu'])]
        if gaps[0]>5e-5 or gaps[1]>5e-4:raise ValueError(f'Independent CPU field replay failed {index}: {gaps}')
        replay.append(dict(index=index,max_field_gaps=gaps))
    result=dict(status='passed',checked_utc=datetime.now(timezone.utc).isoformat(),run=str(run),
        start_step=start,end_step=end,actual_global_condition_ids_exact=True,actual_global_time_draws_exact=True,
        actual_geometry_identities_exact=True,stateless_generated_holes_exact=True,
        generated_geometry_count=len(records),distinct_nonbank_geometries_sampled=int(len(np.unique(actual[actual>=0]))),
        initial_model_and_SOAP_exact=True,initialization=initialization,final_state_finite=True,
        bounded_pool_final_identities_exact=True,cpu_replays=replay,final_metrics=summary['final_metrics'],
        scope='One run, same16frozen original training-domain monitors and8reused native development cases. No new test/FEM/generalization guarantee.')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='final_metrics'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['cpu','audit'],required=True)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config-module',help='Explicit configuration module for the CPU preparation gate')
    args=parser.parse_args()
    if args.action=='cpu':cpu_gate(args.output,args.config_module)
    else:
        if args.run is None:parser.error('--run required for audit')
        audit(args.run,args.output)
