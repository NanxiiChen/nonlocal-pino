"""Parallel geometry/operator-only preparation; never integrates a PDE trajectory.

Canonical entry: train.py --config=.../train_operator_richer512.py
--config.operator.action=prepare. Workers use CPU JAX and isolated meshing.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from ml_collections import ConfigDict
from scipy import sparse,linalg
from scipy.sparse.csgraph import connected_components
from examples.confined_ch2d.geometry import quadrature
from examples.confined_ch2d.operator_problem import (
    validate,sample_geometries,specification,case_config,condition,fixed_initial,evaluation_indices)


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def build_one(config,index,holes,draw_index):
    """One worker owns one mesh/operator/metric and writes only its own files."""
    import jax
    from examples.confined_ch2d.operators import RegionalLaplacian
    jax.config.update('jax_enable_x64',True)
    c=ConfigDict(config);cc=case_config(c,holes,index)
    directory=Path(cc.reference.root);directory.mkdir()
    request=directory/'config.json';request.write_text(cc.to_json())
    before=time.perf_counter()
    env=dict(os.environ);env.pop('VIRTUAL_ENV',None)
    subprocess.run([c.operator.reference_python,'-c',
        'import sys,json,numpy as np; from pathlib import Path; from ml_collections import ConfigDict; '
        'from examples.confined_ch2d.geometry import create_mesh; '
        'c=ConfigDict(json.loads(Path(sys.argv[1]).read_text())); '
        'v,t=create_mesh(c,c.geometry.mesh_size,Path(sys.argv[2])); '
        'np.savez(sys.argv[3],vertices=v,cells=t)',
        str(request),str(directory/'mesh.xdmf'),str(directory/'mesh.npz')],
        check=True,env=env,capture_output=True,text=True)
    with np.load(directory/'mesh.npz',allow_pickle=False) as d:
        vertices,cells=d['vertices'],d['cells']
    x,v=quadrature(vertices,cells)
    if np.any(v<=0) or not np.isfinite(x).all() or not np.isfinite(v).all():
        raise ValueError(f'Invalid mesh/quadrature: {index}')
    mesh_seconds=time.perf_counter()-before;before=time.perf_counter()
    op=RegionalLaplacian(x,v,cc)
    nb,co=np.asarray(op.neighbors),np.asarray(op.coefficients)
    off=sparse.coo_matrix((co.ravel(),(np.repeat(np.arange(len(x)),nb.shape[1]),nb.ravel())),shape=(len(x),len(x))).tocsr()
    off.eliminate_zeros();A=sparse.diags(np.asarray(off.sum(axis=1)).ravel())-off
    if np.any(co<0) or connected_components(A,directed=False,return_labels=False)!=1:
        raise ValueError(f'Invalid/disconnected production operator: {index}')
    weighted=sparse.diags(v)@A
    asymmetry=weighted-weighted.T
    if asymmetry.nnz and abs(asymmetry.data).max()>1e-12:
        raise ValueError(f'Operator is not symmetric in the physical volume inner product: {index}')
    np.testing.assert_allclose(A@np.ones(len(x)),0.,atol=1e-10)
    assembly_seconds=time.perf_counter()-before;before=time.perf_counter()
    rv=np.sqrt(v);S=sparse.diags(rv)@A@sparse.diags(1/rv)
    P=(sparse.eye(len(x))+c.physical.terminal_time*c.physical.mobility*c.physical.epsilon**2*(S.T@S)).toarray()
    inverse=linalg.cho_solve(linalg.cho_factor(P,lower=True),np.eye(len(x)))
    metric=inverse*rv[None,:]/rv[:,None]
    probe=np.column_stack((np.ones(len(x)),np.sin(np.arange(len(x)))))
    np.testing.assert_allclose(P@(inverse@probe),probe,atol=1e-9,rtol=1e-9)
    if not np.isfinite(metric).all():raise ValueError(f'Nonfinite residual metric: {index}')
    metric_seconds=time.perf_counter()-before;before=time.perf_counter()
    features,weights=condition(holes,c)
    target=directory/'operator.npz'
    np.savez_compressed(target,points=x,volumes=v,mesh_points=vertices,mesh_cells=cells,
        holes=holes,neighbors=nb,coefficients=co,metric=metric,initial=fixed_initial(x,c,xp=np),
        condition=features,condition_weights=weights)
    write_seconds=time.perf_counter()-before
    return dict(index=index,draw_index=draw_index,file=f'case_{index:03d}/operator.npz',
        holes=holes.tolist(),nodes=len(x),degree=nb.shape[1],area=float(v.sum()),has_reference=False,
        bytes=target.stat().st_size,seconds=dict(mesh=mesh_seconds,assembly=assembly_seconds,
            metric=metric_seconds,write=write_seconds),
        checks='Positive physical volumes; connected nonnegative pair graph; weighted symmetry/nullspace; resolvent inverse probes')


def prepare(c):
    validate(c)
    if 'precompute' not in c.operator:raise ValueError('Explicit parallel precompute config required')
    if os.environ['JAX_PLATFORMS']!='cpu':raise ValueError('Preparation must use CPU only')
    start=time.perf_counter();source=Path(c.operator.precompute.source_path)
    old=json.loads((source/'manifest.json').read_text())
    old_config=ConfigDict(json.loads((source/'config.json').read_text()))
    if old['status']!='passed' or old['specification']!=specification(old_config):
        raise ValueError('Reuse requires a matching independently validated source bank')
    new_spec=specification(c);old_spec=specification(old_config)
    for key in new_spec:
        if key!='prior' and new_spec[key]!=old_spec[key]:
            raise ValueError(f'Physical/reference input mismatch in reused bank: {key}')
    for key in old_spec['prior']:
        if key not in ('train_conditions','test_conditions') and old_spec['prior'][key]!=new_spec['prior'][key]:
            raise ValueError(f'Geometry prior mismatch: {key}')
    reserved=list(c.operator.precompute.reserved_test_draws)
    _,source_labeled=evaluation_indices(old)
    if 'format' in old:
        if old['format']!='operator_only_v1':raise ValueError('Unsupported source bank format')
        source_records={r['draw_index']:r for r in old['conditions']}
        source_reserved=[old['conditions'][i]['draw_index'] for i in old['test_indices']]
    else:
        source_records={r['index']:r for r in old['conditions']}
        source_reserved=old['test_indices']
    if len(source_records)!=len(old['conditions']):raise ValueError('Source accepted-draw identities must be unique')
    if reserved!=source_reserved:
        raise ValueError('Preserve the exact original held-out accepted draws')
    holes=sample_geometries(c);count=len(holes)
    order=[i for i in range(count) if i not in reserved]+reserved
    train=list(range(c.operator.train_conditions));test=list(range(c.operator.train_conditions,count))
    monitor=list(c.operator.precompute.monitor_train_indices)
    for index in monitor+test:
        draw=order[index]
        if draw not in source_records or source_records[draw]['index'] not in source_labeled:
            raise ValueError(f'No existing refined reference for monitor/validation draw {draw}')
    reuse_mode=c.operator.precompute.reuse_mode if 'reuse_mode' in c.operator.precompute else 'copy'
    root=Path(c.operator.dataset_path);root.mkdir(parents=True,exist_ok=False)
    write_json(root/'config.json',json.loads(c.to_json()))
    manifest=dict(format='operator_only_v1',status='incomplete',specification=new_spec,
        conditions=[],train_indices=train,test_indices=test,monitor_train_indices=monitor,
        positive_time_training_labels=False,pde_trajectory_solves=0,fem_solves=0,
        workers=c.operator.precompute.workers,threads_per_worker=1,source_path=str(source),
        reuse_mode=reuse_mode,
        sampling='Finite continuous-prior bank; reserved old validation draws excluded from training')
    write_json(root/'manifest.json',manifest)
    records={};pending=[];reused_operators=0
    for index,h in enumerate(holes):
        draw=order[index]
        required_reference=index in monitor+test
        if draw not in source_records or (not required_reference and source_records[draw]['index'] in source_labeled):
            pending.append((index,h,draw));continue
        record=dict(source_records[draw])
        np.testing.assert_array_equal(h,np.asarray(record['holes']))
        path=Path(record['file'])
        if path.is_absolute() or '..' in path.parts:raise ValueError('Source file must be inside source bank')
        target=root/f'case_{index:03d}';target.mkdir()
        filename='reference.npz' if required_reference else 'operator.npz'
        if reuse_mode=='hardlink':os.link(source/path,target/filename)
        else:shutil.copyfile(source/path,target/filename)
        record.update(index=index,draw_index=draw,file=f'case_{index:03d}/{filename}',
            has_reference=required_reference,source_file=str(source/path),reused=True,reuse_mode=reuse_mode)
        if not required_reference:reused_operators+=1
        records[index]=record
    # Spawn avoids inheriting JAX thread pools. Meshing is a separate interpreter;
    # no DOLFIN/MPI solve or GPU backend is used anywhere in preparation.
    with ProcessPoolExecutor(max_workers=c.operator.precompute.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(build_one,c.to_dict(),i,h,draw) for i,h,draw in pending]
        for completed,future in enumerate(as_completed(futures),1):
            record=future.result();records[record['index']]=record
            if completed==1 or completed%32==0 or completed==len(pending):
                print(json.dumps(dict(generated=completed,total_new=len(pending),
                    elapsed_seconds=time.perf_counter()-start)),flush=True)
    local_records=[]
    if c.operator.local_validation_indices:
        local=json.loads((source/'local_checks.json').read_text())
    for index in c.operator.local_validation_indices:
        draw=order[index];source_index=source_records[draw]['index']
        matches=[r for r in local if r['index']==source_index]
        if len(matches)!=1 or matches[0]['status']!='passed':
            raise ValueError(f'Missing passed local validation for original draw {draw}')
        shutil.copyfile(source/f'local_{source_index:03d}.npz',root/f'local_{index:03d}.npz')
        local_records.append(dict(matches[0],index=index,source_index=source_index,source_draw_index=draw))
    write_json(root/'local_checks.json',local_records)
    manifest['local_reference_status']='passed' if local_records else 'not_requested'
    manifest['conditions']=[records[i] for i in range(count)]
    manifest.update(status='passed',seconds=time.perf_counter()-start,new_operators=len(pending),
        reused_references=len(monitor)+len(test),bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
    manifest['reused_operators']=reused_operators
    write_json(root/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('conditions','specification')},indent=2),flush=True)
    return root


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args()
    prepare(ConfigDict(json.loads(args.request.read_text())))
