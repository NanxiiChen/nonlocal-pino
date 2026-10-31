"""Geometry bank and independent native/local references; CPU/FEniCS process."""
import argparse
import importlib
import json
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import numpy as np
from ml_collections import ConfigDict
from scipy import linalg
from scipy.sparse import diags, eye
from scipy.sparse.csgraph import connected_components
from examples.confined_ch2d.geometry import create_mesh, quadrature
from examples.confined_ch2d.native_reference import native_matrix, native
from examples.confined_ch2d.problem import relative
from examples.confined_ch2d.operator_problem import (
    validate, sample_geometries, condition, case_config, specification)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare_case(config,index,holes):
    """Independent CPU reference for one geometry; no training/JAX imports."""
    c=ConfigDict(config)
    cc=case_config(c,holes,index)
    directory=Path(cc.reference.root);directory.mkdir()
    vertices,cells=create_mesh(cc,c.geometry.mesh_size,directory/'mesh.xdmf')
    points,volumes=quadrature(vertices,cells)
    A=native_matrix(points,volumes,cc)
    if connected_components(A,directed=False,return_labels=False)!=1:
        raise ValueError(f'Disconnected regional geometry {index}')
    coarse=native(cc,points,volumes,A,c.reference.dt,1e-7)
    fine=native(cc,points,volumes,A,c.reference.refined_dt,1e-9)
    errors={key:max(relative(a,b,volumes) for a,b in zip(coarse[key],fine[key],strict=True))
            for key in ('phi','mu')}
    drift=float(np.max(abs(fine['mass']-fine['mass'][0])))
    increase=float(max(0.,np.max(np.diff(fine['energy']))))
    if errors['phi']>=c.reference.phase_tolerance or errors['mu']>=c.reference.chemical_tolerance or drift>1e-9 or increase>1e-9:
        raise ValueError(f'Native reference failed: {index}, {errors}, mass={drift}, energy={increase}')
    # Independent volume-inner-product biharmonic resolvent, same norm as
    # the existing confined case. It transforms residuals, never labels.
    rv=np.sqrt(volumes)
    S=diags(rv)@A@diags(1/rv)
    P=(eye(len(points))+c.physical.terminal_time*c.physical.mobility*c.physical.epsilon**2*(S.T@S)).toarray()
    inverse=linalg.cho_solve(linalg.cho_factor(P,lower=True),np.eye(len(points)))
    metric=inverse*rv[None,:]/rv[:,None]
    max_degree=int(np.diff(A.indptr).max()-1)
    neighbors=np.repeat(np.arange(len(points))[:,None],max_degree,axis=1)
    coefficients=np.zeros_like(neighbors,dtype=float)
    for row in range(len(points)):
        ids=A.indices[A.indptr[row]:A.indptr[row+1]]
        vals=A.data[A.indptr[row]:A.indptr[row+1]]
        active=ids!=row;ids,vals=ids[active],vals[active]
        neighbors[row,:len(ids)]=ids;coefficients[row,:len(ids)]=-vals
    features,weights=condition(holes,c)
    np.savez_compressed(directory/'reference.npz',points=points,volumes=volumes,
        mesh_points=vertices,mesh_cells=cells,holes=holes,
        time=np.linspace(0,c.physical.terminal_time,c.reference.snapshots),
        phi=fine['phi'],mu=fine['mu'],mass=fine['mass'],energy=fine['energy'],
        condition=features,condition_weights=weights,neighbors=neighbors,
        coefficients=coefficients,metric=metric)
    record=dict(index=index,file=f'case_{index:03d}/reference.npz',holes=holes.tolist(),
        nodes=len(points),degree=max_degree,area=float(volumes.sum()),
        native_time_errors=errors,mass_drift=drift,energy_max_increase=increase,
        frozen_initial_phi_pooled=relative(np.broadcast_to(fine['phi'][0],fine['phi'].shape),fine['phi'],volumes))
    return record


def prepare(c):
    validate(c)
    root=Path(c.operator.dataset_path)
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/'config.json',json.loads(c.to_json()))
    geometries=sample_geometries(c)
    manifest=dict(status='incomplete',specification=specification(c),conditions=[],
        train_indices=list(range(c.operator.train_conditions)),
        test_indices=list(range(c.operator.train_conditions,len(geometries))),
        positive_time_training_labels=False,sampling='finite bank drawn from continuous geometry prior')
    start=time.perf_counter();records={}
    def retain(record):
        records[record['index']]=record
        manifest['conditions']=[records[i] for i in sorted(records)]
        write_json(root/'manifest.json',manifest)
        print(json.dumps(record),flush=True)
    if 'reference_workers' in c.operator:
        manifest['reference_workers']=c.operator.reference_workers
        with ProcessPoolExecutor(max_workers=c.operator.reference_workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(prepare_case,c.to_dict(),i,holes) for i,holes in enumerate(geometries)]
            for future in as_completed(futures):retain(future.result())
    else:
        # Preserve the original serial path for legacy configs.
        for i,holes in enumerate(geometries):retain(prepare_case(c.to_dict(),i,holes))
    manifest.update(status='passed',seconds=time.perf_counter()-start)
    write_json(root/'manifest.json',manifest)
    return root


def local_case(config,index):
    """One independent mixed-P2 local check; isolated FEniCS interpreter."""
    from examples.confined_ch2d.solver import fem
    c=ConfigDict(config);root=Path(c.operator.dataset_path)
    manifest=json.loads((root/'manifest.json').read_text())
    with np.load(root/manifest['conditions'][index]['file'],allow_pickle=False) as data:
        d={key:data[key] for key in data.files}
    cc=case_config(c,d['holes'],index)
    h0,h1=c.reference.mesh_sizes
    space0=fem(cc,h0,c.reference.refined_dt,d['points'],'local-coarse')
    time0=fem(cc,h1,c.reference.dt,d['points'],'local-fine-dt')
    fine=fem(cc,h1,c.reference.refined_dt,d['points'],'local-fine')
    # A fixed ambient IC need not satisfy new hole-wall Neumann conditions
    # at t0. The weak CH evolution is valid; do not assert a classical mu0.
    errors={label:max(relative(a,b,d['volumes']) for a,b in zip(first['phi'],fine['phi'],strict=True))
            for label,first in (('space',space0),('time',time0))}
    drift=float(np.max(abs(fine['mass']-fine['mass'][0])))
    increase=float(max(0.,np.max(np.diff(fine['energy']))))
    passed=max(errors.values())<c.reference.phase_tolerance and drift<1e-8 and increase<1e-8
    report=dict(index=index,status='passed' if passed else 'failed',phase_refinement=errors,
        mass_drift=drift,energy_max_increase=increase,
        native_local_phi_pooled=relative(d['phi'],fine['phi'],d['volumes']),
        native_local_phi_terminal=relative(d['phi'][-1],fine['phi'][-1],d['volumes']),
        chemical_scope='Positive-time diagnostics only; no compatible classical initial mu assumed')
    np.savez_compressed(root/f'local_{index:03d}.npz',phi=fine['phi'],mu=fine['mu'],time=d['time'],points=d['points'])
    return report


def local_checks(c):
    """Separate local PDE bias on held-out geometries; no training labels."""
    validate(c)
    if not c.operator.local_validation_indices:
        raise ValueError('No local validation indices requested; select held-out indices to run FEM')
    root=Path(c.operator.dataset_path)
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='passed' or manifest['specification']!=specification(c):
        raise ValueError('Prepare the matching validated geometry bank first')
    records={};start=time.perf_counter()
    def retain(report):
        records[report['index']]=report
        reports=[records[i] for i in c.operator.local_validation_indices if i in records]
        write_json(root/'local_checks.json',reports)
        print(json.dumps(report),flush=True)
        if report['status']!='passed':
            raise ValueError(f'Local phase/physical reference checks failed: {report}')
    if 'local_reference_workers' in c.operator:
        with ProcessPoolExecutor(max_workers=c.operator.local_reference_workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(local_case,c.to_dict(),i) for i in c.operator.local_validation_indices]
            for future in as_completed(futures):retain(future.result())
        write_json(root/'local_preparation.json',dict(workers=c.operator.local_reference_workers,
            seconds=time.perf_counter()-start,config=json.loads(c.to_json())))
    else:
        for index in c.operator.local_validation_indices:retain(local_case(c.to_dict(),index))
    return [records[i] for i in c.operator.local_validation_indices]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='examples.confined_ch2d.configs.train_operator_mild')
    parser.add_argument('--request',type=Path)
    parser.add_argument('--local',action='store_true')
    args=parser.parse_args()
    c=ConfigDict(json.loads(args.request.read_text())) if args.request else importlib.import_module(args.config).get_config()
    local_checks(c) if args.local else prepare(c)
