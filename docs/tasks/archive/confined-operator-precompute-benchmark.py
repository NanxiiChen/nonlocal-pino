"""Time geometry/operator preparation WITHOUT any PDE/FEM trajectory solve.

Run from root: PYTHONPATH=. JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1 .venv/bin/python docs/tasks/archive/confined-operator-precompute-benchmark.py
"""
import json
from pathlib import Path
import tempfile
import time
import os
import subprocess
import numpy as np
import jax
from scipy import sparse,linalg
from examples.confined_ch2d.configs.train_operator_richer import get_config
from examples.confined_ch2d.geometry import quadrature
from examples.confined_ch2d.operator_problem import sample_geometries,case_config,condition,fixed_initial
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.native_reference import native_matrix


jax.config.update('jax_enable_x64',True)
jax.devices('cpu')  # exclude one-time backend initialization from per-geometry timings
c=get_config();holes=sample_geometries(c);records=[]
with tempfile.TemporaryDirectory(prefix='ch-operator-only-') as directory:
    root=Path(directory)
    for index in (0,16,23):
        cc=case_config(c,holes[index],index)
        request=root/f'geometry-{index}.json';request.write_text(cc.to_json())
        before=time.perf_counter()
        # Meshio is installed in fenicsenv, not .venv. Keep meshing in that
        # separate interpreter; it performs no DOLFIN/PDE solve or GPU work.
        child_env=dict(os.environ);child_env.pop('VIRTUAL_ENV',None)
        subprocess.run([c.operator.reference_python,'-c',
            'import sys,json,numpy as np; from pathlib import Path; from ml_collections import ConfigDict; '
            'from examples.confined_ch2d.geometry import create_mesh; '
            'c=ConfigDict(json.loads(Path(sys.argv[1]).read_text())); '
            'v,t=create_mesh(c,c.geometry.mesh_size,Path(sys.argv[2])); '
            'np.savez(sys.argv[3],vertices=v,cells=t)',
            str(request),str(root/f'mesh-{index}.xdmf'),str(root/f'mesh-{index}.npz')],
            check=True,env=child_env,capture_output=True,text=True)
        with np.load(root/f'mesh-{index}.npz',allow_pickle=False) as data:
            vertices,cells=data['vertices'],data['cells']
        x,v=quadrature(vertices,cells)
        mesh_seconds=time.perf_counter()-before
        before=time.perf_counter()
        operator=RegionalLaplacian(x,v,cc)
        nb,co=np.asarray(operator.neighbors),np.asarray(operator.coefficients)
        off=sparse.coo_matrix((co.ravel(),(np.repeat(np.arange(len(x)),nb.shape[1]),nb.ravel())),shape=(len(x),len(x))).tocsr()
        off.eliminate_zeros()
        A=sparse.diags(np.asarray(off.sum(axis=1)).ravel())-off
        assembly_seconds=time.perf_counter()-before
        before=time.perf_counter()
        rv=np.sqrt(v);S=sparse.diags(rv)@A@sparse.diags(1/rv)
        P=(sparse.eye(len(x))+c.physical.terminal_time*c.physical.mobility*c.physical.epsilon**2*(S.T@S)).toarray()
        inverse=linalg.cho_solve(linalg.cho_factor(P,lower=True),np.eye(len(x)))
        metric=inverse*rv[None,:]/rv[:,None]
        metric_seconds=time.perf_counter()-before
        before=time.perf_counter()
        features,weights=condition(holes[index],c)
        target=root/f'operator-{index}.npz'
        np.savez_compressed(target,points=x,volumes=v,mesh_points=vertices,mesh_cells=cells,
            holes=holes[index],neighbors=nb,coefficients=co,metric=metric,
            initial=fixed_initial(x,c,xp=np),condition=features,condition_weights=weights)
        write_seconds=time.perf_counter()-before
        before=time.perf_counter()
        independent=native_matrix(x,v,cc)
        independent_seconds=time.perf_counter()-before
        for field in (np.ones(len(x)),np.sin(np.arange(len(x))),x[:,0]+x[:,1]**2):
            np.testing.assert_allclose(A@field,independent@field,rtol=2e-12,atol=2e-10)
        records.append(dict(index=index,nodes=len(x),mesh_and_quadrature_seconds=mesh_seconds,
            production_assembly_seconds=assembly_seconds,residual_metric_seconds=metric_seconds,
            compressed_write_seconds=write_seconds,
            training_preparation_seconds=mesh_seconds+assembly_seconds+metric_seconds+write_seconds,
            independent_ordered_assembly_seconds=independent_seconds,compressed_bytes=target.stat().st_size,
            metric_dense_bytes=metric.nbytes,independent_operator_probe_agreement=True))
        print(json.dumps(records[-1]),flush=True)
mean={key:float(np.mean([r[key] for r in records])) for key in records[0] if key.endswith('_seconds') or key.endswith('_bytes')}
report=dict(config='examples/confined_ch2d/configs/train_operator_richer.py',backend='CPU',
    threads='OPENBLAS1/OMP1',pde_time_integration_performed=False,fem_solve_performed=False,
    samples=records,mean=mean,
    serial512_estimate_seconds=512*mean['training_preparation_seconds'],
    compressed512_estimate_bytes=512*mean['compressed_bytes'],
    scope='Three representative samples, serial estimate; not a measured512bank or parallel speed result. Independent ordered assembly is a separate validation cost, excluded from production preparation.')
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2),flush=True)
