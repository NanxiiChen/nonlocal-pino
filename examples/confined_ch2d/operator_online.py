"""Bounded asynchronous input-only geometry production and deterministic refresh.

CPU workers build the same physical graph/metric as the finite bank. Only the
main process owns GPU arrays; replacement values are dynamic fixed-shape inputs.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from ml_collections import ConfigDict
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from examples.confined_ch2d.geometry import quadrature
from examples.confined_ch2d.operator_problem import valid_holes,case_config,condition,fixed_initial
from examples.confined_ch2d.operator_metrics import linear_stable_resolvent


def validate_online(c):
    settings=c.operator.online
    required={'workers','refresh_size','refresh_every','prefetch_batches','max_points',
              'max_neighbors','origin_step'}
    if set(settings)!=required:raise ValueError(f'Online settings must explicitly declare {sorted(required)}')
    for name in required:
        value=settings[name]
        minimum=0 if name=='origin_step' else 1
        if isinstance(value,bool) or not isinstance(value,int) or value<minimum:
            raise ValueError(f'Integer online.{name}>={minimum} required')
    if (c.operator.action!='train' or c.operator.metric!='linear_stable_resolvent'
            or 'augmentation' in c.operator or 'initial_rhs_condition' in c.operator):
        raise ValueError('Online inputs require unaugmented stable-metric train mode without extra domain fields')
    if settings.refresh_size>c.operator.train_conditions or c.operator.train_conditions%settings.refresh_size:
        raise ValueError('Online refresh_size must divide the initial pool capacity')
    if settings.origin_step>=c.train.iterations:
        raise ValueError('Online origin_step must precede the final iteration')
    if settings.origin_step!=0 and 'continuation' not in c.train:
        raise ValueError('Cold online training requires origin_step=0')


def sample_geometry(c,draw_id):
    """Stateless rejection draw from the unchanged continuous parameter prior."""
    if isinstance(draw_id,bool) or not isinstance(draw_id,int) or draw_id<0:
        raise ValueError('Online draw identity must be a nonnegative integer')
    rng=np.random.default_rng(np.random.SeedSequence([c.train.seed,27183,draw_id]))
    for _ in range(c.operator.maximum_sampling_attempts):
        centers=rng.uniform(c.operator.center_min,c.operator.center_max)
        radii=rng.uniform(c.operator.radius_min,c.operator.radius_max)
        holes=np.column_stack((centers,radii))
        if valid_holes(holes,c):return holes
    raise RuntimeError(f'No valid geometry for online draw {draw_id}; prior not modified')


def pad_input(data,c):
    """Only known physical inputs enter the fixed-shape training payload."""
    keys={'points','volumes','initial','neighbors','coefficients','metric','condition','condition_weights'}
    if set(data)!=keys:raise ValueError('Online payload must contain exactly the eight known-input arrays')
    n=len(data['points']);k=data['neighbors'].shape[1]
    nmax,kmax=c.operator.online.max_points,c.operator.online.max_neighbors
    if n>nmax or k>kmax:
        raise ValueError(f'Online padding overflow: nodes={n}/{nmax}, degree={k}/{kmax}; refusing biased resampling/truncation')
    neighbors=np.repeat(np.arange(nmax,dtype=np.int32)[:,None],kmax,axis=1)
    coefficients=np.zeros((nmax,kmax));neighbors[:n,:k]=data['neighbors'];coefficients[:n,:k]=data['coefficients']
    output=dict(points=np.pad(data['points'],((0,nmax-n),(0,0))),volumes=np.pad(data['volumes'],(0,nmax-n)),
        initial=np.pad(data['initial'],(0,nmax-n)),neighbors=neighbors,coefficients=coefficients,
        metric=np.pad(data['metric'],((0,nmax-n),(0,nmax-n))),
        condition=data['condition'],condition_weights=data['condition_weights'])
    return {name:value.astype(np.int32 if name=='neighbors' else np.dtype(c.train.compute_dtype))
            for name,value in output.items()}


def _cpu_worker():
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    import jax
    jax.config.update('jax_platforms','cpu')
    jax.config.update('jax_enable_x64',True)
    if jax.default_backend()!='cpu':raise RuntimeError('Geometry workers must use the CPU backend')


def build_input(config,draw_id):
    """Isolated CPU graph generation; no reference/FEM time integration."""
    from examples.confined_ch2d.operators import RegionalLaplacian
    c=ConfigDict(config);holes=sample_geometry(c,draw_id);cc=case_config(c,holes,draw_id)
    tick=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='confined-online-') as directory:
        directory=Path(directory);mesh=directory/'mesh.npz';request=directory/'request.json'
        request.write_text(json.dumps(dict(config=cc.to_dict(),h=c.geometry.mesh_size,
            path=str(directory/'mesh.xdmf'),output_arrays=str(mesh))))
        # Explicit existing meshing interpreter, isolated from the JAX worker.
        # This module imports Gmsh/NumPy only: no DOLFIN or FEniCS MPI solver.
        env=dict(os.environ);env.pop('VIRTUAL_ENV',None)
        try:
            subprocess.run([c.operator.reference_python,'-m','examples.confined_ch2d.geometry','--input',str(request)],
                check=True,capture_output=True,text=True,env=env,cwd=Path(__file__).resolve().parents[2])
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f'Gmsh generation failed for online draw {draw_id}: {error.stderr}') from error
        with np.load(mesh,allow_pickle=False) as values:
            x,v=quadrature(values['vertices'],values['cells'])
    mesh_seconds=time.perf_counter()-tick;tick=time.perf_counter()
    op=RegionalLaplacian(x,v,cc);nb,co=np.asarray(op.neighbors),np.asarray(op.coefficients)
    n=len(x)
    off=sparse.coo_matrix((co.ravel(),(np.repeat(np.arange(n),nb.shape[1]),nb.ravel())),shape=(n,n)).tocsr()
    off.eliminate_zeros();A=sparse.diags(co.sum(axis=1))-off
    if np.any(co<0) or connected_components(A,directed=False,return_labels=False)!=1:
        raise ValueError(f'Invalid online graph {draw_id}')
    asymmetry=sparse.diags(v)@A-A.T@sparse.diags(v)
    if asymmetry.nnz and abs(asymmetry.data).max()>1e-12:raise ValueError('Online graph loses volume symmetry')
    np.testing.assert_allclose(A@np.ones(n),0.,atol=1e-10)
    initial=fixed_initial(x,c,xp=np);features,weights=condition(holes,c)
    assembly_seconds=time.perf_counter()-tick;tick=time.perf_counter();p=c.physical
    metric,report=linear_stable_resolvent(v,nb,co,initial,p.beta,p.epsilon,p.mobility,p.terminal_time)
    metric_seconds=time.perf_counter()-tick;tick=time.perf_counter()
    data=pad_input(dict(points=x,volumes=v,initial=initial,neighbors=nb,coefficients=co,
        metric=metric,condition=features,condition_weights=weights),c)
    if not all(np.isfinite(a).all() for a in data.values()):raise FloatingPointError('Nonfinite online input')
    record=dict(draw_id=draw_id,holes=holes.tolist(),nodes=n,degree=nb.shape[1],area=float(v.sum()),
        seconds=dict(mesh=mesh_seconds,assembly=assembly_seconds,metric=metric_seconds,padding=time.perf_counter()-tick),
        constant_mode_error=report['constant_mode_error'],pde_solves=0,fem_solves=0)
    return data,record


def replace_slots(bank,slots,updates):
    """Donate the old pool so replacement does not copy its entire dense metric."""
    import jax
    return jax.tree.map(lambda old,new:old.at[slots].set(new),bank,updates)


def refresh_count(step,c):
    if step<c.operator.online.origin_step:raise ValueError('Online step precedes origin')
    return (step-c.operator.online.origin_step+c.operator.online.refresh_every-1)//c.operator.online.refresh_every


def pool_identities(count,capacity,size):
    result=-1-np.arange(capacity,dtype=np.int64)
    for draw in range(max(0,count*size-capacity),count*size):result[draw%capacity]=draw
    return result


class RollingGeometryPool:
    """Host controller; network calls and compiled array updates stay functional."""
    def __init__(self,c,run,start_step,provenance,forbidden_holes):
        validate_online(c)
        import jax
        self.c=c;self.run=Path(run);self.forbidden=np.asarray(forbidden_holes)
        self.capacity=c.operator.train_conditions;self.size=c.operator.online.refresh_size
        self.count=refresh_count(start_step,c);self.draw_ids=pool_identities(0,self.capacity,self.size)
        if self.count:
            if provenance is None or 'online_state' not in provenance:
                raise ValueError('Resuming an online pool requires a paired input-state marker')
            state=json.loads(Path(provenance['online_state']).read_text())
            if state['step']!=start_step or state['refresh_count']!=self.count or state['origin_step']!=c.operator.online.origin_step:
                raise ValueError('Online checkpoint counters disagree')
            self.draw_ids=np.asarray(state['draw_ids'],dtype=np.int64)
            if self.draw_ids.shape!=(self.capacity,):raise ValueError('Online checkpoint pool shape mismatch')
            np.testing.assert_array_equal(self.draw_ids,pool_identities(self.count,self.capacity,self.size))
        self.executor=ProcessPoolExecutor(max_workers=c.operator.online.workers,
            mp_context=multiprocessing.get_context('spawn'),initializer=_cpu_worker)
        self.batches=deque();self.next_batch=self.count;self.refresh_wait_seconds=0.;self.update_seconds=0.
        self.replace=jax.jit(replace_slots,donate_argnums=(0,))

    def _submit(self):
        while len(self.batches)<self.c.operator.online.prefetch_batches:
            begin=self.next_batch*self.size
            jobs=[self.executor.submit(build_input,self.c.to_dict(),draw) for draw in range(begin,begin+self.size)]
            self.batches.append((self.next_batch,jobs));self.next_batch+=1

    def _collect(self,jobs):
        values=[];records=[]
        for job in jobs:
            value,record=job.result();holes=np.asarray(record['holes'])
            if np.any(np.all(self.forbidden==holes,axis=(1,2))):
                raise ValueError('Online geometry exactly duplicates a reserved development domain')
            values.append(value);records.append(record)
        return {name:np.stack([value[name] for value in values]) for name in values[0]},records

    def start(self,bank):
        """Reconstruct only the bounded active pool when restoring an online run."""
        if self.count:
            for begin in range(0,self.capacity,self.size):
                slots=np.arange(begin,begin+self.size);draws=self.draw_ids[slots]
                active=slots[draws>=0]
                if len(active):
                    jobs=[self.executor.submit(build_input,self.c.to_dict(),int(draw)) for draw in draws[draws>=0]]
                    updates,_=self._collect(jobs)
                    bank=self.replace(bank,active,updates)
            for value in bank.values():value.block_until_ready()
        self._submit();return bank

    def refresh(self,bank,step):
        wanted=refresh_count(step,self.c)
        if wanted==self.count:return bank
        if wanted!=self.count+1:raise ValueError('Online refresh skipped a global counter')
        number,jobs=self.batches.popleft()
        if number!=self.count:raise ValueError('Online generation order changed')
        tick=time.perf_counter();updates,records=self._collect(jobs);wait=time.perf_counter()-tick
        slots=(number*self.size+np.arange(self.size))%self.capacity
        tick=time.perf_counter();bank=self.replace(bank,slots,updates)
        for value in bank.values():value.block_until_ready()
        update=time.perf_counter()-tick
        self.draw_ids[slots]=np.asarray([record['draw_id'] for record in records]);self.count=wanted
        self.refresh_wait_seconds+=wait;self.update_seconds+=update
        with (self.run/'online_inputs.jsonl').open('a') as stream:
            for record in records:stream.write(json.dumps(record,allow_nan=False)+'\n')
        record=dict(step=step,batch=number,slots=slots.tolist(),draw_ids=self.draw_ids[slots].tolist(),
            wait_seconds=wait,replace_seconds=update,total_distinct_new_inputs=self.count*self.size)
        with (self.run/'online_refresh.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
        self._submit()
        return bank

    def save_state(self,step):
        if refresh_count(step,self.c)!=self.count:raise ValueError('Checkpoint input counters not synchronized')
        path=self.run/f'online_pool_{step:06d}.json'
        with path.open('x') as stream:
            stream.write(json.dumps(dict(step=step,origin_step=self.c.operator.online.origin_step,
                refresh_count=self.count,draw_ids=self.draw_ids.tolist(),
                identity='Negative -1-slot identifies initial bank input; nonnegative identity uses stateless online geometry draw'),
                indent=2,allow_nan=False)+'\n')
        return path

    def summary(self):
        return dict(kind='asynchronous_rolling_geometry_pool',capacity=self.capacity,
            refresh_count=self.count,distinct_generated_inputs=self.count*self.size,
            refresh_wait_seconds=self.refresh_wait_seconds,pool_replace_seconds=self.update_seconds,
            pde_solves=0,fem_solves=0,reference_labels_in_step=False,
            geometry_order='Stateless per-draw rejection sampling, fixed global refresh schedule, FIFO consumption',
            bounded_prefetch_inputs=self.c.operator.online.prefetch_batches*self.size)

    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=True)
