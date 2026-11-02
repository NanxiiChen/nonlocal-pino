"""Bounded FIFO production of channel geometry/operator inputs, without solves."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import json
import multiprocessing
import os
from pathlib import Path
import time


from examples.channel_nsch2d.geometry import parameter_count,sample_parameters,coefficient_bounds

DRAW_NAMESPACE=314159


def validate_construction_screen(c,path):
    """Match a completed input-only prior screen to an explicit cold online run."""
    from examples.channel_nsch2d.problem import identity
    checked=json.loads(Path(path).read_text())
    required=((c.train.iterations-1)//c.operator.online.refresh_every+c.operator.online.prefetch_batches)*c.operator.online.refresh_size
    if (checked['status']!='passed' or checked['operator']!=c.nl.to_dict()
            or checked['grid']!=[c.sample.Nx,c.sample.Ny]
            or checked['physical_family']!=json.loads(json.dumps(identity(c)))
            or checked['online_draws']<required or checked['run_seed']!=c.train.seed
            or checked['initial_bank_seed']!=c.operator.dataset_seed
            or checked['initial_geometries']!=c.operator.train_conditions
            or checked['maximum_condition']>c.nl.condition_limit
            or checked['maximum_polynomial_error']>5e-9
            or any(checked[k]!=0 for k in ('pde_solves','fem_solves','stored_operator_arrays'))):
        raise ValueError('Input-prior conditioning proof does not cover the chosen online recipe')
    expected={('initial',i) for i in range(c.operator.train_conditions)}|{('online',i) for i in range(checked['online_draws'])}
    actual=[(row['kind'],row['index']) for row in checked['cases']]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('Input-prior screen has missing or duplicate actual construction records')
    result={k:v for k,v in checked.items() if k!='cases'}
    if 'derivatives' in c:
        result['additional_gradient_scope']=('Integral domain/boundary moments are checked in every screened construction'
            if c.nl.method=='integral_moment'else
            'Prior screen covers unchanged Laplacian/prior only; domain/boundary GMLS gradients are rank/reproduction checked in every actual input construction')
    return result


def validate_online(c):
    settings=c.operator.online
    required={'workers','refresh_size','refresh_every','prefetch_batches'}
    if set(settings)!=required:
        raise ValueError(f'Online settings must explicitly declare {sorted(required)}')
    for name in required:
        value=settings[name]
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(f'Positive integer online.{name} required')
    if c.operator.train_conditions<2 or c.operator.train_conditions%settings.refresh_size:
        raise ValueError('Online capacity>=2 must be divisible by refresh_size')
    if 'continuation' in c.train:
        raise ValueError('Online continuation requires full rolling-pool recovery, which is not implemented')
    if c.train.iterations<=settings.refresh_every:
        raise ValueError('Online training budget must include at least one actual pool replacement')


def sample_geometry(c,draw_id):
    """Stateless draws from the declared independent uniform wall prior."""
    import numpy as np
    if isinstance(draw_id,bool) or not isinstance(draw_id,int) or draw_id<0:
        raise ValueError('Online draw identity must be a nonnegative integer')
    rng=np.random.default_rng(np.random.SeedSequence([c.train.seed,DRAW_NAMESPACE,draw_id]))
    return sample_parameters(c,rng)


def validate_payload(data,c):
    """Fixed known-input schema; refuse labels, shape changes or invalid operators."""
    import numpy as np
    from examples.channel_nsch2d.problem import boundary_payload_shapes
    n=(c.sample.Nx+1)*(c.sample.Ny+1);k=c.nl.neighbors
    b=4*(c.boundary_sampling.points_per_side if 'boundary_sampling' in c else c.sample.boundary_points)
    shapes=dict(points=(n,2),volumes=(n,),neighbors=(n,k),laplacian=(n,k),
        interior=(n,),amplitudes=(parameter_count(c),),boundary=(b,5),boundary_weights=(b,),
        domain=(n,3),domain_weights=(n,))
    shapes.update(boundary_payload_shapes(c))
    if 'phase_transport' in c:
        from examples.channel_nsch2d.transport import transport_shapes
        shapes.update(transport_shapes(c))
    if 'derivatives' in c:
        from examples.channel_nsch2d.operators import gradient_shapes
        shapes.update(gradient_shapes(c))
    if set(data)!=set(shapes):
        raise ValueError('Online payload requires the ten known-input arrays and only explicitly configured transport arrays')
    for name,value in data.items():
        expected=np.dtype('int32' if name in ('neighbors','boundary_neighbors') else c.train.compute_dtype)
        if value.shape!=shapes[name] or value.dtype!=expected or not np.isfinite(value).all():
            raise ValueError(f'Invalid online {name}: shape={value.shape},dtype={value.dtype},expected={shapes[name]}/{expected}')
    if np.any(data['neighbors']<0) or np.any(data['neighbors']>=n):
        raise ValueError('Online neighbor index outside physical nodes')
    if 'derivatives' in c and (np.any(data['boundary_neighbors']<0) or np.any(data['boundary_neighbors']>=n)):
        raise ValueError('Online boundary neighbor index outside physical nodes')
    positive=('volumes','boundary_weights','domain_weights')
    if 'boundary_sampling' in c:positive+=('encoder_boundary_weights',)
    if any(np.any(data[name]<=0) for name in positive):
        raise ValueError('Online physical measures must be positive')
    if 'phase_transport' in c:
        if any(np.any(data[name]<=0) for name in ('transport_weight_x','transport_weight_eta','transport_jacobian','transport_measure')):
            raise ValueError('Online SBP measures and mapping Jacobian must be positive')
    if not np.all(np.isin(data['interior'],[0,1])) or not np.all(np.isin(data['boundary'][:,4],[0,1,2])):
        raise ValueError('Online interior masks and boundary types are invalid')
    if 'boundary_sampling' in c:
        from examples.channel_nsch2d.problem import boundary,boundary_loss_quadrature
        # Sampled amplitudes were rounded to training dtype; compare their
        # constructed features within the corresponding rounding tolerance.
        eps=np.finfo(data['points'].dtype).eps
        for names,fn in [(('encoder_boundary','encoder_boundary_weights'),boundary),
                         (('boundary','boundary_weights'),boundary_loss_quadrature)]:
            for name,expected in zip(names,fn(data['amplitudes'],c),strict=True):
                if not np.allclose(data[name],expected,atol=8*eps,rtol=8*eps):
                    raise ValueError(f'Online {name} disagrees with declared geometry/quadrature')
    np.testing.assert_array_equal(data['domain'][:,:2],data['points'])
    np.testing.assert_array_equal(data['domain_weights'],data['volumes'])
    bounds=np.asarray(coefficient_bounds(c),dtype=c.train.compute_dtype)
    if np.any(data['amplitudes']<bounds[0]) or np.any(data['amplitudes']>bounds[1]):
        raise ValueError('Online wall amplitudes outside declared prior')


def build_input(config,draw_id):
    """Canonical FP64 assembly followed by explicit casting; no JAX/FEM calls."""
    import numpy as np
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.data import geometry_data
    c=ConfigDict(config);amplitudes=sample_geometry(c,draw_id);begin=time.perf_counter()
    raw,operator=geometry_data(amplitudes,c)
    data={name:value.astype('int32' if name in ('neighbors','boundary_neighbors') else c.train.compute_dtype)
          for name,value in raw.items()}
    validate_payload(data,c)
    record=dict(draw_id=draw_id,amplitudes=amplitudes.tolist(),
        training_amplitudes=data['amplitudes'].tolist(),nodes=len(data['points']),
        seconds=time.perf_counter()-begin,operator=operator,pde_solves=0,fem_solves=0,
        construction_dtype='float64',training_dtype=c.train.compute_dtype)
    return data,record


def _cpu_worker():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['JAX_PLATFORMS']='cpu'
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'


@contextmanager
def _worker_environment():
    # Spawn inherits these before NumPy/main-module imports; restore the main
    # process environment immediately after the initial bounded submissions.
    keys=('CUDA_VISIBLE_DEVICES','JAX_PLATFORMS','OPENBLAS_NUM_THREADS',
          'OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
    previous={key:os.environ[key] if key in os.environ else None for key in keys}
    _cpu_worker()
    try:yield
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def replace_slots(bank,slots,updates):
    """Pure fixed-shape array update; the training owner chooses JIT/donation."""
    import jax
    return jax.tree.map(lambda old,new:old.at[slots].set(new),bank,updates)


def audit_training_inputs(run,c):
    """Reconstruct actual rolling identities and canonical final inputs on CPU."""
    import numpy as np
    validate_online(c);run=Path(run)
    with np.load(run/'known_inputs.npz',allow_pickle=False) as f:initial={k:f[k] for k in f.files}
    with np.load(run/'known_inputs_final.npz',allow_pickle=False) as f:final={k:f[k] for k in f.files}
    with np.load(run/'sampling.npz',allow_pickle=False) as f:sampled={k:f[k] for k in f.files}
    with np.load(run/'evaluation_inputs.npz',allow_pickle=False) as f:evaluation={k:f[k] for k in f.files}
    with np.load(run/'reserved_geometries.npz',allow_pickle=False) as f:reserved=f['amplitudes']
    with np.load(run/'prediction.npz',allow_pickle=False) as f:development=f['amplitudes']
    expected_reserved=development[None,:]
    if 'validation_manifest' in c.operator:
        from examples.channel_nsch2d.cohort import reserved_amplitudes
        expected_reserved=np.concatenate((expected_reserved,reserved_amplitudes(c)))
    np.testing.assert_array_equal(reserved,expected_reserved)
    validate_payload(evaluation,c)
    records=[json.loads(line) for line in (run/'online_inputs.jsonl').read_text().splitlines()]
    refresh=[json.loads(line) for line in (run/'online_refresh.jsonl').read_text().splitlines()]
    count=(c.train.iterations-1)//c.operator.online.refresh_every
    size=c.operator.online.refresh_size;capacity=c.operator.train_conditions
    if len(refresh)!=count or len(records)!=count*size:raise ValueError('Incomplete online refresh/input log')
    dtype=np.dtype(c.train.compute_dtype)
    amplitudes=initial['amplitudes'].astype(dtype).copy();ids=-1-np.arange(capacity,dtype=np.int64)
    if sampled['draw_ids'].shape!=sampled['indices'].shape or sampled['amplitudes'].shape!=(*sampled['indices'].shape,parameter_count(c)):
        raise ValueError('Sampled rolling geometry identities do not align with actual slot draws')
    if np.any(np.all(amplitudes[:,None,:]==reserved.astype(dtype)[None,:,:],axis=-1)):
        raise ValueError('Initial training bank contains the reserved evaluation geometry')
    event=0
    for step in range(c.train.iterations):
        if event<count and step==(event+1)*c.operator.online.refresh_every:
            row=refresh[event];slots=(event*size+np.arange(size))%capacity
            np.testing.assert_array_equal(row['slots'],slots)
            np.testing.assert_array_equal(row['draw_ids'],np.arange(event*size,(event+1)*size))
            if row['completed_updates']!=step or row['batch']!=event:raise ValueError('Refresh update or FIFO order changed')
            for slot,draw_id,record in zip(slots,row['draw_ids'],records[event*size:(event+1)*size],strict=True):
                draw=record['draw_id'];expected=sample_geometry(c,draw)
                if draw!=draw_id:raise ValueError('Input construction log differs from installed FIFO draw')
                np.testing.assert_array_equal(record['amplitudes'],expected)
                np.testing.assert_array_equal(record['training_amplitudes'],expected.astype(dtype))
                if record['pde_solves']!=0 or record['fem_solves']!=0:raise ValueError('Training inputs include evolved solves')
                if np.any(np.all(reserved.astype(dtype)==expected.astype(dtype),axis=1)):
                    raise ValueError('Online training draw duplicates the reserved evaluation geometry')
                ids[slot]=draw;amplitudes[slot]=expected.astype(dtype)
            event+=1
        selected=sampled['indices'][step]
        np.testing.assert_array_equal(sampled['draw_ids'][step],ids[selected])
        np.testing.assert_array_equal(sampled['amplitudes'][step],amplitudes[selected])
    state=json.loads((run/f'online_pool_{c.train.iterations:06d}.json').read_text())
    np.testing.assert_array_equal(state['draw_ids'],ids)
    if state['completed_updates']!=c.train.iterations or state['refresh_count']!=count:
        raise ValueError('Final pool counters disagree with actual training')
    for slot,draw in enumerate(ids):
        validate_payload({k:v[slot] for k,v in final.items()},c)
        expected=({k:v[slot].astype('int32' if k in ('neighbors','boundary_neighbors') else dtype) for k,v in initial.items()}
                  if draw<0 else build_input(c.to_dict(),int(draw))[0])
        for name in expected:np.testing.assert_array_equal(final[name][slot],expected[name])
    from examples.channel_nsch2d.data import geometry_data
    # Geometry construction uses the original double-precision amplitudes;
    # retain those in the prediction artifact rather than rebuilding from FP32.
    with np.load(run/'prediction.npz',allow_pickle=False) as f:reserved=f['amplitudes']
    raw,_=geometry_data(reserved,c)
    for name,value in raw.items():
        np.testing.assert_array_equal(evaluation[name],value.astype('int32' if name in ('neighbors','boundary_neighbors') else dtype))
    return dict(status='actual_rolling_inputs_reconstructed',refresh_count=count,
        installed_inputs=count*size,distinct_sampled_online_draws=int(np.unique(sampled['draw_ids'][sampled['draw_ids']>=0]).size),
        final_pool_inputs_rebuilt=capacity,evaluation_inputs_independent=True,
        evolved_training_labels=0,scope='Actual input identities and fixed-shape operators; not an accuracy or generalization claim')


class RollingGeometryPool:
    """CPU controller; timing cannot select geometries or change refresh order."""
    def __init__(self,c,run,forbidden_amplitudes):
        import numpy as np
        validate_online(c)
        self.c=c;self.run=Path(run);self.forbidden=np.asarray(forbidden_amplitudes,dtype=np.float64)
        if self.forbidden.ndim!=2 or self.forbidden.shape[1]!=parameter_count(c) or not np.isfinite(self.forbidden).all():
            raise ValueError('Forbidden geometry parameters must match the finite(N,parameter_count) family')
        self.capacity=c.operator.train_conditions;self.size=c.operator.online.refresh_size
        self.draw_ids=-1-np.arange(self.capacity,dtype=np.int64)
        self.count=0;self.last_step=-1;self.next_batch=0;self.jobs=deque()
        self.wait_seconds=0.;self.executor=None

    def _submit(self):
        while len(self.jobs)<self.c.operator.online.prefetch_batches:
            begin=self.next_batch*self.size
            jobs=[self.executor.submit(build_input,self.c.to_dict(),draw)
                  for draw in range(begin,begin+self.size)]
            self.jobs.append((self.next_batch,jobs));self.next_batch+=1

    def start(self):
        if self.executor is not None:raise ValueError('Online producer already started')
        try:
            with _worker_environment():
                self.executor=ProcessPoolExecutor(max_workers=self.c.operator.online.workers,
                    mp_context=multiprocessing.get_context('spawn'),initializer=_cpu_worker)
                self._submit()
        except BaseException:
            self.close()
            raise
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self,exc_type,exc_value,traceback):
        self.close()
        return False

    def before_step(self,completed):
        """Replace after every full refresh_every updates, before the next one."""
        import numpy as np
        if self.executor is None:raise ValueError('Start the producer before consuming inputs')
        if isinstance(completed,bool) or not isinstance(completed,int) or completed<0 or completed<self.last_step:
            raise ValueError('Online update counter must be a monotone nonnegative integer')
        due=(self.count+1)*self.c.operator.online.refresh_every
        if completed>due:raise ValueError('Missed scheduled online refresh')
        self.last_step=completed
        if completed<due:return None
        number,jobs=self.jobs.popleft()
        if number!=self.count:raise ValueError('Online FIFO batch order changed')
        begin=time.perf_counter();data=[];records=[]
        for offset,job in enumerate(jobs):
            value,record=job.result()
            if record['draw_id']!=number*self.size+offset:raise ValueError('Worker draw identity changed')
            validate_payload(value,self.c)
            if (np.any(np.all(self.forbidden==np.asarray(record['amplitudes']),axis=1)) or
                    np.any(np.all(self.forbidden.astype(value['amplitudes'].dtype)==value['amplitudes'],axis=1))):
                raise ValueError('Online draw duplicates a reserved geometry; prior is not silently resampled')
            data.append(value);records.append(record)
        wait=time.perf_counter()-begin;self.wait_seconds+=wait
        slots=(number*self.size+np.arange(self.size,dtype=np.int32))%self.capacity
        updates={name:np.stack([value[name] for value in data]) for name in data[0]}
        self.draw_ids[slots]=[record['draw_id'] for record in records];self.count+=1
        with (self.run/'online_inputs.jsonl').open('a') as stream:
            for record in records:stream.write(json.dumps(record,allow_nan=False)+'\n')
        with (self.run/'online_refresh.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(completed_updates=completed,batch=number,
                slots=slots.tolist(),draw_ids=self.draw_ids[slots].tolist(),wait_seconds=wait),allow_nan=False)+'\n')
        self._submit()
        return slots,updates

    def save_state(self,completed):
        if completed!=self.last_step+1:raise ValueError('Pool state must follow the last consumed training update')
        path=self.run/f'online_pool_{completed:06d}.json'
        with path.open('x') as stream:
            stream.write(json.dumps(dict(completed_updates=completed,refresh_count=self.count,
                draw_ids=self.draw_ids.tolist(),next_refresh=(self.count+1)*self.c.operator.online.refresh_every,
                identity='Negative -1-slot denotes initial bank; nonnegative IDs use stateless continuous geometry draws'),
                indent=2,allow_nan=False)+'\n')
        return path

    def summary(self):
        return dict(kind='asynchronous_rolling_geometry_pool',capacity=self.capacity,
            refresh_count=self.count,installed_inputs=self.count*self.size,
            submitted_inputs=self.next_batch*self.size,
            bounded_prefetch_inputs=self.c.operator.online.prefetch_batches*self.size,
            producer_wait_seconds=self.wait_seconds,pde_solves=0,fem_solves=0,
            draw_namespace=DRAW_NAMESPACE,reference_labels_in_step=False)

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True,cancel_futures=True);self.executor=None
