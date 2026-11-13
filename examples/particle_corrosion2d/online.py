"""Bounded CPU production of fresh geometry and spatial inputs, without solves.

Refresh identities are planned before submission. Workers may finish in any
order, but only the next scheduled event can replace the training pool.
"""
from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
from ml_collections import ConfigDict

from .geometry import GeometrySamplingExhausted, sample_geometry, validate_geometry
from .inputs import conditioning, graph_payload, initial_payload, make_core_sampler, pad_graph
from .problem import validate
from .full_grid import full_graph, is_full, validate_full


GEOMETRY_NAMESPACE = 271828
SPATIAL_NAMESPACE = 161803
FIXED_KEYS = ('domain_tokens', 'domain_weights', 'boundary_tokens', 'boundary_weights',
              'wall_points', 'wall_normals', 'wall_weights', 'mouth_points',
              'mouth_weights', 'ic_points', 'initial_ic', 'ic_weights')
GRAPH_KEYS = ('graph_points', 'core_indices', 'neighbors', 'coefficients', 'pde_weights')
_CACHE = OrderedDict()


def validate_online(c):
    if c.nl.method == 'fem_weak':
        from .fem_weak import validate_fem
        validate_fem(c)
    if is_full(c):
        validate_full(c)
    required = {'workers', 'geometry_every', 'geometry_size', 'spatial_every',
                'spatial_size', 'prefetch', 'halo', 'neighbors', 'worker_cache'}
    if set(c.operator.online) != required:
        raise ValueError(f'Online settings must explicitly declare {sorted(required)}')
    for name in required:
        value = c.operator.online[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'Positive integer operator.online.{name} required')
    o = c.operator.online
    if max(o.geometry_size, o.spatial_size) > c.operator.geometry_count:
        raise ValueError('A refresh cannot contain more slots than the geometry pool')
    if o.halo > c.nl.max_halo or o.neighbors > c.nl.max_neighbors:
        raise ValueError('Online capacities cannot exceed the explicit nonlocal construction bounds')
    if c.train.iterations <= max(o.geometry_every, o.spatial_every):
        raise ValueError('Online budget must include both geometry and spatial refreshes')
    if c.train.compute_dtype != 'float32':
        raise ValueError('The qualified rolling input path uses explicit float32 payloads')


def payload_shapes(c):
    """One complete slot, including every fixed-size spatial pool."""
    domain = int(np.prod(c.operator.domain_shape))
    walls = 5*c.operator.wall_points_per_side+c.particles.count*c.operator.wall_points_per_particle
    mouth, ic = c.operator.mouth_points, c.operator.ic_points
    pools, core = c.sample.K_pool, c.sample.K_core
    halo, neighbors = c.operator.online.halo, c.operator.online.neighbors
    if is_full(c):
        pools, core = 1, c.sample.full_grid.nodes
        halo, neighbors = c.sample.full_grid.nodes, c.sample.full_grid.neighbors
    return dict(domain_tokens=(domain, 3), domain_weights=(domain,),
        boundary_tokens=(walls+mouth, 5), boundary_weights=(walls+mouth,),
        wall_points=(walls, 2), wall_normals=(walls, 2), wall_weights=(walls,),
        mouth_points=(mouth, 2), mouth_weights=(mouth,), ic_points=(ic, 2),
        initial_ic=(ic, 2), ic_weights=(ic,), graph_points=(pools, halo, 2),
        core_indices=(pools, core), neighbors=(pools, core, neighbors),
        coefficients=(pools, core, neighbors), pde_weights=(pools, core))


def validate_payload(data, c):
    shapes = payload_shapes(c)
    if set(data) != set(shapes):
        raise ValueError('Rolling payload permits known geometry/IC/operator arrays only; no fields/labels')
    for name, shape in shapes.items():
        value = data[name]
        dtype = np.dtype('int32' if name in ('core_indices', 'neighbors') else 'float32')
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f'Invalid rolling {name}: expected {shape}/{dtype}, got {value.shape}/{value.dtype}')
    for name in ('domain_weights', 'boundary_weights', 'wall_weights', 'mouth_weights',
                 'ic_weights', 'pde_weights', 'coefficients'):
        if np.any(data[name] < 0) and not (name == 'coefficients' and c.nl.method == 'fem_weak'):
            raise ValueError(f'Negative rolling physical weight: {name}')
        if name != 'coefficients' and np.any(data[name].sum(axis=-1) <= 0):
            raise ValueError(f'Empty rolling physical measure: {name}')
    for name in ('core_indices', 'neighbors'):
        capacity = c.sample.full_grid.nodes if is_full(c) else c.operator.online.halo
        if np.any(data[name] < 0) or np.any(data[name] >= capacity):
            raise ValueError(f'Rolling {name} indexes outside the complete padded halo')
    row_sum = data['coefficients'].sum(axis=-1)
    if is_full(c):
        active = data['pde_weights'] > 0
        if np.any(row_sum[active] <= 0) or np.any(row_sum[~active] != 0):
            raise ValueError('Full-grid rows require complete active neighborhoods and zero inactive edges')
        if np.any(data['core_indices'][~active] != 0):
            raise ValueError('Inactive full-grid core indices must be zero')
        active_indices = data['core_indices'][active]
        if len(np.unique(active_indices)) != len(active_indices):
            raise ValueError('Full-grid PDE core nodes cannot be repeated')
    elif np.any(row_sum <= 0):
        raise ValueError('Every core row must retain a nonempty complete neighborhood')


def pad_initial_bank(bank, c):
    """Increase zero padding only; preserve every existing physical input bit."""
    validate_online(c)
    if is_full(c):
        raise ValueError('Full-grid initialization requires explicit prepare_initial_full_bank, not padding sampled cores')
    if set(bank) != set(FIXED_KEYS+GRAPH_KEYS):
        raise ValueError('Unexpected initial bank schema')
    count = c.operator.geometry_count
    if any(value.shape[0] != count for value in bank.values()):
        raise ValueError('Initial bank size differs from the explicit pool capacity')
    old_halo, old_neighbors = bank['graph_points'].shape[2], bank['neighbors'].shape[3]
    if old_halo > c.operator.online.halo or old_neighbors > c.operator.online.neighbors:
        raise ValueError('Online padding is smaller than the initial bank; truncation is forbidden')
    result = {name: value.copy() for name, value in bank.items()}
    points = np.repeat(bank['graph_points'][:, :, :1], c.operator.online.halo, axis=2)
    points[:, :, :old_halo] = bank['graph_points']
    result['graph_points'] = points
    shape = (*bank['neighbors'].shape[:3], c.operator.online.neighbors)
    for name in ('neighbors', 'coefficients'):
        value = np.zeros(shape, dtype=bank[name].dtype)
        value[..., :old_neighbors] = bank[name]
        result[name] = value
    for slot in range(count):
        validate_payload({name: value[slot] for name, value in result.items()}, c)
    return result


def geometry_identity(record):
    """Direct canonical parameter representation, invariant to particle order."""
    return (record['schema'], tuple(sorted(json.dumps(p, sort_keys=True, separators=(',', ':'))
                                          for p in record['particles'])))


def build_input(config, seed, geometry_id, revision, initial_record):
    """Rebuild a refreshed slot from explicit IDs, without solution trajectories.

    Initial negative IDs with revision zero denote the supplied historical bank;
    those unchanged arrays must be read from that bank rather than regenerated.
    The process-local LRU changes construction cost, never sample identities.
    """
    from .mesh import build_mesh
    from .operators import build_operator
    c = ConfigDict(config)
    validate_online(c)
    for name, value in (('seed', seed), ('geometry_id', geometry_id), ('revision', revision)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'Integer {name} required')
    if seed < 0 or revision < 0 or (geometry_id < 0 and revision == 0):
        raise ValueError('Invalid refreshed input identity; initial revision zero belongs to the saved bank')
    if (geometry_id < 0) != (initial_record is not None):
        raise ValueError('Negative geometry IDs require their exact initial record; fresh IDs forbid one')
    started = time.perf_counter()
    # Identify the geometry before prior rejection, so cached spatial refreshes
    # skip geometry sampling as well as meshing/assembly. No artifact hashes.
    key = json.dumps([c.geometry.to_dict(), c.particles.to_dict(), c.validation.to_dict(),
                      c.mesh.to_dict(), c.nl.to_dict(), seed, GEOMETRY_NAMESPACE,
                      geometry_id, initial_record], sort_keys=True, separators=(',', ':'))
    cache_hit = key in _CACHE
    if cache_hit:
        record, mesh, matrix, generation = _CACHE.pop(key)
    else:
        if geometry_id < 0:
            record = deepcopy(initial_record)
            validate_geometry(record, c)
            generation = dict(origin='provided_initial_record', sampling_calls=0,
                budget_restarts=0, layout_attempts=0, particle_proposals=0,
                accepted_particle_proposals=0, rejections={})
        else:
            rng = np.random.default_rng(np.random.SeedSequence([seed, GEOMETRY_NAMESPACE, geometry_id]))
            generation = dict(origin='same_rng_rejection_sampling', sampling_calls=0,
                budget_restarts=0, layout_attempts=0, particle_proposals=0,
                accepted_particle_proposals=0, rejections={})
            while True:
                diagnostic = {}
                exhausted = False
                try:
                    record = sample_geometry(c, rng, diagnostics=diagnostic)
                except GeometrySamplingExhausted as error:
                    diagnostic = error.diagnostics
                    exhausted = True
                generation['sampling_calls'] += 1
                for name in ('layout_attempts', 'particle_proposals', 'accepted_particle_proposals'):
                    generation[name] += diagnostic[name]
                for name, count in diagnostic['rejections'].items():
                    if name not in generation['rejections']:
                        generation['rejections'][name] = 0
                    generation['rejections'][name] += count
                if not exhausted:
                    break
                generation['budget_restarts'] += 1
                print(json.dumps(dict(warning='geometry_sampling_budget_exhausted_continuing_same_rng',
                    draw_id=geometry_id, seed=seed, geometry_namespace=GEOMETRY_NAMESPACE,
                    generation_diagnostics=generation), allow_nan=False), flush=True)
        mesh = build_mesh(c, record, c.mesh.spacing)
        if c.nl.method == 'fem_weak':
            from .fem_weak import diffusion_matrix
            matrix = diffusion_matrix(mesh)
        else:
            matrix = build_operator(c, record, mesh['points'], mesh['volumes'])
    _CACHE[key] = (record, mesh, matrix, generation)
    while len(_CACHE) > c.operator.online.worker_cache:
        _CACHE.popitem(last=False)
    namespace_id = -1-geometry_id if geometry_id < 0 else geometry_id
    spatial_seed = [seed, SPATIAL_NAMESPACE, int(geometry_id >= 0), namespace_id, revision]
    rng = np.random.default_rng(np.random.SeedSequence(spatial_seed))
    fixed = {**conditioning(c, record), **initial_payload(c, mesh, rng)}
    if is_full(c):
        graphs = [full_graph(c, mesh, matrix)]
        actual_halo = len(mesh['points'])
        actual_neighbors = int(np.max(np.sum(graphs[0]['coefficients'] > 0, axis=-1)))
    else:
        sample_cores = make_core_sampler(c, mesh)
        graphs = [graph_payload(c, mesh, matrix, rng, sample_cores=sample_cores) for _ in range(c.sample.K_pool)]
        actual_halo = max(len(graph['graph_points']) for graph in graphs)
        actual_neighbors = max(graph['neighbors'].shape[1] for graph in graphs)
        if actual_halo > c.operator.online.halo or actual_neighbors > c.operator.online.neighbors:
            raise ValueError(f'Rolling geometry {geometry_id}, revision {revision} exceeds fixed capacity: '
                             f'halo {actual_halo}/{c.operator.online.halo}, '
                             f'neighbors {actual_neighbors}/{c.operator.online.neighbors}; no truncation/resampling')
        graphs = [pad_graph(graph, c.operator.online.halo, c.operator.online.neighbors) for graph in graphs]
    data = {name: np.asarray(value, dtype=np.float32) for name, value in fixed.items()}
    data.update({name: np.stack([graph[name] for graph in graphs]).astype(
        np.int32 if name in ('core_indices', 'neighbors') else np.float32) for name in GRAPH_KEYS})
    validate_payload(data, c)
    metadata = dict(draw_id=geometry_id, spatial_version=revision, geometry=deepcopy(record),
        seed=seed, geometry_namespace=GEOMETRY_NAMESPACE, spatial_seed_words=spatial_seed,
        mesh_nodes=len(mesh['points']), actual_halo=actual_halo, actual_neighbors=actual_neighbors,
        cache_hit=cache_hit, worker_seconds=time.perf_counter()-started, pde_solves=0,
        fem_solves=0, evolved_fields_generated=False, generation_diagnostics=deepcopy(generation))
    if is_full(c):
        metadata.update(spatial_quadrature='full_mesh_lumped_volume', graph_pool_count=1,
                        active_pde_nodes=int(np.count_nonzero(graphs[0]['pde_weights'])))
    return data, metadata


def plan_events(c, capacity):
    """Cheap identity schedule only; no geometry/payload is generated here."""
    validate_online(c)
    if capacity != c.operator.geometry_count:
        raise ValueError('Schedule capacity must match the declared geometry pool')
    o = c.operator.online
    ids, versions = -1-np.arange(capacity, dtype=np.int64), np.zeros(capacity, dtype=np.int64)
    ticks = sorted(set(range(o.geometry_every, c.train.iterations, o.geometry_every)) |
                   set(range(o.spatial_every, c.train.iterations, o.spatial_every)))
    events = []
    geometry_cursor = spatial_cursor = next_draw = 0
    for index, tick in enumerate(ticks):
        tasks, geometry_slots = [], set()
        if tick % o.geometry_every == 0:
            for offset in range(o.geometry_size):
                slot = (geometry_cursor+offset) % capacity
                ids[slot], versions[slot] = next_draw, 0
                tasks.append(dict(slot=slot, draw_id=next_draw, spatial_version=0, kind='geometry'))
                geometry_slots.add(slot)
                next_draw += 1
            geometry_cursor += o.geometry_size
        skipped = []
        if tick % o.spatial_every == 0:
            for offset in range(o.spatial_size):
                slot = (spatial_cursor+offset) % capacity
                if slot in geometry_slots:
                    skipped.append(slot)
                    continue
                versions[slot] += 1
                tasks.append(dict(slot=slot, draw_id=int(ids[slot]),
                                  spatial_version=int(versions[slot]), kind='spatial'))
            spatial_cursor += o.spatial_size
        events.append(dict(index=index, completed_updates=tick, tasks=tasks,
                           spatial_merged_into_geometry=skipped))
    return events


_CPU_ENV = dict(JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu', CUDA_VISIBLE_DEVICES='',
                OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                NUMEXPR_NUM_THREADS='1')


@contextmanager
def _worker_environment():
    previous = {name: os.environ[name] if name in os.environ else None for name in _CPU_ENV}
    os.environ.update(_CPU_ENV)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _cpu_worker():
    os.environ.update(_CPU_ENV)
    _CACHE.clear()


class RollingPool:
    """FIFO background inputs with deterministic, independent refresh clocks."""
    def __init__(self, c, run, manifest, seed, forbidden_records):
        validate(c)
        validate_online(c)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError('Runtime seed must be a nonnegative integer')
        self.c, self.run, self.seed = c, Path(run), seed
        self.initial_records = manifest['geometries']
        self.capacity = c.operator.geometry_count
        if len(self.initial_records) != self.capacity:
            raise ValueError('Initial geometry records do not match the pool capacity')
        self.forbidden = {geometry_identity(record) for record in forbidden_records}
        if any(geometry_identity(record) in self.forbidden for record in self.initial_records):
            raise ValueError('Initial geometry bank overlaps a forbidden evaluation geometry')
        if (self.run/'online_refresh.jsonl').exists():
            raise ValueError('Online continuation requires full input/state recovery; existing journal refused')
        self.draw_ids = -1-np.arange(self.capacity, dtype=np.int64)
        self.spatial_versions = np.zeros(self.capacity, dtype=np.int64)
        self.records = list(self.initial_records)
        self.events = plan_events(c, self.capacity)
        self.jobs, self.executor = deque(), None
        self.submitted = self.installed = 0
        self.last_step = -1
        self.wait_seconds = self.shutdown_seconds = 0.
        self.worker_seconds = 0.
        self.cache_hits = self.installed_geometry = self.installed_spatial = 0
        self.finished_requests = self.submitted_requests = self.installed_requests = 0
        self.started = False

    def _submit(self):
        with _worker_environment():
            while len(self.jobs) < self.c.operator.online.prefetch and self.submitted < len(self.events):
                event = self.events[self.submitted]
                jobs = []
                for task in event['tasks']:
                    draw = task['draw_id']
                    initial = self.initial_records[-1-draw] if draw < 0 else None
                    jobs.append(self.executor.submit(build_input, self.c.to_dict(), self.seed,
                        draw, task['spatial_version'], initial))
                self.jobs.append((event, jobs))
                self.submitted_requests += len(jobs)
                self.submitted += 1

    def start(self):
        if self.started:
            raise ValueError('RollingPool is single-use; restart requires explicit recovery')
        if not self.run.is_dir():
            raise ValueError('Create the fresh run directory before starting the producer')
        self.started = True
        try:
            with _worker_environment():
                self.executor = ProcessPoolExecutor(max_workers=self.c.operator.online.workers,
                    mp_context=multiprocessing.get_context('spawn'), initializer=_cpu_worker)
                self._submit()
        except BaseException:
            self.close()
            raise
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def before_step(self, completed):
        if self.executor is None:
            raise ValueError('Start the rolling producer before consuming it')
        if isinstance(completed, bool) or not isinstance(completed, int) or completed < max(0, self.last_step):
            raise ValueError('Training updates must be monotone nonnegative integers')
        self.last_step = completed
        if self.installed == len(self.events):
            return None
        due = self.events[self.installed]['completed_updates']
        if completed > due:
            raise ValueError(f'Missed rolling refresh at completed update {due}')
        if completed < due:
            return None
        event, jobs = self.jobs[0]
        if event['index'] != self.installed:
            raise ValueError('Rolling FIFO event identity changed')
        started = time.perf_counter()
        values, records = [], []
        for task, job in zip(event['tasks'], jobs, strict=True):
            data, record = job.result()  # Worker errors propagate; no stale/alternate input fallback.
            if (record['draw_id'] != task['draw_id'] or
                    record['spatial_version'] != task['spatial_version'] or record['seed'] != self.seed):
                raise ValueError('Worker returned a different geometry/spatial identity')
            validate_payload(data, self.c)
            if geometry_identity(record['geometry']) in self.forbidden:
                raise ValueError('Fresh geometry duplicates an evaluation record; no silent resampling')
            if task['kind'] == 'spatial':
                slot = task['slot']
                if (self.draw_ids[slot] != task['draw_id'] or
                        self.spatial_versions[slot]+1 != task['spatial_version'] or
                        geometry_identity(self.records[slot]) != geometry_identity(record['geometry'])):
                    raise ValueError('Stale spatial work does not match the currently installed geometry')
            values.append(data)
            records.append(dict(task, **{key: value for key, value in record.items() if key not in task}))
        wait = time.perf_counter()-started
        slots = np.asarray([task['slot'] for task in event['tasks']], dtype=np.int32)
        updates = {name: np.stack([data[name] for data in values]) for name in values[0]}
        for task, record in zip(event['tasks'], records, strict=True):
            slot = task['slot']
            self.draw_ids[slot], self.spatial_versions[slot] = task['draw_id'], task['spatial_version']
            self.records[slot] = record['geometry']
            self.worker_seconds += record['worker_seconds']
            self.cache_hits += int(record['cache_hit'])
            self.installed_geometry += int(task['kind'] == 'geometry')
            self.installed_spatial += int(task['kind'] == 'spatial')
        self.wait_seconds += wait
        self.installed_requests += len(jobs)
        self.jobs.popleft()
        self.installed += 1
        journal = dict(event=event['index'], completed_updates=completed, slots=slots.tolist(),
            draw_ids=self.draw_ids[slots].tolist(), spatial_versions=self.spatial_versions[slots].tolist(),
            seed=self.seed, geometry_namespace=GEOMETRY_NAMESPACE, spatial_namespace=SPATIAL_NAMESPACE,
            spatial_merged_into_geometry=event['spatial_merged_into_geometry'],
            wait_seconds=wait, records=records)
        with (self.run/'online_refresh.jsonl').open('a') as stream:
            stream.write(json.dumps(journal, allow_nan=False)+'\n')
        self._submit()
        return slots, updates

    def save_state(self, completed):
        if completed != self.last_step+1:
            raise ValueError('Save pool state immediately after the last consumed training update')
        path = self.run/f'online_pool_{completed:06d}.json'
        next_event = self.events[self.installed] if self.installed < len(self.events) else None
        with path.open('x') as stream:
            json.dump(dict(completed_updates=completed, seed=self.seed,
                draw_ids=self.draw_ids.tolist(), spatial_versions=self.spatial_versions.tolist(),
                geometries=self.records, installed_events=self.installed, next_event=next_event,
                geometry_namespace=GEOMETRY_NAMESPACE, spatial_namespace=SPATIAL_NAMESPACE,
                recovery='Identity record only; exact continuation also requires the final payload bank and optimizer'),
                stream, indent=2, allow_nan=False)
            stream.write('\n')
        return path

    def summary(self):
        finished = self.finished_requests
        if self.executor is not None:
            finished += sum(job.done() and not job.cancelled() and job.exception() is None
                            for _, jobs in self.jobs for job in jobs)
        return dict(kind='background_geometry_and_spatial_pool', capacity=self.capacity,
            installed_events=self.installed, submitted_events=self.submitted,
            submitted_requests=self.submitted_requests, successfully_finished_requests=self.installed_requests+finished,
            installed_requests=self.installed_requests, installed_geometries=self.installed_geometry,
            installed_spatial_refreshes=self.installed_spatial,
            maximum_prefetched_events=self.c.operator.online.prefetch,
            maximum_prefetched_requests=self.c.operator.online.prefetch*(self.c.operator.online.geometry_size+self.c.operator.online.spatial_size),
            producer_wait_seconds=self.wait_seconds, installed_worker_seconds=self.worker_seconds,
            installed_cache_hits=self.cache_hits, shutdown_seconds=self.shutdown_seconds,
            pde_solves=0, fem_solves=0, evolved_training_fields=0,
            sampled_identities='Recorded by training per-step draws, not inferred from generated/installed counts')

    def close(self):
        if self.executor is not None:
            started = time.perf_counter()
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.finished_requests = sum(job.done() and not job.cancelled() and job.exception() is None
                                         for _, jobs in self.jobs for job in jobs)
            self.shutdown_seconds += time.perf_counter()-started
            self.executor = None
