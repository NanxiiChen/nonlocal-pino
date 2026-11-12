"""Volume-correct local groups for the conservative-potential CH experiment.

Each group is a cell of a fixed Cartesian partition, clipped by the actual
geometry. Optional wall/bulk strata change allocation, not the PDE measure.
No labels or neural predictions enter this sampler.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np

from nonlocal_pinn.training.spatial_sampling import polyline_band, stratified_indices


def boundary_segments(data, c):
    points = np.asarray(data['condition'])[:, :2]
    counts = (c.geometry.boundary_outer, c.geometry.boundary_per_hole,
              c.geometry.boundary_per_hole)
    if sum(counts) != len(points):
        raise ValueError('Expected the ordered outer boundary and two hole contours')
    pieces = np.split(points, np.cumsum(counts)[:-1])
    return np.concatenate([np.stack((p, np.roll(p, -1, axis=0)), axis=1) for p in pieces])


class VolumeGroups:
    """Immutable partitions; all sampling randomness is supplied by the caller."""
    def __init__(self, data, c, *, cell_width, wall_band):
        points = np.asarray(data['points'])
        self.volume = np.asarray(data['volumes'])
        if (points.shape != (len(self.volume), 2) or not np.isfinite(points).all()
                or not np.isfinite(self.volume).all() or np.any(self.volume < 0)
                or self.volume.sum() <= 0 or not np.isfinite(cell_width) or cell_width <= 0):
            raise ValueError('Finite physical coordinates, positive total volume and cell width required')
        physical = np.flatnonzero(self.volume > 0)
        wall = polyline_band(points, boundary_segments(data, c), wall_band)
        origin = np.asarray(c.geometry.center)-np.asarray(c.geometry.axes)
        cells = np.floor((points-origin)/cell_width).astype(np.int64)
        self.partitions = []
        # The first partition is unsplit. The other two preserve actual masses.
        for chosen in (physical, physical[wall[physical]], physical[~wall[physical]]):
            _, inverse = np.unique(cells[chosen], axis=0, return_inverse=True)
            groups = [chosen[inverse == i] for i in range(inverse.max()+1)] if len(chosen) else []
            groups = [g[np.lexsort((points[g, 1], points[g, 0]))] for g in groups]
            masses = np.asarray([self.volume[g].sum() for g in groups])
            self.partitions.append((groups, masses))

    def draw(self, *, core_count, group_size, wall_groups, rng):
        if (any(type(x) is not int for x in (core_count, group_size, wall_groups))
                or min(core_count, group_size) < 1 or core_count % group_size
                or not 0 <= wall_groups < core_count//group_size):
            raise ValueError('Explicit integral group sizes and a positive bulk allocation required')
        groups_count = core_count//group_size
        allocations = ((0, groups_count),) if wall_groups == 0 else (
            (1, wall_groups), (2, groups_count-wall_groups))
        ids, weights = [], []
        for partition, count in allocations:
            groups, masses = self.partitions[partition]
            if not len(masses) or masses.sum() <= 0:
                raise ValueError('Requested volume stratum is empty; refusing implicit reallocation')
            for index in stratified_indices(masses, count, rng):
                group = groups[index]
                rows = group[stratified_indices(self.volume[group], group_size, rng)]
                ids.append(rows)
                weights.extend([masses.sum()/self.volume.sum()/count/group_size]*group_size)
        return np.concatenate(ids).astype(np.int32), np.asarray(weights, dtype=self.volume.dtype)


def validate_spatial(c):
    s = c.sample.spatial
    required = {'mode','core_count','group_size','cell_width','wall_groups','wall_band',
                'inner_capacity','query_capacity','pools','refresh_every','workers'}
    if set(s) != required or s.mode != 'rolling_groups':
        raise ValueError(f'Explicit rolling_groups settings required: {sorted(required)}')
    for name in required-{'mode','cell_width','wall_band'}:
        if type(s[name]) is not int or s[name] < (0 if name == 'wall_groups' else 1):
            raise ValueError(f'Invalid integral spatial.{name}')
    if (not np.isfinite([s.cell_width,s.wall_band]).all() or min(s.cell_width,s.wall_band) <= 0
            or s.core_count % s.group_size or not 0 <= s.wall_groups < s.core_count//s.group_size
            or s.inner_capacity > s.query_capacity):
        raise ValueError('Invalid groups, wall allocation or complete support capacities')


class RollingSpatialPool:
    """Bounded host-side pools; background work never owns a neural/GPU state.

    Each geometry slot retains at most one epoch's set of spatial pools. The
    next deterministic condition batch is prefetched during the current GPU
    update. Replacing a geometry invalidates its old immutable payloads.
    """
    def __init__(self,c,bank,run):
        import jax
        validate_spatial(c)
        self.c,self.run = c,Path(run)
        raw = jax.device_get(bank)
        self.data = [{key:value[i] for key,value in raw.items()} for i in range(c.operator.train_conditions)]
        self.versions = np.zeros(len(self.data),dtype=np.int64)
        self.cache = {}
        self.partitions = {}
        self.executor = ThreadPoolExecutor(max_workers=c.sample.spatial.workers)
        self.wait_seconds = 0.
        self.selected = []
        self.closed = False

    def install(self,slots,updates):
        """Called with the exact CPU arrays used to replace the GPU geometry."""
        for offset,slot in enumerate(slots):
            slot=int(slot)
            self.data[slot]={key:value[offset] for key,value in updates.items()}
            self.versions[slot]+=1
            if slot in self.cache:
                for job in self.cache.pop(slot)[1].values():job.cancel()
            if slot in self.partitions:self.partitions.pop(slot).cancel()

    def _partition(self,data):
        s=self.c.sample.spatial
        return VolumeGroups(data,self.c,cell_width=s.cell_width,wall_band=s.wall_band)

    def _build_one(self,slot,version,epoch,pool,data,partition):
        from examples.confined_ch2d.operator_potential import compact_payload
        s=self.c.sample.spatial
        groups=partition.result()
        rng=np.random.default_rng(np.random.SeedSequence([self.c.train.seed,58013,slot,version,epoch,pool]))
        core,weights=groups.draw(core_count=s.core_count,group_size=s.group_size,wall_groups=s.wall_groups,rng=rng)
        payload,report=compact_payload(data,core,weights,inner_capacity=s.inner_capacity,query_capacity=s.query_capacity)
        return payload,report,core

    def _choices(self,ids,step):
        rng=np.random.default_rng(np.random.SeedSequence([self.c.train.seed,58014,step]))
        return rng.integers(self.c.sample.spatial.pools,size=len(ids))

    def prefetch(self,ids,step):
        if self.closed:raise RuntimeError('Spatial pool is closed')
        epoch=(step-1)//self.c.sample.spatial.refresh_every
        for slot,pool in zip(ids,self._choices(ids,step),strict=True):
            slot,pool=int(slot),int(pool)
            identity=(int(self.versions[slot]),epoch)
            if slot not in self.partitions:
                self.partitions[slot]=self.executor.submit(self._partition,self.data[slot])
            if slot not in self.cache or self.cache[slot][0]!=identity:
                if slot in self.cache:
                    for job in self.cache.pop(slot)[1].values():job.cancel()
                self.cache[slot]=(identity,{})
            jobs=self.cache[slot][1]
            if pool not in jobs:
                jobs[pool]=self.executor.submit(self._build_one,slot,*identity,pool,
                    self.data[slot],self.partitions[slot])

    def batch(self,ids,step):
        import jax
        import jax.numpy as jnp
        self.prefetch(ids,step)
        choices=self._choices(ids,step)
        batches=[];core_ids=[];reports=[]
        tick=time.perf_counter()
        for slot,pool in zip(ids,choices,strict=True):
            payload,report,core=self.cache[int(slot)][1][int(pool)].result()
            batches.append(payload);core_ids.append(core)
            reports.append(dict(slot=int(slot),version=int(self.versions[slot]),pool=int(pool),**report))
        self.wait_seconds+=time.perf_counter()-tick
        batch={key:jnp.asarray(np.stack([item[key] for item in batches])) for key in batches[0]}
        jax.block_until_ready(batch)
        self.selected.extend(reports)
        with (self.run/'spatial_inputs.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=step,epoch=(step-1)//self.c.sample.spatial.refresh_every,
                samples=reports),allow_nan=False)+'\n')
        return batch,np.stack(core_ids)

    def summary(self):
        return dict(settings=self.c.sample.spatial.to_dict(),wait_seconds=self.wait_seconds,
            selected_count=len(self.selected),encoder_context='Complete original boundary and domain tokens',
            decoder_queries='Complete two-hop potential support; no global mean or dense metric',
            construction='Only selected pool sets, prefetched; partitions reused until geometry changes',
            median_query=float(np.median([r['query'] for r in self.selected])) if self.selected else None,
            maximum_query=max(r['query'] for r in self.selected) if self.selected else None,
            geometry_installations=int(self.versions.sum()),maximum_cache_entries=len(self.data))

    def close(self):
        self.closed=True
        self.executor.shutdown(wait=True,cancel_futures=True)
