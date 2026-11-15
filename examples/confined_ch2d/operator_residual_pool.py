"""Rolling loss rows after the dense CH metric; all field queries stay global.

This is a sampling control, not a compact core/halo implementation. Workers
only read coordinates and volumes. They never copy a dense metric, use labels,
or own a network/JAX random stream.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from examples.confined_ch2d.operator_spatial import residual_rows


def validate_residual_rows(c):
    s = c.sample.residual_rows
    required = {'count', 'pools', 'refresh_every', 'workers'}
    if set(s) != required:
        raise ValueError(f'Residual row settings require exactly {sorted(required)}')
    if any(isinstance(s[k], bool) or not isinstance(s[k], int) or s[k] < 1 for k in required):
        raise ValueError('Residual row counts, pool count, refresh interval and workers must be positive integers')
    if (c.operator.action != 'train' or c.operator.metric != 'linear_stable_resolvent'
            or 'formulation' in c.operator or 'spatial' in c.sample
            or 'continuation' in c.train or 'augmentation' in c.operator):
        raise ValueError('Residual row control requires cold unaugmented direct-phase/stable-metric training')


def _known_arrays(bank):
    # Deliberately extract only these two arrays BEFORE any host conversion.
    points = np.array(bank['points'], copy=True)
    volumes = np.array(bank['volumes'], copy=True)
    if (volumes.ndim != 2 or points.shape != (*volumes.shape, 2)
            or not np.issubdtype(points.dtype, np.floating)
            or not np.issubdtype(volumes.dtype, np.floating)
            or not np.isfinite(points).all() or not np.isfinite(volumes).all()
            or np.any(volumes < 0) or not np.isfinite(np.sum(volumes, axis=1, dtype=np.float64)).all()
            or np.any(np.sum(volumes, axis=1, dtype=np.float64) <= 0)):
        raise ValueError('Residual row pool requires finite (batch,nodes,2) points and positive domain measures')
    points.setflags(write=False)
    volumes.setflags(write=False)
    return points, volumes


class RollingResidualRows:
    """Lazy bounded sets, refreshed by spatial epoch and geometry version.

    Each CDF-stratified set estimates the unchanged volume integral in
    expectation over its construction. Reusing a set does not make each
    adaptive training gradient conditionally unbiased; rolling refreshes
    change that finite quadrature. No wall-area measure is added implicitly.
    """

    def __init__(self, c, bank, run):
        validate_residual_rows(c)
        self.c, self.run = c, Path(run)
        if ('seed' not in c.train or isinstance(c.train.seed, bool)
                or not isinstance(c.train.seed, int) or c.train.seed < 0):
            raise ValueError('Residual pool requires the explicit nonnegative runtime seed')
        if not self.run.is_dir():
            raise FileNotFoundError(self.run)
        points, volumes = _known_arrays(bank)
        self.capacity = c.operator.train_conditions
        if len(points) < self.capacity:
            raise ValueError('Residual row pool has fewer domains than train_conditions')
        self.shape = points.shape[1:]
        self.dtypes = (points.dtype, volumes.dtype)
        self.data = [dict(points=points[i], volumes=volumes[i]) for i in range(self.capacity)]
        self.versions = np.zeros(self.capacity, dtype=np.int64)
        self.input_dir = self.run/'residual_row_geometry'
        self.input_dir.mkdir(exist_ok=False)
        self.installations = 0
        np.savez_compressed(self.input_dir/'initial.npz', points=points[:self.capacity],
            volumes=volumes[:self.capacity], slots=np.arange(self.capacity), versions=self.versions)
        self.cache = {}
        self.executor = ThreadPoolExecutor(max_workers=c.sample.residual_rows.workers)
        self.wait_seconds = 0.
        self.selected_count = 0
        self.distinct_row_sum = 0
        self.closed = False

    def _validate_ids(self, ids):
        ids = np.asarray(ids)
        if (self.closed or ids.ndim != 1 or not len(ids)
                or not np.issubdtype(ids.dtype, np.integer)
                or np.any(ids < 0) or np.any(ids >= self.capacity)):
            raise ValueError('Open residual pool and nonempty valid integer slot IDs required')
        return ids

    def install(self, slots, updates):
        """Invalidate old sets before the same physical arrays enter the GPU bank."""
        slots = self._validate_ids(slots)
        points, volumes = _known_arrays(updates)
        if (len(np.unique(slots)) != len(slots) or len(points) != len(slots)
                or points.shape[1:] != self.shape
                or (points.dtype, volumes.dtype) != self.dtypes):
            raise ValueError('Residual pool replacement must preserve slot uniqueness, shape and dtypes')
        # Only the small known-coordinate/measure arrays are retained for exact
        # independent row replay; no dense metric or evolving field is saved.
        np.savez_compressed(self.input_dir/f'replacement_{self.installations+1:06d}.npz',
            points=points, volumes=volumes, slots=slots, versions=self.versions[slots]+1)
        self.installations += 1
        for offset, slot in enumerate(slots):
            slot = int(slot)
            self.data[slot] = dict(points=points[offset], volumes=volumes[offset])
            self.versions[slot] += 1
            if slot in self.cache:
                for job in self.cache.pop(slot)[1].values():
                    job.cancel()

    def _choices(self, ids, step):
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 1:
            raise ValueError('Residual pool step must be a positive integer')
        rng = np.random.default_rng(np.random.SeedSequence([self.c.train.seed, 58114, int(step)]))
        return rng.integers(self.c.sample.residual_rows.pools, size=len(ids))

    def _build(self, slot, version, epoch, pool, data):
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.c.train.seed, 58113, slot, version, epoch, pool]))
        sampled = residual_rows(data, self.c.sample.residual_rows.count, rng)
        return {name: sampled[name] for name in ('residual_core_indices', 'residual_core_weights')}

    def prefetch(self, ids, step):
        ids = self._validate_ids(ids)
        choices = self._choices(ids, step)
        epoch = (step-1)//self.c.sample.residual_rows.refresh_every
        for slot, pool in zip(ids, choices, strict=True):
            slot, pool = int(slot), int(pool)
            identity = (int(self.versions[slot]), int(epoch))
            if slot not in self.cache or self.cache[slot][0] != identity:
                if slot in self.cache:
                    for job in self.cache.pop(slot)[1].values():
                        job.cancel()
                self.cache[slot] = (identity, {})
            jobs = self.cache[slot][1]
            if pool not in jobs:
                jobs[pool] = self.executor.submit(self._build, slot, *identity, pool, self.data[slot])

    def batch(self, ids, step):
        ids = self._validate_ids(ids)
        self.prefetch(ids, step)
        values, records = [], []
        tick = time.perf_counter()
        for slot, pool in zip(ids, self._choices(ids, step), strict=True):
            identity, jobs = self.cache[int(slot)]
            payload = jobs[int(pool)].result()
            values.append(payload)
            unique = len(np.unique(payload['residual_core_indices']))
            records.append(dict(slot=int(slot), version=identity[0], epoch=identity[1],
                                pool=int(pool), distinct_rows=unique))
        self.wait_seconds += time.perf_counter()-tick
        self.selected_count += len(records)
        self.distinct_row_sum += sum(r['distinct_rows'] for r in records)
        with (self.run/'residual_row_inputs.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=int(step), samples=records), allow_nan=False)+'\n')
        return {name: np.stack([value[name] for value in values]) for name in values[0]}

    def summary(self):
        return dict(settings=self.c.sample.residual_rows.to_dict(), wait_seconds=self.wait_seconds,
            selected_count=self.selected_count, geometry_installations=int(self.versions.sum()),
            maximum_cached_sets=self.capacity*self.c.sample.residual_rows.pools,
            mean_distinct_rows=self.distinct_row_sum/self.selected_count if self.selected_count else None,
            decoder_queries='Complete original physical grid; NOT compact core/halo',
            objective='Volume-weighted selected rows AFTER the unchanged dense stable metric',
            constraints='Original global mass projection and prescribed initial field',
            host_fields=['points', 'volumes'], input_snapshots=self.installations+1,
            pool_clock='Global step and per-slot replacement version')

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)
