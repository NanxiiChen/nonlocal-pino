"""Optional full/fixed/rolling spatial batches on one immutable scalar graph.

Only the main thread transfers arrays to JAX. The bounded background queue
contains NumPy inputs, with deterministic draw identities and complete halos.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

import jax
import numpy as np


def validate_spatial(c):
    s = c.sample.spatial
    if set(s) != {'mode', 'max_nodes', 'max_edges', 'workers', 'prefetch'}:
        raise ValueError('Spatial sampling requires mode/max_nodes/max_edges/workers/prefetch')
    if s.mode not in ('full', 'fixed_pool', 'rolling_pool'):
        raise ValueError('Spatial mode must be full, fixed_pool or rolling_pool')
    for name in ('max_nodes', 'max_edges', 'workers', 'prefetch'):
        value = s[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'spatial.{name} must be a positive integer')
    if c.nl.method != 'sparse' or 'online' not in c.operator or c.loss.use_causal:
        raise ValueError('Spatial controls require noncausal, online-condition sparse training')


def pad_subgraph(raw, max_nodes, max_edges):
    """Strict zero padding: no valid core, neighbor or edge may be truncated."""
    ids, (row, col, values, diagonal), core, count = raw
    ids, row, col = np.asarray(ids), np.asarray(row), np.asarray(col)
    values, diagonal, core = np.asarray(values), np.asarray(diagonal), np.asarray(core)
    if (ids.ndim != 1 or count != len(ids) or diagonal.shape != ids.shape
            or core.shape != ids.shape or core.dtype != np.bool_
            or row.ndim != 1 or col.shape != row.shape or values.shape != row.shape
            or not np.issubdtype(ids.dtype, np.integer)
            or not np.issubdtype(row.dtype, np.integer)
            or not np.issubdtype(col.dtype, np.integer)):
        raise ValueError('Invalid scalar subgraph array shapes/types')
    if (not len(ids) or not np.any(core) or np.any(ids < 0)
            or len(np.unique(ids)) != len(ids)
            or not np.isfinite(values).all() or not np.isfinite(diagonal).all()
            or np.any(values > 0) or np.any(diagonal < 0)
            or np.any(row < 0) or np.any(row >= count)
            or np.any(col < 0) or np.any(col >= count)
            or np.any(~core[row])):
        raise ValueError('Invalid core/halo graph values or edge indices')
    if count > max_nodes or len(row) > max_edges:
        raise ValueError(f'Subgraph overflow: nodes{count}/{max_nodes}, edges{len(row)}/{max_edges}; no truncation')
    nodes, edges = max_nodes-count, max_edges-len(row)
    return (np.pad(ids, (0, nodes)).astype(np.int32),
            (np.pad(row, (0, edges)).astype(np.int32),
             np.pad(col, (0, edges)).astype(np.int32),
             np.pad(values, (0, edges)).astype(np.float32),
             np.pad(diagonal, (0, nodes)).astype(np.float32)),
            np.pad(core, (0, nodes)), count)


@dataclass(frozen=True)
class SpatialBatch:
    arrays: tuple
    core_ids: np.ndarray
    core_mask: np.ndarray
    actual_nodes: int
    build_seconds: float


class SpatialSampler:
    def __init__(self, op, points, free, ic_ids, bc_ids, c):
        validate_spatial(c)
        self.settings = c.sample.spatial
        self.op = op
        self.points = np.asarray(points, dtype=np.float32)
        self.free = np.asarray(free, dtype=bool)
        self.ic_ids = np.asarray(ic_ids, dtype=np.int32)
        self.bc_ids = np.asarray(bc_ids, dtype=np.int32)
        self.core_count, self.capacity = c.sample.K_core, c.sample.K_pool
        n = len(points)
        if self.free.shape != (n,) or not np.any(self.free):
            raise ValueError('Free-node mask must match the fixed graph')
        if self.settings.mode != 'full':
            if (self.settings.max_nodes < min(n, self.core_count*(op._k_max+1))
                    or self.settings.max_edges < self.core_count*op._k_max):
                raise ValueError('Spatial capacities do not cover the proven one-hop degree bounds')
        curve_x = np.linspace(0, 1, c.operator.online.curve_points).astype(np.float32)
        self.nearest = np.searchsorted(curve_x, self.points[:, 0]).astype(np.int32)
        self._host = [self._build(i) for i in range(self.capacity if self.settings.mode != 'full' else 1)]
        self.initial_build_seconds = sum(b.build_seconds for b in self._host)
        transfer_start = time.perf_counter()
        self._device = [self._transfer(b) for b in self._host]
        self.initial_transfer_seconds = time.perf_counter()-transfer_start
        self._executor = None
        self._pending = deque()
        self._next_draw = self.capacity
        self._step = 0
        self._trace = {}
        self._draws = []
        self._covered = np.zeros(n, dtype=bool)
        self.wait_seconds, self.transfer_seconds, self.build_seconds = [], [], []

    def _build(self, draw):
        start = time.perf_counter()
        if self.settings.mode == 'full':
            ids = np.arange(len(self.points), dtype=np.int32)
            sub = (self.op._row_idx_np, self.op._col_idx_np,
                   self.op._vals_np, np.asarray(self.op._diag))
            core, actual = np.ones(len(ids), dtype=bool), len(ids)
        else:
            ids, sub, core, actual = pad_subgraph(self.op.sub_sample(draw, self.core_count),
                self.settings.max_nodes, self.settings.max_edges)
        mask = core & self.free[ids]
        selected = ids[mask]
        if not len(selected):
            raise ValueError('Spatial batch has no free PDE core nodes')
        width = len(self.points) if self.settings.mode == 'full' else self.core_count
        selected_mask = np.arange(width) < len(selected)
        selected = np.pad(selected, (0, width-len(selected)), mode='edge')
        node_ids = np.concatenate([ids, self.ic_ids, self.bc_ids]).astype(np.int32)
        arrays = (node_ids, self.points[node_ids], sub, mask, self.nearest[node_ids])
        return SpatialBatch(arrays, selected, selected_mask, actual, time.perf_counter()-start)

    @staticmethod
    def _transfer(batch):
        arrays = jax.device_put(batch.arrays)
        jax.block_until_ready(arrays)
        return arrays

    def __enter__(self):
        if self.settings.mode == 'rolling_pool':
            self._executor = ThreadPoolExecutor(max_workers=self.settings.workers)
            for _ in range(self.settings.prefetch):
                self._submit()
        return self

    def _submit(self):
        draw = self._next_draw
        self._pending.append((draw, self._executor.submit(self._build, draw)))
        self._next_draw += 1

    def take(self, step):
        if step != self._step:
            raise ValueError(f'Spatial draws require consecutive counters; expected{self._step}, got{step}')
        mode = self.settings.mode
        draw = 0 if mode == 'full' else step % self.capacity if mode == 'fixed_pool' else step
        slot = 0 if mode == 'full' else step % self.capacity
        wait, transfer = 0., 0.
        if mode == 'rolling_pool' and step >= self.capacity:
            if self._executor is None:
                raise RuntimeError('Rolling sampler must be entered as a context manager')
            pending_draw, future = self._pending.popleft()
            if pending_draw != draw:
                raise ValueError('Background producer draw order mismatch')
            start = time.perf_counter()
            batch = future.result()
            wait = time.perf_counter()-start
            start = time.perf_counter()
            device = self._transfer(batch)
            transfer = time.perf_counter()-start
            self._host[slot], self._device[slot] = batch, device
            self._submit()
        batch = self._host[slot]
        if draw not in self._trace:
            self._trace[draw] = (batch.core_ids, batch.core_mask, batch.actual_nodes)
        self._covered[batch.core_ids[batch.core_mask]] = True
        self._draws.append(draw)
        self.wait_seconds.append(wait)
        self.transfer_seconds.append(transfer)
        self.build_seconds.append(batch.build_seconds)
        self._step += 1
        return self._device[slot]

    def trace(self):
        ids = sorted(self._trace)
        return dict(draw_ids=np.asarray(self._draws), entry_draw_ids=np.asarray(ids),
            core_ids=np.stack([self._trace[i][0] for i in ids]),
            core_mask=np.stack([self._trace[i][1] for i in ids]),
            nonlocal_halo_counts=np.asarray([self._trace[i][2] for i in ids]))

    def summary(self):
        return dict(mode=self.settings.mode, entries_used=len(self._trace),
            unique_free_cores=int(self._covered.sum()), free_nodes=int(self.free.sum()),
            coverage=float(self._covered.sum()/self.free.sum()),
            padded_nodes=len(self._host[0].arrays[3]),
            padded_edges=len(self._host[0].arrays[2][0]),
            initial_build_seconds=self.initial_build_seconds,
            initial_transfer_seconds=self.initial_transfer_seconds,
            producer_wait_seconds=float(sum(self.wait_seconds)),
            replacement_transfer_seconds=float(sum(self.transfer_seconds)),
            ordering=('Complete fixed grid' if self.settings.mode == 'full' else
                      'Cyclic initial pool' if self.settings.mode == 'fixed_pool' else
                      'Initial pool followed by global step draw ID; worker completion order ignored'))

    def __exit__(self, kind, value, traceback):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        return False
