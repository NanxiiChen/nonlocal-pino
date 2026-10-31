"""Deterministic spatial pools and complete full-grid chunks on a fixed graph.

Workers build only NumPy metadata. Uniform cores use Generator.choice without
replacement, avoiding the legacy RandomState full-grid permutation per draw.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

import jax
import numpy as np


def validate_spatial(c):
    s = c.sample.spatial
    fields = {'mode', 'max_nodes', 'max_edges', 'workers', 'prefetch', 'full_chunk_core'}
    if set(s) != fields or s.mode not in ('full', 'fixed_pool', 'rolling_pool'):
        raise ValueError('Spatial settings require mode/max_nodes/max_edges/workers/prefetch/full_chunk_core')
    for name in fields - {'mode'}:
        if isinstance(s[name], bool) or not isinstance(s[name], int) or s[name] < 1:
            raise ValueError(f'spatial.{name} must be a positive integer')
    if (s.max_nodes < min(c.sample.Nx**3, 7*c.sample.K_core)
            or s.max_edges < 6*c.sample.K_core):
        raise ValueError('Spatial capacity does not cover proven six-neighbor degree bounds')
    if s.full_chunk_core > c.sample.Nx**3:
        raise ValueError('Full chunk must not exceed grid node count')


def core_subgraph(op, core):
    """Gather every outgoing edge of explicit cores, without an O(N) remap."""
    core = np.asarray(core, dtype=np.int32)
    if (core.ndim != 1 or not len(core) or len(np.unique(core)) != len(core)
            or np.any(core < 0) or np.any(core >= op._N)):
        raise ValueError('Core indices must be nonempty, unique and inside the graph')
    valid = op._nei_mask_np[core]
    neighbors, weights = op._nei_idx_np[core][valid], op._nei_w_np[core][valid]
    ids = np.unique(np.concatenate([core, neighbors]))
    row = np.searchsorted(ids, np.repeat(core, valid.sum(axis=1))).astype(np.int32)
    col = np.searchsorted(ids, neighbors).astype(np.int32)
    mask = np.zeros(len(ids), dtype=bool)
    mask[np.searchsorted(ids, core)] = True
    diagonal = np.bincount(row, weights=weights, minlength=len(ids)).astype(np.float32)
    return ids, (row, col, -weights, diagonal), mask


def pad_graph(raw, max_nodes, max_edges):
    ids, (row, col, values, diagonal), mask = raw
    if len(ids) > max_nodes or len(row) > max_edges:
        raise ValueError(f'Subgraph overflow nodes{len(ids)}/{max_nodes}, edges{len(row)}/{max_edges}; no truncation')
    if (not np.isfinite(values).all() or not np.isfinite(diagonal).all()
            or np.any(values > 0) or np.any(~mask[row])):
        raise ValueError('Invalid scalar core graph weights or source rows')
    return (np.pad(ids, (0, max_nodes-len(ids))).astype(np.int32),
            (np.pad(row, (0, max_edges-len(row))).astype(np.int32),
             np.pad(col, (0, max_edges-len(col))).astype(np.int32),
             np.pad(values, (0, max_edges-len(values))).astype(np.float32),
             np.pad(diagonal, (0, max_nodes-len(ids))).astype(np.float32)),
            np.pad(mask, (0, max_nodes-len(mask))))


@dataclass(frozen=True)
class Batch:
    arrays: tuple
    core: np.ndarray
    actual_nodes: int
    seconds: float


class SpatialSampler:
    def __init__(self, c, op, points):
        validate_spatial(c)
        self.c, self.op = c, op
        self.points = np.asarray(points, dtype=np.float32)
        self.settings, self.capacity = c.sample.spatial, c.sample.K_pool
        self.mode, self._step = self.settings.mode, 0
        start = time.perf_counter()
        if self.mode == 'full':
            size = self.settings.full_chunk_core
            raw = [core_subgraph(op, np.arange(i, min(i+size, op._N), dtype=np.int32))
                   for i in range(0, op._N, size)]
            # All chunks are constructed now; no future graph can exceed these
            # exact fixed-grid capacities. Rolling uses the proven degree bound.
            self.max_nodes = max(len(r[0]) for r in raw)
            self.max_edges = max(len(r[1][0]) for r in raw)
            self._host = [self._pack(r, 0.) for r in raw]
        else:
            self.max_nodes, self.max_edges = self.settings.max_nodes, self.settings.max_edges
            self._host = [self._build(i) for i in range(self.capacity)]
        self.initial_build_seconds = time.perf_counter()-start
        start = time.perf_counter()
        # Full chunks stay on the host; one bounded chunk is transferred at a
        # time. Never allocate B*Nt*all-N network activations on the device.
        self._device = [] if self.mode == 'full' else [self._transfer(b) for b in self._host]
        self.initial_transfer_seconds = time.perf_counter()-start
        self._executor, self._pending, self._next = None, deque(), self.capacity
        self._draws, self._trace = [], {}
        self._covered = np.zeros(op._N, dtype=bool)
        self.wait_seconds, self.transfer_seconds, self.take_seconds = [], [], []

    def _pack(self, raw, seconds):
        ids, sub, mask = pad_graph(raw, self.max_nodes, self.max_edges)
        return Batch((self.points[ids], sub, mask), raw[0][raw[2]], len(raw[0]), seconds)

    def _build(self, draw):
        start = time.perf_counter()
        core = np.random.default_rng(draw).choice(self.op._N, self.c.sample.K_core, replace=False)
        raw = core_subgraph(self.op, core)
        batch = self._pack(raw, 0.)
        return Batch(batch.arrays, batch.core, batch.actual_nodes, time.perf_counter()-start)

    @staticmethod
    def _transfer(batch):
        result = jax.device_put(batch.arrays)
        jax.block_until_ready(result)
        return result

    def __enter__(self):
        if self.mode == 'rolling_pool':
            self._executor = ThreadPoolExecutor(max_workers=self.settings.workers)
            for _ in range(self.settings.prefetch):
                self._submit()
        return self

    def _submit(self):
        self._pending.append((self._next, self._executor.submit(self._build, self._next)))
        self._next += 1

    def take(self, step):
        if self.mode == 'full' or step != self._step:
            raise ValueError('take requires a sampled mode and consecutive update counters')
        start = time.perf_counter()
        slot = step % self.capacity
        draw = slot if self.mode == 'fixed_pool' else step
        wait, transfer = 0., 0.
        if self.mode == 'rolling_pool' and step >= self.capacity:
            pending_draw, future = self._pending.popleft()
            if pending_draw != draw:
                raise ValueError('Background producer draw order mismatch')
            begin = time.perf_counter(); batch = future.result(); wait = time.perf_counter()-begin
            begin = time.perf_counter(); device = self._transfer(batch); transfer = time.perf_counter()-begin
            self._host[slot], self._device[slot] = batch, device
            self._submit()
        batch = self._host[slot]
        self._trace[draw] = (batch.core, batch.actual_nodes)
        self._covered[batch.core] = True
        self._draws.append(draw); self._step += 1
        self.wait_seconds.append(wait); self.transfer_seconds.append(transfer)
        self.take_seconds.append(time.perf_counter()-start)
        return self._device[slot]

    def full_chunks(self):
        if self.mode != 'full':
            raise ValueError('full_chunks requires full mode')
        for batch in self._host:
            yield self._transfer(batch), len(batch.core)/self.op._N

    def trace(self):
        if self.mode == 'full':
            return dict(chunk_offsets=np.r_[0, np.cumsum([len(b.core) for b in self._host])])
        draws = sorted(self._trace)
        return dict(draw_ids=np.asarray(self._draws), entry_draw_ids=np.asarray(draws),
            core_ids=np.stack([self._trace[d][0] for d in draws]),
            actual_nodes=np.asarray([self._trace[d][1] for d in draws]))

    def summary(self):
        return dict(mode=self.mode, chunks_per_update=len(self._host) if self.mode == 'full' else 1,
            entries_used=len(self._host) if self.mode == 'full' else len(self._trace),
            unique_cores=self.op._N if self.mode == 'full' else int(self._covered.sum()),
            grid_nodes=self.op._N, coverage=1. if self.mode == 'full' else float(self._covered.mean()),
            padded_nodes=self.max_nodes, padded_edges=self.max_edges,
            initial_build_seconds=self.initial_build_seconds,
            initial_transfer_seconds=self.initial_transfer_seconds,
            producer_wait_seconds=float(sum(self.wait_seconds)),
            replacement_transfer_seconds=float(sum(self.transfer_seconds)),
            core_rng='NumPy Generator(draw_id).choice(N,K,replace=False); independent of IC/time keys')

    def __exit__(self, kind, value, traceback):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        return False


def make_accumulator(c, optimizer):
    """One exact spatially accumulated gradient and one Adam update per step."""
    import equinox as eqx
    import jax.numpy as jnp
    from examples.icemelting3d.operator_losses import make_loss_fn
    loss = make_loss_fn(c)

    @eqx.filter_jit
    def accumulate(net, total, value, times, images, initial, points, sub, mask, weight):
        part, grad = eqx.filter_value_and_grad(loss)(net, times, images, initial, points, sub, mask)
        total = jax.tree.map(lambda a, b: a+weight*b, total, grad)
        return total, value+weight*part

    @eqx.filter_jit
    def apply(net, state, grad, value):
        finite = jnp.isfinite(value) & jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(grad) if eqx.is_array(a)]))
        updates, state = optimizer.update(grad, state, net)
        return eqx.apply_updates(net, updates), state, value, finite

    return accumulate, apply
