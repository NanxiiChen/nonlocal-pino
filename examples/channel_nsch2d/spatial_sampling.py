"""Area-weighted complete core/halo rows for the random-wall integral recipe."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import time

import numpy as np

NAMESPACE = 730249
ROW_WEIGHTS = ('gradient_x', 'gradient_y', 'laplacian')
GRAPH_KEYS = (*ROW_WEIGHTS, 'points', 'neighbors', 'boundary_neighbors',
              'interior', 'volumes', 'transport_dx', 'transport_deta',
              'transport_weight_x', 'transport_weight_eta', 'transport_jacobian', 'transport_measure')


def validate_spatial(c):
    s = c.sample.spatial
    if (set(s) != {'mode', 'core_count', 'node_capacity', 'pools', 'refresh_every', 'workers'}
            or s.mode not in ('full', 'scatter', 'tiles')):
        raise ValueError('Explicit channel spatial mode, sizes, and refresh interval required')
    for name in ('core_count', 'node_capacity', 'pools', 'refresh_every', 'workers'):
        if type(s[name]) is not int or s[name] < 1:
            raise ValueError(f'Positive integer spatial.{name} required')
    if (c.nl.method != 'integral_moment' or 'time_derivative' not in c or 'online' not in c.operator
            or 'phase_transport' not in c or c.phase_transport.kind != 'integral_conservative'
            or any(k in c for k in ('causal', 'stagger', 'balance', 'phase_metric', 'droplet_metric',
                                    'residual_transform', 'stress_formulation', 'chemical_constraint',
                                    'time_discretization'))
            or any(k in c.train for k in ('refinement', 'continuation', 'initialization'))):
        raise ValueError('Channel spatial experiment requires the cold integral/time-JVP rolling recipe')
    if c.train.iterations < 11:
        raise ValueError('Spatial timing requires at least 11 optimizer updates')
    if (s.node_capacity > (c.sample.Nx+1)*(c.sample.Ny+1)
            or s.core_count > (c.sample.Nx-1)*(c.sample.Ny-1)):
        raise ValueError('Spatial capacity/count exceeds the declared physical grid')
    if s.mode == 'tiles' and (s.core_count % 64 or min(c.sample.Nx, c.sample.Ny) <= 8):
        raise ValueError('Tile mode requires multiples of 64 rows and at least 8 free nodes per axis')


def make_tiles(nx, ny):
    """Partition the interior into local blocks, merging short edge strips."""
    xs, ys = np.arange(1, nx), np.arange(1, ny)
    xb = np.split(xs, np.arange(8, len(xs)-7, 8))
    yb = np.split(ys, np.arange(8, len(ys)-7, 8))
    return [(x[:, None]*(ny+1)+y[None, :]).ravel() for x in xb for y in yb]


def compact_rows(data, core, weights, capacity):
    """Copy active rows exactly, with all their source nodes and BC supports."""
    neighbors = data['neighbors'][core]
    row_values = {k: data[k][core] for k in ROW_WEIGHTS}
    active = np.logical_or.reduce([v != 0 for v in row_values.values()])
    bn = data['boundary_neighbors']
    ba = (data['boundary_gradient_x'] != 0) | (data['boundary_gradient_y'] != 0)
    nodes = np.unique(np.concatenate((core, neighbors[active], bn[ba])))
    count = len(nodes)
    if count > capacity:
        raise ValueError(f'Spatial halo requires {count} nodes, capacity {capacity}; full stencil retained')
    padded = np.pad(nodes, (0, capacity-count), mode='edge')
    result = {k: v for k, v in data.items() if k not in GRAPH_KEYS}
    result.update(row_values, points=data['points'][padded], core_weights=weights.astype(data['points'].dtype),
                  core_indices=np.searchsorted(nodes, core).astype(np.int32),
                  neighbors=np.searchsorted(nodes, np.where(active, neighbors, nodes[0])).astype(np.int32),
                  boundary_neighbors=np.searchsorted(nodes, np.where(ba, bn, nodes[0])).astype(np.int32))
    return result, count


def draw_rows(data, c, rng, *, tiles):
    measure = data['transport_measure'].astype(c.train.compute_dtype).astype(np.float64)
    count = c.sample.spatial.core_count
    if c.sample.spatial.mode == 'scatter':
        free = np.flatnonzero(data['interior'])
        p = measure[free]
        return rng.choice(free, count, replace=True, p=p/p.sum()), np.full(count, 1/count)
    masses = np.array([measure[tile].sum() for tile in tiles])
    groups = count//64
    chosen = rng.choice(len(tiles), groups, replace=True, p=masses/masses.sum())
    rows, weights = [], []
    for index in chosen:
        tile = tiles[index]
        core = tile if len(tile) == 64 else rng.choice(tile, 64, replace=False)
        rows.append(core)
        # A row of a larger edge tile has inclusion probability 64/len(tile).
        weights.append(measure[core]*len(tile)/(masses[index]*64*groups))
    return np.concatenate(rows), np.concatenate(weights)


def _build(data, c, identity, revision, *, tiles):
    rng = np.random.default_rng(np.random.SeedSequence([c.train.seed, NAMESPACE,
        int(identity >= 0), identity if identity >= 0 else -identity-1, revision]))
    sets = []
    for _ in range(c.sample.spatial.pools):
        core, weights = draw_rows(data, c, rng, tiles=tiles)
        batch, count = compact_rows(data, core, weights, c.sample.spatial.node_capacity)
        sets.append((batch, core.astype(np.int32), count))
    return sets


class SpatialPool:
    """Bounded CPU prefetch for each geometry slot, keyed by spatial revision."""

    def __init__(self, c, initial):
        self.c = c
        self.initial = initial
        self.size = c.operator.train_conditions
        tiles = make_tiles(c.sample.Nx, c.sample.Ny) if c.sample.spatial.mode == 'tiles' else ()
        self.build = partial(_build, c=c, tiles=tiles)
        self.slots = {}
        self.wait_seconds = 0.

    def __enter__(self):
        self.workers = ThreadPoolExecutor(max_workers=self.c.sample.spatial.workers)
        try:
            indices = np.arange(self.size)
            self.install(indices, self.initial, -1-indices, 0)
        except BaseException:
            self.workers.shutdown(wait=True, cancel_futures=True)
            raise
        return self

    def install(self, slots, data, identities, step):
        revision = step//self.c.sample.spatial.refresh_every
        for i, (slot, identity) in enumerate(zip(slots, identities, strict=True)):
            if slot in self.slots:
                self.slots[slot][3].cancel()
            item = {k: v[i] for k, v in data.items()}
            job = self.workers.submit(self.build, item, identity=int(identity), revision=revision)
            self.slots[slot] = [item, int(identity), revision, job, None]

    def take(self, indices, step):
        revision = step//self.c.sample.spatial.refresh_every
        rng = np.random.default_rng(np.random.SeedSequence([self.c.train.seed, NAMESPACE, step]))
        choices = rng.integers(self.c.sample.spatial.pools, size=len(indices), dtype=np.int32)
        selected = []
        for slot, choice in zip(indices, choices, strict=True):
            state = self.slots[slot]
            item, identity, pending, job, cache = state
            if cache is None or cache[0] != revision:
                if pending != revision:
                    job.cancel()
                    job = self.workers.submit(self.build, item, identity=identity, revision=revision)
                start = time.perf_counter()
                cache = (revision, job.result())
                self.wait_seconds += time.perf_counter()-start
                job = self.workers.submit(self.build, item, identity=identity, revision=revision+1)
                state[2:] = [revision+1, job, cache]
            selected.append(cache[1][choice])
        payloads, cores, counts = zip(*selected, strict=True)
        batch = {k: np.stack([d[k] for d in payloads]) for k in payloads[0]}
        return batch, choices, np.stack(cores), np.asarray(counts, dtype=np.int32)

    def __exit__(self, *_):
        self.workers.shutdown(wait=True, cancel_futures=True)
