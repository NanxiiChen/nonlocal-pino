"""Complete nonlocal mesh rows with physical-volume quadrature and padding.

Only the PDE spatial measure changes. Mouth nodes remain available as graph
neighbors with the network's predicted values; their PDE rows carry no weight.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing

import numpy as np
from ml_collections import ConfigDict
from scipy import sparse


def is_full(c):
    return 'full_grid' in c.sample


def validate_full(c):
    if not is_full(c) or set(c.sample.full_grid) != {'nodes', 'neighbors'}:
        raise ValueError('Full-grid mode requires explicit sample.full_grid={nodes,neighbors}')
    for name in ('nodes', 'neighbors'):
        value = c.sample.full_grid[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'Positive integer sample.full_grid.{name} required')
    if c.nl.method != 'nonlocal':
        raise ValueError('The full-grid control requires the same nonlocal spatial operator')
    if c.train.compute_dtype != 'float32':
        raise ValueError('Full-grid input finalization requires the qualified float32 path')


def full_graph(c, mesh, matrix):
    """One graph, without a pool axis; nodes/rows both use the explicit capacity.

    Every original mesh point occupies its original index. Free PDE row i uses
    core index i and its lumped volume. Mouth and padded rows contain zero edge
    coefficients, zero weights and core index zero. All negative off-diagonal
    entries of each free row are retained, including edges to mouth nodes.
    """
    validate_full(c)
    points, volumes, mouth = mesh['points'], mesh['volumes'], mesh['mouth']
    n = len(points)
    capacity, width = c.sample.full_grid.nodes, c.sample.full_grid.neighbors
    if (points.shape != (n, 2) or volumes.shape != (n,) or mouth.shape != (n,)
            or mouth.dtype != np.bool_ or n < 2 or not np.isfinite(points).all()
            or not np.isfinite(volumes).all() or np.any(volumes <= 0) or np.all(mouth)):
        raise ValueError('Full graph requires finite points, positive volumes and a nonempty boolean free-node mask')
    if n > capacity:
        raise ValueError(f'Full mesh has {n} nodes, exceeding explicit capacity {capacity}; truncation forbidden')
    if not sparse.issparse(matrix) or matrix.shape != (n, n):
        raise ValueError('Full graph requires the finite square native sparse operator')
    operator = matrix.tocsr(copy=True)
    operator.sum_duplicates()
    operator.sort_indices()
    if not np.isfinite(operator.data).all():
        raise ValueError('Full graph requires the finite square native sparse operator')
    rows = np.repeat(np.arange(n), np.diff(operator.indptr))
    columns = operator.indices
    off_diagonal = rows != columns
    if np.any(operator.data[off_diagonal] > 0) or np.any(operator.diagonal() < 0):
        raise ValueError('Full graph requires the positive negative-Laplacian sign convention')
    scale = max(1., float(np.max(np.abs(operator.diagonal()))))
    if np.max(np.abs(np.asarray(operator.sum(axis=1)).ravel())) > 1e-10*scale:
        raise ValueError('Full operator must annihilate constants before diagonal-free graph conversion')
    selected = off_diagonal & (operator.data < 0) & ~mouth[rows]
    edge_rows, edge_columns = rows[selected], columns[selected]
    coefficients = -operator.data[selected]
    degree = np.bincount(edge_rows, minlength=n)
    if np.any(degree[~mouth] == 0):
        raise ValueError('A free full-grid PDE row has no physical neighbors')
    if degree.max() > width:
        raise ValueError(f'Full graph needs {degree.max()} neighbors, exceeding {width}; truncation forbidden')
    positions = np.arange(len(edge_rows))-np.repeat(np.cumsum(degree)-degree, degree)
    graph_points = np.repeat(points[:1], capacity, axis=0).astype(np.float32)
    graph_points[:n] = points
    core_indices = np.zeros(capacity, dtype=np.int32)
    free = np.flatnonzero(~mouth)
    core_indices[free] = free
    neighbors = np.zeros((capacity, width), dtype=np.int32)
    weights = np.zeros((capacity, width), dtype=np.float32)
    neighbors[edge_rows, positions] = edge_columns
    weights[edge_rows, positions] = coefficients
    pde_weights = np.zeros(capacity, dtype=np.float32)
    pde_weights[free] = volumes[free]
    return dict(graph_points=graph_points, core_indices=core_indices, neighbors=neighbors,
                coefficients=weights, pde_weights=pde_weights)


def build_initial_full_slot(config, record):
    """CPU reconstruction of graph arrays only, with one explicit pool axis."""
    from .mesh import build_mesh
    from .operators import build_operator
    c = ConfigDict(config)
    validate_full(c)
    mesh = build_mesh(c, record, c.mesh.spacing)
    matrix = build_operator(c, record, mesh['points'], mesh['volumes'])
    return {name: value[None] for name, value in full_graph(c, mesh, matrix).items()}


def _initial_job(arguments):
    return build_initial_full_slot(*arguments)


def prepare_initial_full_bank(bank, c, manifest, workers):
    """Replace initial graph arrays while preserving all original IC/BC inputs.

    This explicit CPU operation never runs through pad_initial_bank and never
    resamples the supplied geometries or their historical fixed input arrays.
    """
    from .online import FIXED_KEYS, GRAPH_KEYS, _cpu_worker, _worker_environment, validate_payload
    validate_full(c)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('Full initial-bank construction requires positive integer workers')
    if set(bank) != set(FIXED_KEYS+GRAPH_KEYS):
        raise ValueError('Initial bank must contain only the known fixed and graph input arrays')
    records = manifest['geometries']
    if len(records) != c.operator.geometry_count or any(len(value) != len(records) for value in bank.values()):
        raise ValueError('Initial bank geometry count differs from the explicit pool capacity')
    expected = json.loads(c.to_json())
    for name in ('geometry', 'particles', 'physical', 'mesh'):
        if manifest['config'][name] != expected[name]:
            raise ValueError(f'Full initial-bank {name} differs from the supplied physical bank')
    for name in ('delta', 'visibility_points', 'visibility_tolerance'):
        if manifest['config']['nl'][name] != expected['nl'][name]:
            raise ValueError(f'Full initial-bank operator {name} differs')
    for name in FIXED_KEYS:
        if bank[name].dtype != np.float32 or not np.isfinite(bank[name]).all():
            raise ValueError(f'Invalid historical fixed inputs: {name}')
    arguments = [(c.to_dict(), record) for record in records]
    with _worker_environment():
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_cpu_worker) as pool:
            graphs = list(pool.map(_initial_job, arguments))
    result = {name: bank[name].copy() for name in FIXED_KEYS}
    result.update({name: np.stack([graph[name] for graph in graphs]) for name in GRAPH_KEYS})
    for slot in range(len(records)):
        validate_payload({name: value[slot] for name, value in result.items()}, c)
    return result
