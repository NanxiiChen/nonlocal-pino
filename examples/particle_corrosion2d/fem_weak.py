"""Mass-lumped P1 weak diffusion, including signed element couplings.

D=M_lumped^-1 K approximates -Delta with natural homogeneous Neumann walls.
The existing nodal reaction, physical time JVP and mass-normalized L1 losses
are retained. No spatial AD, solved fields, coefficient clipping or extra
wall loss is introduced. Dirichlet columns remain in all selected free rows.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
from ml_collections import ConfigDict
from scipy import sparse
from scipy.spatial import cKDTree


def validate_fem(c):
    if (c.nl.method != 'fem_weak' or 'fem' not in c
            or c.fem.to_dict() != dict(kind='p1_lumped_nodal', residual_norm='mass_normalized_l1')
            or 'online' not in c.operator or 'full_grid' in c.sample
            or 'method_comparison' not in c.operator
            or c.operator.method_comparison != 'rolling_fem_v1'
            or 'phase' in c or 'boundary' in c
            or 'pde_reduction' in c.loss or 'transport_residual' in c.loss
            or 'time_form' in c.loss):
        raise ValueError('FEM requires explicit P1 lumped/nodal, original L1 fields and rolling_fem_v1')


def diffusion_matrix(mesh):
    mass = np.asarray(mesh['volumes'])
    stiffness = sparse.csr_matrix(mesh['stiffness'], copy=True)
    n = len(mass)
    if (mass.shape != (n,) or n < 2 or np.any(mass <= 0)
            or not np.isfinite(mass).all() or stiffness.shape != (n, n)
            or not np.isfinite(stiffness.data).all()):
        raise ValueError('Positive lumped mass and finite square P1 stiffness required')
    stiffness.sum_duplicates()
    stiffness.eliminate_zeros()
    scale = max(1., float(np.max(np.abs(stiffness.data))))
    symmetry = stiffness-stiffness.T
    if ((symmetry.nnz and abs(symmetry.data).max() > 1e-12*scale)
            or np.max(abs(stiffness@np.ones(n))) > 1e-11*scale
            or np.any(stiffness.diagonal() <= 0)):
        raise ValueError('FEM stiffness must be symmetric, constant-null and have positive diagonals')
    return (sparse.diags(1/mass)@stiffness).tocsr()


def signed_payload(c, mesh, matrix, core, weights):
    """Exactly retain every signed off-diagonal of each selected weak row."""
    points, mouth = np.asarray(mesh['points']), np.asarray(mesh['mouth'])
    core, weights = np.asarray(core), np.asarray(weights)
    if (core.shape != (c.sample.K_core,) or not np.issubdtype(core.dtype, np.integer)
            or np.any(core < 0) or np.any(core >= len(points)) or np.any(mouth[core])
            or weights.shape != core.shape or not np.isfinite(weights).all() or np.any(weights <= 0)
            or matrix.shape != (len(points), len(points))):
        raise ValueError('FEM cores must be free physical nodes with positive sampling weights')
    rows, coefficients = [], []
    for index in core:
        row = matrix.getrow(int(index))
        keep = (row.indices != index) & (row.data != 0)
        ids, values = row.indices[keep], -row.data[keep]
        if (not len(ids) or not np.isfinite(values).all()
                or not np.isclose(values.sum(), matrix[index, index], rtol=1e-11, atol=1e-11)):
            raise ValueError('Complete constant-null FEM row required; no diagonal repair')
        rows.append(ids)
        coefficients.append(values)
    halo = np.unique(np.concatenate((core, *rows)))
    width = max(map(len, rows))
    if len(halo) > c.operator.online.halo or width > c.operator.online.neighbors:
        raise ValueError(f'Complete FEM support overflow: {len(halo)}/{c.operator.online.halo}, '
                         f'{width}/{c.operator.online.neighbors}')
    neighbors = np.zeros((len(core), width), dtype=np.int32)
    values = np.zeros(neighbors.shape)
    for i, (row, coef) in enumerate(zip(rows, coefficients, strict=True)):
        neighbors[i, :len(row)] = np.searchsorted(halo, row)
        values[i, :len(row)] = coef
    return dict(graph_points=points[halo], core_indices=np.searchsorted(halo, core).astype(np.int32),
                neighbors=neighbors, coefficients=values, pde_weights=weights)


def _initial_slot(arguments):
    from .mesh import build_mesh
    from .inputs import pad_graph
    cdict, record, core_points, weights = arguments
    c = ConfigDict(cdict)
    mesh = build_mesh(c, record, c.mesh.spacing)
    matrix = diffusion_matrix(mesh)
    _, ids = cKDTree(mesh['points']).query(core_points)
    if not np.array_equal(mesh['points'][ids].astype(core_points.dtype), core_points):
        raise ValueError('Rebuilt FEM nodes do not exactly recover the supplied physical core coordinates')
    graphs = [pad_graph(signed_payload(c, mesh, matrix, core, weight),
                        c.operator.online.halo, c.operator.online.neighbors)
              for core, weight in zip(ids, weights, strict=True)]
    return {key: np.stack([g[key] for g in graphs]).astype(
        np.int32 if key in ('core_indices', 'neighbors') else np.float32) for key in graphs[0]}


def prepare_initial_bank(bank, c, manifest, workers):
    """Change only initial spatial operators, preserving actual cores/IC/BC."""
    from .online import FIXED_KEYS, GRAPH_KEYS, _worker_environment, _cpu_worker, validate_payload
    validate_fem(c)
    if type(workers) is not int or workers < 1:
        raise ValueError('Positive integer FEM construction worker count required')
    if set(bank) != set(FIXED_KEYS+GRAPH_KEYS) or len(manifest['geometries']) != c.operator.geometry_count:
        raise ValueError('Explicit matching initial geometry bank required')
    args = []
    for slot, record in enumerate(manifest['geometries']):
        points = np.take_along_axis(bank['graph_points'][slot], bank['core_indices'][slot][..., None], axis=1)
        args.append((c.to_dict(), record, points, bank['pde_weights'][slot]))
    with _worker_environment():
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_cpu_worker) as pool:
            graphs = list(pool.map(_initial_slot, args))
    result = {key: bank[key].copy() for key in FIXED_KEYS}
    result.update({key: np.stack([g[key] for g in graphs]) for key in GRAPH_KEYS})
    for slot in range(c.operator.geometry_count):
        validate_payload({key: value[slot] for key, value in result.items()}, c)
    return result
