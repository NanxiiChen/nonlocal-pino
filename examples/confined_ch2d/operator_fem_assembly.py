"""CPU P1 weak CH input assembly; no reference trajectories or JAX backend."""
import numpy as np
from scipy import sparse, linalg

from examples.confined_ch2d.operator_problem import fixed_initial


def validate_fem(c):
    spec = c.operator.fem
    if set(spec) != {'kind', 'max_points', 'max_neighbors', 'reaction', 'evaluation'}:
        raise ValueError('FEM requires explicit topology capacities, reaction and evaluation rules')
    if (spec.kind != 'p1_lumped' or spec.reaction != 'nodal'
            or spec.evaluation != 'neural_phase_p1_chemical'
            or c.operator.metric != 'linear_stable_resolvent'
            or 'formulation' in c.operator or 'spatial' in c.sample or 'residual_rows' in c.sample
            or 'phase' in c or 'initial_rhs_condition' in c.operator or 'augmentation' in c.operator):
        raise ValueError('FEM control requires original full-field mass-projected CH with explicit nodal weak rule')
    if any(type(n) is not int or n < 3 for n in (spec.max_points, spec.max_neighbors)):
        raise ValueError('Positive explicit FEM capacities required')


def assemble(vertices, cells):
    vertices, cells = np.asarray(vertices, dtype=float), np.asarray(cells)
    if (vertices.ndim != 2 or vertices.shape[1] != 2 or not np.isfinite(vertices).all()
            or cells.ndim != 2 or cells.shape[1] != 3 or not len(cells)
            or not np.issubdtype(cells.dtype, np.integer) or cells.min() < 0 or cells.max() >= len(vertices)):
        raise ValueError('P1 requires finite 2D vertices and valid integer triangles')
    x = vertices[cells]
    affine = np.concatenate((np.ones(x.shape[:2]+(1,)), x), axis=2)
    area = np.abs(np.linalg.det(affine))/2
    if np.any(area <= 0) or not np.isfinite(area).all():
        raise ValueError('Degenerate/nonfinite FEM triangle')
    gradients = np.swapaxes(np.linalg.inv(affine)[:, 1:, :], 1, 2)
    element = area[:, None, None]*(gradients @ np.swapaxes(gradients, 1, 2))
    rows = np.repeat(cells, 3, axis=1).ravel()
    cols = np.tile(cells, (1, 3)).ravel()
    stiffness = sparse.coo_matrix((element.ravel(), (rows, cols)), shape=(len(vertices),)*2).tocsr()
    stiffness.eliminate_zeros()
    mass = np.bincount(cells.ravel(), weights=np.repeat(area/3, 3), minlength=len(vertices))
    if np.any(mass <= 0) or not np.isfinite(stiffness.data).all():
        raise ValueError('Unreferenced FEM vertex or nonfinite stiffness')
    np.testing.assert_allclose(stiffness @ np.ones(len(vertices)), 0., atol=1e-11, rtol=0)
    return mass, stiffness, x.mean(axis=1), area


def stable_metric(mass, stiffness, initial, c):
    root = np.sqrt(mass)
    symmetric = stiffness.toarray()/root[:, None]/root[None, :]
    tolerance = 100*np.finfo(float).eps*len(root)*max(1., np.linalg.norm(symmetric, ord=np.inf))
    if np.max(np.abs(symmetric-symmetric.T)) > tolerance:
        raise ValueError('FEM weak operator loses weighted self-adjointness')
    eigenvalues, eigenvectors = linalg.eigh(symmetric, driver='evd', check_finite=False)
    if eigenvalues[0] < -tolerance:
        raise ValueError('FEM stiffness is not positive semidefinite')
    p = c.physical
    mean = mass @ initial/mass.sum()
    rates = p.mobility*eigenvalues*(p.epsilon**2*eigenvalues+p.beta*(3*mean**2-1))
    weights = 1/(1+p.terminal_time*np.maximum(rates, 0.))
    metric = ((eigenvectors*weights) @ eigenvectors.T)*root[None, :]/root[:, None]
    constant_error = float(np.max(np.abs(metric @ np.ones(len(root))-1)))
    if not np.isfinite(metric).all() or constant_error > 1e-8:
        raise ValueError('Nonfinite FEM resolvent or failed constant mode')
    return metric, dict(minimum_eigenvalue=float(eigenvalues[0]),
        maximum_eigenvalue=float(eigenvalues[-1]), constant_error=constant_error,
        minimum_metric_weight=float(weights.min()), initial_nodal_mean=float(mean))


def build_input(vertices, cells, c):
    """Return explicitly padded FEM arrays alongside unchanged encoder inputs."""
    validate_fem(c)
    vertices, cells = np.asarray(vertices), np.asarray(cells)
    mass, stiffness, centroids, areas = assemble(vertices, cells)
    n, nmax, kmax = len(mass), c.operator.fem.max_points, c.operator.fem.max_neighbors
    ncmax = c.operator.online.max_points
    if n > nmax or len(cells) > ncmax:
        raise ValueError(f'FEM mesh exceeds declared capacities {n}/{nmax}, {len(cells)}/{ncmax}')
    neighbors = np.repeat(np.arange(nmax, dtype=np.int32)[:, None], kmax, axis=1)
    coefficients = np.zeros((nmax, kmax), dtype=float)
    positive_off_diagonals = 0
    maximum_degree = 0
    for row in range(n):
        start, stop = stiffness.indptr[row:row+2]
        ids, values = stiffness.indices[start:stop], stiffness.data[start:stop]
        keep = ids != row; ids, values = ids[keep], values[keep]
        if len(ids) > kmax:
            raise ValueError(f'FEM degree {len(ids)} exceeds explicit capacity {kmax}; no truncation')
        neighbors[row, :len(ids)] = ids
        coefficients[row, :len(ids)] = -values/mass[row]
        positive_off_diagonals += int(np.count_nonzero(values > 0))
        maximum_degree = max(maximum_degree, len(ids))
    initial = fixed_initial(vertices, c, xp=np)
    metric, report = stable_metric(mass, stiffness, initial, c)
    dtype = np.dtype(c.train.compute_dtype)
    output = dict(fe_points=np.pad(vertices, ((0, nmax-n), (0, 0))).astype(dtype),
        fe_mass=np.pad(mass, (0, nmax-n)).astype(dtype),
        fe_initial=np.pad(initial, (0, nmax-n)).astype(dtype),
        fe_neighbors=neighbors, fe_coefficients=coefficients.astype(dtype),
        fe_metric=np.pad(metric, ((0, nmax-n), (0, nmax-n))).astype(dtype),
        fe_cells=np.pad(cells, ((0, ncmax-len(cells)), (0, 0))).astype(np.int32))
    report.update(vertices=n, cells=len(cells), maximum_degree=maximum_degree,
        positive_off_diagonals_retained=positive_off_diagonals, area=float(mass.sum()))
    return output, report


def input_job(arguments):
    from ml_collections import ConfigDict
    vertices, cells, config = arguments
    return build_input(vertices, cells, ConfigDict(config))
