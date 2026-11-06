"""Visible, conservative regional diffusion around impermeable particles.

The positive matrix represents ``D u = sum_j V_j K_ij (u_i-u_j)``. In 2D,
``K=120/(pi*delta**4)*(1-r/delta)**2`` has second moment 2 in each coordinate,
so its full-horizon limit is ``-Laplacian``. Symmetric segment visibility and
physical nodal volumes retain weighted symmetry, nonnegative energy and zero
exchange across a closed particle wall. This is a regional nonlocal boundary
model, not exact pointwise local Neumann at a finite horizon.

Visibility uses the SAME polygon as the conforming FEM mesh. Every interval
between segment/polygon crossings is checked; testing endpoints or just the
bond midpoint would miss thin or nonconvex obstructions. Boundary tangency is
permitted. The polygon approximation is separately refined with the mesh.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from examples.particle_corrosion2d.geometry import boundary


def _cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def _settings(c):
    delta = float(c.nl.delta)
    tolerance = float(c.nl.visibility_tolerance)
    count = c.nl.visibility_points
    if not np.isfinite([delta, tolerance]).all() or min(delta, tolerance) <= 0:
        raise ValueError('Nonlocal horizon and visibility tolerance must be positive and finite')
    if (isinstance(count, bool) or not isinstance(count, (int, np.integer))
            or count < 32 or count != c.mesh.boundary_points):
        raise ValueError('Visibility and FEM must use the same explicit polygon count >=32')
    if tolerance >= delta*1e-3:
        raise ValueError('Visibility tolerance must be much smaller than the nonlocal horizon')
    return delta, tolerance, count


def _polygon_interior(points, polygon, tolerance):
    """Strict polygon interior by ray parity, excluding a tolerance-wide wall."""
    points = np.asarray(points, dtype=float)
    result = np.zeros(len(points), dtype=bool)
    start, end = polygon, np.roll(polygon, -1, axis=0)
    edge = end-start
    squared_lengths = np.sum(edge**2, axis=1)
    if np.any(squared_lengths <= 0):
        raise ValueError('Particle polygon has coincident consecutive vertices')
    for offset in range(0, len(points), 512):
        p = points[offset:offset+512]
        q = p[:, None, :]-start
        projection = np.clip(np.sum(q*edge, axis=-1)/squared_lengths, 0, 1)
        distance2 = np.sum((q-projection[..., None]*edge)**2, axis=-1)
        is_wall = np.min(distance2, axis=1) <= tolerance**2
        crossing_y = (start[None, :, 1] > p[:, None, 1]) != (
            end[None, :, 1] > p[:, None, 1])
        ratio = np.zeros(crossing_y.shape)
        np.divide(p[:, None, 1]-start[None, :, 1], edge[None, :, 1],
                  out=ratio, where=edge[None, :, 1] != 0)
        crossing_x = start[None, :, 0]+ratio*edge[None, :, 0]
        parity = np.count_nonzero(crossing_y & (p[:, None, 0] < crossing_x), axis=1) % 2
        result[offset:offset+len(p)] = (parity != 0) & ~is_wall
    return result


def _polygons(record, count):
    if record['schema'] != 'smooth_radial_particles_v1':
        raise ValueError('Unknown particle geometry schema')
    angles = np.arange(count)*(2*np.pi/count)
    return [boundary(p, angles)[0] for p in record['particles']]


def _blocked_segments(start, end, polygon, tolerance):
    """Check all open subsegments between polygon-edge intersection points."""
    result = np.zeros(len(start), dtype=bool)
    edge_start, edge_end = polygon, np.roll(polygon, -1, axis=0)
    edge = edge_end-edge_start
    edge_lengths = np.linalg.norm(edge, axis=1)
    for offset in range(0, len(start), 512):
        a, b = start[offset:offset+512], end[offset:offset+512]
        direction = b-a
        lengths = np.linalg.norm(direction, axis=1)
        if np.any(lengths == 0):
            raise ValueError('Visibility received a zero-length segment')
        relative = edge_start[None, :, :]-a[:, None, :]
        denominator = _cross(direction[:, None, :], edge[None, :, :])
        nonparallel = np.abs(denominator) > (
            32*np.finfo(float).eps*lengths[:, None]*edge_lengths[None, :])
        t, u = np.full(denominator.shape, np.inf), np.full(denominator.shape, np.inf)
        np.divide(_cross(relative, edge[None, :, :]), denominator, out=t, where=nonparallel)
        np.divide(_cross(relative, direction[:, None, :]), denominator, out=u, where=nonparallel)
        t_tolerance, u_tolerance = tolerance/lengths[:, None], tolerance/edge_lengths[None, :]
        intersects = (nonparallel & (t >= -t_tolerance) & (t <= 1+t_tolerance)
                      & (u >= -u_tolerance) & (u <= 1+u_tolerance))
        cuts = np.where(intersects, np.clip(t, 0, 1), np.inf)
        cuts = np.sort(np.column_stack((np.zeros(len(a)), cuts, np.ones(len(a)))), axis=1)
        left, right = cuts[:, :-1], cuts[:, 1:]
        # Avoid inf-inf invalid arithmetic in the unused tail of each cut list.
        finite = np.isfinite(right)
        widths = np.zeros(left.shape)
        np.subtract(right, left, out=widths, where=finite)
        active = finite & (widths > tolerance/lengths[:, None])
        rows, columns = np.nonzero(active)
        middle = (left[rows, columns]+right[rows, columns])/2
        probes = a[rows]+middle[:, None]*direction[rows]
        inside = _polygon_interior(probes, polygon, tolerance)
        blocked = np.zeros(len(a), dtype=bool)
        np.logical_or.at(blocked, rows, inside)
        result[offset:offset+len(a)] = blocked
    return result


def visible_pairs(c, record, points, pairs):
    """Symmetric visibility against the explicit shared FEM particle polygons."""
    _, tolerance, count = _settings(c)
    points, pairs = np.asarray(points, dtype=float), np.asarray(pairs)
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or pairs.ndim != 2 or pairs.shape[1] != 2
            or not np.issubdtype(pairs.dtype, np.integer)
            or np.any(pairs < 0) or np.any(pairs >= len(points))):
        raise ValueError('Visibility requires finite points(N,2) and valid integer pairs(E,2)')
    result = np.ones(len(pairs), dtype=bool)
    start, end = points[pairs[:, 0]], points[pairs[:, 1]]
    if np.any(np.linalg.norm(end-start, axis=1) == 0):
        raise ValueError('Visibility received duplicate points or self-pairs')
    lower, upper = np.minimum(start, end), np.maximum(start, end)
    for polygon in _polygons(record, count):
        candidate = result & np.all(upper >= polygon.min(axis=0)-tolerance, axis=1) & np.all(
            lower <= polygon.max(axis=0)+tolerance, axis=1)
        ids = np.flatnonzero(candidate)
        result[ids] = ~_blocked_segments(start[ids], end[ids], polygon, tolerance)
    return result


def build_operator(c, record, points, volumes):
    """Assemble positive negative-Laplacian CSR using physical nodal measures."""
    delta, tolerance, count = _settings(c)
    points, volumes = np.asarray(points, dtype=float), np.asarray(volumes, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
            or volumes.shape != (len(points),) or not np.isfinite(points).all()
            or not np.isfinite(volumes).all() or np.any(volumes <= 0)):
        raise ValueError('Require finite points(N,2) and strictly positive physical volumes(N,)')
    if (np.any(points < -tolerance)
            or np.any(points > np.array([c.geometry.length, c.geometry.height])+tolerance)):
        raise ValueError('Operator points must lie in the outer rectangle')
    for polygon in _polygons(record, count):
        if _polygon_interior(points, polygon, tolerance).any():
            raise ValueError('Operator points lie inside an impermeable particle polygon')
    pairs = cKDTree(points).query_pairs(delta, output_type='ndarray')
    if not len(pairs):
        raise ValueError('No nonlocal pairs; refine the spatial grid or increase the horizon')
    radii = np.linalg.norm(points[pairs[:, 0]]-points[pairs[:, 1]], axis=1)
    if np.any(radii == 0):
        raise ValueError('Operator points contain duplicate coordinates')
    pairs = pairs[radii < delta]
    pairs = pairs[visible_pairs(c, record, points, pairs)]
    if not len(pairs):
        raise ValueError('No visible nonlocal pairs remain')
    i, j = pairs.T
    radii = np.linalg.norm(points[i]-points[j], axis=1)
    kernel = 120/(np.pi*delta**4)*(1-radii/delta)**2
    exchange = sparse.coo_matrix((np.r_[kernel*volumes[j], kernel*volumes[i]],
        (np.r_[i, j], np.r_[j, i])), shape=(len(points), len(points))).tocsr()
    if np.any(np.diff(exchange.indptr) == 0):
        raise ValueError('Nonlocal graph has isolated points; refine the grid or increase the horizon')
    components = connected_components(exchange, directed=False, return_labels=False)
    if components != 1:
        raise ValueError(f'Connected physical domain produced {components} nonlocal components')
    matrix = sparse.diags(np.asarray(exchange.sum(axis=1)).ravel())-exchange
    matrix = matrix.tocsr()
    matrix.sort_indices()
    return matrix


def operator_diagnostics(matrix, volumes):
    """Small scalar invariants; no dense eigensolve or accelerator allocation."""
    matrix = sparse.csr_matrix(matrix)
    volumes = np.asarray(volumes, dtype=float)
    if (matrix.shape != (len(volumes), len(volumes)) or np.any(volumes <= 0)
            or not np.isfinite(volumes).all() or not np.isfinite(matrix.data).all()):
        raise ValueError('Diagnostic matrix and positive physical volumes must agree')
    weighted = sparse.diags(volumes)@matrix
    asymmetry = weighted-weighted.T
    off = matrix-sparse.diags(matrix.diagonal())
    n = len(volumes)
    return dict(points=n, undirected_pairs=int(off.nnz//2),
        maximum_row_sum_error=float(np.max(np.abs(matrix@np.ones(n)))),
        maximum_weighted_symmetry_error=float(np.max(np.abs(asymmetry.data))) if asymmetry.nnz else 0.,
        maximum_weighted_conservation_error=float(np.max(np.abs(volumes@matrix))),
        minimum_diagonal=float(matrix.diagonal().min()),
        maximum_offdiagonal=float(off.data.max()) if off.nnz else 0.,
        components=int(connected_components(off, directed=False, return_labels=False)))
