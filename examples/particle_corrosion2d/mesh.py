"""Conforming polygon-hole meshes and independently assembled local P1 FEM.

Each ceramic contour is sampled at the explicit ``mesh.boundary_points``
uniform angles. Straight Gmsh edges between those analytic samples define the
actual discrete domain, shared with nonlocal visibility. Refining the polygon
and the triangles are distinct geometry/spatial consistency checks. No element
centroid filtering or point-only hole deletion is used.
"""
import numpy as np
from scipy import sparse

from .geometry import boundary, validate_geometry


def assemble_p1(points, triangles):
    """Return oriented triangles, lumped P1 mass, and weak Laplace stiffness.

    On a triangle T, K_ij=area(T)*grad(N_i).grad(N_j) and lumped M_ii receives
    area(T)/3. The Neumann weak form has no boundary term. A valid FEM stiffness
    need not have nonpositive off-diagonal entries on a general triangle mesh.
    """
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(triangles)
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or cells.ndim != 2 or cells.shape[1] != 3 or len(cells) == 0
            or not np.issubdtype(cells.dtype, np.integer)
            or cells.min() < 0 or cells.max() >= len(points)):
        raise ValueError('P1 assembly requires finite (N,2) points and valid integer (M,3) triangles')
    cells = cells.astype(np.int64, copy=True)
    vertices = points[cells]
    first, second = vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]
    determinants = first[:, 0]*second[:, 1]-first[:, 1]*second[:, 0]
    if np.any(determinants == 0) or not np.isfinite(determinants).all():
        raise ValueError('Degenerate or nonfinite P1 triangle')
    reversed_cells = determinants < 0
    cells[reversed_cells] = cells[reversed_cells][:, [0, 2, 1]]
    vertices = points[cells]
    area = 0.5*np.abs(determinants)
    gradients = np.stack((
        vertices[:, [1, 2, 0], 1]-vertices[:, [2, 0, 1], 1],
        vertices[:, [2, 0, 1], 0]-vertices[:, [1, 2, 0], 0]), axis=-1)/(2*area[:, None, None])
    element = area[:, None, None]*np.einsum('mik,mjk->mij', gradients, gradients)
    row = np.repeat(cells, 3, axis=1).ravel()
    column = np.tile(cells, (1, 3)).ravel()
    stiffness = sparse.coo_matrix((element.ravel(), (row, column)),
                                  shape=(len(points), len(points))).tocsr()
    stiffness.sum_duplicates()
    volumes = np.bincount(cells.ravel(), weights=np.repeat(area/3, 3), minlength=len(points))
    if np.any(volumes <= 0):
        raise ValueError('Every mesh point must belong to a positive-area triangle')
    scale = max(float(np.max(np.abs(stiffness.data))), 1.0)
    difference = stiffness-stiffness.T
    if ((difference.nnz and np.max(np.abs(difference.data)) > 1e-12*scale)
            or np.max(np.abs(stiffness@np.ones(len(points)))) > 1e-11*scale):
        raise ValueError('P1 stiffness must be symmetric and annihilate constants')
    return cells, volumes, stiffness


def _polygon_area(points):
    following = np.roll(points, -1, axis=0)
    return 0.5*float(np.sum(points[:, 0]*following[:, 1]-points[:, 1]*following[:, 0]))


def _boundary_data(points, triangles, edges, groups):
    """Determine outward normals from each boundary edge's incident triangle."""
    ordered = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    canonical = np.sort(ordered, axis=1)
    unique, inverse, counts = np.unique(canonical, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('Nonmanifold triangle edge')
    triangle_ids = np.tile(np.arange(len(triangles)), 3)
    owner = np.full(len(unique), -1, dtype=np.int64)
    owner[inverse] = triangle_ids
    edge_lookup = {tuple(edge): index for index, edge in enumerate(unique)}
    canonical_boundary = np.sort(edges, axis=1)
    if len(np.unique(canonical_boundary, axis=0)) != len(edges):
        raise ValueError('Duplicate meshed boundary edge')
    if set(map(tuple, canonical_boundary)) != set(map(tuple, unique[counts == 1])):
        raise ValueError('Gmsh line entities do not exactly cover the triangulated boundary')
    indices = np.array([edge_lookup[tuple(edge)] for edge in canonical_boundary])
    if np.any(counts[indices] != 1):
        raise ValueError('A declared boundary edge has more than one incident triangle')
    endpoints = points[edges]
    tangent = endpoints[:, 1]-endpoints[:, 0]
    lengths = np.linalg.norm(tangent, axis=1)
    if np.any(lengths <= 0):
        raise ValueError('Zero-length boundary edge')
    midpoints = endpoints.mean(axis=1)
    normals = np.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)/lengths[:, None]
    inward = points[triangles[owner[indices]]].mean(axis=1)-midpoints
    normals[np.sum(normals*inward, axis=1) > 0] *= -1
    return dict(boundary_edges=edges, boundary_midpoints=midpoints,
                boundary_normals=normals, boundary_lengths=lengths, boundary_groups=groups)


def build_mesh(c, record, spacing):
    """Mesh the rectangle minus the exact shared sampled ceramic polygons.

    Returns points(N,2), triangles(M,3), volumes(N,), stiffness(CSR), mouth(N,),
    hole_polygons(K,P,2), and edge quadrature/normals. Boundary group0 denotes
    closed exterior,1 the reservoir mouth, and2+k ceramic hole k. Corners have
    separate edge normals; no averaged corner normal is invented.
    """
    import gmsh

    validate_geometry(record, c)
    count = c.mesh.boundary_points
    if (isinstance(count, bool) or not isinstance(count, int) or count < 32
            or not np.isfinite(spacing) or spacing <= 0):
        raise ValueError('Mesh requires explicit boundary_points>=32 and positive finite spacing')
    if gmsh.isInitialized():
        raise RuntimeError('build_mesh owns a separate Gmsh session; do not call inside an existing session')
    theta = np.arange(count)*(2*np.pi/count)
    polygons = np.stack([boundary(particle, theta)[0] for particle in record['particles']])
    g = c.geometry
    outer = np.array([[0, 0], [g.pit_x-g.mouth_half_width, 0],
        [g.pit_x+g.mouth_half_width, 0], [g.length, 0],
        [g.length, g.height], [0, g.height]], dtype=np.float64)
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('General.NumThreads', 1)
        gmsh.option.setNumber('Mesh.MaxNumThreads1D', 1)
        gmsh.option.setNumber('Mesh.MaxNumThreads2D', 1)
        gmsh.option.setNumber('Mesh.Algorithm', 6)
        gmsh.option.setNumber('Mesh.ElementOrder', 1)
        gmsh.option.setNumber('Mesh.MeshSizeMin', spacing)
        gmsh.option.setNumber('Mesh.MeshSizeMax', spacing)
        # The finely sampled polygon fixes boundary geometry, not a global
        # interior mesh size. Disable propagation of its short edge lengths.
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
        gmsh.model.add('particle_corrosion_local_p1')
        geo = gmsh.model.geo
        curve_groups = []
        outer_points = [geo.addPoint(float(x), float(y), 0, spacing) for x, y in outer]
        outer_curves = []
        for i, start in enumerate(outer_points):
            curve = geo.addLine(start, outer_points[(i+1) % len(outer_points)])
            outer_curves.append(curve)
            curve_groups.append((curve, 1 if i == 1 else 0))
        loops = [geo.addCurveLoop(outer_curves)]
        hole_curves = []
        for index, polygon in enumerate(polygons):
            nodes = [geo.addPoint(float(x), float(y), 0, spacing) for x, y in polygon]
            curves = [geo.addLine(node, nodes[(i+1) % len(nodes)]) for i, node in enumerate(nodes)]
            loops.append(geo.addCurveLoop([-curve for curve in curves[::-1]]))
            hole_curves.extend(curves)
            curve_groups.extend((curve, index+2) for curve in curves)
        geo.addPlaneSurface(loops)
        geo.synchronize()
        # Keep every hole wall identical to the specified visibility polygon.
        for curve in hole_curves:
            gmsh.model.mesh.setTransfiniteCurve(curve, 2)
        gmsh.model.mesh.generate(2)
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tags)
        sorted_tags = np.asarray(tags)[order]
        points = np.asarray(coordinates).reshape(-1, 3)[order, :2].copy()

        def indices(node_tags):
            node_tags = np.asarray(node_tags)
            ids = np.searchsorted(sorted_tags, node_tags)
            if np.any(ids >= len(sorted_tags)) or not np.array_equal(sorted_tags[ids], node_tags):
                raise ValueError('Gmsh connectivity refers to an unknown node')
            return ids.astype(np.int64)

        types, _, node_lists = gmsh.model.mesh.getElements(2)
        if list(types) != [2]:
            raise ValueError('Independent reference requires exclusively linear triangular elements')
        triangles = indices(np.asarray(node_lists[0]).reshape(-1, 3))
        boundary_edges, boundary_groups = [], []
        for curve, group in curve_groups:
            types, _, node_lists = gmsh.model.mesh.getElements(1, curve)
            if list(types) != [1]:
                raise ValueError('Reference boundary requires exclusively linear edge elements')
            edges = indices(np.asarray(node_lists[0]).reshape(-1, 2))
            if group >= 2 and len(edges) != 1:
                raise ValueError('Ceramic polygon edge was unexpectedly subdivided')
            boundary_edges.append(edges)
            boundary_groups.append(np.full(len(edges), group, dtype=np.int32))
    finally:
        gmsh.finalize()
    triangles, volumes, stiffness = assemble_p1(points, triangles)
    edges = np.concatenate(boundary_edges)
    groups = np.concatenate(boundary_groups)
    boundary_data = _boundary_data(points, triangles, edges, groups)
    mouth = np.zeros(len(points), dtype=bool)
    mouth[np.unique(edges[groups == 1])] = True
    if not mouth.any() or mouth.all():
        raise ValueError('Mesh must resolve reservoir and free nodes separately')
    expected_area = g.length*g.height-sum(_polygon_area(polygon) for polygon in polygons)
    if not np.isclose(volumes.sum(), expected_area, atol=1e-11, rtol=1e-11):
        raise ValueError('Triangulated area does not equal rectangle minus shared ceramic polygons')
    return dict(points=points, triangles=triangles, volumes=volumes, stiffness=stiffness,
                mouth=mouth, hole_polygons=polygons, **boundary_data,
                mesh_info=dict(spacing=float(spacing), boundary_points=count,
                    nodes=len(points), triangles=len(triangles), area=float(volumes.sum()),
                    geometry='straight edges through analytic contour samples',
                    discretization='P1 FEM stiffness with row-sum-lumped P1 mass'))
