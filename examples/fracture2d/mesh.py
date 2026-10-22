"""Validated local-P1 geometry with explicitly selected damage discretization."""
import json
from pathlib import Path
import numpy as np
from pinn.operators import ExplicitVolumeSparseD
from examples.fracture2d import mesh_geometry as direct
from examples.fracture2d.local_mechanics import LocalMechanics
from examples.fracture2d.storage import atomic_json

def visibility_graph(coordinates, triangles, delta, tolerance):
    """Exact frozen polygon predicate with a conservative rectangle broad phase.

    A segment inside the polygon's enclosing rectangle can leave the polygon
    only by crossing a non-rectangle boundary edge. Such edges are enclosed
    by the obstruction box; every potentially crossing pair uses the original
    predicate. All input points are checked in the polygon closure first.
    """
    polygon = direct._g_boundary_polygon_from_triangles(coordinates, triangles)
    predicate = direct._g_UnionPolygonPredicate(polygon, tolerance)
    tol = predicate._effective_tolerance(delta)
    if any(predicate._location(point, tol) == 0 for point in coordinates):
        raise ValueError("visibility coordinates lie outside their mesh polygon")
    pairs = direct._g_all_strict_pairs(coordinates, delta)
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    start, end = polygon, np.roll(polygon, -1, axis=0)
    rectangle = np.zeros(len(polygon), bool)
    for axis in (0, 1):
        for bound in (low[axis], high[axis]):
            rectangle |= (abs(start[:, axis] - bound) <= tol) & (abs(end[:, axis] - bound) <= tol)
    flags = np.ones(len(pairs), bool)
    if not np.all(rectangle):
        internal = np.concatenate((start[~rectangle], end[~rectangle]))
        obstacle_low, obstacle_high = internal.min(axis=0) - tol, internal.max(axis=0) + tol
        segment_low = np.minimum(coordinates[pairs[:, 0]], coordinates[pairs[:, 1]])
        segment_high = np.maximum(coordinates[pairs[:, 0]], coordinates[pairs[:, 1]])
        risky = np.flatnonzero(np.all(segment_high >= obstacle_low, axis=1)
                               & np.all(segment_low <= obstacle_high, axis=1))
        for index in risky:
            i, j = pairs[index]
            flags[index] = predicate(coordinates[i], coordinates[j])
    candidate = direct._g_directed_edges_from_pairs(pairs)
    visible = direct._g_directed_edges_from_pairs(pairs[flags])
    removed = direct._g_directed_edges_from_pairs(pairs[~flags])
    direct._g_strict_graph_validation(coordinates, visible, delta, "split visible graph")
    return polygon, candidate, visible, removed

def read_cloud(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: archive[name] for name in
                ('coordinates', 'triangles', 'volumes', 'bottom', 'top', 'anchor_particle')}


def geometry_identity(c):
    identity = dict(cloud=str(c.data.cloud), delta_d=c.nl.delta_d,
                visibility_tolerance=c.nl.visibility_tolerance,
                grip_extent=c.geometry.grip_extent, h_grip=c.geometry.h_grip,
                E=c.physical.E, nu=c.physical.nu, k_res=c.physical.k_res,
                schema='p1-local-volume-grips-amor-lumped-v1')
    if 'boundary' in c:
        if (c.boundary.mode not in ('horizontal_clamped_edges', 'l_shaped_panel')
                or c.damage.discretization != 'fem' or c.geometry.grip_extent != 0):
            raise ValueError('edge clamps require explicit FEM damage and zero grip extent')
        identity.update(boundary=c.boundary.to_dict(), schema='p1-local-edge-clamps-amor-lumped-v1')
        if c.boundary.mode == 'l_shaped_panel':
            identity.update(width=c.geometry.width, height=c.geometry.height,
                            schema='p1-local-l-panel-amor-lumped-v1')
    return identity


def shear_boundary(cloud):
    """Top ux=U,uy=0; bottom ux=uy=0 on original material edges."""
    points = cloud['coordinates']
    for name in ('bottom', 'top'):
        if cloud[name].dtype != bool or cloud[name].shape != (len(points),):
            raise ValueError('shear boundary masks must be material-node Boolean arrays')
    bottom = direct._g_validate_interface_coordinates(points, cloud['bottom'], 0., 'shear bottom')
    top = direct._g_validate_interface_coordinates(points, cloud['top'], 1., 'shear top')
    tolerance = direct._g_geometry_tolerance(points)
    if (not np.array_equal(cloud['bottom'], abs(points[:, 1]) <= tolerance)
            or not np.array_equal(cloud['top'], abs(points[:, 1]-1.) <= tolerance)):
        raise ValueError('shear masks must cover the complete y=0 and y=1 edges')
    nodes = np.concatenate((bottom, top))
    constrained = np.sort((2*nodes[:, None]+np.arange(2)).ravel()).astype(np.int64)
    direction = np.zeros(2*len(points), np.float64)
    direction[2*top] = 1.
    return constrained, direction


def l_panel_boundary(cloud, c):
    """Validate the L polygon; clamp bottom and lift only the underside patch."""
    b = c.boundary
    width, height = c.geometry.width, c.geometry.height
    cx, cy, start, end = b.corner_x, b.corner_y, b.load_start, b.load_end
    if (not np.isfinite([width, height, cx, cy, start, end]).all()
            or not 0 < cx < width or not 0 < cy < height
            or not cx < start < end <= width):
        raise ValueError('invalid explicit L-panel dimensions or load interval')
    points, cells = cloud['coordinates'], cloud['triangles']
    if (points.dtype != np.float64 or points.ndim != 2 or points.shape[1] != 2
            or not np.isfinite(points).all() or cells.dtype != np.int64):
        raise ValueError('invalid L-panel coordinates or connectivity dtype')
    tol = direct._g_geometry_tolerance(points)
    inside = ((points[:, 0] >= -tol) & (points[:, 0] <= width+tol)
              & (points[:, 1] >= -tol) & (points[:, 1] <= height+tol)
              & ((points[:, 0] <= cx+tol) | (points[:, 1] >= cy-tol)))
    if not np.all(inside):
        raise ValueError('L-panel mesh contains points outside the material')
    polygon = direct._g_boundary_polygon_from_triangles(points, cells)
    vertices = np.array([[0., 0.], [cx, 0.], [cx, cy],
                         [width, cy], [width, height], [0., height]])
    a, z = polygon, np.roll(polygon, -1, axis=0)
    on_edge = np.zeros(len(polygon), bool)
    for p, q in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        axis = int(np.flatnonzero(p == q)[0])
        other = 1-axis
        lo, hi = sorted((p[other], q[other]))
        on_edge |= ((abs(a[:, axis]-p[axis]) <= tol) & (abs(z[:, axis]-p[axis]) <= tol)
                    & (a[:, other] >= lo-tol) & (a[:, other] <= hi+tol)
                    & (z[:, other] >= lo-tol) & (z[:, other] <= hi+tol))
    area = .5*abs(np.sum(a[:, 0]*z[:, 1]-a[:, 1]*z[:, 0]))
    expected_area = width*height-(width-cx)*cy
    if not np.all(on_edge) or not np.isclose(area, expected_area, rtol=1e-12, atol=1e-14):
        raise ValueError('prepared material boundary is not the configured L polygon')
    bottom = abs(points[:, 1]) <= tol
    loaded = ((abs(points[:, 1]-cy) <= tol)
              & (points[:, 0] >= start-tol) & (points[:, 0] <= end+tol))
    for name, expected in (('bottom', bottom), ('top', loaded)):
        if (cloud[name].dtype != bool or cloud[name].shape != (len(points),)
                or not np.array_equal(cloud[name], expected) or not np.any(expected)):
            raise ValueError('L-panel fixed/loaded masks differ from the complete configured edges')
    fixed = (2*np.flatnonzero(bottom)[:, None]+np.arange(2)).ravel()
    driven = 2*np.flatnonzero(loaded)+1
    constrained = np.sort(np.concatenate((fixed, driven))).astype(np.int64)
    direction = np.zeros(2*len(points), np.float64)
    direction[driven] = 1.
    return constrained, direction


def edge_boundary(cloud, c):
    if c.boundary.mode == 'horizontal_clamped_edges':
        return shear_boundary(cloud)
    if c.boundary.mode == 'l_shaped_panel':
        return l_panel_boundary(cloud, c)
    raise ValueError('unknown explicit edge-boundary mode')

def prepare(c):
    identity = geometry_identity(c)
    root = Path(c.mesh.prepared_dir)
    root.mkdir()
    cloud = read_cloud(c.data.cloud)
    points, tri, V = cloud['coordinates'], cloud['triangles'], cloud['volumes']
    print(json.dumps(dict(event='prepare', material_nodes=len(points), cells=len(tri))), flush=True)
    if 'boundary' in c:
        constrained, direction = edge_boundary(cloud, c)
        np.savez_compressed(root/'mesh.npz', coordinates=points, triangles=tri,
            n_material=np.int64(len(points)), n_material_cells=np.int64(len(tri)),
            volumes=V, constrained=constrained, load_direction=direction)
        atomic_json(root/'identity.json', dict(config=identity,
            mesh_path=str(root/'mesh.npz'), operator_path=None))
        print(json.dumps(dict(event='prepared', root=str(root), material_only=True,
            constrained_dofs=len(constrained), boundary=c.boundary.mode)), flush=True)
        return
    offsets = np.arange(c.geometry.h_grip, c.geometry.grip_extent, c.geometry.h_grip)
    offsets = np.concatenate((offsets[offsets < c.geometry.grip_extent], [c.geometry.grip_extent]))
    coordinates = points.tolist()
    bottom = direct._g_validate_interface_coordinates(points, cloud['bottom'], 0., 'local bottom')
    top = direct._g_validate_interface_coordinates(points, cloud['top'], 1., 'local top')
    btri, _ = direct._g_structured_grip_triangles(points, bottom, offsets, -1., coordinates, 'bottom')
    ttri, _ = direct._g_structured_grip_triangles(points, top, offsets, 1., coordinates, 'top')
    coordinates = np.asarray(coordinates, np.float64)
    triangles = np.concatenate((tri, btri, ttri)).astype(np.int64)
    collar = np.arange(len(coordinates)) >= len(points)
    bnodes = np.flatnonzero(collar & (coordinates[:, 1] < 0))
    tnodes = np.flatnonzero(collar & (coordinates[:, 1] > 1))
    constrained = np.sort(np.concatenate((2*bnodes+1, 2*tnodes+1, [0]))).astype(np.int64)
    direction = np.zeros(2*len(coordinates))
    direction[2*tnodes+1] = 1.
    _, _, edges, _ = visibility_graph(points, tri, c.nl.delta_d, c.nl.visibility_tolerance)
    operator = ExplicitVolumeSparseD(points, V, edges, c.nl.delta_d, 'cubic')
    from scipy.sparse import save_npz
    save_npz(root/'damage_operator.npz', operator.to_scipy())
    np.savez_compressed(root/'mesh.npz', coordinates=coordinates, triangles=triangles,
        n_material=np.int64(len(points)), n_material_cells=np.int64(len(tri)),
        volumes=V, constrained=constrained, load_direction=direction, damage_edges=edges)
    atomic_json(root/'identity.json', dict(config=geometry_identity(c),
        mesh_path=str(root/'mesh.npz'), operator_path=str(root/'damage_operator.npz')))
    print(json.dumps(dict(event='prepared', root=str(root), union_nodes=len(coordinates), damage_edges=len(edges))), flush=True)

def load_geometry(c):
    root = Path(c.mesh.prepared_dir)
    identity = json.loads((root/'identity.json').read_text())
    # Compare numerical settings; historical integrity metadata is ignored.
    expected = geometry_identity(c)
    prepared = identity['config']
    for key in expected:
        if key != 'cloud' and (key not in prepared or prepared[key] != expected[key]):
            raise ValueError(f'prepared geometry setting differs: {key}')
    with np.load(root/'mesh.npz', allow_pickle=False) as f:
        values = {name: f[name].copy() for name in f.files}
    cloud = read_cloud(c.data.cloud)
    n, nc = int(values['n_material']), int(values['n_material_cells'])
    if 'boundary' in c:
        constrained, direction = edge_boundary(cloud, c)
        if (len(values['coordinates']) != n or len(values['triangles']) != nc
                or not np.array_equal(values['constrained'], constrained)
                or not np.array_equal(values['load_direction'], direction)):
            raise ValueError('prepared edge domain or boundary differs from configuration')
    for actual, expected_values in (
            (values['coordinates'][:n], cloud['coordinates']),
            (values['triangles'][:nc], cloud['triangles']),
            (values['volumes'], cloud['volumes'])):
        if not np.array_equal(actual, expected_values):
            raise ValueError('prepared material mesh differs from the configured cloud arrays')
    mechanics = LocalMechanics(values['coordinates'], values['triangles'], int(values['n_material']),
        int(values['n_material_cells']), values['volumes'], E=c.physical.E, nu=c.physical.nu,
        k_res=c.physical.k_res, constrained=values['constrained'], load_direction=values['load_direction'], range_atol=c.validation.range_atol)
    if c.damage.discretization == 'fem':
        from examples.fracture2d.fem_damage import P1DamageOperator
        operator = P1DamageOperator(mechanics)
        return mechanics, operator, dict(config=expected, mesh_path=str(root/'mesh.npz'),
            damage_discretization='material-p1-lumped', operator_path=None)
    if c.damage.discretization != 'nonlocal':
        raise ValueError('unknown explicit damage discretization')
    operator = ExplicitVolumeSparseD(mechanics.coordinates[:mechanics.n_material], mechanics.volumes,
                                    values['damage_edges'], c.nl.delta_d, 'cubic')
    from scipy.sparse import load_npz
    rebuilt, prepared = operator.to_scipy(), load_npz(root/'damage_operator.npz')
    difference = rebuilt-prepared
    coefficient_limit = c.validation.roundoff_factor*np.finfo(float).eps*np.max(abs(prepared.data))
    if (not np.array_equal(rebuilt.indices, prepared.indices)
            or not np.array_equal(rebuilt.indptr, prepared.indptr)
            or difference.nnz and np.max(abs(difference.data)) > coefficient_limit):
        raise ValueError('rebuilt damage operator differs from prepared coefficients')
    return mechanics, operator, dict(config=expected, mesh_path=str(root/'mesh.npz'),
                                      operator_path=str(root/'damage_operator.npz'))
