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
    return dict(cloud=str(c.data.cloud), delta_d=c.nl.delta_d,
                visibility_tolerance=c.nl.visibility_tolerance,
                grip_extent=c.geometry.grip_extent, h_grip=c.geometry.h_grip,
                E=c.physical.E, nu=c.physical.nu, k_res=c.physical.k_res,
                schema='p1-local-volume-grips-amor-lumped-v1')

def prepare(c):
    root = Path(c.mesh.prepared_dir)
    root.mkdir()
    cloud = read_cloud(c.data.cloud)
    points, tri, V = cloud['coordinates'], cloud['triangles'], cloud['volumes']
    print(json.dumps(dict(event='prepare', material_nodes=len(points), cells=len(tri))), flush=True)
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
        if key != 'cloud' and prepared[key] != expected[key]:
            raise ValueError(f'prepared geometry setting differs: {key}')
    with np.load(root/'mesh.npz', allow_pickle=False) as f:
        values = {name: f[name].copy() for name in f.files}
    cloud = read_cloud(c.data.cloud)
    n, nc = int(values['n_material']), int(values['n_material_cells'])
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
