"""Independent A2/AT2 horizon validation branch; frozen baseline is reused intact.

Geometry/constraints retain the engineering volume-grip BVP. Mechanics and
damage use separately declared strict reciprocal visibility graphs. No
damage-dependent support, new constitutive term or hidden rank fallback exists.
"""

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from pinn.operators import ExplicitVolumeSparseD
from examples.fracture2d import nonlocal_baseline as direct


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


@dataclass(frozen=True)
class SplitHorizonGeometry(direct.UnionGripGeometry):
    """Same physical phase quadrature/grips, with explicit independent radii.

    Inherited ``delta`` is the mechanics radius. ``grip_extent`` is the
    physical collar thickness; ``damage_delta`` is material scalar radius.
    The base geometry validator checks structural/phase data at the enclosing
    grip extent. Stricter actual mechanics/damage graph bounds are then checked
    separately, and the constructor supplies complete predicate-built graphs.
    """

    grip_extent: float
    damage_delta: float
    material_damage_edges: np.ndarray

    def __post_init__(self):
        delta_u = direct._g_scalar_float(self.delta, "mechanics horizon")
        extent = direct._g_scalar_float(self.grip_extent, "physical grip extent")
        delta_d = direct._g_scalar_float(self.damage_delta, "damage horizon")
        if not 0 < self.h_g < delta_u <= extent or delta_d <= 0:
            raise ValueError("split geometry requires 0 < h_grip < delta_u <= grip_extent and delta_d > 0")
        values = {field.name: getattr(self, field.name) for field in fields(direct.UnionGripGeometry)}
        values["delta"] = extent
        checked = direct.UnionGripGeometry(**values)
        # Adopt immutable validated structural arrays, not mutated old objects.
        for field in fields(direct.UnionGripGeometry):
            object.__setattr__(self, field.name, getattr(checked, field.name))
        object.__setattr__(self, "delta", delta_u)
        object.__setattr__(self, "grip_extent", extent)
        object.__setattr__(self, "damage_delta", delta_d)
        for name in ("candidate_edges", "directed_edges", "material_reference_edges"):
            points = self.material_reference_coordinates if name == "material_reference_edges" else self.coordinates
            direct._g_strict_graph_validation(points, getattr(self, name), delta_u, "split " + name)
        if len(self.removed_edges):
            direct._g_strict_graph_validation(self.coordinates, self.removed_edges, delta_u, "split removed graph")
        damage_edges = direct._g_require_int64(self.material_damage_edges, "damage edges")
        direct._g_strict_graph_validation(self.material_reference_coordinates, damage_edges, delta_d, "damage graph")
        object.__setattr__(self, "material_damage_edges", direct._g_immutable(damage_edges))
        if not np.isclose(self.grip_area, 2 * extent, rtol=0, atol=1e-12):
            raise ValueError("split geometry changed the declared physical grip areas")


def build_geometry(cloud, delta_u, delta_d, grip_extent, h_grip, tolerance):
    coords = direct._g_require_float64(cloud["coordinates"], "cloud coordinates")
    triangles = direct._g_require_int64(cloud["triangles"], "cloud triangles")
    volumes = direct._g_require_float64(cloud["volumes"], "cloud volumes", (len(coords),))
    bottom = direct._g_require_bool(cloud["bottom"], "cloud bottom", (len(coords),))
    top = direct._g_require_bool(cloud["top"], "cloud top", (len(coords),))
    if int(cloud["anchor_particle"]) != 0 or not np.array_equal(coords[0], [0, 0]):
        raise ValueError("split cloud must retain the physical anchor as particle zero")
    if not 0 < h_grip < delta_u <= grip_extent or delta_d <= 0:
        raise ValueError("invalid explicit split horizons/grip geometry")
    bottom_indices = direct._g_validate_interface_coordinates(coords, bottom, 0.0, "bottom interface")
    top_indices = direct._g_validate_interface_coordinates(coords, top, 1.0, "top interface")
    offsets = []
    k = 1
    while k * h_grip < grip_extent:
        offsets.append(float(k * h_grip))
        k += 1
    offsets = np.asarray([*offsets, grip_extent], np.float64)
    new_coordinates = coords.tolist()
    bottom_tri, bottom_rows = direct._g_structured_grip_triangles(
        coords, bottom_indices, offsets, -1.0, new_coordinates, "bottom")
    top_tri, top_rows = direct._g_structured_grip_triangles(
        coords, top_indices, offsets, 1.0, new_coordinates, "top")
    grip_tri = np.vstack((bottom_tri, top_tri)).astype(np.int64)
    union_coords = np.asarray(new_coordinates, np.float64)
    union_tri = np.vstack((triangles, grip_tri)).astype(np.int64)
    material_areas, Vm, Mm = direct._g_triangle_quadrature(union_coords, triangles, "material triangles")
    grip_areas, Vg, Mg = direct._g_triangle_quadrature(union_coords, grip_tri, "grip triangles")
    union_areas, Vu, Mu = direct._g_triangle_quadrature(union_coords, union_tri, "union triangles")
    if not np.array_equal(Vm[:len(coords)], volumes):
        raise ValueError("split cloud material quadrature is not exactly its triangle quadrature")
    print(json.dumps({"event": "split_layout", "material_nodes": len(coords),
                      "union_nodes": len(union_coords), "delta_u": delta_u, "delta_d": delta_d,
                      "grip_extent": grip_extent}), flush=True)
    _, _, material_u, _ = visibility_graph(coords, triangles, delta_u, tolerance)
    print(json.dumps({"event": "split_material_mechanics_graph", "edges": len(material_u)}), flush=True)
    if delta_d == delta_u:
        material_d = material_u
    else:
        _, _, material_d, _ = visibility_graph(coords, triangles, delta_d, tolerance)
    print(json.dumps({"event": "split_material_damage_graph", "edges": len(material_d)}), flush=True)
    polygon, candidate, visible, removed = visibility_graph(union_coords, union_tri, delta_u, tolerance)
    induced = visible[(visible[:, 0] < len(coords)) & (visible[:, 1] < len(coords))]
    np.testing.assert_array_equal(induced, material_u)
    n = len(union_coords)
    masks = []
    for indices in (bottom_indices, top_indices, bottom_rows[-1], top_rows[-1]):
        mask = np.zeros(n, bool)
        mask[indices] = True
        masks.append(mask)
    return SplitHorizonGeometry(
        delta=delta_u, h_g=h_grip, collar_offsets=offsets, visibility_tolerance=tolerance,
        n_material=len(coords), coordinates=union_coords, material_indices=np.arange(len(coords), dtype=np.int64),
        material_reference_coordinates=coords, material_reference_triangles=triangles,
        material_reference_volumes=volumes, material_reference_edges=material_u,
        material_triangles=triangles, grip_triangles=grip_tri, union_triangles=union_tri,
        material_triangle_areas=material_areas, grip_triangle_areas=grip_areas, union_triangle_areas=union_areas,
        material_volumes=Vm, grip_volumes=Vg, union_volumes=Vu,
        material_first_moments=Mm, grip_first_moments=Mg, union_first_moments=Mu,
        boundary_polygon=polygon, candidate_edges=candidate, directed_edges=visible, removed_edges=removed,
        neighbor_counts=np.bincount(visible[:, 0], minlength=n).astype(np.int64),
        material_bottom_mask=masks[0], material_top_mask=masks[1],
        interface_bottom_mask=masks[0], interface_top_mask=masks[1], interface_mask=masks[0] | masks[1],
        outer_bottom_mask=masks[2], outer_top_mask=masks[3],
        material_bottom_indices=bottom_indices, material_top_indices=top_indices,
        outer_bottom_indices=bottom_rows[-1], outer_top_indices=top_rows[-1],
        grip_extent=grip_extent, damage_delta=delta_d, material_damage_edges=material_d)


@dataclass(frozen=True)
class SplitHorizonModel(direct.EngineeringGripModel):
    """Explicitly validated extension; damage graph is not the mechanics graph."""

    geometry: SplitHorizonGeometry

    def __post_init__(self):
        if not isinstance(self.geometry, SplitHorizonGeometry):
            raise TypeError("split model requires separately validated SplitHorizonGeometry")
        geometry = self.geometry
        if not isinstance(self.static, direct.A2StaticMechanics) or not isinstance(self.constraints, direct.AffineConstraints):
            raise TypeError("split model requires the validated A2 static/constraint types")
        np.testing.assert_array_equal(self.static.directed_edges, geometry.directed_edges)
        if self.static.delta != geometry.delta or self.constraints.n_particles != geometry.n_particles:
            raise ValueError("split static horizon or constraint particle count differs")
        expected = direct.build_volume_grip_constraints(geometry, 0)
        for name in ("constrained_dofs", "free_dofs", "q_zero", "load_direction", "outer_bottom_mask", "outer_top_mask"):
            np.testing.assert_array_equal(getattr(self.constraints, name), getattr(expected, name))
        for line, side in ((self.top_line, "top"), (self.bottom_line, "bottom")):
            canonical = direct.physical_interface_line_weights(geometry, side)
            np.testing.assert_array_equal(line.indices, canonical.indices)
            np.testing.assert_array_equal(line.weights, canonical.weights)
        if not isinstance(self.material_operator, ExplicitVolumeSparseD):
            raise TypeError("split damage operator must be ExplicitVolumeSparseD")
        op = self.material_operator
        if op.shape != (geometry.n_material, geometry.n_material) or op.kernel != "cubic" or op.delta != geometry.damage_delta:
            raise ValueError("split damage operator shape/kernel/horizon differs")
        for actual, expected_values in ((op.coordinates, geometry.material_reference_coordinates),
                                       (op.volumes, geometry.material_reference_volumes),
                                       (op.directed_edges, geometry.material_damage_edges)):
            np.testing.assert_array_equal(np.asarray(actual), expected_values)
        Gc, l0 = direct._e_scalar(self.Gc, "split Gc"), direct._e_scalar(self.l0, "split l0")
        if min(Gc, l0) <= 0:
            raise ValueError("split Gc and l0 must be positive")
        object.__setattr__(self, "Gc", Gc)
        object.__setattr__(self, "l0", l0)
        mask = direct._e_bool(self.crack_region_mask, "split diagnostic crack region", (geometry.n_material,))
        object.__setattr__(self, "crack_region_mask", direct._e_immutable(mask))
        material_hash, union_hash = direct._e_source_digest(self)
        if self.material_hash != material_hash or self.union_hash != union_hash:
            raise ValueError("split model source hashes differ")

    @property
    def model_digest(self):
        return direct._e_hash_named((
            ("schema", "fracture2d-independent-horizons-v1"),
            ("engineering_value_digest", direct._e_engineering_model_digest(self)),
            ("grip_extent", self.geometry.grip_extent),
            ("delta_u", self.geometry.delta), ("delta_d", self.geometry.damage_delta)))


def build_model(cloud, *, delta_u, delta_d, grip_extent, h_grip, visibility_tolerance,
                E, nu, k_res, rank_rtol, Gc, l0):
    if not jax.config.jax_enable_x64:
        raise ValueError("split model requires JAX x64 before operator construction")
    geometry = build_geometry(cloud, delta_u, delta_d, grip_extent, h_grip, visibility_tolerance)
    constraints = direct.build_volume_grip_constraints(geometry, 0)
    top_line = direct.physical_interface_line_weights(geometry, "top")
    bottom_line = direct.physical_interface_line_weights(geometry, "bottom")
    static = direct.A2StaticMechanics.from_geometry(geometry, constraints, top_line, bottom_line, E, nu, k_res, rank_rtol)
    print(json.dumps({"event": "split_BA_validated", "edges": static.n_edges,
                      "max_condition": float(max(static.shape_condition)),
                      "minimum_rank": int(min(static.shape_rank))}), flush=True)
    operator = ExplicitVolumeSparseD(geometry.material_reference_coordinates, geometry.material_reference_volumes,
                                    geometry.material_damage_edges, delta_d, "cubic")
    material_hash = direct._e_hash_named((
        ("coordinates", geometry.material_reference_coordinates), ("volumes", geometry.material_reference_volumes),
        ("triangles", geometry.material_reference_triangles), ("edges", geometry.material_reference_edges),
        ("bottom", geometry.material_bottom_mask[:geometry.n_material]),
        ("top", geometry.material_top_mask[:geometry.n_material]), ("anchor", 0)))
    union_hash = direct._e_hash_named((
        ("coordinates", geometry.coordinates), ("material_phase_volumes", geometry.material_volumes),
        ("grip_phase_volumes", geometry.grip_volumes), ("union_volumes", geometry.union_volumes),
        ("union_triangles", geometry.union_triangles), ("candidate_edges", geometry.candidate_edges),
        ("visible_edges", geometry.directed_edges), ("removed_edges", geometry.removed_edges),
        ("volume_bottom_mask", constraints.outer_bottom_mask), ("volume_top_mask", constraints.outer_top_mask),
        ("visibility_tolerance", np.asarray(visibility_tolerance, np.float64))))
    return SplitHorizonModel(geometry, static, constraints, top_line, bottom_line, operator, Gc, l0,
        direct.build_crack_region_mask(geometry.material_reference_coordinates, delta_d), material_hash, union_hash)


def read_cloud(path, expected_sha256):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"split cloud SHA256 differs: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in ("coordinates", "triangles", "volumes", "bottom", "top", "anchor_particle")}
