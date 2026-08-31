"""Candidate A2 frozen-mechanics grip diagnostic for fracture2d.

This module is deliberately a diagnostic branch.  It does not modify the
material-only bond-associated mechanics, the AT2 damage operator, the
continuation controller, any JAX/PINN path, or a production configuration.
All numerical arrays in this module are NumPy ``float64``/SciPy data.

Candidate A2 is the project-owned phase-measured elastic-grip counterfactual
specified in ``docs/tasks/fracture2d-nonlocal-boundary-formulation.md``.  The
material particle prefix is copied exactly, structured exterior strips are
added without remeshing the material triangles, and one conservative energy is
used for the fixed-damage mechanics residual and tangent:

``Pi = sum_i W_i sum_j eta_ij psi_i|j``

with ``W_i = V_i^m g(d_i) + V_i^g g(0)``.  The existing target-inclusive
bond-associated kinematics are reused only after their family masses,
supports, shape tensors, and center factors have been independently rebuilt
and checked on the union graph.

The published/project provenance distinction is explicit.  Chen--Spencer's
bond-associated support/shape/force construction and the plane-strain linear
elastic map are the source-backed basis.  The phase-measured center energy,
structured collar, outer-face constraints, mapped-field comparison, and all
union-cloud visibility/mesh choices are project-owned diagnostic choices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.linalg import LinearOperator, cg, lobpcg

from .nonlocal_geometry import ParticleCloud
from .nonlocal_mechanics import (
    BondAssociatedMechanics,
    FixedDamageBondAssociatedMechanics,
    MechanicsEvaluation,
)


__all__ = [
    "A2MechanicsSolveResult",
    "A2StaticMechanics",
    "A2TransitionResult",
    "AffineConstraints",
    "CandidateA2GateFailure",
    "CandidateA2Run",
    "CandidateA2Sources",
    "FixedDamagePhaseMeasuredMechanics",
    "IndependentUnionBAReference",
    "LineQuadrature",
    "MaterialMetrics",
    "UnionGripGeometry",
    "build_affine_constraints",
    "build_volume_grip_constraints",
    "build_union_geometry",
    "build_union_geometry_from_arrays",
    "compute_material_metrics",
    "compute_h_used",
    "independent_union_ba_reference",
    "load_frozen_sources",
    "physical_interface_line_weights",
    "physical_interface_opening",
    "run_manufactured_gates",
    "run_frozen_transition",
    "run_staged_diagnostic",
    "run_union_hard_gates",
    "solve_fixed_damage_a2",
    "transition_decision_arithmetic",
]


_FLOAT64 = np.dtype(np.float64)
_INT64 = np.dtype(np.int64)
_BOOL = np.dtype(bool)
_GEOMETRY_EPS_FACTOR = 64.0
_AREA_RECONSTRUCTION_EPS_FACTOR = 512.0
_INTERFACE_INPUT_TOLERANCE = 1.0e-12
_DELTA_D048 = 0.048
_TARGET_OPENING = 0.0056
# The authoritative load solve retains the executable contract's
# cg_rtol=1e-7/cg_atol=1e-10.  Centered reaction/work checks use independent,
# explicitly stricter re-solves so their finite-difference error does not get
# confused with the primary solve residual.
_WORK_CG_RTOL = 1.0e-10
_WORK_CG_ATOL = 1.0e-13
_RAW_DEFAULT = Path("/tmp/opencode/fracture2d_d048_warning_ufinal_0075.raw.npz")
_CLOUD_DEFAULT = Path("/tmp/opencode/stage3_cloud_float64.npz")
_FEM_DEFAULT = Path(
    "/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz"
)
_RAW_SHA256 = "48d92621955289c51710296569362c50db30e28f63d622ee447c11a2317f66e5"
_FEM_SHA256 = "e9634f1f8fdb21116b50ad453d37115746c4a5a09f828e76245a9c59f6107d17"
_ARTIFACT_SCHEMA = "candidate-a2-run-v2"


def _immutable(values: Any, dtype: Any | None = None) -> np.ndarray:
    """Copy values behind a bytes buffer so read-only flags cannot be undone."""

    source = np.array(values, dtype=dtype, copy=True, order="C")
    raw = source.tobytes(order="C")
    result = np.frombuffer(raw, dtype=source.dtype, count=source.size)
    result = result.reshape(source.shape)
    result.setflags(write=False)
    return result


def _scalar_float(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar, got shape {array.shape}")
    try:
        result = float(array)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a scalar number") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _require_float64(
    values: Any, name: str, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != _FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _require_int64(
    values: Any, name: str, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != _INT64:
        raise ValueError(f"{name} must use exact int64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _require_bool(
    values: Any, name: str, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != _BOOL:
        raise ValueError(f"{name} must use exact boolean dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _geometry_tolerance(coordinates: np.ndarray) -> float:
    scale = max(1.0, float(np.max(np.abs(coordinates))))
    # The 1e-12 input floor is the declared tolerance needed to accept the
    # unchanged, unsnapped Gmsh boundary coordinates.  The machine component
    # remains scale-aware and is never used to alter coordinates.
    return max(
        _AREA_RECONSTRUCTION_EPS_FACTOR * np.finfo(np.float64).eps * scale,
        _INTERFACE_INPUT_TOLERANCE * scale,
    )


def _relative_error(first: float, second: float) -> float:
    return abs(float(first) - float(second)) / max(
        abs(float(first)), abs(float(second)), 1.0e-30
    )


def _array_relative_error(first: np.ndarray, second: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(first) - np.asarray(second)))
    denominator = max(
        float(np.linalg.norm(np.asarray(first))),
        float(np.linalg.norm(np.asarray(second))),
        1.0e-30,
    )
    return numerator / denominator


def _strict_graph_validation(
    coordinates: np.ndarray,
    directed_edges: np.ndarray,
    delta: float,
    name: str,
) -> None:
    edges = _require_int64(directed_edges, name)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"{name} must have shape (E, 2), got {edges.shape}")
    if len(edges) == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(edges < 0) or np.any(edges >= len(coordinates)):
        raise ValueError(f"{name} contains an out-of-range particle index")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError(f"{name} must not contain self edges")
    expected_order = np.lexsort((edges[:, 1], edges[:, 0]))
    if not np.array_equal(expected_order, np.arange(len(edges))):
        raise ValueError(f"{name} must be lexicographically sorted")
    if len(np.unique(edges, axis=0)) != len(edges):
        raise ValueError(f"{name} contains duplicate directed edges")
    edge_set = {tuple(value) for value in edges.tolist()}
    missing = [(i, j) for i, j in edge_set if (j, i) not in edge_set]
    if missing:
        raise ValueError(f"{name} is not reciprocal; missing reverse of {missing[0]}")
    distance = np.linalg.norm(coordinates[edges[:, 1]] - coordinates[edges[:, 0]], axis=1)
    if np.any(~np.isfinite(distance)) or np.any(distance <= 0.0):
        raise ValueError(f"{name} contains a zero/non-finite edge distance")
    if np.any(distance >= delta):
        bad = int(np.flatnonzero(distance >= delta)[0])
        raise ValueError(
            f"{name} must satisfy strict 0 < distance < delta; bad edge "
            f"{edges[bad].tolist()} has distance {distance[bad]:.17g}"
        )


def _triangle_quadrature(
    coordinates: np.ndarray,
    triangles: np.ndarray,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = _require_int64(triangles, name)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError(f"{name} must have non-empty shape (T, 3)")
    if np.any(triangles < 0) or np.any(triangles >= len(coordinates)):
        raise ValueError(f"{name} contains an out-of-range node index")
    if np.any(
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 0] == triangles[:, 2])
        | (triangles[:, 1] == triangles[:, 2])
    ):
        raise ValueError(f"{name} contains a repeated node")
    p0 = coordinates[triangles[:, 0]]
    p1 = coordinates[triangles[:, 1]]
    p2 = coordinates[triangles[:, 2]]
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (
        p1[:, 1] - p0[:, 1]
    ) * (p2[:, 0] - p0[:, 0])
    areas = 0.5 * np.abs(cross)
    if np.any(~np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError(f"{name} contains a non-positive/non-finite triangle area")
    volumes = np.zeros(len(coordinates), dtype=np.float64)
    contribution = areas / 3.0
    np.add.at(volumes, triangles[:, 0], contribution)
    np.add.at(volumes, triangles[:, 1], contribution)
    np.add.at(volumes, triangles[:, 2], contribution)
    if np.any(~np.isfinite(volumes)) or np.any(volumes < 0.0):
        raise ValueError(f"{name} produced negative/non-finite nodal volumes")
    used = np.unique(triangles)
    if np.any(volumes[used] <= 0.0):
        raise ValueError(f"{name} produced non-positive volume on a used node")
    if not np.isclose(
        float(np.sum(volumes)),
        float(np.sum(areas)),
        rtol=0.0,
        atol=32.0 * np.finfo(np.float64).eps,
    ):
        raise ValueError(f"{name} nodal-volume sum does not equal triangle area")
    first_moments = np.sum(volumes[:, None] * coordinates, axis=0)
    if not np.all(np.isfinite(first_moments)):
        raise ValueError(f"{name} first moments are non-finite")
    return areas, volumes, first_moments


def _directed_edges_from_pairs(pairs: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must have shape (P, 2), got {pairs.shape}")
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    directed = np.vstack((pairs, pairs[:, ::-1]))
    order = np.lexsort((directed[:, 1], directed[:, 0]))
    return np.asarray(directed[order], dtype=np.int64)


def _all_strict_pairs(coordinates: np.ndarray, delta: float) -> np.ndarray:
    raw = np.asarray(
        cKDTree(coordinates).query_pairs(delta, output_type="ndarray"),
        dtype=np.int64,
    )
    if raw.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    pairs = raw.reshape((-1, 2))
    distance = np.linalg.norm(coordinates[pairs[:, 1]] - coordinates[pairs[:, 0]], axis=1)
    pairs = pairs[(distance > 0.0) & (distance < delta)]
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return np.asarray(pairs[order], dtype=np.int64)


def _boundary_polygon_from_triangles(
    coordinates: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    incidence: dict[tuple[int, int], int] = defaultdict(int)
    for triangle in triangles:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            key = tuple(sorted((int(first), int(second))))
            incidence[key] += 1
    nonmanifold = [item for item in incidence.items() if item[1] > 2]
    if nonmanifold:
        raise ValueError(
            "union triangle mesh is non-manifold at edge "
            f"{nonmanifold[0][0]}"
        )
    boundary_edges = [edge for edge, count in incidence.items() if count == 1]
    if not boundary_edges:
        raise ValueError("union triangle mesh has no boundary")
    adjacency: dict[int, list[int]] = defaultdict(list)
    for first, second in boundary_edges:
        adjacency[first].append(second)
        adjacency[second].append(first)
    bad_degree = [
        (node, len(neighbors))
        for node, neighbors in adjacency.items()
        if len(neighbors) != 2
    ]
    if bad_degree:
        raise ValueError(
            "union boundary must be one closed polygon; particle "
            f"{bad_degree[0][0]} has degree {bad_degree[0][1]}"
        )
    start = min(adjacency, key=lambda index: (coordinates[index, 0], coordinates[index, 1], index))
    loop: list[int] = []
    previous = -1
    current = start
    while True:
        loop.append(current)
        choices = sorted(adjacency[current])
        next_node = choices[0] if choices[0] != previous else choices[1]
        previous, current = current, next_node
        if current == start:
            break
        if len(loop) > len(boundary_edges) + 1:
            raise ValueError("union boundary traversal did not close")
    if len(loop) != len(adjacency):
        raise ValueError("union triangle mesh has more than one boundary loop")
    polygon = np.asarray(coordinates[np.asarray(loop, dtype=np.int64)], dtype=np.float64)
    signed_area = 0.5 * float(
        np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - polygon[:, 1] * np.roll(polygon[:, 0], -1)
        )
    )
    if not np.isfinite(signed_area) or signed_area <= 0.0:
        # Traversal starts at the lower-left corner and chooses the lower/right
        # branch, which is CCW for the structured mesh.  Reverse only if a
        # caller supplies a valid mesh whose deterministic ordering is opposite.
        polygon = polygon[::-1].copy()
        signed_area = -signed_area
    if signed_area <= 0.0:
        raise ValueError("union boundary polygon has non-positive area")
    return polygon


class _UnionPolygonPredicate:
    """Independent segment-in-closure predicate for one simple union polygon."""

    def __init__(self, polygon: np.ndarray, input_tolerance: float) -> None:
        polygon = _require_float64(polygon, "union boundary polygon")
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError("union boundary polygon must have shape (N, 2), N >= 3")
        tolerance = _scalar_float(input_tolerance, "visibility_tolerance")
        if tolerance < 0.0:
            raise ValueError("visibility_tolerance must be non-negative")
        self.polygon = polygon
        self.start = polygon
        self.end = np.roll(polygon, -1, axis=0)
        self.direction = self.end - self.start
        self.length = np.linalg.norm(self.direction, axis=1)
        if np.any(self.length <= 0.0):
            raise ValueError("union boundary contains a zero-length edge")
        self.input_tolerance = tolerance
        self.polygon_scale = max(
            float(np.linalg.norm(np.ptp(polygon, axis=0))),
            float(np.max(self.length)),
        )

    def _effective_tolerance(self, segment_length: float) -> float:
        return max(
            self.input_tolerance,
            _GEOMETRY_EPS_FACTOR
            * np.finfo(np.float64).eps
            * max(self.polygon_scale, segment_length),
        )

    def _location(self, point: np.ndarray, tolerance: float) -> int:
        offset = point[None, :] - self.start
        cross = offset[:, 0] * self.direction[:, 1] - offset[:, 1] * self.direction[:, 0]
        projection = np.sum(offset * self.direction, axis=1)
        on_boundary = (
            np.abs(cross) <= tolerance * self.length
        ) & (projection >= -tolerance * self.length) & (
            projection <= self.length**2 + tolerance * self.length
        )
        if np.any(on_boundary):
            return -1
        upper = self.start[:, 1] > point[1]
        lower = self.end[:, 1] > point[1]
        crossing = upper != lower
        denominator = self.end[:, 1] - self.start[:, 1]
        crossing_x = np.empty(len(self.start), dtype=np.float64)
        crossing_x[:] = np.inf
        nonzero = denominator != 0.0
        crossing_x[nonzero] = self.start[nonzero, 0] + (
            self.direction[nonzero, 0]
            * (point[1] - self.start[nonzero, 1])
            / denominator[nonzero]
        )
        return 1 if int(np.count_nonzero(crossing & (crossing_x > point[0]))) % 2 else 0

    def __call__(self, start: np.ndarray, end: np.ndarray) -> bool:
        start = _require_float64(start, "visibility start", (2,))
        end = _require_float64(end, "visibility end", (2,))
        difference = end - start
        segment_length = float(np.linalg.norm(difference))
        if segment_length <= 0.0:
            raise ValueError("visibility segment endpoints must be distinct")
        tolerance = self._effective_tolerance(segment_length)
        segment_min = np.minimum(start, end) - tolerance
        segment_max = np.maximum(start, end) + tolerance
        candidate = (
            (np.maximum(self.start[:, 0], self.end[:, 0]) >= segment_min[0])
            & (np.minimum(self.start[:, 0], self.end[:, 0]) <= segment_max[0])
            & (np.maximum(self.start[:, 1], self.end[:, 1]) >= segment_min[1])
            & (np.minimum(self.start[:, 1], self.end[:, 1]) <= segment_max[1])
        )
        edge_start = self.start[candidate]
        edge_direction = self.direction[candidate]
        edge_length = self.length[candidate]
        parameters: list[float] = [0.0, 1.0]
        denominator = difference[0] * edge_direction[:, 1] - difference[1] * edge_direction[:, 0]
        offset = edge_start - start
        determinant_floor = _GEOMETRY_EPS_FACTOR * np.finfo(np.float64).eps * segment_length * edge_length
        nonparallel = np.abs(denominator) > determinant_floor
        if np.any(nonparallel):
            den = denominator[nonparallel]
            off = offset[nonparallel]
            direction = edge_direction[nonparallel]
            t = (off[:, 0] * direction[:, 1] - off[:, 1] * direction[:, 0]) / den
            u = (off[:, 0] * difference[1] - off[:, 1] * difference[0]) / den
            edge_tolerance = tolerance / np.maximum(edge_length[nonparallel], np.finfo(float).tiny)
            accepted = (
                (t >= -tolerance / segment_length)
                & (t <= 1.0 + tolerance / segment_length)
                & (u >= -edge_tolerance)
                & (u <= 1.0 + edge_tolerance)
            )
            parameters.extend(np.clip(t[accepted], 0.0, 1.0).tolist())
        parallel = ~nonparallel
        if np.any(parallel):
            off = offset[parallel]
            direction = edge_direction[parallel]
            collinear = np.abs(
                off[:, 0] * difference[1] - off[:, 1] * difference[0]
            ) <= tolerance * segment_length
            if np.any(collinear):
                col_start = edge_start[parallel][collinear]
                col_direction = direction[collinear]
                denominator_t = float(difference @ difference)
                t0 = (col_start - start) @ difference / denominator_t
                t1 = (col_start + col_direction - start) @ difference / denominator_t
                accepted = (
                    np.maximum(t0, t1) >= -tolerance / segment_length
                ) & (np.minimum(t0, t1) <= 1.0 + tolerance / segment_length)
                if np.any(accepted):
                    parameters.extend(
                        np.clip(
                            np.column_stack((t0[accepted], t1[accepted])).reshape(-1),
                            0.0,
                            1.0,
                        ).tolist()
                    )
        unique = np.unique(np.asarray(parameters, dtype=np.float64))
        for lower, upper in zip(unique[:-1], unique[1:]):
            if upper <= lower:
                continue
            midpoint = start + 0.5 * (lower + upper) * difference
            if self._location(midpoint, tolerance) == 0:
                return False
        return True


def _build_union_visibility_graph(
    coordinates: np.ndarray,
    union_triangles: np.ndarray,
    delta: float,
    visibility_tolerance: float,
    material_count: int,
    reference_material_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the union graph without calling Stage 1's material predicate."""

    polygon = _boundary_polygon_from_triangles(coordinates, union_triangles)
    predicate = _UnionPolygonPredicate(polygon, visibility_tolerance)
    pairs = _all_strict_pairs(coordinates, delta)
    visible_flags = np.asarray(
        [predicate(coordinates[i], coordinates[j]) for i, j in pairs], dtype=bool
    )
    visible_pairs = pairs[visible_flags]
    removed_pairs = pairs[~visible_flags]
    candidate_edges = _directed_edges_from_pairs(pairs)
    visible_edges = _directed_edges_from_pairs(visible_pairs)
    removed_edges = _directed_edges_from_pairs(removed_pairs)
    _strict_graph_validation(coordinates, candidate_edges, delta, "union candidate_edges")
    _strict_graph_validation(coordinates, visible_edges, delta, "union directed_edges")
    if len(removed_edges):
        _strict_graph_validation(coordinates, removed_edges, delta, "union removed_edges")
    induced = visible_edges[
        (visible_edges[:, 0] < material_count)
        & (visible_edges[:, 1] < material_count)
    ]
    if not np.array_equal(induced, reference_material_edges):
        raise ValueError(
            "independent union visibility changed the induced material d048 graph"
        )
    neighbor_counts = np.bincount(
        visible_edges[:, 0], minlength=len(coordinates)
    ).astype(np.int64, copy=False)
    if np.any(neighbor_counts <= 0):
        bad = np.flatnonzero(neighbor_counts <= 0)
        raise ValueError(
            "union visibility leaves particles without outgoing neighbors: "
            f"{bad[:10].tolist()}"
        )
    return polygon, candidate_edges, visible_edges, removed_edges, neighbor_counts


@dataclass(frozen=True)
class UnionGripGeometry:
    """Immutable material-prefix-preserving A2 union geometry and graph."""

    delta: float
    h_g: float
    collar_offsets: np.ndarray
    visibility_tolerance: float
    n_material: int
    coordinates: np.ndarray
    material_indices: np.ndarray
    material_reference_coordinates: np.ndarray
    material_reference_triangles: np.ndarray
    material_reference_volumes: np.ndarray
    material_reference_edges: np.ndarray
    material_triangles: np.ndarray
    grip_triangles: np.ndarray
    union_triangles: np.ndarray
    material_triangle_areas: np.ndarray
    grip_triangle_areas: np.ndarray
    union_triangle_areas: np.ndarray
    material_volumes: np.ndarray
    grip_volumes: np.ndarray
    union_volumes: np.ndarray
    material_first_moments: np.ndarray
    grip_first_moments: np.ndarray
    union_first_moments: np.ndarray
    boundary_polygon: np.ndarray
    candidate_edges: np.ndarray
    directed_edges: np.ndarray
    removed_edges: np.ndarray
    neighbor_counts: np.ndarray
    material_bottom_mask: np.ndarray
    material_top_mask: np.ndarray
    interface_bottom_mask: np.ndarray
    interface_top_mask: np.ndarray
    interface_mask: np.ndarray
    outer_bottom_mask: np.ndarray
    outer_top_mask: np.ndarray
    material_bottom_indices: np.ndarray
    material_top_indices: np.ndarray
    outer_bottom_indices: np.ndarray
    outer_top_indices: np.ndarray

    def __post_init__(self) -> None:
        delta = _scalar_float(self.delta, "geometry.delta")
        h_g = _scalar_float(self.h_g, "geometry.h_g")
        if delta <= 0.0 or h_g <= 0.0 or h_g >= delta:
            raise ValueError("geometry delta/h_g must satisfy 0 < h_g < delta")
        if not isinstance(self.n_material, Integral) or isinstance(self.n_material, bool):
            raise ValueError("geometry.n_material must be an integer")
        n_material = int(self.n_material)
        coordinates = _require_float64(self.coordinates, "geometry.coordinates")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) <= n_material:
            raise ValueError("geometry.coordinates must have shape (N, 2) with a nonempty collar")
        if len(np.unique(coordinates, axis=0)) != len(coordinates):
            raise ValueError("union coordinates must be unique")
        reference_coordinates = _require_float64(
            self.material_reference_coordinates,
            "geometry.material_reference_coordinates",
            (n_material, 2),
        )
        reference_triangles = _require_int64(
            self.material_reference_triangles, "geometry.material_reference_triangles"
        )
        reference_volumes = _require_float64(
            self.material_reference_volumes,
            "geometry.material_reference_volumes",
            (n_material,),
        )
        reference_edges = _require_int64(
            self.material_reference_edges, "geometry.material_reference_edges"
        )
        if not np.array_equal(coordinates[:n_material], reference_coordinates):
            raise ValueError("union coordinate prefix is not exactly the material cloud")
        if not np.array_equal(self.material_triangles, reference_triangles):
            raise ValueError("material triangles were changed in the union geometry")
        _strict_graph_validation(reference_coordinates, reference_edges, delta, "material reference edges")
        if not np.array_equal(
            np.asarray(self.directed_edges)[
                (np.asarray(self.directed_edges)[:, 0] < n_material)
                & (np.asarray(self.directed_edges)[:, 1] < n_material)
            ],
            reference_edges,
        ):
            raise ValueError("union graph induced material edges differ from d048")

        array_specs = (
            ("collar_offsets", _FLOAT64),
            ("material_indices", _INT64),
            ("material_triangles", _INT64),
            ("grip_triangles", _INT64),
            ("union_triangles", _INT64),
            ("material_triangle_areas", _FLOAT64),
            ("grip_triangle_areas", _FLOAT64),
            ("union_triangle_areas", _FLOAT64),
            ("material_volumes", _FLOAT64),
            ("grip_volumes", _FLOAT64),
            ("union_volumes", _FLOAT64),
            ("material_first_moments", _FLOAT64),
            ("grip_first_moments", _FLOAT64),
            ("union_first_moments", _FLOAT64),
            ("boundary_polygon", _FLOAT64),
            ("candidate_edges", _INT64),
            ("directed_edges", _INT64),
            ("removed_edges", _INT64),
            ("neighbor_counts", _INT64),
            ("material_bottom_indices", _INT64),
            ("material_top_indices", _INT64),
            ("outer_bottom_indices", _INT64),
            ("outer_top_indices", _INT64),
        )
        for name, dtype in array_specs:
            array = np.asarray(getattr(self, name))
            if array.dtype != dtype:
                raise ValueError(f"geometry.{name} must use dtype {dtype}, got {array.dtype}")
            if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
                raise ValueError(f"geometry.{name} contains non-finite values")
            object.__setattr__(self, name, _immutable(array))
        mask_specs = (
            "material_bottom_mask",
            "material_top_mask",
            "interface_bottom_mask",
            "interface_top_mask",
            "interface_mask",
            "outer_bottom_mask",
            "outer_top_mask",
        )
        for name in mask_specs:
            mask = _require_bool(getattr(self, name), f"geometry.{name}", (len(coordinates),))
            object.__setattr__(self, name, _immutable(mask))
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "h_g", h_g)
        collar_offsets = self.collar_offsets
        if collar_offsets.ndim != 1 or len(collar_offsets) == 0:
            raise ValueError("geometry.collar_offsets must be nonempty and one-dimensional")
        expected_offsets: list[float] = []
        offset_index = 1
        while offset_index * h_g < delta:
            expected_offsets.append(float(offset_index * h_g))
            offset_index += 1
        expected_offsets.append(delta)
        if not np.array_equal(collar_offsets, np.asarray(expected_offsets, dtype=np.float64)):
            raise ValueError("geometry.collar_offsets do not match the executable h_g rule")
        object.__setattr__(self, "collar_offsets", _immutable(collar_offsets))
        object.__setattr__(self, "visibility_tolerance", _scalar_float(self.visibility_tolerance, "visibility_tolerance"))
        object.__setattr__(self, "n_material", n_material)
        object.__setattr__(self, "coordinates", _immutable(coordinates))
        object.__setattr__(self, "material_reference_coordinates", _immutable(reference_coordinates))
        object.__setattr__(self, "material_reference_triangles", _immutable(reference_triangles))
        object.__setattr__(self, "material_reference_volumes", _immutable(reference_volumes))
        object.__setattr__(self, "material_reference_edges", _immutable(reference_edges))

        if not np.array_equal(self.material_indices, np.arange(n_material, dtype=np.int64)):
            raise ValueError("geometry.material_indices must be the original prefix")
        if not np.array_equal(self.union_triangles, np.vstack((self.material_triangles, self.grip_triangles))):
            raise ValueError("union triangles are not material-prefix followed by grip triangles")
        if len(np.unique(np.sort(self.union_triangles, axis=1), axis=0)) != len(self.union_triangles):
            raise ValueError("union phase triangles contain duplicates")
        for name, array in (
            ("material_volumes", self.material_volumes),
            ("grip_volumes", self.grip_volumes),
            ("union_volumes", self.union_volumes),
        ):
            if array.shape != (len(coordinates),) or np.any(array < 0.0) or not np.all(np.isfinite(array)):
                raise ValueError(f"geometry.{name} must be finite and shape (N,)")
        if np.any(self.material_volumes[:n_material] <= 0.0):
            raise ValueError("all original material particles must have positive material volume")
        if np.any(self.grip_volumes[n_material:] <= 0.0):
            raise ValueError("all collar particles must have positive grip volume")
        if not np.array_equal(self.material_volumes[:n_material], reference_volumes):
            raise ValueError("copied material volumes are not bitwise identical")
        if not np.allclose(self.material_volumes[n_material:], 0.0, rtol=0.0, atol=0.0):
            raise ValueError("new collar particles must have zero material volume")
        volume_tolerance = _AREA_RECONSTRUCTION_EPS_FACTOR * np.finfo(np.float64).eps * max(
            1.0, float(np.max(self.union_volumes))
        )
        if not np.allclose(
            self.union_volumes,
            self.material_volumes + self.grip_volumes,
            rtol=0.0,
            atol=volume_tolerance,
        ):
            raise ValueError("union volumes are not the exact phase-volume sum")
        if np.any(self.union_volumes <= 0.0):
            raise ValueError("union volumes must be strictly positive")
        phase_data = (
            (self.material_triangles, self.material_triangle_areas, "material"),
            (self.grip_triangles, self.grip_triangle_areas, "grip"),
            (self.union_triangles, self.union_triangle_areas, "union"),
        )
        for triangles, areas, name in phase_data:
            if areas.shape != (len(triangles),) or np.any(areas <= 0.0):
                raise ValueError(f"{name} triangle areas are inconsistent")
            _, volumes, moments = _triangle_quadrature(coordinates, triangles, name + " triangles")
            expected_volumes = self.material_volumes if name == "material" else self.grip_volumes if name == "grip" else self.union_volumes
            if not np.array_equal(areas, _triangle_quadrature(coordinates, triangles, name + " area check")[0]):
                raise ValueError(f"{name} triangle areas changed from coordinates")
            if not np.allclose(volumes, expected_volumes, rtol=0.0, atol=volume_tolerance):
                raise ValueError(f"{name} phase volumes do not match its triangles")
            expected_moments = self.material_first_moments if name == "material" else self.grip_first_moments if name == "grip" else self.union_first_moments
            if not np.allclose(moments, expected_moments, rtol=0.0, atol=volume_tolerance):
                raise ValueError(f"{name} first moments do not match its volumes")
        if not np.array_equal(self.union_triangle_areas, np.concatenate((self.material_triangle_areas, self.grip_triangle_areas))):
            raise ValueError("union triangle areas are not phase-concatenated")
        if not np.allclose(
            self.union_first_moments,
            self.material_first_moments + self.grip_first_moments,
            rtol=0.0,
            atol=volume_tolerance,
        ):
            raise ValueError("union first moments are not the phase sum")

        for name in ("candidate_edges", "directed_edges", "removed_edges"):
            values = getattr(self, name)
            if name == "removed_edges" and len(values) == 0:
                if values.shape != (0, 2):
                    raise ValueError("empty geometry.removed_edges must have shape (0, 2)")
            else:
                _strict_graph_validation(coordinates, values, delta, "geometry." + name)
        candidate_set = set(map(tuple, self.candidate_edges.tolist()))
        visible_set = set(map(tuple, self.directed_edges.tolist()))
        removed_set = set(map(tuple, self.removed_edges.tolist()))
        if not visible_set.issubset(candidate_set):
            raise ValueError("visible union edges are not a subset of candidates")
        if visible_set & removed_set:
            raise ValueError("visible and removed union edges overlap")
        if visible_set | removed_set != candidate_set:
            raise ValueError("visible and removed union edges are not complete over candidates")
        if self.neighbor_counts.shape != (len(coordinates),) or np.any(self.neighbor_counts <= 0):
            raise ValueError("union neighbor counts must be positive for every particle")
        if not np.array_equal(
            self.neighbor_counts,
            np.bincount(self.directed_edges[:, 0], minlength=len(coordinates)).astype(np.int64),
        ):
            raise ValueError("union neighbor counts do not match the graph")
        if self.boundary_polygon.ndim != 2 or self.boundary_polygon.shape[1] != 2:
            raise ValueError("geometry.boundary_polygon must have shape (B, 2)")
        expected_boundary = _boundary_polygon_from_triangles(coordinates, self.union_triangles)
        if not np.array_equal(self.boundary_polygon, expected_boundary):
            raise ValueError("boundary polygon ownership/order does not match union triangles")
        object.__setattr__(self, "boundary_polygon", _immutable(self.boundary_polygon))
        index_specs = (
            ("material_bottom_indices", self.material_bottom_indices),
            ("material_top_indices", self.material_top_indices),
            ("outer_bottom_indices", self.outer_bottom_indices),
            ("outer_top_indices", self.outer_top_indices),
        )
        for name, indices in index_specs:
            if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(coordinates)):
                raise ValueError(f"geometry.{name} contains invalid/duplicate indices")
        expected_material_bottom = np.zeros(len(coordinates), dtype=bool)
        expected_material_bottom[self.material_bottom_indices] = True
        expected_material_top = np.zeros(len(coordinates), dtype=bool)
        expected_material_top[self.material_top_indices] = True
        expected_outer_bottom = np.zeros(len(coordinates), dtype=bool)
        expected_outer_bottom[self.outer_bottom_indices] = True
        expected_outer_top = np.zeros(len(coordinates), dtype=bool)
        expected_outer_top[self.outer_top_indices] = True
        for name, actual, expected in (
            ("material_bottom_mask", self.material_bottom_mask, expected_material_bottom),
            ("material_top_mask", self.material_top_mask, expected_material_top),
            ("interface_bottom_mask", self.interface_bottom_mask, expected_material_bottom),
            ("interface_top_mask", self.interface_top_mask, expected_material_top),
            ("interface_mask", self.interface_mask, expected_material_bottom | expected_material_top),
            ("outer_bottom_mask", self.outer_bottom_mask, expected_outer_bottom),
            ("outer_top_mask", self.outer_top_mask, expected_outer_top),
        ):
            if not np.array_equal(actual, expected):
                raise ValueError(f"geometry.{name} does not match its explicit particle indices")

    @property
    def n_particles(self) -> int:
        return int(len(self.coordinates))

    @property
    def n_dofs(self) -> int:
        return 2 * self.n_particles

    @property
    def material_area(self) -> float:
        return float(np.sum(self.material_volumes))

    @property
    def grip_area(self) -> float:
        return float(np.sum(self.grip_volumes))

    @property
    def bottom_grip_area(self) -> float:
        count = 2 * (len(self.material_bottom_indices) - 1) * len(self.collar_offsets)
        return float(np.sum(self.grip_triangle_areas[:count]))

    @property
    def top_grip_area(self) -> float:
        count = 2 * (len(self.material_bottom_indices) - 1) * len(self.collar_offsets)
        return float(np.sum(self.grip_triangle_areas[count:]))

    @property
    def offsets(self) -> np.ndarray:
        return self.collar_offsets

    @property
    def union_area(self) -> float:
        return float(np.sum(self.union_volumes))

    @property
    def shape_condition_summary(self) -> Mapping[str, float]:
        return {}


def _validate_interface_coordinates(
    coordinates: np.ndarray,
    mask: np.ndarray,
    expected_y: float,
    name: str,
) -> np.ndarray:
    indices = np.flatnonzero(mask).astype(np.int64)
    if len(indices) < 2:
        raise ValueError(f"{name} must contain at least two ordered particles")
    values = coordinates[indices]
    if np.any(np.abs(values[:, 1] - expected_y) > _geometry_tolerance(coordinates)):
        raise ValueError(f"{name} particles are not on y={expected_y}")
    order = np.lexsort((values[:, 1], values[:, 0]))
    indices = indices[order]
    x = coordinates[indices, 0]
    if np.any(np.diff(x) <= 0.0) or len(np.unique(x)) != len(x):
        raise ValueError(f"{name} x coordinates must be strictly ordered and unique")
    spacing = np.diff(x)
    median = float(np.median(spacing))
    tolerance = _geometry_tolerance(coordinates)
    if np.any(np.abs(spacing - median) > tolerance):
        raise ValueError(
            f"{name} spacing is not uniform within the declared float64 geometry tolerance"
        )
    return indices


def _structured_grip_triangles(
    coordinates: np.ndarray,
    interface_indices: np.ndarray,
    offsets: np.ndarray,
    sign: float,
    new_coordinates: list[list[float]],
    name: str,
) -> tuple[np.ndarray, list[np.ndarray]]:
    interface = coordinates[interface_indices]
    rows: list[np.ndarray] = [interface_indices]
    for offset in offsets:
        indices = np.arange(
            len(new_coordinates), len(new_coordinates) + len(interface), dtype=np.int64
        )
        row = interface + np.asarray([0.0, sign * float(offset)], dtype=np.float64)
        new_coordinates.extend(row.tolist())
        rows.append(indices)
    triangles: list[list[int]] = []
    for row_index in range(len(rows) - 1):
        first = rows[row_index]
        second = rows[row_index + 1]
        if sign < 0.0:
            lower, upper = second, first
        else:
            lower, upper = first, second
        for segment in range(len(interface) - 1):
            ll, lr = int(lower[segment]), int(lower[segment + 1])
            ul, ur = int(upper[segment]), int(upper[segment + 1])
            if segment % 2 == 0:
                triangles.extend(((ll, lr, ur), (ll, ur, ul)))
            else:
                triangles.extend(((ll, lr, ul), (lr, ur, ul)))
    result = np.asarray(triangles, dtype=np.int64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} grip strip produced no triangles")
    return result, rows[1:]


def build_union_geometry_from_arrays(
    coordinates: Any,
    triangles: Any,
    volumes: Any,
    bottom_mask: Any,
    top_mask: Any,
    anchor_particle: Any,
    directed_edges: Any,
    delta: Any,
    h_g: Any,
    visibility_tolerance: Any,
) -> UnionGripGeometry:
    """Build one exact-prefix A2 union from explicit material arrays.

    This public array constructor is also the manufactured-cloud entry point.
    ``directed_edges`` is the unchanged material reference graph; for the
    production call it is exactly the serialized ``d048`` graph.
    """

    coords = _require_float64(coordinates, "material coordinates")
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0:
        raise ValueError("material coordinates must have nonempty shape (N, 2)")
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError("material coordinates must be unique")
    material_triangles = _require_int64(triangles, "material triangles")
    material_volumes_input = _require_float64(volumes, "material volumes", (len(coords),))
    if np.any(material_volumes_input <= 0.0):
        raise ValueError("material volumes must be strictly positive")
    material_bottom = _require_bool(bottom_mask, "material bottom mask", (len(coords),))
    material_top = _require_bool(top_mask, "material top mask", (len(coords),))
    if not np.any(material_bottom) or not np.any(material_top) or np.any(material_bottom & material_top):
        raise ValueError("material top/bottom masks must be nonempty and disjoint")
    if not isinstance(anchor_particle, Integral) or isinstance(anchor_particle, bool):
        raise ValueError("anchor_particle must be an integer")
    anchor = int(anchor_particle)
    if anchor < 0 or anchor >= len(coords):
        raise ValueError("anchor_particle is outside the material particle range")
    if not np.array_equal(coords[anchor], np.asarray([0.0, 0.0], dtype=np.float64)):
        raise ValueError("the original anchor must be exactly the material particle (0, 0)")
    delta_value = _scalar_float(delta, "delta")
    h_value = _scalar_float(h_g, "h_g")
    if delta_value <= 0.0 or h_value <= 0.0 or h_value >= delta_value:
        raise ValueError("delta/h_g must satisfy 0 < h_g < delta")
    edges = _require_int64(directed_edges, "material reference edges")
    _strict_graph_validation(coords, edges, delta_value, "material reference edges")
    bottom_indices = _validate_interface_coordinates(coords, material_bottom, 0.0, "material bottom interface")
    top_indices = _validate_interface_coordinates(coords, material_top, 1.0, "material top interface")

    offsets_list: list[float] = []
    k = 1
    while k * h_value < delta_value:
        offsets_list.append(float(k * h_value))
        k += 1
    offsets_list.append(delta_value)
    offsets = np.asarray(offsets_list, dtype=np.float64)
    if len(np.unique(offsets)) != len(offsets) or np.any(offsets <= 0.0) or np.any(offsets > delta_value):
        raise ValueError("collar offsets must be unique, positive, and end at exact delta")

    new_coordinates: list[list[float]] = coords.tolist()
    bottom_grip_triangles, bottom_rows = _structured_grip_triangles(
        coords, bottom_indices, offsets, -1.0, new_coordinates, "bottom"
    )
    top_grip_triangles, top_rows = _structured_grip_triangles(
        coords, top_indices, offsets, 1.0, new_coordinates, "top"
    )
    grip_triangles = np.vstack((bottom_grip_triangles, top_grip_triangles)).astype(np.int64, copy=False)
    union_coordinates = np.asarray(new_coordinates, dtype=np.float64)
    if not np.array_equal(union_coordinates[: len(coords)], coords):
        raise ValueError("structured collar construction changed the material coordinate prefix")
    union_triangles = np.vstack((material_triangles, grip_triangles)).astype(np.int64, copy=False)
    if len(np.unique(np.sort(union_triangles, axis=1), axis=0)) != len(union_triangles):
        raise ValueError("structured collar construction produced duplicate triangles")

    material_areas, material_phase_volumes, material_moments = _triangle_quadrature(
        union_coordinates, material_triangles, "material triangles"
    )
    grip_areas, grip_phase_volumes, grip_moments = _triangle_quadrature(
        union_coordinates, grip_triangles, "grip triangles"
    )
    union_areas, union_phase_volumes, union_moments = _triangle_quadrature(
        union_coordinates, union_triangles, "union triangles"
    )
    if not np.array_equal(material_phase_volumes[: len(coords)], material_volumes_input):
        raise ValueError("material triangle reconstruction differs from serialized material volumes")
    if not np.allclose(material_phase_volumes[len(coords) :], 0.0, rtol=0.0, atol=0.0):
        raise ValueError("material phase has volume on a non-material collar particle")
    if not np.array_equal(material_phase_volumes, np.concatenate((material_volumes_input, np.zeros(len(union_coordinates) - len(coords), dtype=np.float64)))):
        raise ValueError("material phase volume storage is not an exact material prefix")
    phase_tolerance = _AREA_RECONSTRUCTION_EPS_FACTOR * np.finfo(np.float64).eps * max(
        1.0, float(np.max(union_phase_volumes))
    )
    if not np.allclose(
        union_phase_volumes,
        material_phase_volumes + grip_phase_volumes,
        rtol=0.0,
        atol=phase_tolerance,
    ):
        raise ValueError("union quadrature does not equal the two phase measures")
    if not np.isclose(
        float(np.sum(union_phase_volumes)),
        float(np.sum(material_volumes_input) + 2.0 * delta_value),
        rtol=0.0,
        atol=_AREA_RECONSTRUCTION_EPS_FACTOR * np.finfo(np.float64).eps * max(1.0, delta_value),
    ):
        raise ValueError("union area does not equal material area plus two full-width grips")
    if not np.isclose(
        float(np.sum(grip_phase_volumes)), 2.0 * delta_value,
        rtol=0.0,
        atol=_AREA_RECONSTRUCTION_EPS_FACTOR * np.finfo(np.float64).eps * max(1.0, delta_value),
    ):
        raise ValueError("each full-width grip pair must have total area 2*delta")
    bottom_grip_count = 2 * (len(bottom_indices) - 1) * len(offsets)
    bottom_grip_area = float(np.sum(grip_areas[:bottom_grip_count]))
    top_grip_area = float(np.sum(grip_areas[bottom_grip_count:]))
    grip_area_tolerance = _AREA_RECONSTRUCTION_EPS_FACTOR * np.finfo(np.float64).eps * max(
        1.0, delta_value
    )
    if abs(bottom_grip_area - delta_value) > grip_area_tolerance or abs(top_grip_area - delta_value) > grip_area_tolerance:
        raise ValueError(
            "bottom/top structured grip areas do not each equal exact delta: "
            f"bottom={bottom_grip_area:.17g}, top={top_grip_area:.17g}, delta={delta_value:.17g}"
        )

    material_count = len(coords)
    polygon, candidate_edges, union_edges, removed_edges, neighbor_counts = _build_union_visibility_graph(
        union_coordinates,
        union_triangles,
        delta_value,
        _scalar_float(visibility_tolerance, "visibility_tolerance"),
        material_count,
        edges,
    )

    n_union = len(union_coordinates)
    material_volume_full = np.asarray(material_phase_volumes, dtype=np.float64)
    grip_volume_full = np.asarray(grip_phase_volumes, dtype=np.float64)
    interface_bottom_mask = np.zeros(n_union, dtype=bool)
    interface_bottom_mask[bottom_indices] = True
    interface_top_mask = np.zeros(n_union, dtype=bool)
    interface_top_mask[top_indices] = True
    interface_mask = interface_bottom_mask | interface_top_mask
    outer_bottom_indices = np.asarray(bottom_rows[-1], dtype=np.int64)
    outer_top_indices = np.asarray(top_rows[-1], dtype=np.int64)
    outer_bottom_mask = np.zeros(n_union, dtype=bool)
    outer_bottom_mask[outer_bottom_indices] = True
    outer_top_mask = np.zeros(n_union, dtype=bool)
    outer_top_mask[outer_top_indices] = True
    if np.any(outer_bottom_mask & outer_top_mask) or np.any(interface_mask & (outer_bottom_mask | outer_top_mask)):
        raise ValueError("interface and outer-face masks must be disjoint")

    return UnionGripGeometry(
        delta=delta_value,
        h_g=h_value,
        collar_offsets=offsets,
        visibility_tolerance=_scalar_float(visibility_tolerance, "visibility_tolerance"),
        n_material=material_count,
        coordinates=union_coordinates,
        material_indices=np.arange(material_count, dtype=np.int64),
        material_reference_coordinates=coords,
        material_reference_triangles=material_triangles,
        material_reference_volumes=material_volumes_input,
        material_reference_edges=edges,
        material_triangles=material_triangles,
        grip_triangles=grip_triangles,
        union_triangles=union_triangles,
        material_triangle_areas=material_areas,
        grip_triangle_areas=grip_areas,
        union_triangle_areas=union_areas,
        material_volumes=material_volume_full,
        grip_volumes=grip_volume_full,
        union_volumes=union_phase_volumes,
        material_first_moments=material_moments,
        grip_first_moments=grip_moments,
        union_first_moments=union_moments,
        boundary_polygon=polygon,
        candidate_edges=candidate_edges,
        directed_edges=union_edges,
        removed_edges=removed_edges,
        neighbor_counts=neighbor_counts,
        material_bottom_mask=np.pad(material_bottom, (0, n_union - material_count)),
        material_top_mask=np.pad(material_top, (0, n_union - material_count)),
        interface_bottom_mask=interface_bottom_mask,
        interface_top_mask=interface_top_mask,
        interface_mask=interface_mask,
        outer_bottom_mask=outer_bottom_mask,
        outer_top_mask=outer_top_mask,
        material_bottom_indices=bottom_indices,
        material_top_indices=top_indices,
        outer_bottom_indices=outer_bottom_indices,
        outer_top_indices=outer_top_indices,
    )


def build_union_geometry(
    cloud: ParticleCloud,
    h_g: Any,
    *,
    visibility_tolerance: Any,
    delta: Any,
) -> UnionGripGeometry:
    """Build A2 from a Stage 1 cloud while preserving every material field."""

    if not isinstance(cloud, ParticleCloud):
        raise TypeError("cloud must be a ParticleCloud")
    delta_value = _scalar_float(delta, "delta")
    if delta_value != _DELTA_D048:
        raise ValueError("Candidate A2 production geometry requires delta exactly 0.048")
    if _scalar_float(cloud.horizon, "cloud.horizon") != delta_value:
        raise ValueError("the Stage 1 cloud horizon must equal A2 delta exactly")
    if not np.array_equal(cloud.visible_edges, np.asarray(cloud.visible_edges, dtype=np.int64)):
        raise ValueError("cloud visible edges are not exact int64 material data")
    return build_union_geometry_from_arrays(
        cloud.coordinates,
        cloud.triangles,
        cloud.volumes,
        cloud.boundary.bottom,
        cloud.boundary.top,
        cloud.boundary.anchor_particle,
        cloud.visible_edges,
        delta_value,
        h_g,
        visibility_tolerance,
    )


@dataclass(frozen=True)
class IndependentUnionBAReference:
    """Independently reconstructed union BA support and family data."""

    source: np.ndarray
    target: np.ndarray
    distance: np.ndarray
    omega: np.ndarray
    family_mass: np.ndarray
    support_indices: tuple[np.ndarray, ...]
    beta: tuple[np.ndarray, ...]
    xi: tuple[np.ndarray, ...]
    shape_inverse: tuple[np.ndarray, ...]
    eta: np.ndarray
    shape_rank: np.ndarray
    condition_numbers: np.ndarray
    material_damage: np.ndarray
    material_degradation: np.ndarray
    grip_degradation: float
    phase_center_weights: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "source",
            "target",
            "distance",
            "omega",
            "family_mass",
            "eta",
            "shape_rank",
            "condition_numbers",
            "material_damage",
            "material_degradation",
            "phase_center_weights",
        ):
            array = np.asarray(getattr(self, name))
            if name in {"source", "target", "shape_rank"}:
                if array.dtype != _INT64:
                    raise ValueError(f"reference.{name} must be int64")
            else:
                if array.dtype != _FLOAT64:
                    raise ValueError(f"reference.{name} must be float64")
            object.__setattr__(self, name, _immutable(array))
        grip_degradation = _scalar_float(self.grip_degradation, "reference.grip_degradation")
        if grip_degradation <= 0.0:
            raise ValueError("reference.grip_degradation must be positive")
        object.__setattr__(self, "grip_degradation", grip_degradation)
        object.__setattr__(
            self,
            "support_indices",
            tuple(_immutable(value, _INT64) for value in self.support_indices),
        )
        object.__setattr__(
            self,
            "beta",
            tuple(_immutable(value, _FLOAT64) for value in self.beta),
        )
        object.__setattr__(
            self,
            "xi",
            tuple(_immutable(value, _FLOAT64) for value in self.xi),
        )
        object.__setattr__(
            self,
            "shape_inverse",
            tuple(_immutable(value, _FLOAT64) for value in self.shape_inverse),
        )
        if len(self.source) != len(self.target) or len(self.source) != len(self.eta):
            raise ValueError("reference edge arrays have incompatible lengths")
        if len(self.support_indices) != len(self.source) or len(self.beta) != len(self.source) or len(self.xi) != len(self.source) or len(self.shape_inverse) != len(self.source):
            raise ValueError("reference support arrays have incompatible lengths")


def independent_union_ba_reference(
    geometry: UnionGripGeometry,
    material_damage: Any,
    k_res: Any,
    rank_rtol: Any,
) -> IndependentUnionBAReference:
    """Rebuild union BA data and phase weights without using mechanics state.

    The phase-weight portion is intentionally reconstructed from the immutable
    geometry phase measures, the raw material damage, and ``k_res``.  In
    particular, this reference never reads ``mechanics.center_weights`` or any
    other mechanics-derived phase quantity.
    """

    if not isinstance(geometry, UnionGripGeometry):
        raise TypeError("geometry must be a UnionGripGeometry")
    rank_value = _scalar_float(rank_rtol, "rank_rtol")
    if rank_value <= 0.0 or rank_value >= 1.0:
        raise ValueError("rank_rtol must satisfy 0 < rank_rtol < 1")
    damage = _validate_material_damage(material_damage, geometry.n_material)
    k_res_value = _scalar_float(k_res, "k_res")
    if k_res_value < 0.0:
        raise ValueError("k_res must be non-negative")
    material_degradation = (1.0 - damage) ** 2 + k_res_value
    grip_degradation = 1.0 + k_res_value
    material_degradation_full = np.pad(
        material_degradation,
        (0, geometry.n_particles - geometry.n_material),
    )
    phase_center_weights = np.asarray(
        geometry.material_volumes * material_degradation_full
        + geometry.grip_volumes * grip_degradation,
        dtype=np.float64,
    )
    if np.any(~np.isfinite(phase_center_weights)) or np.any(phase_center_weights <= 0.0):
        raise FloatingPointError("independent phase center weights are invalid")
    edges = geometry.directed_edges
    source = edges[:, 0]
    target = edges[:, 1]
    distance = np.linalg.norm(
        geometry.coordinates[target] - geometry.coordinates[source], axis=1
    )
    omega = 120.0 / (np.pi * geometry.delta**4) * (1.0 - distance / geometry.delta) ** 2
    if np.any(~np.isfinite(omega)) or np.any(omega <= 0.0):
        raise ValueError("independent union kernel weights are invalid")
    family_mass = np.bincount(
        source,
        weights=omega * geometry.union_volumes[target],
        minlength=geometry.n_particles,
    ).astype(np.float64, copy=False)
    if np.any(~np.isfinite(family_mass)) or np.any(family_mass <= 0.0):
        raise ValueError("independent union family masses are invalid")
    adjacency: list[set[int]] = [set() for _ in range(geometry.n_particles)]
    weights: dict[tuple[int, int], float] = {}
    for (i_raw, j_raw), value in zip(edges, omega):
        i, j = int(i_raw), int(j_raw)
        adjacency[i].add(j)
        weights[i, j] = float(value)
    supports: list[np.ndarray] = []
    betas: list[np.ndarray] = []
    xis: list[np.ndarray] = []
    inverses: list[np.ndarray] = []
    ranks: list[int] = []
    conditions: list[float] = []
    eta = np.empty(len(edges), dtype=np.float64)
    for edge_index, (i_raw, j_raw) in enumerate(edges):
        i, j = int(i_raw), int(j_raw)
        common = sorted(adjacency[i].intersection(adjacency[j]) - {i, j})
        support = np.asarray([j, *common], dtype=np.int64)
        beta = np.asarray(
            [weights[i, j] * geometry.union_volumes[j]]
            + [weights[i, k] * geometry.union_volumes[k] for k in common],
            dtype=np.float64,
        )
        xi = geometry.coordinates[support] - geometry.coordinates[i]
        shape = np.einsum("s,sa,sb->ab", beta, xi, xi, optimize=True)
        singular_values = np.linalg.svd(shape, compute_uv=False)
        threshold = rank_value * float(singular_values[0])
        rank = int(np.count_nonzero(singular_values > threshold))
        if rank < 2:
            raise ValueError(
                "independent union BA support is rank-deficient at edge "
                f"{edge_index} ({i}, {j})"
            )
        condition = float(singular_values[0] / singular_values[-1])
        if not np.isfinite(condition):
            raise ValueError("independent union BA condition number is non-finite")
        supports.append(support)
        betas.append(beta)
        xis.append(xi)
        inverses.append(np.linalg.inv(shape))
        ranks.append(rank)
        conditions.append(condition)
        eta[edge_index] = beta[0] / family_mass[i]
    if np.any(~np.isfinite(eta)) or np.any(eta <= 0.0):
        raise ValueError("independent union eta values are invalid")
    return IndependentUnionBAReference(
        source=source,
        target=target,
        distance=distance,
        omega=omega,
        family_mass=family_mass,
        support_indices=tuple(supports),
        beta=tuple(betas),
        xi=tuple(xis),
        shape_inverse=tuple(inverses),
        eta=eta,
        shape_rank=np.asarray(ranks, dtype=np.int64),
        condition_numbers=np.asarray(conditions, dtype=np.float64),
        material_damage=damage,
        material_degradation=material_degradation,
        grip_degradation=grip_degradation,
        phase_center_weights=phase_center_weights,
    )


def _validate_material_damage(
    damage: Any, n_material: int, name: str = "material_damage"
) -> np.ndarray:
    values = _require_float64(damage, name, (n_material,))
    # The raw archive is a transplanted warning diagnostic.  Range clipping or
    # rejection would change its provenance, so only finiteness is required.
    return values


class FixedDamagePhaseMeasuredMechanics(BondAssociatedMechanics):
    """Immutable phase-measured A2 fixed-damage mechanics view.

    ``BondAssociatedMechanics`` supplies the exact target-inclusive kinematics,
    constitutive map, explicit endpoint residual, and tangent implementation.
    Only its outer center factor is replaced by the A2 ``W_i*eta_ij`` factor;
    no material-only object is mutated or reinterpreted.
    """

    _CACHE_FIELDS = frozenset({"_tangent_block_diagonal_cache", "_tangent_diagonal_cache"})

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_a2_locked", False) and name not in self._CACHE_FIELDS:
            raise AttributeError(
                "FixedDamagePhaseMeasuredMechanics state is immutable; construct a new view"
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if self.__dict__.get("_a2_locked", False) and name not in self._CACHE_FIELDS:
            raise AttributeError(
                "FixedDamagePhaseMeasuredMechanics state is immutable; construct a new view"
            )
        object.__delattr__(self, name)

    def __init__(
        self,
        geometry: UnionGripGeometry,
        material_damage: Any,
        youngs_modulus: Any,
        poisson_ratio: Any,
        k_res: Any,
        rank_rtol: Any,
    ) -> None:
        if not isinstance(geometry, UnionGripGeometry):
            raise TypeError("geometry must be a UnionGripGeometry")
        damage = _validate_material_damage(material_damage, geometry.n_material)
        super().__init__(
            geometry.coordinates,
            geometry.union_volumes,
            geometry.directed_edges,
            geometry.delta,
            youngs_modulus,
            poisson_ratio,
            k_res,
            rank_rtol,
        )
        material_degradation = (1.0 - damage) ** 2 + self.k_res
        grip_degradation = 1.0 + self.k_res
        center_weights = np.asarray(
            geometry.material_volumes * np.pad(
                material_degradation,
                (0, geometry.n_particles - geometry.n_material),
            )
            + geometry.grip_volumes * grip_degradation,
            dtype=np.float64,
        )
        # The expression above has zero material contribution on new collar
        # particles, and material/grip contributions on interface particles.
        effective = center_weights / geometry.union_volumes
        if np.any(~np.isfinite(center_weights)) or np.any(center_weights <= 0.0):
            raise FloatingPointError("A2 phase center weights are invalid")
        if np.any(~np.isfinite(effective)) or np.any(effective <= 0.0):
            raise FloatingPointError("A2 effective center degradation is invalid")
        direct_center_factor = center_weights[self.source] * self.eta
        equivalent_center_factor = (
            geometry.union_volumes[self.source] * effective[self.source] * self.eta
        )
        center_tolerance = _AREA_RECONSTRUCTION_EPS_FACTOR * np.finfo(np.float64).eps * max(
            1.0, float(np.max(np.abs(direct_center_factor)))
        )
        if not np.allclose(
            direct_center_factor,
            equivalent_center_factor,
            rtol=0.0,
            atol=center_tolerance,
        ):
            raise ValueError("A2 W/Vstar center-factor algebra is not equivalent")
        self.geometry = geometry
        self.material_damage = _immutable(damage)
        self.damage_union = _immutable(
            np.concatenate((damage, np.zeros(geometry.n_particles - geometry.n_material, dtype=np.float64)))
        )
        self.material_degradation = _immutable(material_degradation)
        self.center_weights = _immutable(center_weights)
        self.effective_center_degradation = _immutable(effective)
        self.degradation = self.effective_center_degradation
        self.edge_degradation = _immutable(effective[self.source])
        self.center_factor = _immutable(direct_center_factor)
        self.center_factor_vstar_reference = _immutable(equivalent_center_factor)
        self.grip_degradation = grip_degradation
        self._a2_locked = True

    def _effective_stress_fields(
        self,
        edge_stress: np.ndarray,
        particle_stress: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.edge_degradation[:, None, None] * edge_stress,
            self.effective_center_degradation[:, None, None] * particle_stress,
        )


@dataclass(frozen=True)
class AffineConstraints:
    """Generic immutable outer-face affine constraint contract for A2."""

    n_particles: int
    constrained_dofs: np.ndarray
    free_dofs: np.ndarray
    q_zero: np.ndarray
    load_direction: np.ndarray
    outer_bottom_mask: np.ndarray
    outer_top_mask: np.ndarray
    anchor_particle: int
    anchor_coordinate: np.ndarray
    outer_top_selector: np.ndarray
    outer_bottom_selector: np.ndarray
    anchor_selector: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.n_particles, Integral) or isinstance(self.n_particles, bool):
            raise ValueError("constraints.n_particles must be an integer")
        n_particles = int(self.n_particles)
        if n_particles <= 0:
            raise ValueError("constraints.n_particles must be positive")
        n_dofs = 2 * n_particles
        bottom = _require_bool(self.outer_bottom_mask, "outer_bottom_mask", (n_particles,))
        top = _require_bool(self.outer_top_mask, "outer_top_mask", (n_particles,))
        if not np.any(bottom) or not np.any(top) or np.any(bottom & top):
            raise ValueError("outer top/bottom masks must be nonempty and disjoint")
        if not isinstance(self.anchor_particle, Integral) or isinstance(self.anchor_particle, bool):
            raise ValueError("anchor_particle must be an integer")
        anchor = int(self.anchor_particle)
        if anchor < 0 or anchor >= n_particles:
            raise ValueError("anchor_particle is outside the union particle range")
        anchor_coordinate = _require_float64(self.anchor_coordinate, "anchor_coordinate", (2,))
        if not np.array_equal(anchor_coordinate, np.asarray([0.0, 0.0], dtype=np.float64)):
            raise ValueError("A2 anchor coordinate must remain exactly (0, 0)")
        constrained = _require_int64(self.constrained_dofs, "constrained_dofs")
        free = _require_int64(self.free_dofs, "free_dofs")
        if constrained.ndim != 1 or free.ndim != 1:
            raise ValueError("constrained_dofs/free_dofs must be one-dimensional int64")
        if len(constrained) and (np.min(constrained) < 0 or np.max(constrained) >= n_dofs):
            raise ValueError("constrained_dofs contains an out-of-range DOF")
        if len(free) and (np.min(free) < 0 or np.max(free) >= n_dofs):
            raise ValueError("free_dofs contains an out-of-range DOF")
        if len(np.unique(constrained)) != len(constrained) or len(np.unique(free)) != len(free):
            raise ValueError("constrained/free DOFs must be unique")
        if not np.array_equal(constrained, np.sort(constrained)):
            raise ValueError("constrained_dofs must be sorted")
        if np.intersect1d(constrained, free).size:
            raise ValueError("constrained/free DOFs overlap")
        if not np.array_equal(np.sort(np.concatenate((constrained, free))), np.arange(n_dofs, dtype=np.int64)):
            raise ValueError("constrained/free DOFs must partition every union DOF")
        q_zero = _require_float64(self.q_zero, "q_zero", (n_dofs,))
        load_direction = _require_float64(self.load_direction, "load_direction", (n_dofs,))
        selectors = (
            ("outer_top_selector", self.outer_top_selector),
            ("outer_bottom_selector", self.outer_bottom_selector),
            ("anchor_selector", self.anchor_selector),
        )
        for name, values in selectors:
            _require_float64(values, name, (n_dofs,))
        expected_top = np.zeros(n_dofs, dtype=np.float64)
        expected_top[2 * np.flatnonzero(top) + 1] = 1.0
        expected_bottom = np.zeros(n_dofs, dtype=np.float64)
        expected_bottom[2 * np.flatnonzero(bottom) + 1] = 1.0
        expected_anchor = np.zeros(n_dofs, dtype=np.float64)
        expected_anchor[2 * anchor] = 1.0
        expected_constrained = np.sort(
            np.concatenate((
                np.flatnonzero(expected_bottom).astype(np.int64),
                np.flatnonzero(expected_top).astype(np.int64),
                np.asarray([2 * anchor], dtype=np.int64),
            ))
        )
        if not np.array_equal(constrained, expected_constrained):
            raise ValueError("constraints contain a non-contract DOF or omit a required DOF")
        if not np.array_equal(q_zero, np.zeros(n_dofs, dtype=np.float64)):
            raise ValueError("q_zero must be exactly zero for A2")
        if not np.array_equal(load_direction, expected_top):
            raise ValueError("load_direction must be unit vertical displacement only on outer top")
        if not np.array_equal(np.asarray(self.outer_top_selector), expected_top):
            raise ValueError("outer_top_selector is inconsistent with outer_top_mask")
        if not np.array_equal(np.asarray(self.outer_bottom_selector), expected_bottom):
            raise ValueError("outer_bottom_selector is inconsistent with outer_bottom_mask")
        if not np.array_equal(np.asarray(self.anchor_selector), expected_anchor):
            raise ValueError("anchor_selector is inconsistent with anchor_particle")
        for name, values in (
            ("constrained_dofs", constrained),
            ("free_dofs", free),
            ("q_zero", q_zero),
            ("load_direction", load_direction),
            ("outer_bottom_mask", bottom),
            ("outer_top_mask", top),
            ("anchor_coordinate", anchor_coordinate),
            ("outer_top_selector", self.outer_top_selector),
            ("outer_bottom_selector", self.outer_bottom_selector),
            ("anchor_selector", self.anchor_selector),
        ):
            object.__setattr__(self, name, _immutable(values))
        object.__setattr__(self, "n_particles", n_particles)
        object.__setattr__(self, "anchor_particle", anchor)

    @property
    def n_dofs(self) -> int:
        return 2 * int(self.n_particles)

    @property
    def free_positions(self) -> np.ndarray:
        positions = np.full(self.n_dofs, -1, dtype=np.int64)
        positions[self.free_dofs] = np.arange(len(self.free_dofs), dtype=np.int64)
        return _immutable(positions)

    @property
    def a_ext(self) -> np.ndarray:
        return self.load_direction

    @property
    def a_top(self) -> np.ndarray:
        return self.outer_top_selector

    @property
    def a_bottom(self) -> np.ndarray:
        return self.outer_bottom_selector

    @property
    def a_anchor(self) -> np.ndarray:
        return self.anchor_selector

    def prescribed_vector(self, U_ext: Any) -> np.ndarray:
        value = _scalar_float(U_ext, "U_ext")
        return np.asarray(self.q_zero + value * self.load_direction, dtype=np.float64)


def build_affine_constraints(
    coordinates: Any,
    outer_bottom_mask: Any,
    outer_top_mask: Any,
    anchor_particle: Any,
    delta: Any,
) -> AffineConstraints:
    """Construct and validate the generic A2 outer-face constraint object."""

    coords = _require_float64(coordinates, "constraint coordinates")
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0:
        raise ValueError("constraint coordinates must have shape (N, 2)")
    bottom = _require_bool(outer_bottom_mask, "outer_bottom_mask", (len(coords),))
    top = _require_bool(outer_top_mask, "outer_top_mask", (len(coords),))
    delta_value = _scalar_float(delta, "delta")
    if not np.any(bottom) or not np.any(top):
        raise ValueError("outer top/bottom masks must be nonempty")
    tolerance = _geometry_tolerance(coords)
    if np.any(np.abs(coords[bottom, 1] + delta_value) > tolerance):
        raise ValueError("outer_bottom_mask does not select the exact y=-delta face")
    if np.any(np.abs(coords[top, 1] - (1.0 + delta_value)) > tolerance):
        raise ValueError("outer_top_mask does not select the exact y=1+delta face")
    if not isinstance(anchor_particle, Integral) or isinstance(anchor_particle, bool):
        raise ValueError("anchor_particle must be an integer")
    anchor = int(anchor_particle)
    if anchor < 0 or anchor >= len(coords):
        raise ValueError("anchor_particle is outside the union particle range")
    constrained = np.sort(
        np.concatenate(
            (
                2 * np.flatnonzero(bottom) + 1,
                2 * np.flatnonzero(top) + 1,
                np.asarray([2 * anchor], dtype=np.int64),
            )
        ).astype(np.int64)
    )
    free = np.setdiff1d(np.arange(2 * len(coords), dtype=np.int64), constrained)
    q_zero = np.zeros(2 * len(coords), dtype=np.float64)
    load_direction = np.zeros(2 * len(coords), dtype=np.float64)
    load_direction[2 * np.flatnonzero(top) + 1] = 1.0
    bottom_selector = np.zeros(2 * len(coords), dtype=np.float64)
    bottom_selector[2 * np.flatnonzero(bottom) + 1] = 1.0
    anchor_selector = np.zeros(2 * len(coords), dtype=np.float64)
    anchor_selector[2 * anchor] = 1.0
    return AffineConstraints(
        n_particles=len(coords),
        constrained_dofs=constrained,
        free_dofs=free,
        q_zero=q_zero,
        load_direction=load_direction,
        outer_bottom_mask=bottom,
        outer_top_mask=top,
        anchor_particle=anchor,
        anchor_coordinate=coords[anchor],
        outer_top_selector=load_direction,
        outer_bottom_selector=bottom_selector,
        anchor_selector=anchor_selector,
    )


def build_volume_grip_constraints(
    geometry: UnionGripGeometry,
    anchor_particle: Any,
) -> AffineConstraints:
    """Build the direct engineering-volume-grip constraint contract.

    The A2 diagnostic constrains only the two outermost collar rows.  The
    engineering-grip baseline instead transfers the prescribed vertical
    displacement through every added, nonmaterial collar particle.  The
    material prefix is deliberately inspected separately: material particles
    on ``y=0`` and ``y=1`` are interface particles and remain completely free
    in the vertical direction.  Only the original material anchor has a
    constrained horizontal degree of freedom.
    """

    if not isinstance(geometry, UnionGripGeometry):
        raise TypeError("geometry must be a UnionGripGeometry")
    coordinates = _require_float64(
        geometry.coordinates, "volume-grip constraint coordinates"
    )
    n_material = int(geometry.n_material)
    if n_material <= 0 or n_material >= len(coordinates):
        raise ValueError("volume-grip geometry must contain a material prefix and collars")
    if not np.array_equal(
        coordinates[:n_material], geometry.material_reference_coordinates
    ):
        raise ValueError("volume-grip geometry material prefix is not unchanged")

    material = np.arange(len(coordinates), dtype=np.int64) < n_material
    collar = ~material
    bottom = collar & (coordinates[:, 1] < 0.0)
    top = collar & (coordinates[:, 1] > 1.0)
    if not np.any(bottom) or not np.any(top):
        raise ValueError("volume-grip geometry must contain bottom and top collars")
    if np.any(collar & ~(bottom | top)):
        raise ValueError(
            "every nonmaterial particle must lie strictly below y=0 or above y=1"
        )
    if np.any(material & ((coordinates[:, 1] < 0.0) | (coordinates[:, 1] > 1.0))):
        raise ValueError("material particles must remain in the unit-square y range")

    if not isinstance(anchor_particle, Integral) or isinstance(anchor_particle, bool):
        raise ValueError("anchor_particle must be an integer")
    anchor = int(anchor_particle)
    if anchor < 0 or anchor >= n_material:
        raise ValueError("volume-grip anchor must be an original material particle")
    if not np.array_equal(
        coordinates[anchor], np.asarray([0.0, 0.0], dtype=np.float64)
    ):
        raise ValueError("the volume-grip anchor must be the original particle (0, 0)")

    n_dofs = 2 * len(coordinates)
    bottom_dofs = 2 * np.flatnonzero(bottom) + 1
    top_dofs = 2 * np.flatnonzero(top) + 1
    constrained = np.sort(
        np.concatenate((bottom_dofs, top_dofs, np.asarray([2 * anchor], dtype=np.int64)))
    ).astype(np.int64, copy=False)
    free = np.setdiff1d(
        np.arange(n_dofs, dtype=np.int64), constrained, assume_unique=False
    )
    q_zero = np.zeros(n_dofs, dtype=np.float64)
    load_direction = np.zeros(n_dofs, dtype=np.float64)
    load_direction[top_dofs] = 1.0
    bottom_selector = np.zeros(n_dofs, dtype=np.float64)
    bottom_selector[bottom_dofs] = 1.0
    anchor_selector = np.zeros(n_dofs, dtype=np.float64)
    anchor_selector[2 * anchor] = 1.0
    return AffineConstraints(
        n_particles=len(coordinates),
        constrained_dofs=constrained,
        free_dofs=free,
        q_zero=q_zero,
        load_direction=load_direction,
        outer_bottom_mask=bottom,
        outer_top_mask=top,
        anchor_particle=anchor,
        anchor_coordinate=coordinates[anchor],
        outer_top_selector=load_direction,
        outer_bottom_selector=bottom_selector,
        anchor_selector=anchor_selector,
    )


def _a2_hash_primitives(values: Iterable[tuple[str, Any]]) -> str:
    """Hash named primitive arrays without depending on Python object identity."""

    digest = hashlib.sha256()
    for name, value in values:
        if not isinstance(name, str) or not name:
            raise ValueError("A2 identity field names must be non-empty strings")
        if isinstance(value, str):
            encoded = value.encode("utf-8")
            digest.update(name.encode("utf-8"))
            digest.update(b"|str|")
            digest.update(np.asarray([len(encoded)], dtype=np.int64).tobytes())
            digest.update(encoded)
            continue
        array = np.asarray(value)
        if array.dtype.kind == "O":
            raise ValueError(f"A2 identity field {name!r} must not have object dtype")
        digest.update(name.encode("utf-8"))
        digest.update(b"|array|")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _a2_line_primitives(line: Any, name: str) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Validate the small line-quadrature contract used by static identities."""

    if not isinstance(line, LineQuadrature):
        raise TypeError(f"{name} must be a LineQuadrature instance")
    indices = _require_int64(line.indices, f"{name}.indices")
    weights = _require_float64(line.weights, f"{name}.weights")
    if indices.ndim != 1 or weights.shape != indices.shape or len(indices) < 2:
        raise ValueError(f"{name} indices/weights have incompatible shapes")
    return indices, weights, _scalar_float(line.length, f"{name}.length"), line.side


@dataclass(frozen=True)
class A2StaticMechanics:
    """Explicit immutable shared BA data for the coupled A2 controller.

    The object deliberately contains a closed primitive whitelist.  A fixed
    damage view may borrow these arrays, but it may not copy an arbitrary
    ``BondAssociatedMechanics.__dict__``: damage-dependent factors and caches
    belong to each fresh view.  ``runtime_identity`` is process-local and is
    never used by archive replay; ``deterministic_digest`` is the cross-process
    identity of the static model.
    """

    coordinates: np.ndarray
    union_volumes: np.ndarray
    directed_edges: np.ndarray
    source: np.ndarray
    target: np.ndarray
    edge_distance: np.ndarray
    edge_omega: np.ndarray
    family_mass: np.ndarray
    eta: np.ndarray
    support_indices: np.ndarray
    support_mask: np.ndarray
    support_sizes: np.ndarray
    beta: np.ndarray
    xi: np.ndarray
    shape_inverse: np.ndarray
    shape_rank: np.ndarray
    shape_condition: np.ndarray
    target_beta: np.ndarray
    gradient_coefficient: np.ndarray
    youngs_modulus: float
    poisson_ratio: float
    lame_lambda: float
    shear_modulus: float
    k_res: float
    rank_rtol: float
    delta: float
    h_g: float
    n_particles: int
    n_edges: int
    n_dofs: int
    material_count: int
    material_prefix_map: np.ndarray
    material_damage_volumes: np.ndarray
    union_material_phase_volumes: np.ndarray
    grip_phase_volumes: np.ndarray
    constrained_dofs: np.ndarray
    free_dofs: np.ndarray
    q_zero: np.ndarray
    load_direction: np.ndarray
    outer_top_selector: np.ndarray
    outer_bottom_selector: np.ndarray
    anchor_selector: np.ndarray
    outer_bottom_mask: np.ndarray
    outer_top_mask: np.ndarray
    anchor_particle: int
    top_line_indices: np.ndarray
    top_line_weights: np.ndarray
    top_line_length: float
    bottom_line_indices: np.ndarray
    bottom_line_weights: np.ndarray
    bottom_line_length: float
    runtime_identity: object
    deterministic_digest: str

    SHARED_WHITELIST = frozenset(
        {
            "coordinates",
            "union_volumes",
            "directed_edges",
            "source",
            "target",
            "edge_distance",
            "edge_omega",
            "family_mass",
            "eta",
            "support_indices",
            "support_mask",
            "support_sizes",
            "beta",
            "xi",
            "shape_inverse",
            "shape_rank",
            "shape_condition",
            "target_beta",
            "gradient_coefficient",
            "youngs_modulus",
            "poisson_ratio",
            "lame_lambda",
            "shear_modulus",
            "k_res",
            "rank_rtol",
            "delta",
            "h_g",
            "n_particles",
            "n_edges",
            "n_dofs",
            "material_count",
            "material_prefix_map",
            "material_damage_volumes",
            "union_material_phase_volumes",
            "grip_phase_volumes",
            "constrained_dofs",
            "free_dofs",
            "q_zero",
            "load_direction",
            "outer_top_selector",
            "outer_bottom_selector",
            "anchor_selector",
            "outer_bottom_mask",
            "outer_top_mask",
            "anchor_particle",
            "top_line_indices",
            "top_line_weights",
            "top_line_length",
            "bottom_line_indices",
            "bottom_line_weights",
            "bottom_line_length",
            "runtime_identity",
            "deterministic_digest",
        }
    )

    @classmethod
    def from_geometry(
        cls,
        geometry: UnionGripGeometry,
        constraints: AffineConstraints,
        top_line: LineQuadrature,
        bottom_line: LineQuadrature,
        youngs_modulus: Any,
        poisson_ratio: Any,
        k_res: Any,
        rank_rtol: Any,
    ) -> "A2StaticMechanics":
        """Build shared geometry/BA data exactly once from an A2 union."""

        if not isinstance(geometry, UnionGripGeometry):
            raise TypeError("geometry must be a UnionGripGeometry")
        if not isinstance(constraints, AffineConstraints):
            raise TypeError("constraints must be an AffineConstraints instance")
        if constraints.n_particles != geometry.n_particles:
            raise ValueError("A2 constraints and union geometry have different particle counts")
        top_indices, top_weights, top_length, top_side = _a2_line_primitives(
            top_line, "top_line"
        )
        bottom_indices, bottom_weights, bottom_length, bottom_side = _a2_line_primitives(
            bottom_line, "bottom_line"
        )
        if top_side != "top" or bottom_side != "bottom":
            raise ValueError("A2 line quadratures must be top then bottom")
        if not np.array_equal(top_indices, np.flatnonzero(geometry.interface_top_mask)):
            raise ValueError("top line indices are not the exact material interface")
        if not np.array_equal(bottom_indices, np.flatnonzero(geometry.interface_bottom_mask)):
            raise ValueError("bottom line indices are not the exact material interface")

        E = _scalar_float(youngs_modulus, "A2 youngs_modulus")
        nu = _scalar_float(poisson_ratio, "A2 poisson_ratio")
        kres = _scalar_float(k_res, "A2 k_res")
        rank = _scalar_float(rank_rtol, "A2 rank_rtol")
        if E <= 0.0 or not -1.0 < nu < 0.5 or kres <= 0.0:
            raise ValueError("A2 constitutive parameters are outside their strict ranges")
        if rank <= 0.0 or rank >= 1.0:
            raise ValueError("A2 rank_rtol must satisfy 0 < rank_rtol < 1")

        zero_damage = np.zeros(geometry.n_material, dtype=np.float64)
        reference = independent_union_ba_reference(geometry, zero_damage, kres, rank)
        n_edges = len(reference.source)
        max_support = max(len(support) for support in reference.support_indices)
        support_indices = np.full((n_edges, max_support), -1, dtype=np.int64)
        support_mask = np.zeros((n_edges, max_support), dtype=bool)
        beta = np.zeros((n_edges, max_support), dtype=np.float64)
        xi = np.zeros((n_edges, max_support, 2), dtype=np.float64)
        for edge_index, support in enumerate(reference.support_indices):
            count = len(support)
            support_indices[edge_index, :count] = support
            support_mask[edge_index, :count] = True
            beta[edge_index, :count] = reference.beta[edge_index]
            xi[edge_index, :count] = reference.xi[edge_index]
        shape_inverse = np.asarray(reference.shape_inverse, dtype=np.float64)
        source = np.asarray(reference.source, dtype=np.int64)
        target = np.asarray(reference.target, dtype=np.int64)
        gradient_coefficient = np.einsum(
            "es,esb,ebc->esc",
            beta,
            xi,
            shape_inverse,
            optimize=True,
        )
        lame_lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        shear_modulus = E / (2.0 * (1.0 + nu))
        prefix_map = np.arange(geometry.n_material, dtype=np.int64)

        digest_fields = (
            ("coordinates", geometry.coordinates),
            ("union_volumes", geometry.union_volumes),
            ("directed_edges", geometry.directed_edges),
            ("source", source),
            ("target", target),
            ("edge_distance", reference.distance),
            ("edge_omega", reference.omega),
            ("family_mass", reference.family_mass),
            ("eta", reference.eta),
            ("support_indices", support_indices),
            ("support_mask", support_mask),
            ("support_sizes", np.asarray([len(s) for s in reference.support_indices], dtype=np.int64)),
            ("beta", beta),
            ("xi", xi),
            ("shape_inverse", shape_inverse),
            ("shape_rank", reference.shape_rank),
            ("shape_condition", reference.condition_numbers),
            ("target_beta", beta[:, 0]),
            ("gradient_coefficient", gradient_coefficient),
            ("youngs_modulus", E),
            ("poisson_ratio", nu),
            ("lame_lambda", lame_lambda),
            ("shear_modulus", shear_modulus),
            ("k_res", kres),
            ("rank_rtol", rank),
            ("delta", geometry.delta),
            ("h_g", geometry.h_g),
            ("material_count", geometry.n_material),
            ("material_prefix_map", prefix_map),
            ("material_damage_volumes", geometry.material_reference_volumes),
            ("union_material_phase_volumes", geometry.material_volumes),
            ("grip_phase_volumes", geometry.grip_volumes),
            ("constrained_dofs", constraints.constrained_dofs),
            ("free_dofs", constraints.free_dofs),
            ("q_zero", constraints.q_zero),
            ("load_direction", constraints.load_direction),
            ("outer_top_selector", constraints.outer_top_selector),
            ("outer_bottom_selector", constraints.outer_bottom_selector),
            ("anchor_selector", constraints.anchor_selector),
            ("outer_bottom_mask", constraints.outer_bottom_mask),
            ("outer_top_mask", constraints.outer_top_mask),
            ("anchor_particle", constraints.anchor_particle),
            ("top_line_indices", top_indices),
            ("top_line_weights", top_weights),
            ("top_line_length", top_length),
            ("bottom_line_indices", bottom_indices),
            ("bottom_line_weights", bottom_weights),
            ("bottom_line_length", bottom_length),
        )
        return cls(
            coordinates=geometry.coordinates,
            union_volumes=geometry.union_volumes,
            directed_edges=geometry.directed_edges,
            source=source,
            target=target,
            edge_distance=reference.distance,
            edge_omega=reference.omega,
            family_mass=reference.family_mass,
            eta=reference.eta,
            support_indices=support_indices,
            support_mask=support_mask,
            support_sizes=np.asarray(
                [len(support) for support in reference.support_indices], dtype=np.int64
            ),
            beta=beta,
            xi=xi,
            shape_inverse=shape_inverse,
            shape_rank=reference.shape_rank,
            shape_condition=reference.condition_numbers,
            target_beta=beta[:, 0],
            gradient_coefficient=gradient_coefficient,
            youngs_modulus=E,
            poisson_ratio=nu,
            lame_lambda=lame_lambda,
            shear_modulus=shear_modulus,
            k_res=kres,
            rank_rtol=rank,
            delta=geometry.delta,
            h_g=geometry.h_g,
            n_particles=geometry.n_particles,
            n_edges=n_edges,
            n_dofs=geometry.n_dofs,
            material_count=geometry.n_material,
            material_prefix_map=prefix_map,
            material_damage_volumes=geometry.material_reference_volumes,
            union_material_phase_volumes=geometry.material_volumes,
            grip_phase_volumes=geometry.grip_volumes,
            constrained_dofs=constraints.constrained_dofs,
            free_dofs=constraints.free_dofs,
            q_zero=constraints.q_zero,
            load_direction=constraints.load_direction,
            outer_top_selector=constraints.outer_top_selector,
            outer_bottom_selector=constraints.outer_bottom_selector,
            anchor_selector=constraints.anchor_selector,
            outer_bottom_mask=constraints.outer_bottom_mask,
            outer_top_mask=constraints.outer_top_mask,
            anchor_particle=constraints.anchor_particle,
            top_line_indices=top_indices,
            top_line_weights=top_weights,
            top_line_length=top_length,
            bottom_line_indices=bottom_indices,
            bottom_line_weights=bottom_weights,
            bottom_line_length=bottom_length,
            runtime_identity=object(),
            deterministic_digest=_a2_hash_primitives(digest_fields),
        )

    def __post_init__(self) -> None:
        array_specs = (
            ("coordinates", _FLOAT64),
            ("union_volumes", _FLOAT64),
            ("directed_edges", _INT64),
            ("source", _INT64),
            ("target", _INT64),
            ("edge_distance", _FLOAT64),
            ("edge_omega", _FLOAT64),
            ("family_mass", _FLOAT64),
            ("eta", _FLOAT64),
            ("support_indices", _INT64),
            ("support_mask", _BOOL),
            ("support_sizes", _INT64),
            ("beta", _FLOAT64),
            ("xi", _FLOAT64),
            ("shape_inverse", _FLOAT64),
            ("shape_rank", _INT64),
            ("shape_condition", _FLOAT64),
            ("target_beta", _FLOAT64),
            ("gradient_coefficient", _FLOAT64),
            ("material_prefix_map", _INT64),
            ("material_damage_volumes", _FLOAT64),
            ("union_material_phase_volumes", _FLOAT64),
            ("grip_phase_volumes", _FLOAT64),
            ("constrained_dofs", _INT64),
            ("free_dofs", _INT64),
            ("q_zero", _FLOAT64),
            ("load_direction", _FLOAT64),
            ("outer_top_selector", _FLOAT64),
            ("outer_bottom_selector", _FLOAT64),
            ("anchor_selector", _FLOAT64),
            ("outer_bottom_mask", _BOOL),
            ("outer_top_mask", _BOOL),
            ("top_line_indices", _INT64),
            ("top_line_weights", _FLOAT64),
            ("bottom_line_indices", _INT64),
            ("bottom_line_weights", _FLOAT64),
        )
        for name, dtype in array_specs:
            array = np.asarray(getattr(self, name))
            if array.dtype != dtype:
                raise ValueError(f"A2StaticMechanics.{name} must use dtype {dtype}")
            if dtype.kind == "f" and not np.all(np.isfinite(array)):
                raise ValueError(f"A2StaticMechanics.{name} must be finite")
            object.__setattr__(self, name, _immutable(array))
        n_particles = int(self.n_particles)
        n_edges = int(self.n_edges)
        n_dofs = int(self.n_dofs)
        material_count = int(self.material_count)
        if n_particles <= 0 or n_edges <= 0 or n_dofs != 2 * n_particles:
            raise ValueError("A2StaticMechanics particle/edge/DOF counts are inconsistent")
        if material_count <= 0 or material_count >= n_particles:
            raise ValueError("A2StaticMechanics must contain a nonempty material prefix and grips")
        if self.coordinates.shape != (n_particles, 2) or self.union_volumes.shape != (n_particles,):
            raise ValueError("A2StaticMechanics coordinates/volumes have incompatible shapes")
        if self.directed_edges.shape != (n_edges, 2):
            raise ValueError("A2StaticMechanics directed_edges have an incompatible shape")
        if self.source.shape != (n_edges,) or self.target.shape != (n_edges,):
            raise ValueError("A2StaticMechanics source/target have incompatible shapes")
        if self.support_indices.shape[0] != n_edges or self.support_mask.shape != self.support_indices.shape:
            raise ValueError("A2StaticMechanics support arrays have incompatible shapes")
        if self.beta.shape != self.support_indices.shape or self.xi.shape != (*self.support_indices.shape, 2):
            raise ValueError("A2StaticMechanics beta/xi have incompatible shapes")
        if self.shape_inverse.shape != (n_edges, 2, 2):
            raise ValueError("A2StaticMechanics shape_inverse has an incompatible shape")
        for name, value in (
            ("youngs_modulus", self.youngs_modulus),
            ("poisson_ratio", self.poisson_ratio),
            ("lame_lambda", self.lame_lambda),
            ("shear_modulus", self.shear_modulus),
            ("k_res", self.k_res),
            ("rank_rtol", self.rank_rtol),
            ("delta", self.delta),
            ("h_g", self.h_g),
            ("top_line_length", self.top_line_length),
            ("bottom_line_length", self.bottom_line_length),
        ):
            value = _scalar_float(value, f"A2StaticMechanics.{name}")
            object.__setattr__(self, name, value)
        if self.youngs_modulus <= 0.0 or self.shear_modulus <= 0.0 or self.k_res <= 0.0:
            raise ValueError("A2StaticMechanics constitutive values must be positive")
        if not -1.0 < self.poisson_ratio < 0.5 or self.rank_rtol <= 0.0 or self.rank_rtol >= 1.0:
            raise ValueError("A2StaticMechanics constitutive/rank values are invalid")
        if self.union_material_phase_volumes.shape != (n_particles,) or self.grip_phase_volumes.shape != (n_particles,):
            raise ValueError("A2StaticMechanics phase volumes must have union shape")
        if self.material_damage_volumes.shape != (material_count,):
            raise ValueError("A2StaticMechanics material damage volumes must have material shape")
        if not np.array_equal(
            self.union_material_phase_volumes[:material_count], self.material_damage_volumes
        ) or not np.array_equal(
            self.union_material_phase_volumes[material_count:],
            np.zeros(n_particles - material_count, dtype=np.float64),
        ):
            raise ValueError("A2StaticMechanics material phase is not the exact material prefix")
        volume_tolerance = _AREA_RECONSTRUCTION_EPS_FACTOR * np.finfo(np.float64).eps * max(
            1.0, float(np.max(self.union_volumes))
        )
        if not np.allclose(
            self.union_volumes,
            self.union_material_phase_volumes + self.grip_phase_volumes,
            rtol=0.0,
            atol=volume_tolerance,
        ):
            raise ValueError("A2StaticMechanics union volumes are not the phase sum")
        object.__setattr__(self, "n_particles", n_particles)
        object.__setattr__(self, "n_edges", n_edges)
        object.__setattr__(self, "n_dofs", n_dofs)
        object.__setattr__(self, "material_count", material_count)
        if self.runtime_identity is None:
            raise ValueError("A2StaticMechanics runtime identity is missing")
        digest = _a2_hash_primitives(
            (name, getattr(self, name))
            for name in (
                "coordinates", "union_volumes", "directed_edges", "source", "target",
                "edge_distance", "edge_omega", "family_mass", "eta", "support_indices",
                "support_mask", "support_sizes", "beta", "xi", "shape_inverse",
                "shape_rank", "shape_condition", "target_beta", "gradient_coefficient",
                "youngs_modulus", "poisson_ratio", "lame_lambda", "shear_modulus", "k_res",
                "rank_rtol", "delta", "h_g", "material_count", "material_prefix_map", "material_damage_volumes",
                "union_material_phase_volumes", "grip_phase_volumes", "constrained_dofs",
                "free_dofs", "q_zero", "load_direction", "outer_top_selector",
                "outer_bottom_selector", "anchor_selector", "outer_bottom_mask", "outer_top_mask",
                "anchor_particle", "top_line_indices", "top_line_weights", "top_line_length",
                "bottom_line_indices", "bottom_line_weights", "bottom_line_length",
            )
        )
        if not isinstance(self.deterministic_digest, str) or self.deterministic_digest != digest:
            raise ValueError("A2StaticMechanics deterministic identity mismatch")

    @property
    def static_digest(self) -> str:
        return self.deterministic_digest

    @property
    def runtime_token(self) -> object:
        return self.runtime_identity

    @property
    def material_prefix_count(self) -> int:
        return self.material_count

    @property
    def material_phase_volumes(self) -> np.ndarray:
        return self.union_material_phase_volumes

    @property
    def n_material(self) -> int:
        return self.material_count


@dataclass(frozen=True)
class A2MechanicsSolveResult:
    """Immutable result of the generic A2 fixed-damage solve."""

    displacement: np.ndarray
    evaluation: MechanicsEvaluation
    constraints: AffineConstraints
    U_ext: float
    free_residual: np.ndarray
    free_residual_norm: float
    free_rhs_norm: float
    free_residual_relative_norm: float
    cg_convergence_limit: float
    cg_iterations: int
    cg_info: int
    outer_reaction: float
    bottom_reaction: float
    anchor_reaction: float
    total_force: np.ndarray
    total_moment: float

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation, MechanicsEvaluation):
            raise TypeError("A2 solve evaluation must be MechanicsEvaluation")
        if not isinstance(self.constraints, AffineConstraints):
            raise TypeError("A2 solve constraints must be AffineConstraints")
        for name, shape in (
            ("displacement", (self.constraints.n_dofs,)),
            ("free_residual", (len(self.constraints.free_dofs),)),
            ("total_force", (2,)),
        ):
            array = _require_float64(getattr(self, name), name, shape)
            object.__setattr__(self, name, _immutable(array))
        for name in (
            "U_ext",
            "free_residual_norm",
            "free_rhs_norm",
            "free_residual_relative_norm",
            "cg_convergence_limit",
            "outer_reaction",
            "bottom_reaction",
            "anchor_reaction",
            "total_moment",
        ):
            value = _scalar_float(getattr(self, name), name)
            object.__setattr__(self, name, value)
        for name in ("cg_iterations", "cg_info"):
            raw = np.asarray(getattr(self, name))
            if raw.ndim != 0 or not np.issubdtype(raw.dtype, np.integer):
                raise ValueError(f"{name} must be an integer")
            if int(raw) < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, int(raw))
        if abs(float(self.free_residual_norm) - float(np.linalg.norm(self.free_residual))) > 512.0 * np.finfo(float).eps * max(1.0, self.free_residual_norm):
            raise ValueError("A2 free residual norm is inconsistent with its vector")
        expected_relative = self.free_residual_norm / max(self.free_rhs_norm, 1.0e-300)
        if abs(self.free_residual_relative_norm - expected_relative) > 512.0 * np.finfo(float).eps * max(1.0, abs(expected_relative)):
            raise ValueError("A2 free residual relative norm is inconsistent")
        if self.cg_convergence_limit <= 0.0 or self.free_residual_norm >= self.cg_convergence_limit:
            raise ValueError("A2 solve did not satisfy its authoritative free-residual gate")

    @property
    def energy(self) -> float:
        return self.evaluation.energy

    @property
    def residual(self) -> np.ndarray:
        return self.evaluation.residual

    @property
    def R_ext(self) -> float:
        return self.outer_reaction


def _validate_cg_parameters(cg_rtol: Any, cg_atol: Any, cg_maxiter: Any) -> tuple[float, float, int]:
    rtol = _scalar_float(cg_rtol, "cg_rtol")
    atol = _scalar_float(cg_atol, "cg_atol")
    if rtol < 0.0 or atol < 0.0 or (rtol == 0.0 and atol == 0.0):
        raise ValueError("cg_rtol/cg_atol must be non-negative and not both zero")
    raw = np.asarray(cg_maxiter)
    if raw.ndim != 0 or not np.issubdtype(raw.dtype, np.integer) or int(raw) <= 0:
        raise ValueError("cg_maxiter must be a positive integer")
    return rtol, atol, int(raw)


def solve_fixed_damage_a2(
    mechanics: FixedDamagePhaseMeasuredMechanics,
    constraints: AffineConstraints,
    U_ext: Any,
    cg_rtol: Any,
    cg_atol: Any,
    cg_maxiter: Any,
) -> A2MechanicsSolveResult:
    """Solve the A2 fixed-damage linear mechanics problem by matrix-free CG."""

    if not isinstance(mechanics, FixedDamagePhaseMeasuredMechanics):
        raise TypeError("mechanics must be FixedDamagePhaseMeasuredMechanics")
    if not isinstance(constraints, AffineConstraints):
        raise TypeError("constraints must be AffineConstraints")
    if constraints.n_particles != mechanics.n_particles:
        raise ValueError("constraints and A2 mechanics have different particle counts")
    load = _scalar_float(U_ext, "U_ext")
    rtol, atol, maxiter = _validate_cg_parameters(cg_rtol, cg_atol, cg_maxiter)
    prescribed = constraints.prescribed_vector(load)
    free = constraints.free_dofs

    def inject(values: np.ndarray) -> np.ndarray:
        values = _require_float64(values, "CG free vector", (len(free),))
        result = np.zeros(mechanics.n_dofs, dtype=np.float64)
        result[free] = values
        return result

    def free_action(values: np.ndarray) -> np.ndarray:
        return mechanics.tangent_action(inject(values))[free]

    prescribed_residual = mechanics.residual(prescribed)
    rhs = -prescribed_residual[free]
    if not np.all(np.isfinite(rhs)):
        raise FloatingPointError("A2 CG right-hand side is non-finite")
    rhs_norm = float(np.linalg.norm(rhs))
    convergence_limit = max(atol, rtol * rhs_norm)
    callback_iterations: list[int] = []

    def callback(_: np.ndarray) -> None:
        callback_iterations.append(1)

    block_diagonal = mechanics.tangent_block_diagonal
    positions = constraints.free_positions
    preconditioner_blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for particle in range(mechanics.n_particles):
        components = np.flatnonzero(positions[2 * particle : 2 * particle + 2] >= 0)
        if len(components) == 0:
            continue
        free_positions = positions[2 * particle + components]
        block = block_diagonal[particle][np.ix_(components, components)]
        eigenvalues = np.linalg.eigvalsh(block)
        if np.any(~np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0.0):
            raise ValueError("A2 free block-Jacobi preconditioner is not positive definite")
        preconditioner_blocks.append((free_positions, np.linalg.inv(block)))

    def preconditioner(values: np.ndarray) -> np.ndarray:
        values = _require_float64(values, "CG preconditioner vector", (len(free),))
        result = np.zeros_like(values)
        for block_positions, inverse in preconditioner_blocks:
            result[block_positions] = inverse @ values[block_positions]
        return result

    if rhs_norm == 0.0:
        free_values = np.zeros(len(free), dtype=np.float64)
        info = 0
    else:
        operator = LinearOperator(
            shape=(len(free), len(free)), matvec=free_action, dtype=np.float64
        )
        preconditioner_operator = LinearOperator(
            shape=(len(free), len(free)), matvec=preconditioner, dtype=np.float64
        )
        free_values, info = cg(
            operator,
            rhs,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            callback=callback,
            M=preconditioner_operator,
        )
        info = int(info)
        if info != 0:
            raise RuntimeError(
                "A2 fixed-damage CG did not converge: "
                f"info={info}, iterations={len(callback_iterations)}"
            )
    free_values = _require_float64(np.asarray(free_values), "CG solution", (len(free),))
    displacement = prescribed + inject(free_values)
    evaluation = mechanics.evaluate(displacement)
    free_residual = evaluation.residual[free]
    free_residual_norm = float(np.linalg.norm(free_residual))
    free_relative = free_residual_norm / max(rhs_norm, 1.0e-300)
    if not np.isfinite(free_residual_norm) or free_residual_norm >= convergence_limit:
        raise RuntimeError(
            "A2 CG returned a solution outside the authoritative free-residual gate: "
            f"norm={free_residual_norm}, limit={convergence_limit}"
        )
    residual_matrix = evaluation.residual.reshape(mechanics.n_particles, 2)
    total_force = np.sum(residual_matrix, axis=0)
    total_moment = float(
        np.sum(
            mechanics.coordinates[:, 0] * residual_matrix[:, 1]
            - mechanics.coordinates[:, 1] * residual_matrix[:, 0]
        )
    )
    if not np.all(np.isfinite(total_force)) or not np.isfinite(total_moment):
        raise FloatingPointError("A2 force/moment diagnostics are non-finite")
    return A2MechanicsSolveResult(
        displacement=displacement,
        evaluation=evaluation,
        constraints=constraints,
        U_ext=load,
        free_residual=free_residual,
        free_residual_norm=free_residual_norm,
        free_rhs_norm=rhs_norm,
        free_residual_relative_norm=free_relative,
        cg_convergence_limit=convergence_limit,
        cg_iterations=len(callback_iterations),
        cg_info=info,
        outer_reaction=float(constraints.outer_top_selector @ evaluation.residual),
        bottom_reaction=float(constraints.outer_bottom_selector @ evaluation.residual),
        anchor_reaction=float(constraints.anchor_selector @ evaluation.residual),
        total_force=total_force,
        total_moment=total_moment,
    )


def _scale_a2_solve_result(
    unit_result: A2MechanicsSolveResult,
    mechanics: FixedDamagePhaseMeasuredMechanics,
    scale: float,
    cg_rtol: float,
    cg_atol: float,
) -> A2MechanicsSolveResult:
    """Scale the authoritative unit-load solution without a second load solve.

    A2 fixed-damage mechanics is exactly linear.  Scaling the already
    converged unit result is the executable contract's prescribed load
    operation and avoids reintroducing a CG reduction error into the exact
    ``Delta_m=target`` check.  The evaluation, residual, reactions, and free
    residual are nevertheless recomputed independently at the scaled vector.
    """

    factor = _scalar_float(scale, "A2 solution scale")
    displacement = np.asarray(unit_result.displacement * factor, dtype=np.float64)
    evaluation = mechanics.evaluate(displacement)
    free = unit_result.constraints.free_dofs
    free_residual = evaluation.residual[free]
    free_residual_norm = float(np.linalg.norm(free_residual))
    free_rhs_norm = abs(factor) * unit_result.free_rhs_norm
    convergence_limit = max(cg_atol, cg_rtol * free_rhs_norm)
    if free_residual_norm >= convergence_limit:
        raise RuntimeError(
            "scaled A2 unit solution failed the authoritative free-residual gate: "
            f"norm={free_residual_norm}, limit={convergence_limit}"
        )
    residual_matrix = evaluation.residual.reshape(mechanics.n_particles, 2)
    total_force = np.sum(residual_matrix, axis=0)
    total_moment = float(
        np.sum(
            mechanics.coordinates[:, 0] * residual_matrix[:, 1]
            - mechanics.coordinates[:, 1] * residual_matrix[:, 0]
        )
    )
    return A2MechanicsSolveResult(
        displacement=displacement,
        evaluation=evaluation,
        constraints=unit_result.constraints,
        U_ext=factor,
        free_residual=free_residual,
        free_residual_norm=free_residual_norm,
        free_rhs_norm=free_rhs_norm,
        free_residual_relative_norm=free_residual_norm / max(free_rhs_norm, 1.0e-300),
        cg_convergence_limit=convergence_limit,
        cg_iterations=unit_result.cg_iterations,
        cg_info=unit_result.cg_info,
        outer_reaction=float(unit_result.constraints.outer_top_selector @ evaluation.residual),
        bottom_reaction=float(unit_result.constraints.outer_bottom_selector @ evaluation.residual),
        anchor_reaction=float(unit_result.constraints.anchor_selector @ evaluation.residual),
        total_force=total_force,
        total_moment=total_moment,
    )


def _reaction_balance_metrics(
    result: A2MechanicsSolveResult,
    mechanics: FixedDamagePhaseMeasuredMechanics,
) -> dict[str, float]:
    """Decompose constrained reactions and balance them against free residuals."""

    if result.constraints.n_particles != mechanics.n_particles:
        raise ValueError("reaction result/mechanics particle counts differ")
    residual = result.residual.reshape(mechanics.n_particles, 2)
    constrained = np.zeros_like(result.residual)
    constrained[result.constraints.constrained_dofs] = result.residual[
        result.constraints.constrained_dofs
    ]
    free = result.residual - constrained
    constrained_matrix = constrained.reshape(mechanics.n_particles, 2)
    free_matrix = free.reshape(mechanics.n_particles, 2)
    constrained_force = np.sum(constrained_matrix, axis=0)
    free_force = np.sum(free_matrix, axis=0)
    constrained_moment = float(
        np.sum(
            mechanics.coordinates[:, 0] * constrained_matrix[:, 1]
            - mechanics.coordinates[:, 1] * constrained_matrix[:, 0]
        )
    )
    free_moment = float(
        np.sum(
            mechanics.coordinates[:, 0] * free_matrix[:, 1]
            - mechanics.coordinates[:, 1] * free_matrix[:, 0]
        )
    )
    equilibrium_force = constrained_force + free_force
    equilibrium_moment = constrained_moment + free_moment
    expected_constrained_force = np.asarray(
        [result.anchor_reaction, result.outer_reaction + result.bottom_reaction],
        dtype=np.float64,
    )
    if not np.allclose(constrained_force, expected_constrained_force, rtol=0.0, atol=1.0e-12):
        raise ValueError("constrained reaction force decomposition is inconsistent")
    return {
        "top_reaction": result.outer_reaction,
        "bottom_reaction": result.bottom_reaction,
        "anchor_reaction": result.anchor_reaction,
        "constrained_reaction_force_x": float(constrained_force[0]),
        "constrained_reaction_force_y": float(constrained_force[1]),
        "free_residual_force_x": float(free_force[0]),
        "free_residual_force_y": float(free_force[1]),
        "equilibrium_force_x": float(equilibrium_force[0]),
        "equilibrium_force_y": float(equilibrium_force[1]),
        "constrained_reaction_reference_moment": constrained_moment,
        "free_residual_reference_moment": free_moment,
        "equilibrium_reference_moment": equilibrium_moment,
        "equilibrium_force_error": float(np.linalg.norm(equilibrium_force)),
        "equilibrium_reference_moment_error": abs(equilibrium_moment),
    }


@dataclass(frozen=True)
class LineQuadrature:
    """Trapezoidal line weights on one unchanged physical interface."""

    indices: np.ndarray
    weights: np.ndarray
    length: float
    side: str

    def __post_init__(self) -> None:
        indices = _require_int64(self.indices, "line indices")
        weights = _require_float64(self.weights, "line weights")
        if indices.ndim != 1 or weights.shape != indices.shape or len(indices) < 2:
            raise ValueError("line indices/weights must be one-dimensional with at least two points")
        if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError("line weights must be finite and positive")
        length = _scalar_float(self.length, "line length")
        if length <= 0.0:
            raise ValueError("line length must be positive")
        if not np.isclose(np.sum(weights), length, rtol=0.0, atol=_geometry_tolerance(np.asarray([[0.0, 0.0], [length, 0.0]], dtype=np.float64))):
            raise ValueError("trapezoidal line weights do not sum to the boundary length")
        object.__setattr__(self, "indices", _immutable(indices))
        object.__setattr__(self, "weights", _immutable(weights))
        object.__setattr__(self, "length", length)


def physical_interface_line_weights(
    geometry: UnionGripGeometry,
    side: str,
) -> LineQuadrature:
    """Return exact-prefix trapezoidal weights on the physical top/bottom line."""

    if not isinstance(geometry, UnionGripGeometry):
        raise TypeError("geometry must be a UnionGripGeometry")
    if side not in ("top", "bottom"):
        raise ValueError("side must be 'top' or 'bottom'")
    mask = geometry.interface_top_mask if side == "top" else geometry.interface_bottom_mask
    indices = np.flatnonzero(mask).astype(np.int64)
    order = np.lexsort((geometry.coordinates[indices, 1], geometry.coordinates[indices, 0]))
    indices = indices[order]
    x = geometry.coordinates[indices, 0]
    spacing = np.diff(x)
    if np.any(spacing <= 0.0):
        raise ValueError("physical interface x coordinates are not strictly ordered")
    median = float(np.median(spacing))
    if np.any(np.abs(spacing - median) > _geometry_tolerance(geometry.coordinates)):
        raise ValueError("physical interface spacing is not uniformly validated")
    weights = np.empty(len(indices), dtype=np.float64)
    weights[0] = 0.5 * spacing[0]
    weights[-1] = 0.5 * spacing[-1]
    if len(indices) > 2:
        weights[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    length = float(np.sum(spacing))
    if not np.isclose(length, 1.0, rtol=0.0, atol=_geometry_tolerance(geometry.coordinates)):
        raise ValueError(f"physical {side} interface length is not one: {length:.17g}")
    return LineQuadrature(indices=indices, weights=weights, length=length, side=side)


def physical_interface_opening(
    displacement: Any,
    geometry: UnionGripGeometry,
    top_line: LineQuadrature,
    bottom_line: LineQuadrature,
) -> float:
    """Evaluate the weighted physical-interface opening ``Delta_m``."""

    if not isinstance(geometry, UnionGripGeometry):
        raise TypeError("geometry must be a UnionGripGeometry")
    if top_line.side != "top" or bottom_line.side != "bottom":
        raise ValueError("opening line quadratures must be top then bottom")
    values = _require_float64(displacement, "displacement", (geometry.n_dofs,)).reshape(-1, 2)
    top_mean = float(
        np.sum(top_line.weights * values[top_line.indices, 1]) / top_line.length
    )
    bottom_mean = float(
        np.sum(bottom_line.weights * values[bottom_line.indices, 1]) / bottom_line.length
    )
    opening = top_mean - bottom_mean
    if not np.isfinite(opening):
        raise FloatingPointError("physical interface opening is non-finite")
    return opening


def _amor_positive_energy(
    edge_strain: np.ndarray,
    lame_lambda: float,
    shear_modulus: float,
) -> np.ndarray:
    strain = _require_float64(edge_strain, "edge_strain")
    if strain.ndim != 3 or strain.shape[1:] != (2, 2):
        raise ValueError("edge_strain must have shape (E, 2, 2)")
    skew = strain - np.swapaxes(strain, 1, 2)
    if np.max(np.abs(skew), initial=0.0) > 32.0 * np.finfo(float).eps * max(1.0, np.max(np.abs(strain), initial=0.0)):
        raise ValueError("edge_strain must be symmetric for the Amor driver")
    lam = _scalar_float(lame_lambda, "lame_lambda")
    mu = _scalar_float(shear_modulus, "shear_modulus")
    if mu <= 0.0:
        raise ValueError("shear_modulus must be positive")
    embedded = np.zeros((len(strain), 3, 3), dtype=np.float64)
    embedded[:, :2, :2] = strain
    identity = np.eye(3, dtype=np.float64)
    trace = np.trace(embedded, axis1=1, axis2=2)
    bulk = lam + 2.0 * mu / 3.0
    deviatoric = embedded - (trace / 3.0)[:, None, None] * identity
    result = 0.5 * bulk * np.maximum(trace, 0.0) ** 2 + mu * np.sum(deviatoric * deviatoric, axis=(1, 2))
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("Amor positive energy is invalid")
    return _immutable(result)


@dataclass(frozen=True)
class MaterialMetrics:
    """Material-only instantaneous and frozen history fields for A2 reporting."""

    psi_plus: np.ndarray
    H_used: np.ndarray
    grip_psi_plus: np.ndarray

    def __post_init__(self) -> None:
        for name in ("psi_plus", "H_used", "grip_psi_plus"):
            values = _require_float64(getattr(self, name), name)
            if values.ndim != 1 or np.any(values < 0.0):
                raise ValueError(f"{name} must be finite, non-negative, and one-dimensional")
            object.__setattr__(self, name, _immutable(values))


def compute_h_used(psi_plus: Any, material_history: Any) -> np.ndarray:
    """Combine instantaneous ``Psi_plus`` with the inherited history floor."""

    psi = _require_float64(psi_plus, "psi_plus")
    history = _require_float64(material_history, "material_history", psi.shape)
    if psi.ndim != 1 or np.any(psi < 0.0) or np.any(history < 0.0):
        raise ValueError("psi_plus/material_history must be finite, non-negative vectors")
    return _immutable(np.maximum(history, psi))


def compute_material_metrics(
    mechanics: FixedDamagePhaseMeasuredMechanics,
    displacement: Any,
    material_history: Any,
) -> MaterialMetrics:
    """Report ``Psi_plus``/``H_used`` only on original material particles."""

    if not isinstance(mechanics, FixedDamagePhaseMeasuredMechanics):
        raise TypeError("mechanics must be FixedDamagePhaseMeasuredMechanics")
    history = _require_float64(material_history, "material_history", (mechanics.geometry.n_material,))
    if np.any(history < 0.0):
        raise ValueError("material_history must be non-negative")
    evaluation = mechanics.evaluate(displacement)
    edge_psi = _amor_positive_energy(
        evaluation.edge_strain, mechanics.lame_lambda, mechanics.shear_modulus
    )
    driver = np.zeros(mechanics.n_particles, dtype=np.float64)
    np.add.at(driver, mechanics.source, mechanics.eta * edge_psi)
    material_driver = driver[: mechanics.geometry.n_material]
    H_used = compute_h_used(material_driver, history)
    grip_driver = driver[mechanics.geometry.n_material :]
    if np.any(~np.isfinite(material_driver)) or np.any(~np.isfinite(H_used)):
        raise FloatingPointError("material-only A2 metrics are non-finite")
    return MaterialMetrics(
        psi_plus=material_driver,
        H_used=H_used,
        grip_psi_plus=grip_driver,
    )


def transition_decision_arithmetic(
    top_ratio: Any,
    top_a2_error: Any,
    top_material_error: Any,
    no_new_global_max: Any,
    reaction_fem_error: Any,
) -> dict[str, float | bool]:
    """Apply the predeclared A2 comparison arithmetic without changing gates."""

    ratio = _scalar_float(top_ratio, "top_ratio")
    a2_error = _scalar_float(top_a2_error, "top_a2_error")
    material_error = _scalar_float(top_material_error, "top_material_error")
    reaction_error = _scalar_float(reaction_fem_error, "reaction_fem_error")
    if ratio < 0.0 or a2_error < 0.0 or material_error < 0.0 or reaction_error < 0.0:
        raise ValueError("transition decision inputs must be non-negative")
    if not isinstance(no_new_global_max, (bool, np.bool_)):
        raise ValueError("no_new_global_max must be boolean")
    reduction_factor = a2_error / max(material_error, 1.0e-30)
    reduction_fraction = 1.0 - reduction_factor
    return {
        "top_error_reduction_factor": reduction_factor,
        "top_error_reduction_fraction": reduction_fraction,
        "diagnostically_promising": bool(
            ratio < 1.0
            and a2_error <= 0.5 * material_error
            and bool(no_new_global_max)
            and reaction_error <= 0.10
        ),
    }


def _reference_response(
    geometry: UnionGripGeometry,
    reference: IndependentUnionBAReference,
    displacement: np.ndarray,
    youngs_modulus: float,
    poisson_ratio: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Independent loop for energy, residual, F, and strain."""

    values = _require_float64(displacement, "reference displacement", (geometry.n_dofs,)).reshape(-1, 2)
    weights = reference.phase_center_weights
    lam = youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    residual = np.zeros(geometry.n_dofs, dtype=np.float64)
    energy = 0.0
    fields_f: list[np.ndarray] = []
    fields_strain: list[np.ndarray] = []
    for edge_index, i_raw in enumerate(reference.source):
        i = int(i_raw)
        support = reference.support_indices[edge_index]
        beta = reference.beta[edge_index]
        xi = reference.xi[edge_index]
        inverse = reference.shape_inverse[edge_index]
        moment = np.zeros((2, 2), dtype=np.float64)
        for local, k_raw in enumerate(support):
            k = int(k_raw)
            moment += beta[local] * np.outer(xi[local] + values[k] - values[i], xi[local])
        deformation_gradient = moment @ inverse
        increment = deformation_gradient - np.eye(2, dtype=np.float64)
        strain = 0.5 * (increment + increment.T)
        trace = float(np.trace(strain))
        stress = lam * trace * np.eye(2, dtype=np.float64) + 2.0 * mu * strain
        density = 0.5 * lam * trace**2 + mu * float(np.sum(strain * strain))
        center_factor = weights[i] * reference.eta[edge_index]
        energy += center_factor * density
        for local, k_raw in enumerate(support):
            k = int(k_raw)
            endpoint = beta[local] * (stress @ inverse.T @ xi[local])
            residual[2 * k : 2 * k + 2] += center_factor * endpoint
            residual[2 * i : 2 * i + 2] -= center_factor * endpoint
        fields_f.append(deformation_gradient)
        fields_strain.append(strain)
    return (
        float(energy),
        residual,
        np.asarray(fields_f, dtype=np.float64),
        np.asarray(fields_strain, dtype=np.float64),
    )


def _compare_reference_to_mechanics(
    geometry: UnionGripGeometry,
    mechanics: FixedDamagePhaseMeasuredMechanics,
    reference: IndependentUnionBAReference,
) -> dict[str, float]:
    if not np.array_equal(mechanics.source, reference.source) or not np.array_equal(mechanics.target, reference.target):
        raise ValueError("independent union reference graph differs from mechanics graph")
    support_max = 0.0
    beta_max = 0.0
    xi_max = 0.0
    inverse_max = 0.0
    for edge_index, support in enumerate(reference.support_indices):
        count = len(support)
        if not np.array_equal(mechanics.support_indices[edge_index, :count], support):
            raise ValueError(f"union support mismatch at edge {edge_index}")
        if int(mechanics.support_sizes[edge_index]) != count:
            raise ValueError(f"union support size mismatch at edge {edge_index}")
        support_max = max(support_max, float(np.max(np.abs(mechanics.support_indices[edge_index, :count] - support))))
        beta_max = max(beta_max, float(np.max(np.abs(mechanics.beta[edge_index, :count] - reference.beta[edge_index]))))
        xi_max = max(xi_max, float(np.max(np.abs(mechanics.xi[edge_index, :count] - reference.xi[edge_index]))))
        inverse_max = max(inverse_max, float(np.max(np.abs(mechanics.shape_inverse[edge_index] - reference.shape_inverse[edge_index]))))
    W_eta = reference.phase_center_weights[reference.source] * reference.eta
    center_defect = float(np.max(np.abs(W_eta - mechanics.center_factor)))
    center_equivalence = float(np.max(np.abs(mechanics.center_factor - mechanics.center_factor_vstar_reference)))
    phase_weight_defect = float(
        np.max(np.abs(reference.phase_center_weights - mechanics.center_weights))
    )
    eta_sum = np.bincount(reference.source, weights=reference.eta, minlength=geometry.n_particles)
    eta_sum_defect = float(np.max(np.abs(eta_sum - 1.0)))
    return {
        "support_index_max_abs_defect": support_max,
        "beta_max_abs_defect": beta_max,
        "xi_max_abs_defect": xi_max,
        "shape_inverse_max_abs_defect": inverse_max,
        "center_factor_W_eta_max_abs_defect": center_defect,
        "center_factor_W_over_Vstar_max_abs_defect": center_equivalence,
        "phase_center_weight_max_abs_defect": phase_weight_defect,
        "eta_sum_max_abs_defect": eta_sum_defect,
        "shape_rank_min": int(np.min(reference.shape_rank)),
        "shape_condition_min": float(np.min(reference.condition_numbers)),
        "shape_condition_max": float(np.max(reference.condition_numbers)),
    }


def _dense_tangent(mechanics: FixedDamagePhaseMeasuredMechanics) -> np.ndarray:
    return np.column_stack(
        [
            mechanics.tangent_action(np.eye(mechanics.n_dofs, dtype=np.float64)[:, column])
            for column in range(mechanics.n_dofs)
        ]
    )


def _rigid_modes(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    translation_x = np.zeros(2 * len(coordinates), dtype=np.float64)
    translation_x[0::2] = 1.0
    translation_y = np.zeros(2 * len(coordinates), dtype=np.float64)
    translation_y[1::2] = 1.0
    rotation = np.empty(2 * len(coordinates), dtype=np.float64)
    rotation[0::2] = -coordinates[:, 1]
    rotation[1::2] = coordinates[:, 0]
    return translation_x, translation_y, rotation


def _bilinear_symmetry_defect(
    mechanics: FixedDamagePhaseMeasuredMechanics,
    free: np.ndarray | None,
) -> float:
    rng = np.random.default_rng(20260824)
    values: list[float] = []
    for _ in range(5):
        first = rng.normal(size=mechanics.n_dofs).astype(np.float64)
        second = rng.normal(size=mechanics.n_dofs).astype(np.float64)
        if free is None:
            left = mechanics.tangent_action(second)
            right = mechanics.tangent_action(first)
            lhs = float(first @ left)
            rhs = float(second @ right)
        else:
            first_free = np.zeros(mechanics.n_dofs, dtype=np.float64)
            second_free = np.zeros(mechanics.n_dofs, dtype=np.float64)
            first_free[free] = first[free]
            second_free[free] = second[free]
            left = mechanics.tangent_action(second_free)
            right = mechanics.tangent_action(first_free)
            lhs = float(first[free] @ left[free])
            rhs = float(second[free] @ right[free])
        values.append(abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0))
    return float(max(values))


def _manufactured_union_case() -> UnionGripGeometry:
    n_side = 5
    axis = np.linspace(0.0, 1.0, n_side, dtype=np.float64)
    coordinates = np.asarray(
        [(x, y) for x in axis for y in axis], dtype=np.float64
    )
    triangles: list[tuple[int, int, int]] = []
    for ix in range(n_side - 1):
        for iy in range(n_side - 1):
            ll = ix * n_side + iy
            lr = (ix + 1) * n_side + iy
            ul = ix * n_side + iy + 1
            ur = (ix + 1) * n_side + iy + 1
            triangles.extend(((ll, lr, ur), (ll, ur, ul)))
    triangles_array = np.asarray(triangles, dtype=np.int64)
    _, volumes, _ = _triangle_quadrature(coordinates, triangles_array, "manufactured material")
    bottom = coordinates[:, 1] == 0.0
    top = coordinates[:, 1] == 1.0
    pairs: list[tuple[int, int]] = []
    delta = 0.48
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            distance = float(np.linalg.norm(coordinates[j] - coordinates[i]))
            if 0.0 < distance < delta:
                pairs.append((i, j))
    edges = _directed_edges_from_pairs(np.asarray(pairs, dtype=np.int64))
    return build_union_geometry_from_arrays(
        coordinates,
        triangles_array,
        volumes,
        bottom,
        top,
        0,
        edges,
        delta,
        0.20,
        0.0,
    )


def run_manufactured_gates() -> dict[str, float | int | bool]:
    """Run the independent finite manufactured A2 contract before real data."""

    geometry = _manufactured_union_case()
    damage = np.asarray(
        0.1 + 0.5 * np.linspace(0.0, 1.0, geometry.n_material), dtype=np.float64
    )
    mechanics = FixedDamagePhaseMeasuredMechanics(
        geometry, damage, 2.0, 0.3, 1.0e-8, 1.0e-12
    )
    reference = independent_union_ba_reference(geometry, damage, 1.0e-8, 1.0e-12)
    algebra = _compare_reference_to_mechanics(geometry, mechanics, reference)
    if max(algebra[key] for key in algebra if key.endswith("defect")) > 1.0e-10:
        raise CandidateA2GateFailure("manufactured independent A2 algebra gate failed", algebra)
    affine_f = np.asarray([[1.12, 0.035], [-0.018, 0.93]], dtype=np.float64)
    translation = np.asarray([0.13, -0.22], dtype=np.float64)
    affine_q = (geometry.coordinates @ (affine_f - np.eye(2)).T + translation).ravel()
    evaluation = mechanics.evaluate(affine_q)
    affine_error = float(np.max(np.abs(evaluation.edge_F - affine_f)))
    tx, ty, rotation = _rigid_modes(geometry.coordinates)
    rigid_tangent = max(
        float(np.linalg.norm(mechanics.tangent_action(mode)))
        for mode in (tx, ty, rotation)
    )
    dense = _dense_tangent(mechanics)
    eigenvalues = np.linalg.eigvalsh(dense)
    spectral_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    zero_tolerance = 2.0e-10 * spectral_scale
    rigid_zero_modes = int(np.count_nonzero(np.abs(eigenvalues) <= zero_tolerance))
    if affine_error > 5.0e-12 or rigid_zero_modes != 3:
        raise CandidateA2GateFailure(
            "manufactured affine/rigid spectrum gate failed",
            {"affine_error": affine_error, "rigid_zero_modes": rigid_zero_modes},
        )
    constraints = build_affine_constraints(
        geometry.coordinates,
        geometry.outer_bottom_mask,
        geometry.outer_top_mask,
        0,
        geometry.delta,
    )
    solve = solve_fixed_damage_a2(mechanics, constraints, 0.08, 1.0e-10, 1.0e-12, 3000)
    free_tangent = dense[np.ix_(constraints.free_dofs, constraints.free_dofs)]
    free_eigenvalues = np.linalg.eigvalsh(free_tangent)
    if float(free_eigenvalues[0]) <= 0.0:
        raise CandidateA2GateFailure("manufactured constrained spectrum is not positive", {})
    rng = np.random.default_rng(20260824)
    displacement = np.asarray(1.0e-2 * rng.normal(size=geometry.n_dofs), dtype=np.float64)
    direction = np.asarray(rng.normal(size=geometry.n_dofs), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    epsilon = 2.0e-5
    energy_fd = (
        mechanics.energy(displacement + epsilon * direction)
        - mechanics.energy(displacement - epsilon * direction)
    ) / (2.0 * epsilon)
    energy_error = _relative_error(energy_fd, float(mechanics.residual(displacement) @ direction))
    residual_jvp = (
        mechanics.residual(displacement + epsilon * direction)
        - mechanics.residual(displacement - epsilon * direction)
    ) / (2.0 * epsilon)
    jvp_error = _array_relative_error(residual_jvp, mechanics.tangent_action(direction))
    dense_symmetry = _array_relative_error(dense, dense.T)
    top_line = physical_interface_line_weights(geometry, "top")
    bottom_line = physical_interface_line_weights(geometry, "bottom")
    unit = solve_fixed_damage_a2(mechanics, constraints, 1.0, 1.0e-10, 1.0e-12, 3000)
    opening_slope = physical_interface_opening(unit.displacement, geometry, top_line, bottom_line)
    if opening_slope <= 0.0:
        raise CandidateA2GateFailure("manufactured opening slope is not positive", {})
    work_plus = solve_fixed_damage_a2(mechanics, constraints, 0.08 + 1.0e-5, 1.0e-10, 1.0e-12, 3000)
    work_minus = solve_fixed_damage_a2(mechanics, constraints, 0.08 - 1.0e-5, 1.0e-10, 1.0e-12, 3000)
    R_fd = (work_plus.energy - work_minus.energy) / (2.0e-5)
    D_plus = physical_interface_opening(work_plus.displacement, geometry, top_line, bottom_line)
    D_minus = physical_interface_opening(work_minus.displacement, geometry, top_line, bottom_line)
    R_eff_fd = (work_plus.energy - work_minus.energy) / (D_plus - D_minus)
    work_error = _relative_error(R_fd, solve.outer_reaction)
    opening_chain_error = _relative_error(R_eff_fd, solve.outer_reaction / opening_slope)
    metrics: dict[str, float | int | bool] = {
        **algebra,
        "material_particles": geometry.n_material,
        "union_particles": geometry.n_particles,
        "material_area": geometry.material_area,
        "grip_area": geometry.grip_area,
        "union_area": geometry.union_area,
        "affine_error": affine_error,
        "rigid_tangent_max_norm": rigid_tangent,
        "rigid_zero_modes": rigid_zero_modes,
        "first_constrained_eigenvalue": float(free_eigenvalues[0]),
        "energy_gradient_relative_error": energy_error,
        "residual_jvp_relative_error": jvp_error,
        "full_tangent_symmetry_relative_error": dense_symmetry,
        "opening_slope": opening_slope,
        "reaction_work_relative_error": work_error,
        "opening_chain_relative_error": opening_chain_error,
        "hard_gates_pass": bool(
            affine_error <= 5.0e-12
            and rigid_zero_modes == 3
            and energy_error <= 1.0e-10
            and jvp_error <= 1.0e-10
            and dense_symmetry <= 1.0e-10
            and work_error <= 1.0e-7
            and opening_chain_error <= 1.0e-7
        ),
    }
    if not metrics["hard_gates_pass"]:
        raise CandidateA2GateFailure("manufactured A2 numerical gate failed", metrics)
    return metrics


@dataclass(frozen=True)
class CandidateA2Sources:
    """Exact raw/cloud/FEM provenance and transplanted material fields."""

    coordinates: np.ndarray
    volumes: np.ndarray
    triangles: np.ndarray
    directed_edges: np.ndarray
    bottom_mask: np.ndarray
    top_mask: np.ndarray
    anchor_particle: int
    committed_load: float
    material_damage: np.ndarray
    material_history: np.ndarray
    fem_displacement: np.ndarray
    fem_history: np.ndarray
    fem_damage: np.ndarray
    fem_reaction: float
    raw_path: str
    cloud_path: str
    fem_path: str
    raw_sha256: str
    cloud_sha256: str
    fem_sha256: str
    raw_validated: bool
    raw_index: int
    fem_snapshot_index: int

    def __post_init__(self) -> None:
        coordinates = _require_float64(self.coordinates, "sources.coordinates")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("sources.coordinates must have shape (N, 2)")
        n = len(coordinates)
        for name, shape in (
            ("volumes", (n,)),
            ("bottom_mask", (n,)),
            ("top_mask", (n,)),
            ("material_damage", (n,)),
            ("material_history", (n,)),
            ("fem_displacement", (n, 2)),
            ("fem_history", (n,)),
            ("fem_damage", (n,)),
        ):
            if name.endswith("mask"):
                array = _require_bool(getattr(self, name), "sources." + name, shape)
            elif name == "volumes" or name in {"material_damage", "material_history", "fem_history", "fem_damage", "fem_displacement"}:
                array = _require_float64(getattr(self, name), "sources." + name, shape)
            else:
                array = np.asarray(getattr(self, name))
            object.__setattr__(self, name, _immutable(array))
        triangles = _require_int64(self.triangles, "sources.triangles")
        edges = _require_int64(self.directed_edges, "sources.directed_edges")
        object.__setattr__(self, "triangles", _immutable(triangles))
        object.__setattr__(self, "directed_edges", _immutable(edges))
        object.__setattr__(self, "coordinates", _immutable(coordinates))
        object.__setattr__(self, "anchor_particle", int(self.anchor_particle))
        object.__setattr__(self, "committed_load", _scalar_float(self.committed_load, "committed_load"))
        object.__setattr__(self, "fem_reaction", _scalar_float(self.fem_reaction, "fem_reaction"))
        object.__setattr__(self, "raw_validated", bool(self.raw_validated))
        object.__setattr__(self, "raw_index", int(self.raw_index))
        object.__setattr__(self, "fem_snapshot_index", int(self.fem_snapshot_index))
        for name in ("raw_path", "cloud_path", "fem_path", "raw_sha256", "cloud_sha256", "fem_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"sources.{name} must be a nonempty string")


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"provenance file does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_exact_load(values: np.ndarray, target: float, name: str) -> int:
    values = _require_float64(values, name)
    matches = np.flatnonzero(np.isclose(values, target, rtol=0.0, atol=1.0e-14))
    if len(matches) != 1 or abs(float(values[matches[0]] - target)) > 1.0e-14:
        raise ValueError(f"{name} must contain exactly one load {target!r}; matches={matches.tolist()}")
    return int(matches[0])


def _map_fem_cell_values(
    source_centroids: np.ndarray,
    target_centroids: np.ndarray,
    values: np.ndarray,
    name: str,
) -> np.ndarray:
    distances, indices = cKDTree(target_centroids).query(source_centroids, k=1)
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if len(np.unique(indices)) != len(indices) or np.max(distances, initial=0.0) > 1.0e-12:
        raise ValueError(f"{name} DG0 centroid mapping is not an exact bijection")
    mapped = np.empty(len(target_centroids), dtype=np.float64)
    mapped[indices] = _require_float64(values, name + " values", (len(source_centroids),))
    return mapped


def _lump_cell_scalar(
    triangles: np.ndarray,
    areas: np.ndarray,
    volumes: np.ndarray,
    cell_values: np.ndarray,
) -> np.ndarray:
    result = np.zeros(len(volumes), dtype=np.float64)
    contribution = areas * cell_values / 3.0
    for column in range(3):
        np.add.at(result, triangles[:, column], contribution)
    return result / volumes


def load_frozen_sources(
    raw_path: str | Path,
    cloud_path: str | Path,
    fem_path: str | Path,
) -> CandidateA2Sources:
    """Load and independently validate the exact nonproduction source files."""

    raw = Path(raw_path)
    cloud = Path(cloud_path)
    fem = Path(fem_path)
    raw_hash = _sha256(raw)
    fem_hash = _sha256(fem)
    if raw_hash != _RAW_SHA256:
        raise ValueError(f"raw source SHA256 mismatch: expected {_RAW_SHA256}, got {raw_hash}")
    if fem_hash != _FEM_SHA256:
        raise ValueError(f"FEM source SHA256 mismatch: expected {_FEM_SHA256}, got {fem_hash}")
    cloud_hash = _sha256(cloud)
    with np.load(raw, allow_pickle=False) as archive:
        required = {
            "validated", "coordinates", "triangles", "directed_edges", "u_bar",
            "damage", "history", "displacement",
        }
        if not required.issubset(archive.files):
            raise ValueError(f"raw source is missing required fields: {sorted(required - set(archive.files))}")
        validated = np.asarray(archive["validated"])
        if validated.shape != () or validated.dtype != _BOOL or bool(validated):
            raise ValueError("raw source must carry the exact validated=False flag")
        raw_coordinates = _require_float64(archive["coordinates"], "raw coordinates")
        raw_triangles = _require_int64(archive["triangles"], "raw triangles")
        raw_edges = _require_int64(archive["directed_edges"], "raw directed_edges")
        raw_loads = _require_float64(archive["u_bar"], "raw u_bar")
        raw_damage = _require_float64(archive["damage"], "raw damage")
        raw_history = _require_float64(archive["history"], "raw history")
        raw_displacement = _require_float64(archive["displacement"], "raw displacement")
        raw_index = _require_exact_load(raw_loads, 0.0055, "raw u_bar")
    with np.load(cloud, allow_pickle=False) as archive:
        required = {"coordinates", "volumes", "triangles", "bottom", "top", "anchor_particle", "d048"}
        if not required.issubset(archive.files):
            raise ValueError(f"cloud source is missing required fields: {sorted(required - set(archive.files))}")
        coordinates = _require_float64(archive["coordinates"], "cloud coordinates")
        volumes = _require_float64(archive["volumes"], "cloud volumes")
        triangles = _require_int64(archive["triangles"], "cloud triangles")
        bottom = _require_bool(archive["bottom"], "cloud bottom mask")
        top = _require_bool(archive["top"], "cloud top mask")
        edges = _require_int64(archive["d048"], "cloud d048 edges")
        anchor_raw = np.asarray(archive["anchor_particle"])
        if anchor_raw.shape != () or not np.issubdtype(anchor_raw.dtype, np.integer):
            raise ValueError("cloud anchor_particle must be a scalar integer")
        anchor = int(anchor_raw)
    if raw_coordinates.shape != coordinates.shape or not np.array_equal(raw_coordinates, coordinates):
        raise ValueError("raw/cloud coordinates are not bitwise identical")
    if not np.array_equal(raw_triangles, triangles):
        raise ValueError("raw/cloud triangles are not exactly identical")
    if not np.array_equal(raw_edges, edges):
        raise ValueError("raw/cloud d048 directed edges are not exactly identical")
    if raw_damage.ndim != 2 or raw_damage.shape[1] != len(coordinates) or raw_damage.shape[0] != len(raw_loads):
        raise ValueError("raw damage has an incompatible load/particle shape")
    if raw_history.shape != raw_damage.shape or raw_displacement.shape != (len(raw_loads), 2 * len(coordinates)):
        raise ValueError("raw history/displacement shapes are incompatible")
    _strict_graph_validation(coordinates, edges, _DELTA_D048, "cloud d048 edges")
    _, reconstructed_volumes, reconstructed_moments = _triangle_quadrature(
        coordinates, triangles, "cloud material triangles"
    )
    if not np.array_equal(reconstructed_volumes, volumes):
        raise ValueError("reconstructed/cloud material volumes are not bitwise identical")
    if np.any(raw_history[raw_index] < 0.0):
        raise ValueError("raw committed history is negative")

    with np.load(fem, allow_pickle=False) as archive:
        fem_coordinates = _require_float64(archive["coords"], "FEM coordinates")
        fem_triangles_raw = np.asarray(archive["triangles"])
        if not np.issubdtype(fem_triangles_raw.dtype, np.integer):
            raise ValueError("FEM triangles must contain integer indices")
        fem_triangles = np.asarray(fem_triangles_raw, dtype=np.int64)
        fem_snapshot_loads = _require_float64(archive["snapshot_u_bar"], "FEM snapshot_u_bar")
        fem_loads = _require_float64(archive["u_bar"], "FEM u_bar")
        fem_u = _require_float64(archive["u"], "FEM displacement")
        fem_d = _require_float64(archive["d"], "FEM damage")
        fem_h_coords = _require_float64(archive["H_coords"], "FEM H_coords")
        fem_H = _require_float64(archive["H"], "FEM H")
        fem_reactions = _require_float64(archive["reaction_y"], "FEM reactions")
        fem_snapshot_index = _require_exact_load(fem_snapshot_loads, 0.0056, "FEM snapshot_u_bar")
        fem_history_index = _require_exact_load(fem_loads, 0.0056, "FEM u_bar")
    if fem_coordinates.shape != coordinates.shape or not np.array_equal(fem_coordinates, coordinates):
        raise ValueError("FEM/cloud coordinates are not bitwise identical")
    if fem_u.shape[1:] != (len(coordinates), 2) or fem_d.shape[1:] != (len(coordinates),):
        raise ValueError("FEM particle field shapes are incompatible with cloud")
    cloud_keys = {tuple(sorted(map(int, triangle))): index for index, triangle in enumerate(triangles)}
    mapped_triangles: list[int] = []
    for triangle in fem_triangles:
        key = tuple(sorted(map(int, triangle)))
        if key not in cloud_keys:
            raise ValueError("FEM triangle is absent from the exact cloud mesh")
        mapped_triangles.append(cloud_keys[key])
    if len(mapped_triangles) != len(triangles) or len(set(mapped_triangles)) != len(triangles):
        raise ValueError("FEM/cloud triangle mapping is not bijective")
    fem_areas, fem_centroids, _ = _triangle_geometry(fem_coordinates, fem_triangles)
    cloud_areas, cloud_centroids, _ = _triangle_geometry(coordinates, triangles)
    if not np.allclose(np.sort(fem_areas), np.sort(cloud_areas), rtol=0.0, atol=1.0e-14):
        raise ValueError("FEM/cloud triangle areas differ")
    fem_H_in_fem_order = fem_H[fem_history_index]
    fem_H_cloud = _map_fem_cell_values(fem_h_coords, cloud_centroids, fem_H_in_fem_order, "FEM H")
    mapped_history = _lump_cell_scalar(triangles, cloud_areas, volumes, fem_H_cloud)
    cell_mass = float(np.sum(cloud_areas * fem_H_cloud))
    particle_mass = float(np.sum(volumes * mapped_history))
    if abs(cell_mass - particle_mass) > 2.0e-12 * max(cell_mass, 1.0):
        raise ValueError("FEM DG0-to-particle history mass is not conserved")
    return CandidateA2Sources(
        coordinates=coordinates,
        volumes=volumes,
        triangles=triangles,
        directed_edges=edges,
        bottom_mask=bottom,
        top_mask=top,
        anchor_particle=anchor,
        committed_load=float(raw_loads[raw_index]),
        material_damage=raw_damage[raw_index],
        material_history=raw_history[raw_index],
        fem_displacement=fem_u[fem_snapshot_index],
        fem_history=mapped_history,
        fem_damage=fem_d[fem_snapshot_index],
        fem_reaction=float(fem_reactions[fem_history_index]),
        raw_path=str(raw),
        cloud_path=str(cloud),
        fem_path=str(fem),
        raw_sha256=raw_hash,
        cloud_sha256=cloud_hash,
        fem_sha256=fem_hash,
        raw_validated=False,
        raw_index=raw_index,
        fem_snapshot_index=fem_snapshot_index,
    )


def _triangle_geometry(
    coordinates: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0 = coordinates[triangles[:, 0]]
    p1 = coordinates[triangles[:, 1]]
    p2 = coordinates[triangles[:, 2]]
    signed = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (
        p1[:, 1] - p0[:, 1]
    ) * (p2[:, 0] - p0[:, 0])
    return 0.5 * np.abs(signed), (p0 + p1 + p2) / 3.0, signed


class CandidateA2GateFailure(RuntimeError):
    """Raised when a diagnostic hard gate fails before the next stage."""

    def __init__(self, message: str, metrics: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.metrics = dict(metrics)


@dataclass(frozen=True)
class CandidateA2Run:
    """In-memory successful hard-gate result used by the staged driver."""

    geometry: UnionGripGeometry
    mechanics: FixedDamagePhaseMeasuredMechanics
    constraints: AffineConstraints
    sources: CandidateA2Sources
    metrics: Mapping[str, Any]
    unit_result: A2MechanicsSolveResult
    scaled_result: A2MechanicsSolveResult
    opening_slope: float
    target_opening: float


def _actual_spectrum_gates(
    mechanics: FixedDamagePhaseMeasuredMechanics,
    constraints: AffineConstraints,
) -> dict[str, float | int]:
    modes = _rigid_modes(mechanics.coordinates)
    rigid_norms = [float(np.linalg.norm(mechanics.tangent_action(mode))) for mode in modes]
    rigid_scale = max(float(np.linalg.norm(mechanics.tangent_action(np.ones(mechanics.n_dofs, dtype=np.float64)))), 1.0)
    if max(rigid_norms) > 5.0e-10 * rigid_scale:
        raise CandidateA2GateFailure("actual union rigid-mode tangent gate failed", {"rigid_norms": rigid_norms})
    mode_matrix = np.column_stack(modes)
    q_modes, _ = np.linalg.qr(mode_matrix)
    n_dofs = mechanics.n_dofs

    if n_dofs <= 1000:
        # The finite manufactured contract is intentionally small enough for
        # an exact dense audit.  Production union clouds take the matrix-free
        # branch below and make no dense zero-mode count claim.
        dense = _dense_tangent(mechanics)
        eigenvalues = np.linalg.eigvalsh(dense)
        positive = eigenvalues[eigenvalues > 1.0e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))]
        if len(positive) == 0:
            raise CandidateA2GateFailure("manufactured spectrum has no positive Ritz value", {})
        free_dense = dense[np.ix_(constraints.free_dofs, constraints.free_dofs)]
        free_values = np.linalg.eigvalsh(free_dense)
        if float(free_values[0]) <= 0.0:
            raise CandidateA2GateFailure("manufactured constrained spectrum is not positive", {})
        return {
            "rigid_translation_x_tangent_norm": rigid_norms[0],
            "rigid_translation_y_tangent_norm": rigid_norms[1],
            "rigid_rotation_tangent_norm": rigid_norms[2],
            "projected_unconstrained_positive_ritz": float(np.min(positive)),
            "projected_unconstrained_ritz_relative_residual": 0.0,
            "projected_unconstrained_ritz_absolute_residual": 0.0,
            "constrained_minimum_ritz": float(free_values[0]),
            "constrained_ritz_relative_residual": 0.0,
            "constrained_ritz_absolute_residual": 0.0,
        }

    def full_action(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if values.dtype != _FLOAT64 or values.ndim not in (1, 2) or values.shape[0] != n_dofs:
            raise ValueError("spectrum vector/matrix must be exact float64 with leading size n_dofs")
        if not np.all(np.isfinite(values)):
            raise ValueError("spectrum vector/matrix must be finite")
        one_column = values.ndim == 1
        if one_column:
            values = values[:, None]
        projected = values - q_modes @ (q_modes.T @ values)
        result = np.column_stack(
            [mechanics.tangent_action(projected[:, column]) for column in range(projected.shape[1])]
        )
        result = result - q_modes @ (q_modes.T @ result)
        return result[:, 0] if one_column else result

    full_operator = LinearOperator(
        (n_dofs, n_dofs), matvec=full_action, matmat=full_action, dtype=np.float64
    )
    full_blocks = mechanics.tangent_block_diagonal
    full_inverse = tuple(np.linalg.inv(block) for block in full_blocks)

    def full_preconditioner(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        one_column = values.ndim == 1
        if one_column:
            values = values[:, None]
        reshaped = values.reshape(mechanics.n_particles, 2, -1)
        result = np.empty_like(reshaped)
        for particle, inverse in enumerate(full_inverse):
            result[particle] = inverse @ reshaped[particle]
        result = result.reshape(n_dofs, -1)
        return result[:, 0] if one_column else result

    full_preconditioner_operator = LinearOperator(
        (n_dofs, n_dofs), matvec=full_preconditioner, matmat=full_preconditioner, dtype=np.float64
    )
    # These fixed seeds are part of the diagnostic reproducibility contract;
    # the first spectrum block was independently observed to converge from
    # this pair rather than relying on an accidental random start.
    full_rng = np.random.default_rng(1)
    full_initial = full_rng.normal(size=(n_dofs, 2)).astype(np.float64)
    full_initial -= q_modes @ (q_modes.T @ full_initial)
    full_initial, _ = np.linalg.qr(full_initial)
    with warnings.catch_warnings():
        # LOBPCG can report a harmless 1e-10-level Gram symmetry warning from
        # the matrix-free projection.  The authoritative residual below is
        # not bypassed or relaxed.
        warnings.simplefilter("ignore", UserWarning)
        full_values, full_vectors = lobpcg(
            full_operator,
            full_initial,
            M=full_preconditioner_operator,
            Y=q_modes,
            largest=False,
            tol=1.0e-7,
            maxiter=600,
            verbosityLevel=0,
        )
    projected_positive = float(np.min(full_values))
    selected = int(np.argmin(full_values))
    selected_vector = full_vectors[:, selected]
    full_residual_absolute = float(
        np.linalg.norm(full_action(selected_vector) - projected_positive * selected_vector)
    )
    full_residual_norm = full_residual_absolute / max(abs(projected_positive), 1.0e-300)
    if full_residual_norm >= 1.0e-7 or projected_positive - full_residual_absolute <= 0.0:
        raise CandidateA2GateFailure(
            "projected unconstrained union spectrum lower-bound gate failed",
            {
                "projected_positive": projected_positive,
                "projected_residual": full_residual_norm,
                "projected_absolute_residual": full_residual_absolute,
            },
        )

    free = constraints.free_dofs

    def free_action(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if values.dtype != _FLOAT64 or values.ndim not in (1, 2) or values.shape[0] != len(free):
            raise ValueError("free spectrum vector/matrix has an invalid float64 shape")
        one_column = values.ndim == 1
        if one_column:
            values = values[:, None]
        full = np.zeros((n_dofs, values.shape[1]), dtype=np.float64)
        full[free] = values
        result = np.column_stack(
            [mechanics.tangent_action(full[:, column])[free] for column in range(values.shape[1])]
        )
        return result[:, 0] if one_column else result

    free_operator = LinearOperator(
        (len(free), len(free)), matvec=free_action, matmat=free_action, dtype=np.float64
    )
    free_inverse_blocks: list[tuple[np.ndarray, np.ndarray]] = []
    positions = constraints.free_positions
    for particle in range(mechanics.n_particles):
        components = np.flatnonzero(positions[2 * particle : 2 * particle + 2] >= 0)
        if len(components):
            free_inverse_blocks.append(
                (
                    positions[2 * particle + components],
                    np.linalg.inv(full_blocks[particle][np.ix_(components, components)]),
                )
            )

    def free_preconditioner(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        one_column = values.ndim == 1
        if one_column:
            values = values[:, None]
        result = np.zeros_like(values)
        for block_positions, inverse in free_inverse_blocks:
            result[block_positions] = inverse @ values[block_positions]
        return result[:, 0] if one_column else result

    free_preconditioner_operator = LinearOperator(
        (len(free), len(free)), matvec=free_preconditioner, matmat=free_preconditioner, dtype=np.float64
    )
    free_rng = np.random.default_rng(3)
    free_initial = free_rng.normal(size=(len(free), 2)).astype(np.float64)
    free_initial, _ = np.linalg.qr(free_initial)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        constrained_values, constrained_vectors = lobpcg(
            free_operator,
            free_initial,
            M=free_preconditioner_operator,
            largest=False,
            tol=1.0e-7,
            maxiter=600,
            verbosityLevel=0,
        )
    constrained_minimum = float(np.min(constrained_values))
    constrained_index = int(np.argmin(constrained_values))
    constrained_vector = constrained_vectors[:, constrained_index]
    constrained_residual = free_action(constrained_vector) - constrained_minimum * constrained_vector
    constrained_absolute_residual = float(np.linalg.norm(constrained_residual))
    constrained_residual_norm = constrained_absolute_residual / max(abs(constrained_minimum), 1.0e-300)
    if constrained_residual_norm >= 1.0e-7 or constrained_minimum - constrained_absolute_residual <= 0.0:
        raise CandidateA2GateFailure(
            "constrained union spectrum lower-bound gate failed",
            {
                "constrained_minimum": constrained_minimum,
                "constrained_residual": constrained_residual_norm,
                "constrained_absolute_residual": constrained_absolute_residual,
            },
        )
    return {
        "rigid_translation_x_tangent_norm": rigid_norms[0],
        "rigid_translation_y_tangent_norm": rigid_norms[1],
        "rigid_rotation_tangent_norm": rigid_norms[2],
        "projected_unconstrained_positive_ritz": projected_positive,
        "projected_unconstrained_ritz_relative_residual": full_residual_norm,
        "projected_unconstrained_ritz_absolute_residual": full_residual_absolute,
        "constrained_minimum_ritz": constrained_minimum,
        "constrained_ritz_relative_residual": constrained_residual_norm,
        "constrained_ritz_absolute_residual": constrained_absolute_residual,
    }


def run_union_hard_gates(
    sources: CandidateA2Sources,
    h_g: Any,
    *,
    youngs_modulus: Any,
    poisson_ratio: Any,
    k_res: Any,
    rank_rtol: Any,
    cg_rtol: Any,
    cg_atol: Any,
    cg_maxiter: Any,
    visibility_tolerance: Any,
    target_opening: Any,
) -> CandidateA2Run:
    """Run every A2 hard gate on one collar spacing.

    This function intentionally raises on the first mathematical/structural
    failure.  The staged driver calls it at ``h_g=0.02`` before authorizing
    ``h_g=0.01`` or any transition comparison.
    """

    if not isinstance(sources, CandidateA2Sources):
        raise TypeError("sources must be CandidateA2Sources")
    geometry = build_union_geometry_from_arrays(
        sources.coordinates,
        sources.triangles,
        sources.volumes,
        sources.bottom_mask,
        sources.top_mask,
        sources.anchor_particle,
        sources.directed_edges,
        _DELTA_D048,
        h_g,
        visibility_tolerance,
    )
    expected_area = 1.0722509374813683
    expected_moments = np.asarray([0.5421241109732579, 0.5361254687406843], dtype=np.float64)
    area_defect = abs(geometry.union_area - expected_area)
    moment_defect = float(np.max(np.abs(geometry.union_first_moments - expected_moments)))
    area_limit = _AREA_RECONSTRUCTION_EPS_FACTOR * np.finfo(np.float64).eps * max(1.0, expected_area)
    if area_defect > area_limit or moment_defect > area_limit:
        raise CandidateA2GateFailure(
            "actual union area/first-moment gate failed",
            {"area_defect": area_defect, "moment_defect": moment_defect, "area_limit": area_limit},
        )
    mechanics = FixedDamagePhaseMeasuredMechanics(
        geometry,
        sources.material_damage,
        youngs_modulus,
        poisson_ratio,
        k_res,
        rank_rtol,
    )
    reference = independent_union_ba_reference(
        geometry,
        sources.material_damage,
        k_res,
        rank_rtol,
    )
    algebra = _compare_reference_to_mechanics(geometry, mechanics, reference)
    if algebra["shape_rank_min"] < 2 or not np.isfinite(algebra["shape_condition_max"]):
        raise CandidateA2GateFailure("actual union support rank/condition gate failed", algebra)
    for key in (
        "support_index_max_abs_defect",
        "beta_max_abs_defect",
        "xi_max_abs_defect",
        "shape_inverse_max_abs_defect",
        "center_factor_W_eta_max_abs_defect",
        "center_factor_W_over_Vstar_max_abs_defect",
        "phase_center_weight_max_abs_defect",
        "eta_sum_max_abs_defect",
    ):
        if algebra[key] > 1.0e-10:
            raise CandidateA2GateFailure(f"actual independent algebra gate failed: {key}", algebra)
    constraints = build_affine_constraints(
        geometry.coordinates,
        geometry.outer_bottom_mask,
        geometry.outer_top_mask,
        sources.anchor_particle,
        geometry.delta,
    )
    if np.any(np.isin(geometry.material_bottom_indices, constraints.constrained_dofs // 2)) or np.any(np.isin(geometry.material_top_indices, constraints.constrained_dofs // 2)):
        # This branch checks particle membership only for a clearer failure
        # message; component ownership is checked below.
        material_interface_y_dofs = np.concatenate((2 * geometry.material_bottom_indices + 1, 2 * geometry.material_top_indices + 1))
        if np.any(np.isin(material_interface_y_dofs, constraints.constrained_dofs)):
            raise CandidateA2GateFailure("physical material interface DOFs were constrained", {})
    top_line = physical_interface_line_weights(geometry, "top")
    bottom_line = physical_interface_line_weights(geometry, "bottom")
    unit = solve_fixed_damage_a2(mechanics, constraints, 1.0, cg_rtol, cg_atol, cg_maxiter)
    opening_slope = physical_interface_opening(unit.displacement, geometry, top_line, bottom_line)
    if not np.isfinite(opening_slope) or opening_slope <= 0.0:
        raise CandidateA2GateFailure("actual physical opening slope is not positive", {"opening_slope": opening_slope})
    target = _scalar_float(target_opening, "target_opening")
    if target <= 0.0:
        raise ValueError("target_opening must be positive")
    U_ext = target / opening_slope
    scaled = _scale_a2_solve_result(
        unit,
        mechanics,
        U_ext,
        _scalar_float(cg_rtol, "cg_rtol"),
        _scalar_float(cg_atol, "cg_atol"),
    )
    scaled_opening = physical_interface_opening(scaled.displacement, geometry, top_line, bottom_line)
    opening_scale_defect = abs(scaled_opening - target)
    opening_scale_limit = _AREA_RECONSTRUCTION_EPS_FACTOR * np.finfo(np.float64).eps * max(1.0, abs(target))
    if opening_scale_defect > opening_scale_limit:
        raise CandidateA2GateFailure("scaled physical opening gate failed", {"opening": scaled_opening, "target": target, "defect": opening_scale_defect})
    # Independent energy/residual checks use a deterministic normalized
    # direction and the exact epsilon specified by the executable contract.
    rng = np.random.default_rng(20260824)
    displacement = np.asarray(1.0e-2 * rng.normal(size=geometry.n_dofs), dtype=np.float64)
    direction = np.asarray(rng.normal(size=geometry.n_dofs), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    epsilon = 2.0e-5
    reference_energy, reference_residual, reference_f, reference_strain = _reference_response(
        geometry,
        reference,
        displacement,
        mechanics.youngs_modulus,
        mechanics.poisson_ratio,
    )
    evaluation = mechanics.evaluate(displacement)
    independent_energy_defect = _relative_error(evaluation.energy, reference_energy)
    independent_residual_defect = _array_relative_error(evaluation.residual, reference_residual)
    independent_f_defect = _array_relative_error(evaluation.edge_F, reference_f)
    independent_strain_defect = _array_relative_error(evaluation.edge_strain, reference_strain)
    energy_fd = (
        mechanics.energy(displacement + epsilon * direction)
        - mechanics.energy(displacement - epsilon * direction)
    ) / (2.0 * epsilon)
    energy_gradient_error = _relative_error(energy_fd, float(evaluation.residual @ direction))
    residual_jvp = (
        mechanics.residual(displacement + epsilon * direction)
        - mechanics.residual(displacement - epsilon * direction)
    ) / (2.0 * epsilon)
    residual_jvp_error = _array_relative_error(residual_jvp, mechanics.tangent_action(direction))
    full_symmetry = _bilinear_symmetry_defect(mechanics, None)
    free_symmetry = _bilinear_symmetry_defect(mechanics, constraints.free_dofs)
    if max(independent_energy_defect, independent_residual_defect, independent_f_defect, independent_strain_defect) > 1.0e-10:
        raise CandidateA2GateFailure(
            "actual independent energy/residual reference gate failed",
            {
                "independent_energy_defect": independent_energy_defect,
                "independent_residual_defect": independent_residual_defect,
                "independent_F_defect": independent_f_defect,
                "independent_strain_defect": independent_strain_defect,
            },
        )
    if energy_gradient_error > 1.0e-7 or residual_jvp_error > 1.0e-7 or full_symmetry > 1.0e-10 or free_symmetry > 1.0e-10:
        raise CandidateA2GateFailure(
            "actual variational/symmetry gate failed",
            {
                "energy_gradient_error": energy_gradient_error,
                "residual_jvp_error": residual_jvp_error,
                "full_symmetry": full_symmetry,
                "free_symmetry": free_symmetry,
            },
        )
    tx, ty, rotation = _rigid_modes(geometry.coordinates)
    affine_f = np.asarray([[1.12, 0.035], [-0.018, 0.93]], dtype=np.float64)
    translation = np.asarray([0.13, -0.22], dtype=np.float64)
    affine_q = (geometry.coordinates @ (affine_f - np.eye(2)).T + translation).ravel()
    affine_error = float(np.max(np.abs(mechanics.evaluate(affine_q).edge_F - affine_f)))
    rigid_tangent_errors = [float(np.linalg.norm(mechanics.tangent_action(mode))) for mode in (tx, ty, rotation)]
    if affine_error > 5.0e-12:
        raise CandidateA2GateFailure("actual affine patch gate failed", {"affine_error": affine_error})
    spectrum = _actual_spectrum_gates(mechanics, constraints)
    work_epsilon = 1.0e-5
    plus = solve_fixed_damage_a2(
        mechanics,
        constraints,
        U_ext + work_epsilon,
        _WORK_CG_RTOL,
        _WORK_CG_ATOL,
        cg_maxiter,
    )
    minus = solve_fixed_damage_a2(
        mechanics,
        constraints,
        U_ext - work_epsilon,
        _WORK_CG_RTOL,
        _WORK_CG_ATOL,
        cg_maxiter,
    )
    R_ext_fd = (plus.energy - minus.energy) / (2.0 * work_epsilon)
    D_plus = physical_interface_opening(plus.displacement, geometry, top_line, bottom_line)
    D_minus = physical_interface_opening(minus.displacement, geometry, top_line, bottom_line)
    R_eff_fd = (plus.energy - minus.energy) / (D_plus - D_minus)
    R_eff = scaled.outer_reaction / opening_slope
    reaction_work_error = _relative_error(R_ext_fd, scaled.outer_reaction)
    opening_chain_error = _relative_error(R_eff_fd, R_eff)
    force_error = float(np.linalg.norm(scaled.total_force))
    moment_error = abs(scaled.total_moment)
    reaction_balance = _reaction_balance_metrics(scaled, mechanics)
    force_limit = 1.0e-7 * max(1.0, abs(scaled.outer_reaction))
    if reaction_work_error > 1.0e-7 or opening_chain_error > 1.0e-7:
        raise CandidateA2GateFailure(
            "actual reaction/opening work gate failed",
            {
                "reaction_work_error": reaction_work_error,
                "opening_chain_error": opening_chain_error,
                "R_ext": scaled.outer_reaction,
                "R_eff": R_eff,
            },
        )
    if (
        force_error > force_limit
        or moment_error > force_limit
        or reaction_balance["equilibrium_force_error"] > force_limit
        or reaction_balance["equilibrium_reference_moment_error"] > force_limit
    ):
        raise CandidateA2GateFailure(
            "actual force/moment conservation gate failed",
            {
                "force_error": force_error,
                "moment_error": moment_error,
                "reaction_balance": reaction_balance,
                "limit": force_limit,
            },
        )
    material_metrics = compute_material_metrics(mechanics, scaled.displacement, sources.material_history)
    metrics: dict[str, Any] = {
        "nonproduction": True,
        "validated": False,
        "h_g": geometry.h_g,
        "delta": geometry.delta,
        "material_particles": geometry.n_material,
        "union_particles": geometry.n_particles,
        "material_directed_edges": len(sources.directed_edges),
        "union_directed_edges": len(geometry.directed_edges),
        "union_removed_directed_edges": len(geometry.removed_edges),
        "material_area": geometry.material_area,
        "grip_area": geometry.grip_area,
        "union_area": geometry.union_area,
        "union_area_defect": area_defect,
        "union_first_moment_x": geometry.union_first_moments[0],
        "union_first_moment_y": geometry.union_first_moments[1],
        "union_first_moment_max_defect": moment_defect,
        "area_limit": area_limit,
        "outer_bottom_particles": int(np.count_nonzero(geometry.outer_bottom_mask)),
        "outer_top_particles": int(np.count_nonzero(geometry.outer_top_mask)),
        "interface_bottom_particles": int(np.count_nonzero(geometry.interface_bottom_mask)),
        "interface_top_particles": int(np.count_nonzero(geometry.interface_top_mask)),
        "opening_slope": opening_slope,
        "U_ext": U_ext,
        "Delta_m": scaled_opening,
        "R_ext": scaled.outer_reaction,
        "R_eff": R_eff,
        "R_ext_work_fd": R_ext_fd,
        "R_eff_work_fd": R_eff_fd,
        "reaction_work_relative_error": reaction_work_error,
        "opening_chain_relative_error": opening_chain_error,
        "free_residual_norm": scaled.free_residual_norm,
        "free_residual_relative_norm": scaled.free_residual_relative_norm,
        "cg_iterations": scaled.cg_iterations,
        "cg_info": scaled.cg_info,
        "work_cg_rtol": _WORK_CG_RTOL,
        "work_cg_atol": _WORK_CG_ATOL,
        "force_error": force_error,
        "moment_error": moment_error,
        "force_limit": force_limit,
        **reaction_balance,
        "affine_error": affine_error,
        "translation_x_tangent_norm": rigid_tangent_errors[0],
        "translation_y_tangent_norm": rigid_tangent_errors[1],
        "rotation_tangent_norm": rigid_tangent_errors[2],
        "independent_energy_defect": independent_energy_defect,
        "independent_residual_defect": independent_residual_defect,
        "independent_F_defect": independent_f_defect,
        "independent_strain_defect": independent_strain_defect,
        "energy_gradient_relative_error": energy_gradient_error,
        "residual_jvp_relative_error": residual_jvp_error,
        "full_tangent_symmetry_relative_error": full_symmetry,
        "free_tangent_symmetry_relative_error": free_symmetry,
        **algebra,
        **spectrum,
        "material_psi_plus_min": float(np.min(material_metrics.psi_plus)),
        "material_psi_plus_max": float(np.max(material_metrics.psi_plus)),
        "material_H_used_min": float(np.min(material_metrics.H_used)),
        "material_H_used_max": float(np.max(material_metrics.H_used)),
        "hard_gates_pass": True,
    }
    return CandidateA2Run(
        geometry=geometry,
        mechanics=mechanics,
        constraints=constraints,
        sources=sources,
        metrics=metrics,
        unit_result=unit,
        scaled_result=scaled,
        opening_slope=opening_slope,
        target_opening=target,
    )


@dataclass(frozen=True)
class A2TransitionResult:
    """Frozen transition comparison at matched physical interface opening."""

    metrics: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]


def _material_only_transition(sources: CandidateA2Sources) -> dict[str, Any]:
    intact = BondAssociatedMechanics(
        sources.coordinates,
        sources.volumes,
        sources.directed_edges,
        _DELTA_D048,
        210.0e3,
        0.3,
        1.0e-8,
        1.0e-12,
    )
    constraints = intact.build_constraints(sources.bottom_mask, sources.top_mask, sources.anchor_particle)
    fixed = FixedDamageBondAssociatedMechanics(intact, sources.material_damage)
    result = fixed.solve(constraints, 0.0056, 1.0e-7, 1.0e-10, 3000)
    evaluation = result.evaluation
    edge_psi = _amor_positive_energy(evaluation.edge_strain, fixed.lame_lambda, fixed.shear_modulus)
    driver = np.zeros(len(sources.coordinates), dtype=np.float64)
    np.add.at(driver, fixed.source, fixed.eta * edge_psi)
    H_used = compute_h_used(driver, sources.material_history)
    return {
        "result": result,
        "psi_plus": driver,
        "H_used": H_used,
    }


def run_frozen_transition(
    run: CandidateA2Run,
) -> A2TransitionResult:
    """Compare A2, material-only BA, and mapped FEM without any damage solve."""

    sources = run.sources
    geometry = run.geometry
    metrics = compute_material_metrics(run.mechanics, run.scaled_result.displacement, sources.material_history)
    material_only = _material_only_transition(sources)
    material_result = material_only["result"]
    material_psi = np.asarray(material_only["psi_plus"], dtype=np.float64)
    material_H = np.asarray(material_only["H_used"], dtype=np.float64)
    n = geometry.n_material
    top = sources.top_mask
    tip_matches = np.flatnonzero(np.all(sources.coordinates == np.asarray([0.5, 0.5]), axis=1))
    if len(tip_matches) != 1:
        raise ValueError("the original material notch tip must be one exact particle at (0.5, 0.5)")
    tip = int(tip_matches[0])
    a2_H = metrics.H_used
    fem_H = sources.fem_history
    top_a2_error = float(np.max(np.abs(a2_H[top] - fem_H[top])))
    top_material_error = float(np.max(np.abs(material_H[top] - fem_H[top])))
    top_ratio = float(np.max(a2_H[top]) / max(a2_H[tip], 1.0e-30))
    global_index = int(np.argmax(a2_H))
    distance_to_tip = float(np.linalg.norm(sources.coordinates[global_index] - sources.coordinates[tip]))
    interface_global = bool(geometry.interface_mask[global_index])
    side_corner = bool(
        (sources.coordinates[global_index, 0] == 0.0 or sources.coordinates[global_index, 0] == 1.0)
        and (sources.coordinates[global_index, 1] == 0.0 or sources.coordinates[global_index, 1] == 1.0)
    )
    no_new_max = distance_to_tip <= geometry.delta and not interface_global and not side_corner
    reaction_fem_error = _relative_error(run.scaled_result.outer_reaction / run.opening_slope, sources.fem_reaction)
    reaction_material_error = _relative_error(run.scaled_result.outer_reaction / run.opening_slope, material_result.top_reaction)
    psi_difference = metrics.psi_plus - material_psi
    psi_linf_difference = float(np.max(np.abs(psi_difference)))
    psi_linf_relative_difference = _array_relative_error(metrics.psi_plus, material_psi)
    decision = transition_decision_arithmetic(
        top_ratio,
        top_a2_error,
        top_material_error,
        no_new_max,
        reaction_fem_error,
    )
    a2_displacement = run.scaled_result.displacement[: 2 * n].reshape(n, 2)
    material_displacement = material_result.displacement.reshape(n, 2)
    arrays = {
        "a2_displacement_material": a2_displacement,
        "a2_particle_strain_material": run.scaled_result.evaluation.particle_strain[:n],
        "a2_psi_plus": metrics.psi_plus,
        "a2_H_used": metrics.H_used,
        "a2_grip_psi_plus": metrics.grip_psi_plus,
        "material_only_displacement": material_displacement,
        "material_only_particle_strain": material_result.evaluation.particle_strain,
        "material_only_psi_plus": material_psi,
        "material_only_H_used": material_H,
        "fem_displacement": sources.fem_displacement,
        "fem_history_mapped": fem_H,
    }
    for name, value in arrays.items():
        arrays[name] = _immutable(_require_float64(value, name))
    transition_metrics: dict[str, Any] = {
        "nonproduction": True,
        "validated": False,
        "comparison_load_physical_opening": _TARGET_OPENING,
        "h_g": geometry.h_g,
        "U_ext": run.scaled_result.U_ext,
        "Delta_m": physical_interface_opening(
            run.scaled_result.displacement,
            geometry,
            physical_interface_line_weights(geometry, "top"),
            physical_interface_line_weights(geometry, "bottom"),
        ),
        "R_ext": run.scaled_result.outer_reaction,
        "R_eff": run.scaled_result.outer_reaction / run.opening_slope,
        "R_material_only": material_result.top_reaction,
        "R_fem": sources.fem_reaction,
        "R_eff_relative_error_vs_fem": reaction_fem_error,
        "R_eff_relative_error_vs_material_only": reaction_material_error,
        "material_top_psi_plus": float(np.max(metrics.psi_plus[top])),
        "material_tip_psi_plus": float(metrics.psi_plus[tip]),
        "material_only_top_psi_plus": float(np.max(material_psi[top])),
        "material_only_tip_psi_plus": float(material_psi[tip]),
        "material_top_psi_plus_delta_vs_material_only": float(np.max(psi_difference[top])),
        "material_tip_psi_plus_delta_vs_material_only": float(psi_difference[tip]),
        "material_psi_plus_linf_difference_vs_material_only": psi_linf_difference,
        "material_psi_plus_linf_relative_difference_vs_material_only": psi_linf_relative_difference,
        "material_history_floor_min": float(np.min(sources.material_history)),
        "material_history_floor_max": float(np.max(sources.material_history)),
        "material_top_history_floor_max": float(np.max(sources.material_history[top])),
        "material_tip_history_floor": float(sources.material_history[tip]),
        "material_top_H_used": float(np.max(a2_H[top])),
        "material_tip_H_used": float(a2_H[tip]),
        "material_top_to_tip_H_used_ratio": top_ratio,
        "material_top_H_used_linf_error_vs_mapped_FEM": top_a2_error,
        "material_only_top_H_used_linf_error_vs_mapped_FEM": top_material_error,
        **decision,
        "instantaneous_psi_plus_comparison": {
            "a2_top_max": float(np.max(metrics.psi_plus[top])),
            "a2_tip": float(metrics.psi_plus[tip]),
            "material_only_top_max": float(np.max(material_psi[top])),
            "material_only_tip": float(material_psi[tip]),
            "linf_difference": psi_linf_difference,
            "linf_relative_difference": psi_linf_relative_difference,
        },
        "irreversible_H_used_comparison": {
            "inherited_history_floor_min": float(np.min(sources.material_history)),
            "inherited_history_floor_max": float(np.max(sources.material_history)),
            "top_history_floor_max": float(np.max(sources.material_history[top])),
            "tip_history_floor": float(sources.material_history[tip]),
            "a2_top_max": float(np.max(a2_H[top])),
            "a2_tip": float(a2_H[tip]),
            "a2_top_error_vs_mapped_FEM": top_a2_error,
            "material_only_top_error_vs_mapped_FEM": top_material_error,
            "top_error_reduction_factor": decision["top_error_reduction_factor"],
            "top_error_reduction_fraction": decision["top_error_reduction_fraction"],
        },
        "comparison_interpretation": {
            "H_used_includes_inherited_history_floor": True,
            "H_used_error_reduction_is_not_Psi_plus_mechanics_correction": True,
        },
        "global_H_used_max_index": global_index,
        "global_H_used_max_coordinate_x": float(sources.coordinates[global_index, 0]),
        "global_H_used_max_coordinate_y": float(sources.coordinates[global_index, 1]),
        "global_H_used_max_distance_to_tip": distance_to_tip,
        "global_H_used_max_on_interface": interface_global,
        "global_H_used_max_on_side_corner": side_corner,
        "no_new_global_H_used_max": no_new_max,
        "diagnostically_promising": decision["diagnostically_promising"],
    }
    return A2TransitionResult(metrics=transition_metrics, arrays=arrays)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _source_code_manifest() -> dict[str, str]:
    source_path = Path(__file__).resolve()
    return {
        "diagnostic_module": str(source_path),
        "diagnostic_module_sha256": _sha256(source_path),
    }


def _array_schema(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {"dtype": str(np.asarray(value).dtype), "shape": list(np.asarray(value).shape)}
        for name, value in arrays.items()
    }


def _write_run_artifacts(
    run: CandidateA2Run,
    output_prefix: Path,
    transition: A2TransitionResult | None,
    run_set_id: str,
) -> dict[str, Any]:
    output_prefix = output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_prefix.with_suffix(".npz")
    json_path = output_prefix.with_suffix(".json")
    png_path = output_prefix.with_suffix(".png")
    geometry = run.geometry
    arrays: dict[str, np.ndarray] = {
        "coordinates": geometry.coordinates,
        "material_indices": geometry.material_indices,
        "material_triangles": geometry.material_triangles,
        "grip_triangles": geometry.grip_triangles,
        "union_triangles": geometry.union_triangles,
        "material_volumes": geometry.material_volumes,
        "grip_volumes": geometry.grip_volumes,
        "union_volumes": geometry.union_volumes,
        "candidate_edges": geometry.candidate_edges,
        "directed_edges": geometry.directed_edges,
        "removed_edges": geometry.removed_edges,
        "material_bottom_mask": geometry.material_bottom_mask,
        "material_top_mask": geometry.material_top_mask,
        "interface_mask": geometry.interface_mask,
        "outer_bottom_mask": geometry.outer_bottom_mask,
        "outer_top_mask": geometry.outer_top_mask,
        "unit_displacement": run.unit_result.displacement,
        "scaled_displacement": run.scaled_result.displacement,
        "material_damage": run.sources.material_damage,
        "material_history": run.sources.material_history,
    }
    if transition is not None:
        arrays.update(transition.arrays)
    np.savez_compressed(npz_path, **arrays)
    source_code = _source_code_manifest()
    report: dict[str, Any] = {
        "canonical": True,
        "run_set_id": run_set_id,
        "artifact_schema": _ARTIFACT_SCHEMA,
        "nonproduction": True,
        "validated": False,
        "model": "Candidate A2 phase-measured elastic grip frozen mechanics",
        "source_backed": [
            "Chen and Spencer (2019), DOI 10.1002/nme.5973, target-inclusive bond-associated support/shape/force basis",
            "plane-strain isotropic linear elasticity and Amor positive energy map as used by the existing fracture2d baseline",
        ],
        "project_owned": [
            "phase-measured W_i center energy and intact d=0 grips",
            "structured normal-only collar mesh and independent union visibility graph",
            "outer-face affine constraints and physical-interface opening comparison",
            "frozen raw/cloud/FEM provenance transfer and nonproduction comparison criteria",
        ],
        "geometry": {
            "h_g": geometry.h_g,
            "delta": geometry.delta,
            "material_particles": geometry.n_material,
            "union_particles": geometry.n_particles,
            "material_area": geometry.material_area,
            "grip_area": geometry.grip_area,
            "union_area": geometry.union_area,
            "union_first_moments": geometry.union_first_moments,
            "material_directed_edges": len(run.sources.directed_edges),
            "union_directed_edges": len(geometry.directed_edges),
            "removed_directed_edges": len(geometry.removed_edges),
        },
        "sources": {
            "raw_path": run.sources.raw_path,
            "cloud_path": run.sources.cloud_path,
            "fem_path": run.sources.fem_path,
            "raw_sha256": run.sources.raw_sha256,
            "cloud_sha256": run.sources.cloud_sha256,
            "fem_sha256": run.sources.fem_sha256,
            "raw_validated": run.sources.raw_validated,
            "raw_index": run.sources.raw_index,
            "committed_load": run.sources.committed_load,
            "fem_snapshot_index": run.sources.fem_snapshot_index,
        },
        "hard_gate_metrics": run.metrics,
        "transition": None if transition is None else transition.metrics,
        "source_code": source_code,
        "array_schema": _array_schema(arrays),
        "artifact_paths": {
            "json": str(json_path),
            "npz": str(npz_path),
            "png": str(png_path),
        },
        "artifact_digests": {
            "npz_sha256": _sha256(npz_path),
        },
        "artifact_npz": str(npz_path),
    }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri

        figure, axis = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
        axis.triplot(
            mtri.Triangulation(
                geometry.coordinates[:, 0], geometry.coordinates[:, 1], geometry.union_triangles
            ),
            color="#3b4a5a",
            linewidth=0.25,
        )
        axis.scatter(
            geometry.coordinates[geometry.outer_bottom_mask, 0],
            geometry.coordinates[geometry.outer_bottom_mask, 1],
            s=5,
            label="outer bottom",
        )
        axis.scatter(
            geometry.coordinates[geometry.outer_top_mask, 0],
            geometry.coordinates[geometry.outer_top_mask, 1],
            s=5,
            label="outer top",
        )
        axis.set_aspect("equal")
        axis.set_title(f"Candidate A2 union diagnostic, h_g={geometry.h_g:g} (nonproduction)")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.legend(loc="best", fontsize=8)
        figure.savefig(png_path, dpi=180, facecolor="white")
        plt.close(figure)
    except ImportError as error:
        raise RuntimeError("Candidate A2 artifact plotting requires matplotlib") from error
    report["artifact_digests"]["png_sha256"] = _sha256(png_path)
    json_path.write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "canonical": True,
        "run_set_id": run_set_id,
        "h_g": geometry.h_g,
        "artifact_schema": _ARTIFACT_SCHEMA,
        "source_code": source_code,
        "json_path": str(json_path),
        "npz_path": str(npz_path),
        "png_path": str(png_path),
        "json_sha256": _sha256(json_path),
        "npz_sha256": _sha256(npz_path),
        "png_sha256": _sha256(png_path),
        "array_count": len(arrays),
    }


def run_staged_diagnostic(
    raw_path: str | Path,
    cloud_path: str | Path,
    fem_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Run manufactured gates, then h_g=.02, then authorize .01/transition."""

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_set_id = output.name
    sources = load_frozen_sources(raw_path, cloud_path, fem_path)
    manufactured = run_manufactured_gates()
    common = {
        "youngs_modulus": 210.0e3,
        "poisson_ratio": 0.3,
        "k_res": 1.0e-8,
        "rank_rtol": 1.0e-12,
        "cg_rtol": 1.0e-7,
        "cg_atol": 1.0e-10,
        "cg_maxiter": 3000,
        "visibility_tolerance": 0.0,
        "target_opening": _TARGET_OPENING,
    }
    first = run_union_hard_gates(sources, 0.02, **common)
    second = run_union_hard_gates(sources, 0.01, **common)
    first_transition = run_frozen_transition(first)
    second_transition = run_frozen_transition(second)
    first_manifest = _write_run_artifacts(
        first,
        output / "fracture2d_candidate_a2_hg_002",
        first_transition,
        run_set_id,
    )
    second_manifest = _write_run_artifacts(
        second,
        output / "fracture2d_candidate_a2_hg_001",
        second_transition,
        run_set_id,
    )
    comparison_path = output / "fracture2d_candidate_a2_canonical_comparison.json"
    summary = {
        "canonical": True,
        "artifact_schema": _ARTIFACT_SCHEMA,
        "run_set_id": run_set_id,
        "nonproduction": True,
        "validated": False,
        "source_code": _source_code_manifest(),
        "manufactured": manufactured,
        "h_g_0.02": first.metrics,
        "h_g_0.01": second.metrics,
        "transition_h_g_0.02": first_transition.metrics,
        "transition_h_g_0.01": second_transition.metrics,
        "transition_authorized": True,
        "transition_diagnostic_usefulness": {
            "h_g_0.02": first_transition.metrics["diagnostically_promising"],
            "h_g_0.01": second_transition.metrics["diagnostically_promising"],
        },
        "canonical_artifacts": {
            "h_g_0.02": first_manifest,
            "h_g_0.01": second_manifest,
        },
        "comparison_report_path": str(comparison_path),
        "artifacts_directory": str(output),
    }
    comparison_path.write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = output / "fracture2d_candidate_a2_canonical_manifest.json"
    manifest = {
        "canonical": True,
        "artifact_schema": _ARTIFACT_SCHEMA,
        "run_set_id": run_set_id,
        "source_code": _source_code_manifest(),
        "comparison_report": {
            "path": str(comparison_path),
            "sha256": _sha256(comparison_path),
        },
        "run_artifacts": {
            "h_g=0.02": first_manifest,
            "h_g=0.01": second_manifest,
        },
    }
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["canonical_manifest_path"] = str(manifest_path)
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Candidate A2 frozen-mechanics diagnostics")
    parser.add_argument("--raw", type=Path, default=_RAW_DEFAULT)
    parser.add_argument("--cloud", type=Path, default=_CLOUD_DEFAULT)
    parser.add_argument("--fem", type=Path, default=_FEM_DEFAULT)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("/tmp/opencode/fracture2d_candidate_a2_canonical"),
    )
    arguments = parser.parse_args()
    try:
        summary = run_staged_diagnostic(
            arguments.raw,
            arguments.cloud,
            arguments.fem,
            arguments.output_directory,
        )
    except CandidateA2GateFailure as error:
        failure = {
            "nonproduction": True,
            "validated": False,
            "transition_authorized": False,
            "failure": str(error),
            "metrics": error.metrics,
        }
        path = arguments.output_directory / "fracture2d_candidate_a2_failure.json"
        arguments.output_directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_ready(failure), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    print(json.dumps(_json_ready(summary), sort_keys=True))


if __name__ == "__main__":
    _main()
