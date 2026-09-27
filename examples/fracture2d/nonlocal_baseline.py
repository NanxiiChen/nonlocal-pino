"""Single-entry direct nonlocal solver for the fracture2d engineering-grip model.

Adopted scheme (the best complete scheme from the research record):

* unchanged original BA/Amor/AT2 equations on the 3395-node material cloud
  (horizon ``delta=0.048``, ``l0=0.024``, material data below);
* staggered continuation with fixed damped-Picard relaxation ``omega=0.5``,
  nominal increment ``du=1e-4``, ``du_min=6.25e-6``, up to 144 sweeps per
  attempt, and honest stabilized-**warning** acceptance (every state keeps
  its true strict/warning status; no gate is renamed);
* load path ``0 -> 0.0075``; the closest-to-FEM complete solution is the
  archived run ``engineering_grip_damped_warning_o5_u7p5e3_dumin6p25e6_s144``
  (mirrored as a compact solution by ``verify``/``plot``).

Commands
--------

``run``     fresh zero-start warning continuation; writes a compact solution
            NPZ (loads, fields, reactions, statuses, geometry, metadata).
``verify``  quick identity/response audit of a compact solution or of a
            legacy stabilized archive (state count, warnings, peak force,
            terminal reaction, drift statistics, drift-excluded ranges).
``plot``    field plots for selected loads whose color ranges are computed
            over the non-drift material points only (``|u|/u_bar <= 1.08``
            by default). Drift points remain in the solution and are drawn,
            but they no longer set the plotted minimum/maximum.

This module intentionally contains no experimental controllers: only the
adopted scheme above, in one self-contained GPU-only file."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from dataclasses import dataclass, fields as dataclass_fields
from numbers import Integral
from pathlib import Path
from pinn.operators import ExplicitVolumeSparseD


from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse import coo_matrix, diags
from scipy.sparse import issparse

from scipy.sparse.linalg import MatrixRankWarning, spsolve
from scipy.spatial import cKDTree
from typing import Any
from typing import Any, Callable, Mapping, Sequence
from typing import Any, Iterable
from typing import Any, Iterable, Mapping
import argparse
import hashlib
import jax
import jax.numpy as jnp
import json
import math
import numpy as np


import sys
import time
import warnings
# ===== nonlocal_grip_diagnostic =====








_g_FLOAT64 = np.dtype(np.float64)
_g_INT64 = np.dtype(np.int64)
_g_BOOL = np.dtype(bool)
_g_GEOMETRY_EPS_FACTOR = 64.0
_g_AREA_RECONSTRUCTION_EPS_FACTOR = 512.0
_g_INTERFACE_INPUT_TOLERANCE = 1.0e-12

def _g_immutable(values: Any, dtype: Any | None = None) -> np.ndarray:
    """Copy values behind a bytes buffer so read-only flags cannot be undone."""

    source = np.array(values, dtype=dtype, copy=True, order="C")
    raw = source.tobytes(order="C")
    result = np.frombuffer(raw, dtype=source.dtype, count=source.size)
    result = result.reshape(source.shape)
    result.setflags(write=False)
    return result

def _g_scalar_float(value: Any, name: str) -> float:
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

def _g_require_float64(
    values: Any, name: str, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != _g_FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array

def _g_require_int64(
    values: Any, name: str, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != _g_INT64:
        raise ValueError(f"{name} must use exact int64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array

def _g_require_bool(
    values: Any, name: str, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != _g_BOOL:
        raise ValueError(f"{name} must use exact boolean dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array

def _g_geometry_tolerance(coordinates: np.ndarray) -> float:
    scale = max(1.0, float(np.max(np.abs(coordinates))))
    # The 1e-12 input floor is the declared tolerance needed to accept the
    # unchanged, unsnapped Gmsh boundary coordinates.  The machine component
    # remains scale-aware and is never used to alter coordinates.
    return max(
        _g_AREA_RECONSTRUCTION_EPS_FACTOR * np.finfo(np.float64).eps * scale,
        _g_INTERFACE_INPUT_TOLERANCE * scale,
    )

def _g_strict_graph_validation(
    coordinates: np.ndarray,
    directed_edges: np.ndarray,
    delta: float,
    name: str,
) -> None:
    edges = _g_require_int64(directed_edges, name)
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

def _g_triangle_quadrature(
    coordinates: np.ndarray,
    triangles: np.ndarray,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = _g_require_int64(triangles, name)
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

def _g_directed_edges_from_pairs(pairs: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must have shape (P, 2), got {pairs.shape}")
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    directed = np.vstack((pairs, pairs[:, ::-1]))
    order = np.lexsort((directed[:, 1], directed[:, 0]))
    return np.asarray(directed[order], dtype=np.int64)

def _g_all_strict_pairs(coordinates: np.ndarray, delta: float) -> np.ndarray:
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

def _g_boundary_polygon_from_triangles(
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

class _g_UnionPolygonPredicate:
    """Independent segment-in-closure predicate for one simple union polygon."""

    def __init__(self, polygon: np.ndarray, input_tolerance: float) -> None:
        polygon = _g_require_float64(polygon, "union boundary polygon")
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError("union boundary polygon must have shape (N, 2), N >= 3")
        tolerance = _g_scalar_float(input_tolerance, "visibility_tolerance")
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
            _g_GEOMETRY_EPS_FACTOR
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
        start = _g_require_float64(start, "visibility start", (2,))
        end = _g_require_float64(end, "visibility end", (2,))
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
        determinant_floor = _g_GEOMETRY_EPS_FACTOR * np.finfo(np.float64).eps * segment_length * edge_length
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

def _g_build_union_visibility_graph(
    coordinates: np.ndarray,
    union_triangles: np.ndarray,
    delta: float,
    visibility_tolerance: float,
    material_count: int,
    reference_material_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the union graph without calling Stage 1's material predicate."""

    polygon = _g_boundary_polygon_from_triangles(coordinates, union_triangles)
    predicate = _g_UnionPolygonPredicate(polygon, visibility_tolerance)
    pairs = _g_all_strict_pairs(coordinates, delta)
    visible_flags = np.asarray(
        [predicate(coordinates[i], coordinates[j]) for i, j in pairs], dtype=bool
    )
    visible_pairs = pairs[visible_flags]
    removed_pairs = pairs[~visible_flags]
    candidate_edges = _g_directed_edges_from_pairs(pairs)
    visible_edges = _g_directed_edges_from_pairs(visible_pairs)
    removed_edges = _g_directed_edges_from_pairs(removed_pairs)
    _g_strict_graph_validation(coordinates, candidate_edges, delta, "union candidate_edges")
    _g_strict_graph_validation(coordinates, visible_edges, delta, "union directed_edges")
    if len(removed_edges):
        _g_strict_graph_validation(coordinates, removed_edges, delta, "union removed_edges")
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
        delta = _g_scalar_float(self.delta, "geometry.delta")
        h_g = _g_scalar_float(self.h_g, "geometry.h_g")
        if delta <= 0.0 or h_g <= 0.0 or h_g >= delta:
            raise ValueError("geometry delta/h_g must satisfy 0 < h_g < delta")
        if not isinstance(self.n_material, Integral) or isinstance(self.n_material, bool):
            raise ValueError("geometry.n_material must be an integer")
        n_material = int(self.n_material)
        coordinates = _g_require_float64(self.coordinates, "geometry.coordinates")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) <= n_material:
            raise ValueError("geometry.coordinates must have shape (N, 2) with a nonempty collar")
        if len(np.unique(coordinates, axis=0)) != len(coordinates):
            raise ValueError("union coordinates must be unique")
        reference_coordinates = _g_require_float64(
            self.material_reference_coordinates,
            "geometry.material_reference_coordinates",
            (n_material, 2),
        )
        reference_triangles = _g_require_int64(
            self.material_reference_triangles, "geometry.material_reference_triangles"
        )
        reference_volumes = _g_require_float64(
            self.material_reference_volumes,
            "geometry.material_reference_volumes",
            (n_material,),
        )
        reference_edges = _g_require_int64(
            self.material_reference_edges, "geometry.material_reference_edges"
        )
        if not np.array_equal(coordinates[:n_material], reference_coordinates):
            raise ValueError("union coordinate prefix is not exactly the material cloud")
        if not np.array_equal(self.material_triangles, reference_triangles):
            raise ValueError("material triangles were changed in the union geometry")
        _g_strict_graph_validation(reference_coordinates, reference_edges, delta, "material reference edges")
        if not np.array_equal(
            np.asarray(self.directed_edges)[
                (np.asarray(self.directed_edges)[:, 0] < n_material)
                & (np.asarray(self.directed_edges)[:, 1] < n_material)
            ],
            reference_edges,
        ):
            raise ValueError("union graph induced material edges differ from d048")

        array_specs = (
            ("collar_offsets", _g_FLOAT64),
            ("material_indices", _g_INT64),
            ("material_triangles", _g_INT64),
            ("grip_triangles", _g_INT64),
            ("union_triangles", _g_INT64),
            ("material_triangle_areas", _g_FLOAT64),
            ("grip_triangle_areas", _g_FLOAT64),
            ("union_triangle_areas", _g_FLOAT64),
            ("material_volumes", _g_FLOAT64),
            ("grip_volumes", _g_FLOAT64),
            ("union_volumes", _g_FLOAT64),
            ("material_first_moments", _g_FLOAT64),
            ("grip_first_moments", _g_FLOAT64),
            ("union_first_moments", _g_FLOAT64),
            ("boundary_polygon", _g_FLOAT64),
            ("candidate_edges", _g_INT64),
            ("directed_edges", _g_INT64),
            ("removed_edges", _g_INT64),
            ("neighbor_counts", _g_INT64),
            ("material_bottom_indices", _g_INT64),
            ("material_top_indices", _g_INT64),
            ("outer_bottom_indices", _g_INT64),
            ("outer_top_indices", _g_INT64),
        )
        for name, dtype in array_specs:
            array = np.asarray(getattr(self, name))
            if array.dtype != dtype:
                raise ValueError(f"geometry.{name} must use dtype {dtype}, got {array.dtype}")
            if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
                raise ValueError(f"geometry.{name} contains non-finite values")
            object.__setattr__(self, name, _g_immutable(array))
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
            mask = _g_require_bool(getattr(self, name), f"geometry.{name}", (len(coordinates),))
            object.__setattr__(self, name, _g_immutable(mask))
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
        object.__setattr__(self, "collar_offsets", _g_immutable(collar_offsets))
        object.__setattr__(self, "visibility_tolerance", _g_scalar_float(self.visibility_tolerance, "visibility_tolerance"))
        object.__setattr__(self, "n_material", n_material)
        object.__setattr__(self, "coordinates", _g_immutable(coordinates))
        object.__setattr__(self, "material_reference_coordinates", _g_immutable(reference_coordinates))
        object.__setattr__(self, "material_reference_triangles", _g_immutable(reference_triangles))
        object.__setattr__(self, "material_reference_volumes", _g_immutable(reference_volumes))
        object.__setattr__(self, "material_reference_edges", _g_immutable(reference_edges))

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
        volume_tolerance = _g_AREA_RECONSTRUCTION_EPS_FACTOR * np.finfo(np.float64).eps * max(
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
            _, volumes, moments = _g_triangle_quadrature(coordinates, triangles, name + " triangles")
            expected_volumes = self.material_volumes if name == "material" else self.grip_volumes if name == "grip" else self.union_volumes
            if not np.array_equal(areas, _g_triangle_quadrature(coordinates, triangles, name + " area check")[0]):
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
                _g_strict_graph_validation(coordinates, values, delta, "geometry." + name)
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
        expected_boundary = _g_boundary_polygon_from_triangles(coordinates, self.union_triangles)
        if not np.array_equal(self.boundary_polygon, expected_boundary):
            raise ValueError("boundary polygon ownership/order does not match union triangles")
        object.__setattr__(self, "boundary_polygon", _g_immutable(self.boundary_polygon))
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

def _g_validate_interface_coordinates(
    coordinates: np.ndarray,
    mask: np.ndarray,
    expected_y: float,
    name: str,
) -> np.ndarray:
    indices = np.flatnonzero(mask).astype(np.int64)
    if len(indices) < 2:
        raise ValueError(f"{name} must contain at least two ordered particles")
    values = coordinates[indices]
    if np.any(np.abs(values[:, 1] - expected_y) > _g_geometry_tolerance(coordinates)):
        raise ValueError(f"{name} particles are not on y={expected_y}")
    order = np.lexsort((values[:, 1], values[:, 0]))
    indices = indices[order]
    x = coordinates[indices, 0]
    if np.any(np.diff(x) <= 0.0) or len(np.unique(x)) != len(x):
        raise ValueError(f"{name} x coordinates must be strictly ordered and unique")
    spacing = np.diff(x)
    median = float(np.median(spacing))
    tolerance = _g_geometry_tolerance(coordinates)
    if np.any(np.abs(spacing - median) > tolerance):
        raise ValueError(
            f"{name} spacing is not uniform within the declared float64 geometry tolerance"
        )
    return indices

def _g_structured_grip_triangles(
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

    coords = _g_require_float64(coordinates, "material coordinates")
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0:
        raise ValueError("material coordinates must have nonempty shape (N, 2)")
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError("material coordinates must be unique")
    material_triangles = _g_require_int64(triangles, "material triangles")
    material_volumes_input = _g_require_float64(volumes, "material volumes", (len(coords),))
    if np.any(material_volumes_input <= 0.0):
        raise ValueError("material volumes must be strictly positive")
    material_bottom = _g_require_bool(bottom_mask, "material bottom mask", (len(coords),))
    material_top = _g_require_bool(top_mask, "material top mask", (len(coords),))
    if not np.any(material_bottom) or not np.any(material_top) or np.any(material_bottom & material_top):
        raise ValueError("material top/bottom masks must be nonempty and disjoint")
    if not isinstance(anchor_particle, Integral) or isinstance(anchor_particle, bool):
        raise ValueError("anchor_particle must be an integer")
    anchor = int(anchor_particle)
    if anchor < 0 or anchor >= len(coords):
        raise ValueError("anchor_particle is outside the material particle range")
    if not np.array_equal(coords[anchor], np.asarray([0.0, 0.0], dtype=np.float64)):
        raise ValueError("the original anchor must be exactly the material particle (0, 0)")
    delta_value = _g_scalar_float(delta, "delta")
    h_value = _g_scalar_float(h_g, "h_g")
    if delta_value <= 0.0 or h_value <= 0.0 or h_value >= delta_value:
        raise ValueError("delta/h_g must satisfy 0 < h_g < delta")
    edges = _g_require_int64(directed_edges, "material reference edges")
    _g_strict_graph_validation(coords, edges, delta_value, "material reference edges")
    bottom_indices = _g_validate_interface_coordinates(coords, material_bottom, 0.0, "material bottom interface")
    top_indices = _g_validate_interface_coordinates(coords, material_top, 1.0, "material top interface")

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
    bottom_grip_triangles, bottom_rows = _g_structured_grip_triangles(
        coords, bottom_indices, offsets, -1.0, new_coordinates, "bottom"
    )
    top_grip_triangles, top_rows = _g_structured_grip_triangles(
        coords, top_indices, offsets, 1.0, new_coordinates, "top"
    )
    grip_triangles = np.vstack((bottom_grip_triangles, top_grip_triangles)).astype(np.int64, copy=False)
    union_coordinates = np.asarray(new_coordinates, dtype=np.float64)
    if not np.array_equal(union_coordinates[: len(coords)], coords):
        raise ValueError("structured collar construction changed the material coordinate prefix")
    union_triangles = np.vstack((material_triangles, grip_triangles)).astype(np.int64, copy=False)
    if len(np.unique(np.sort(union_triangles, axis=1), axis=0)) != len(union_triangles):
        raise ValueError("structured collar construction produced duplicate triangles")

    material_areas, material_phase_volumes, material_moments = _g_triangle_quadrature(
        union_coordinates, material_triangles, "material triangles"
    )
    grip_areas, grip_phase_volumes, grip_moments = _g_triangle_quadrature(
        union_coordinates, grip_triangles, "grip triangles"
    )
    union_areas, union_phase_volumes, union_moments = _g_triangle_quadrature(
        union_coordinates, union_triangles, "union triangles"
    )
    if not np.array_equal(material_phase_volumes[: len(coords)], material_volumes_input):
        raise ValueError("material triangle reconstruction differs from serialized material volumes")
    if not np.allclose(material_phase_volumes[len(coords) :], 0.0, rtol=0.0, atol=0.0):
        raise ValueError("material phase has volume on a non-material collar particle")
    if not np.array_equal(material_phase_volumes, np.concatenate((material_volumes_input, np.zeros(len(union_coordinates) - len(coords), dtype=np.float64)))):
        raise ValueError("material phase volume storage is not an exact material prefix")
    phase_tolerance = _g_AREA_RECONSTRUCTION_EPS_FACTOR * np.finfo(np.float64).eps * max(
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
        atol=_g_AREA_RECONSTRUCTION_EPS_FACTOR * np.finfo(np.float64).eps * max(1.0, delta_value),
    ):
        raise ValueError("union area does not equal material area plus two full-width grips")
    if not np.isclose(
        float(np.sum(grip_phase_volumes)), 2.0 * delta_value,
        rtol=0.0,
        atol=_g_AREA_RECONSTRUCTION_EPS_FACTOR * np.finfo(np.float64).eps * max(1.0, delta_value),
    ):
        raise ValueError("each full-width grip pair must have total area 2*delta")
    bottom_grip_count = 2 * (len(bottom_indices) - 1) * len(offsets)
    bottom_grip_area = float(np.sum(grip_areas[:bottom_grip_count]))
    top_grip_area = float(np.sum(grip_areas[bottom_grip_count:]))
    grip_area_tolerance = _g_AREA_RECONSTRUCTION_EPS_FACTOR * np.finfo(np.float64).eps * max(
        1.0, delta_value
    )
    if abs(bottom_grip_area - delta_value) > grip_area_tolerance or abs(top_grip_area - delta_value) > grip_area_tolerance:
        raise ValueError(
            "bottom/top structured grip areas do not each equal exact delta: "
            f"bottom={bottom_grip_area:.17g}, top={top_grip_area:.17g}, delta={delta_value:.17g}"
        )

    material_count = len(coords)
    polygon, candidate_edges, union_edges, removed_edges, neighbor_counts = _g_build_union_visibility_graph(
        union_coordinates,
        union_triangles,
        delta_value,
        _g_scalar_float(visibility_tolerance, "visibility_tolerance"),
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
        visibility_tolerance=_g_scalar_float(visibility_tolerance, "visibility_tolerance"),
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
                if array.dtype != _g_INT64:
                    raise ValueError(f"reference.{name} must be int64")
            else:
                if array.dtype != _g_FLOAT64:
                    raise ValueError(f"reference.{name} must be float64")
            object.__setattr__(self, name, _g_immutable(array))
        grip_degradation = _g_scalar_float(self.grip_degradation, "reference.grip_degradation")
        if grip_degradation <= 0.0:
            raise ValueError("reference.grip_degradation must be positive")
        object.__setattr__(self, "grip_degradation", grip_degradation)
        object.__setattr__(
            self,
            "support_indices",
            tuple(_g_immutable(value, _g_INT64) for value in self.support_indices),
        )
        object.__setattr__(
            self,
            "beta",
            tuple(_g_immutable(value, _g_FLOAT64) for value in self.beta),
        )
        object.__setattr__(
            self,
            "xi",
            tuple(_g_immutable(value, _g_FLOAT64) for value in self.xi),
        )
        object.__setattr__(
            self,
            "shape_inverse",
            tuple(_g_immutable(value, _g_FLOAT64) for value in self.shape_inverse),
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
    rank_value = _g_scalar_float(rank_rtol, "rank_rtol")
    if rank_value <= 0.0 or rank_value >= 1.0:
        raise ValueError("rank_rtol must satisfy 0 < rank_rtol < 1")
    damage = _g_validate_material_damage(material_damage, geometry.n_material)
    k_res_value = _g_scalar_float(k_res, "k_res")
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

def _g_validate_material_damage(
    damage: Any, n_material: int, name: str = "material_damage"
) -> np.ndarray:
    values = _g_require_float64(damage, name, (n_material,))
    # The raw archive is a transplanted warning diagnostic.  Range clipping or
    # rejection would change its provenance, so only finiteness is required.
    return values

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
        bottom = _g_require_bool(self.outer_bottom_mask, "outer_bottom_mask", (n_particles,))
        top = _g_require_bool(self.outer_top_mask, "outer_top_mask", (n_particles,))
        if not np.any(bottom) or not np.any(top) or np.any(bottom & top):
            raise ValueError("outer top/bottom masks must be nonempty and disjoint")
        if not isinstance(self.anchor_particle, Integral) or isinstance(self.anchor_particle, bool):
            raise ValueError("anchor_particle must be an integer")
        anchor = int(self.anchor_particle)
        if anchor < 0 or anchor >= n_particles:
            raise ValueError("anchor_particle is outside the union particle range")
        anchor_coordinate = _g_require_float64(self.anchor_coordinate, "anchor_coordinate", (2,))
        if not np.array_equal(anchor_coordinate, np.asarray([0.0, 0.0], dtype=np.float64)):
            raise ValueError("A2 anchor coordinate must remain exactly (0, 0)")
        constrained = _g_require_int64(self.constrained_dofs, "constrained_dofs")
        free = _g_require_int64(self.free_dofs, "free_dofs")
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
        q_zero = _g_require_float64(self.q_zero, "q_zero", (n_dofs,))
        load_direction = _g_require_float64(self.load_direction, "load_direction", (n_dofs,))
        selectors = (
            ("outer_top_selector", self.outer_top_selector),
            ("outer_bottom_selector", self.outer_bottom_selector),
            ("anchor_selector", self.anchor_selector),
        )
        for name, values in selectors:
            _g_require_float64(values, name, (n_dofs,))
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
            object.__setattr__(self, name, _g_immutable(values))
        object.__setattr__(self, "n_particles", n_particles)
        object.__setattr__(self, "anchor_particle", anchor)

    @property
    def n_dofs(self) -> int:
        return 2 * int(self.n_particles)

    @property
    def free_positions(self) -> np.ndarray:
        positions = np.full(self.n_dofs, -1, dtype=np.int64)
        positions[self.free_dofs] = np.arange(len(self.free_dofs), dtype=np.int64)
        return _g_immutable(positions)

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
        value = _g_scalar_float(U_ext, "U_ext")
        return np.asarray(self.q_zero + value * self.load_direction, dtype=np.float64)

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
    coordinates = _g_require_float64(
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

def _g_a2_hash_primitives(values: Iterable[tuple[str, Any]]) -> str:
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

def _g_a2_line_primitives(line: Any, name: str) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Validate the small line-quadrature contract used by static identities."""

    if not isinstance(line, LineQuadrature):
        raise TypeError(f"{name} must be a LineQuadrature instance")
    indices = _g_require_int64(line.indices, f"{name}.indices")
    weights = _g_require_float64(line.weights, f"{name}.weights")
    if indices.ndim != 1 or weights.shape != indices.shape or len(indices) < 2:
        raise ValueError(f"{name} indices/weights have incompatible shapes")
    return indices, weights, _g_scalar_float(line.length, f"{name}.length"), line.side

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
        top_indices, top_weights, top_length, top_side = _g_a2_line_primitives(
            top_line, "top_line"
        )
        bottom_indices, bottom_weights, bottom_length, bottom_side = _g_a2_line_primitives(
            bottom_line, "bottom_line"
        )
        if top_side != "top" or bottom_side != "bottom":
            raise ValueError("A2 line quadratures must be top then bottom")
        if not np.array_equal(top_indices, np.flatnonzero(geometry.interface_top_mask)):
            raise ValueError("top line indices are not the exact material interface")
        if not np.array_equal(bottom_indices, np.flatnonzero(geometry.interface_bottom_mask)):
            raise ValueError("bottom line indices are not the exact material interface")

        E = _g_scalar_float(youngs_modulus, "A2 youngs_modulus")
        nu = _g_scalar_float(poisson_ratio, "A2 poisson_ratio")
        kres = _g_scalar_float(k_res, "A2 k_res")
        rank = _g_scalar_float(rank_rtol, "A2 rank_rtol")
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
            deterministic_digest=_g_a2_hash_primitives(digest_fields),
        )

    def __post_init__(self) -> None:
        array_specs = (
            ("coordinates", _g_FLOAT64),
            ("union_volumes", _g_FLOAT64),
            ("directed_edges", _g_INT64),
            ("source", _g_INT64),
            ("target", _g_INT64),
            ("edge_distance", _g_FLOAT64),
            ("edge_omega", _g_FLOAT64),
            ("family_mass", _g_FLOAT64),
            ("eta", _g_FLOAT64),
            ("support_indices", _g_INT64),
            ("support_mask", _g_BOOL),
            ("support_sizes", _g_INT64),
            ("beta", _g_FLOAT64),
            ("xi", _g_FLOAT64),
            ("shape_inverse", _g_FLOAT64),
            ("shape_rank", _g_INT64),
            ("shape_condition", _g_FLOAT64),
            ("target_beta", _g_FLOAT64),
            ("gradient_coefficient", _g_FLOAT64),
            ("material_prefix_map", _g_INT64),
            ("material_damage_volumes", _g_FLOAT64),
            ("union_material_phase_volumes", _g_FLOAT64),
            ("grip_phase_volumes", _g_FLOAT64),
            ("constrained_dofs", _g_INT64),
            ("free_dofs", _g_INT64),
            ("q_zero", _g_FLOAT64),
            ("load_direction", _g_FLOAT64),
            ("outer_top_selector", _g_FLOAT64),
            ("outer_bottom_selector", _g_FLOAT64),
            ("anchor_selector", _g_FLOAT64),
            ("outer_bottom_mask", _g_BOOL),
            ("outer_top_mask", _g_BOOL),
            ("top_line_indices", _g_INT64),
            ("top_line_weights", _g_FLOAT64),
            ("bottom_line_indices", _g_INT64),
            ("bottom_line_weights", _g_FLOAT64),
        )
        for name, dtype in array_specs:
            array = np.asarray(getattr(self, name))
            if array.dtype != dtype:
                raise ValueError(f"A2StaticMechanics.{name} must use dtype {dtype}")
            if dtype.kind == "f" and not np.all(np.isfinite(array)):
                raise ValueError(f"A2StaticMechanics.{name} must be finite")
            object.__setattr__(self, name, _g_immutable(array))
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
            value = _g_scalar_float(value, f"A2StaticMechanics.{name}")
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
        volume_tolerance = _g_AREA_RECONSTRUCTION_EPS_FACTOR * np.finfo(np.float64).eps * max(
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
        digest = _g_a2_hash_primitives(
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
class LineQuadrature:
    """Trapezoidal line weights on one unchanged physical interface."""

    indices: np.ndarray
    weights: np.ndarray
    length: float
    side: str

    def __post_init__(self) -> None:
        indices = _g_require_int64(self.indices, "line indices")
        weights = _g_require_float64(self.weights, "line weights")
        if indices.ndim != 1 or weights.shape != indices.shape or len(indices) < 2:
            raise ValueError("line indices/weights must be one-dimensional with at least two points")
        if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError("line weights must be finite and positive")
        length = _g_scalar_float(self.length, "line length")
        if length <= 0.0:
            raise ValueError("line length must be positive")
        if not np.isclose(np.sum(weights), length, rtol=0.0, atol=_g_geometry_tolerance(np.asarray([[0.0, 0.0], [length, 0.0]], dtype=np.float64))):
            raise ValueError("trapezoidal line weights do not sum to the boundary length")
        object.__setattr__(self, "indices", _g_immutable(indices))
        object.__setattr__(self, "weights", _g_immutable(weights))
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
    if np.any(np.abs(spacing - median) > _g_geometry_tolerance(geometry.coordinates)):
        raise ValueError("physical interface spacing is not uniformly validated")
    weights = np.empty(len(indices), dtype=np.float64)
    weights[0] = 0.5 * spacing[0]
    weights[-1] = 0.5 * spacing[-1]
    if len(indices) > 2:
        weights[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    length = float(np.sum(spacing))
    if not np.isclose(length, 1.0, rtol=0.0, atol=_g_geometry_tolerance(geometry.coordinates)):
        raise ValueError(f"physical {side} interface length is not one: {length:.17g}")
    return LineQuadrature(indices=indices, weights=weights, length=length, side=side)






_g_FLOAT64 = np.dtype(np.float64)

def _g_readonly_copy(values: Any, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    """Copy an array into immutable bytes-backed storage.

    A normal read-only ndarray can be made writable again with
    ``setflags(write=True)`` when it owns a writable allocation.  Mechanics
    data can be retained by accepted continuation records, so the immutable
    copy must also make that NumPy operation fail at the buffer boundary.
    """

    source = np.array(values, dtype=dtype, copy=True, order="C")
    raw = source.tobytes(order="C")
    result = np.frombuffer(raw, dtype=source.dtype, count=source.size)
    result = result.reshape(source.shape)
    result.setflags(write=False)
    return result

@dataclass(frozen=True)
class MechanicsEvaluation:
    """Response fields at one mechanics displacement state.

    ``edge_stress`` and ``particle_stress`` are the undegraded constitutive
    plane-strain stress ``P`` from Section 0.4.  The corresponding effective
    stress fields are exposed separately as ``edge_effective_stress`` and
    ``particle_effective_stress``; they equal the view's immutable degradation
    factor times ``P``.
    """

    energy: float
    residual: np.ndarray
    internal_force: np.ndarray
    edge_deformation_gradient: np.ndarray
    edge_strain: np.ndarray
    edge_stress: np.ndarray
    particle_strain: np.ndarray
    particle_stress: np.ndarray
    edge_effective_stress: np.ndarray
    particle_effective_stress: np.ndarray
    edge_energy_density: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy", float(self.energy))
        array_names = (
            "residual",
            "internal_force",
            "edge_deformation_gradient",
            "edge_strain",
            "edge_stress",
            "particle_strain",
            "particle_stress",
            "edge_effective_stress",
            "particle_effective_stress",
            "edge_energy_density",
        )
        for name in array_names:
            array = np.asarray(getattr(self, name))
            if array.dtype != _g_FLOAT64:
                raise ValueError(f"MechanicsEvaluation.{name} must use float64")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"MechanicsEvaluation.{name} must be finite")
            object.__setattr__(self, name, _g_readonly_copy(array))
        if not np.isfinite(self.energy):
            raise ValueError("MechanicsEvaluation.energy must be finite")

    @property
    def edge_F(self) -> np.ndarray:
        """Alias for the edge deformation gradients."""

        return self.edge_deformation_gradient

    @property
    def strain(self) -> np.ndarray:
        """Alias for source-centered particle strain."""

        return self.particle_strain

    @property
    def stress(self) -> np.ndarray:
        """Alias for source-centered undegraded constitutive stress."""

        return self.particle_stress

    def __getitem__(self, key: str) -> Any:
        """Provide lightweight mapping-style access for diagnostics."""

        aliases = {
            "edge_F": "edge_deformation_gradient",
            "F": "edge_deformation_gradient",
            "strain": "particle_strain",
            "stress": "particle_stress",
        }
        return getattr(self, aliases.get(key, key))
# ===== nonlocal_a2_coupled =====








_a_FLOAT64 = np.dtype(np.float64)
_a_INT64 = np.dtype(np.int64)

_a_IDENTITY_2 = np.eye(2, dtype=np.float64)
_a_IDENTITY_3 = np.eye(3, dtype=np.float64)
_a_EPS64 = np.finfo(np.float64).eps
_a_REPLAY_BOUND = 512.0 * _a_EPS64

def _a_immutable(values: Any, dtype: Any | None = None) -> np.ndarray:
    """Make a bytes-backed immutable copy."""

    source = np.array(values, dtype=dtype, copy=True, order="C")
    raw = source.tobytes(order="C")
    result = np.frombuffer(raw, dtype=source.dtype, count=source.size)
    result = result.reshape(source.shape)
    result.setflags(write=False)
    return result

def _a_scalar(value: Any, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0:
        raise ValueError(f"{name} must be scalar, got shape {raw.shape}")
    try:
        result = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a scalar number") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result

def _a_integer(value: Any, name: str, *, positive: bool = False) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must be an integer")
    result = int(raw)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result

def _a_f64(values: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != _a_FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array

def _a_max_abs(values: Any) -> float:
    array = np.asarray(values)
    return float(np.max(np.abs(array))) if array.size else 0.0

def _a_digest_primitives(values: Iterable[tuple[str, Any]]) -> str:
    return _g_a2_hash_primitives(tuple(values))

def _a_damage_digest(damage: np.ndarray) -> str:
    return _a_digest_primitives((("material_damage", damage),))

def _a_unit_system_digest(static_digest: str, damage_digest: str) -> str:
    return _a_digest_primitives(
        (("static_digest", static_digest), ("damage_digest", damage_digest), ("unit", 1.0))
    )

def _a_static_digest_from_object(static: Any) -> str:
    """Recompute the frozen static digest from its declared primitive fields."""

    return _a_digest_primitives(
        (
            ("coordinates", static.coordinates),
            ("union_volumes", static.union_volumes),
            ("directed_edges", static.directed_edges),
            ("source", static.source),
            ("target", static.target),
            ("edge_distance", static.edge_distance),
            ("edge_omega", static.edge_omega),
            ("family_mass", static.family_mass),
            ("eta", static.eta),
            ("support_indices", static.support_indices),
            ("support_mask", static.support_mask),
            ("support_sizes", static.support_sizes),
            ("beta", static.beta),
            ("xi", static.xi),
            ("shape_inverse", static.shape_inverse),
            ("shape_rank", static.shape_rank),
            ("shape_condition", static.shape_condition),
            ("target_beta", static.target_beta),
            ("gradient_coefficient", static.gradient_coefficient),
            ("youngs_modulus", static.youngs_modulus),
            ("poisson_ratio", static.poisson_ratio),
            ("lame_lambda", static.lame_lambda),
            ("shear_modulus", static.shear_modulus),
            ("k_res", static.k_res),
            ("rank_rtol", static.rank_rtol),
            ("delta", static.delta),
            ("h_g", static.h_g),
            ("material_count", static.material_count),
            ("material_prefix_map", static.material_prefix_map),
            ("material_damage_volumes", static.material_damage_volumes),
            ("union_material_phase_volumes", static.union_material_phase_volumes),
            ("grip_phase_volumes", static.grip_phase_volumes),
            ("constrained_dofs", static.constrained_dofs),
            ("free_dofs", static.free_dofs),
            ("q_zero", static.q_zero),
            ("load_direction", static.load_direction),
            ("outer_top_selector", static.outer_top_selector),
            ("outer_bottom_selector", static.outer_bottom_selector),
            ("anchor_selector", static.anchor_selector),
            ("outer_bottom_mask", static.outer_bottom_mask),
            ("outer_top_mask", static.outer_top_mask),
            ("anchor_particle", static.anchor_particle),
            ("top_line_indices", static.top_line_indices),
            ("top_line_weights", static.top_line_weights),
            ("top_line_length", static.top_line_length),
            ("bottom_line_indices", static.bottom_line_indices),
            ("bottom_line_weights", static.bottom_line_weights),
            ("bottom_line_length", static.bottom_line_length),
        )
    )

def _a_check_exact_constraints(
    displacement: np.ndarray, constraints: AffineConstraints, U_ext: float, name: str
) -> None:
    prescribed = constraints.prescribed_vector(U_ext)
    if not np.array_equal(
        displacement[constraints.constrained_dofs], prescribed[constraints.constrained_dofs]
    ):
        raise ValueError(f"{name} does not satisfy exact constrained values")

class A2FixedDamageView:
    """Fresh phase-measured fixed-damage view over one shared static object."""

    _CACHE_FIELDS = frozenset(
        {
            "_gradient_operator_cache",
            "_tangent_block_diagonal_cache",
            "_tangent_diagonal_cache",
        }
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_locked", False) and name not in self._CACHE_FIELDS:
            raise AttributeError("A2FixedDamageView is immutable; construct a fresh view")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if self.__dict__.get("_locked", False) and name not in self._CACHE_FIELDS:
            raise AttributeError("A2FixedDamageView is immutable; construct a fresh view")
        object.__delattr__(self, name)

    def __init__(self, static: A2StaticMechanics, material_damage: Any) -> None:
        if not isinstance(static, A2StaticMechanics):
            raise TypeError("static must be an A2StaticMechanics instance")
        if _a_static_digest_from_object(static) != static.deterministic_digest:
            raise ValueError("A2 static primitive digest does not match deterministic_digest")
        damage = _a_f64(material_damage, "material_damage", (static.material_count,))
        self.static = static
        self.material_damage = _a_immutable(damage)
        self.damage = self.material_damage
        self.damage_digest = _a_damage_digest(self.material_damage)
        self.material_damage_runtime_identity = object()
        self.runtime_identity = (static.runtime_identity, self.material_damage_runtime_identity)
        self.static_digest = static.deterministic_digest

        material_degradation = (1.0 - damage) ** 2 + static.k_res
        grip_degradation = 1.0 + static.k_res
        material_full = np.pad(
            material_degradation,
            (0, static.n_particles - static.material_count),
        )
        center_weights = np.asarray(
            static.union_material_phase_volumes * material_full
            + static.grip_phase_volumes * grip_degradation,
            dtype=np.float64,
        )
        effective = center_weights / static.union_volumes
        if not np.all(np.isfinite(center_weights)) or np.any(center_weights <= 0.0):
            raise FloatingPointError("A2 phase center weights are invalid")
        if not np.all(np.isfinite(effective)) or np.any(effective <= 0.0):
            raise FloatingPointError("A2 effective degradation is invalid")
        self.damage_union = _a_immutable(
            np.concatenate((damage, np.zeros(static.n_particles - static.material_count)))
        )
        self.material_degradation = _a_immutable(material_degradation)
        self.grip_degradation = float(grip_degradation)
        self.center_weights = _a_immutable(center_weights)
        self.effective_center_degradation = _a_immutable(effective)
        self.degradation = self.effective_center_degradation
        self.edge_degradation = _a_immutable(effective[static.source])
        self.center_factor = _a_immutable(static.source * 0.0 + center_weights[static.source] * static.eta)
        self.center_factor_vstar_reference = _a_immutable(
            static.union_volumes[static.source] * effective[static.source] * static.eta
        )
        self.damage_min = float(np.min(damage))
        self.damage_max = float(np.max(damage))
        self.damage_below_zero_count = int(np.count_nonzero(damage < 0.0))
        self.damage_above_one_count = int(np.count_nonzero(damage > 1.0))
        self._locked = True

    @property
    def n_particles(self) -> int:
        return self.static.n_particles

    @property
    def n_material(self) -> int:
        return self.static.material_count

    @property
    def n_edges(self) -> int:
        return self.static.n_edges

    @property
    def n_dofs(self) -> int:
        return self.static.n_dofs

    @property
    def coordinates(self) -> np.ndarray:
        return self.static.coordinates

    @property
    def volumes(self) -> np.ndarray:
        return self.static.union_volumes

    @property
    def directed_edges(self) -> np.ndarray:
        return self.static.directed_edges

    @property
    def edges(self) -> np.ndarray:
        return self.static.directed_edges

    @property
    def source(self) -> np.ndarray:
        return self.static.source

    @property
    def target(self) -> np.ndarray:
        return self.static.target

    @property
    def eta(self) -> np.ndarray:
        return self.static.eta

    @property
    def lame_lambda(self) -> float:
        return self.static.lame_lambda

    @property
    def shear_modulus(self) -> float:
        return self.static.shear_modulus

    @property
    def youngs_modulus(self) -> float:
        return self.static.youngs_modulus

    @property
    def poisson_ratio(self) -> float:
        return self.static.poisson_ratio

    @property
    def k_res(self) -> float:
        return self.static.k_res

    @property
    def delta(self) -> float:
        return self.static.delta

    @property
    def rank_rtol(self) -> float:
        return self.static.rank_rtol

    @property
    def support_indices(self) -> np.ndarray:
        return self.static.support_indices

    @property
    def support_mask(self) -> np.ndarray:
        return self.static.support_mask

    @property
    def support_sizes(self) -> np.ndarray:
        return self.static.support_sizes

    @property
    def beta(self) -> np.ndarray:
        return self.static.beta

    @property
    def xi(self) -> np.ndarray:
        return self.static.xi

    @property
    def shape_inverse(self) -> np.ndarray:
        return self.static.shape_inverse

    @property
    def tangent_block_diagonal(self) -> np.ndarray:
        return self._get_tangent_block_diagonal()

    @property
    def tangent_diagonal(self) -> np.ndarray:
        return self._get_tangent_diagonal()

    def _safe_support(self) -> np.ndarray:
        return np.where(self.static.support_mask, self.static.support_indices, 0)

    def _get_gradient_operator(self) -> csr_matrix:
        cached = getattr(self, "_gradient_operator_cache", None)
        if cached is not None:
            return cached

        edge, slot = np.nonzero(self.static.support_mask)
        support = self.static.support_indices[edge, slot]
        coefficient = self.static.gradient_coefficient[edge, slot]
        n_support_entries = len(edge)
        n_entries = 2 * (n_support_entries + self.n_edges)
        rows = np.empty(n_entries, dtype=np.int64)
        columns = np.empty(n_entries, dtype=np.int64)
        data = np.empty(n_entries, dtype=np.float64)

        support_end = 2 * n_support_entries
        rows[:support_end] = np.column_stack((2 * edge, 2 * edge + 1)).reshape(-1)
        columns[:support_end] = np.repeat(support, 2)
        data[:support_end] = coefficient.reshape(-1)

        edge_index = np.arange(self.n_edges, dtype=np.int64)
        rows[support_end:] = np.column_stack(
            (2 * edge_index, 2 * edge_index + 1)
        ).reshape(-1)
        columns[support_end:] = np.repeat(self.static.source, 2)
        data[support_end:] = -np.sum(self.static.gradient_coefficient, axis=1).reshape(-1)

        operator = coo_matrix(
            (data, (rows, columns)),
            shape=(2 * self.n_edges, self.n_particles),
            dtype=np.float64,
        ).tocsr()
        operator.sort_indices()
        operator.data.setflags(write=False)
        operator.indices.setflags(write=False)
        operator.indptr.setflags(write=False)
        self._gradient_operator_cache = operator
        return operator

    def _deformation_gradient(self, displacement: np.ndarray) -> np.ndarray:
        support = self._safe_support()
        values = displacement.reshape(self.n_particles, 2)
        source_values = values[self.static.source[:, None]]
        current = self.static.xi + values[support] - source_values
        moment = np.einsum(
            "es,esa,esb->eab", self.static.beta, current, self.static.xi, optimize=True
        )
        return np.einsum(
            "eab,ebc->eac", moment, self.static.shape_inverse, optimize=True
        )

    def _strain_stress(self, deformation_gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        increment = deformation_gradient - _a_IDENTITY_2
        strain = 0.5 * (increment + np.swapaxes(increment, 1, 2))
        trace = np.trace(strain, axis1=1, axis2=2)
        stress = (
            self.lame_lambda * trace[:, None, None] * _a_IDENTITY_2[None, :, :]
            + 2.0 * self.shear_modulus * strain
        )
        return strain, stress

    def _density(self, strain: np.ndarray) -> np.ndarray:
        trace = np.trace(strain, axis1=1, axis2=2)
        return 0.5 * self.lame_lambda * trace**2 + self.shear_modulus * np.sum(
            strain * strain, axis=(1, 2)
        )

    def _endpoint_residual(self, stress: np.ndarray) -> np.ndarray:
        weighted_stress = (
            self.center_factor[:, None, None] * stress
        ).transpose(0, 2, 1).reshape(2 * self.n_edges, 2)
        return np.asarray(
            self._get_gradient_operator().T @ weighted_stress, dtype=np.float64
        ).reshape(-1)

    def _delta_gradient(self, direction: np.ndarray) -> np.ndarray:
        values = direction.reshape(self.n_particles, 2)
        return (self._get_gradient_operator() @ values).reshape(
            self.n_edges, 2, 2
        ).transpose(0, 2, 1)

    def _constitutive_tangent(self, delta_gradient: np.ndarray) -> np.ndarray:
        delta_strain = 0.5 * (delta_gradient + np.swapaxes(delta_gradient, 1, 2))
        trace = np.trace(delta_strain, axis1=1, axis2=2)
        return (
            self.lame_lambda * trace[:, None, None] * _a_IDENTITY_2[None, :, :]
            + 2.0 * self.shear_modulus * delta_strain
        )

    def _get_tangent_block_diagonal(self) -> np.ndarray:
        cached = getattr(self, "_tangent_block_diagonal_cache", None)
        if cached is not None:
            return cached
        coefficient = self.static.gradient_coefficient
        cx = coefficient[:, :, 0]
        cy = coefficient[:, :, 1]
        factor = self.center_factor[:, None]
        hxx = factor * (
            (self.lame_lambda + 2.0 * self.shear_modulus) * cx * cx
            + self.shear_modulus * cy * cy
        )
        hxy = factor * (self.lame_lambda + self.shear_modulus) * cx * cy
        hyy = factor * (
            self.shear_modulus * cx * cx
            + (self.lame_lambda + 2.0 * self.shear_modulus) * cy * cy
        )
        support = self._safe_support().reshape(-1)
        blocks = np.zeros((self.n_particles, 2, 2), dtype=np.float64)
        blocks[:, 0, 0] = np.bincount(support, weights=hxx.reshape(-1), minlength=self.n_particles)
        blocks[:, 0, 1] = np.bincount(support, weights=hxy.reshape(-1), minlength=self.n_particles)
        blocks[:, 1, 0] = blocks[:, 0, 1]
        blocks[:, 1, 1] = np.bincount(support, weights=hyy.reshape(-1), minlength=self.n_particles)
        source_coefficient = np.sum(coefficient, axis=1)
        scx, scy = source_coefficient[:, 0], source_coefficient[:, 1]
        source_hxx = self.center_factor * (
            (self.lame_lambda + 2.0 * self.shear_modulus) * scx**2
            + self.shear_modulus * scy**2
        )
        source_hxy = self.center_factor * (self.lame_lambda + self.shear_modulus) * scx * scy
        source_hyy = self.center_factor * (
            self.shear_modulus * scx**2
            + (self.lame_lambda + 2.0 * self.shear_modulus) * scy**2
        )
        blocks[:, 0, 0] += np.bincount(self.static.source, weights=source_hxx, minlength=self.n_particles)
        source_xy = np.bincount(self.static.source, weights=source_hxy, minlength=self.n_particles)
        blocks[:, 0, 1] += source_xy
        blocks[:, 1, 0] += source_xy
        blocks[:, 1, 1] += np.bincount(self.static.source, weights=source_hyy, minlength=self.n_particles)
        if not np.all(np.isfinite(blocks)):
            raise FloatingPointError("A2 tangent block cache is non-finite")
        for block in blocks:
            if np.any(np.linalg.eigvalsh(block) <= 0.0):
                raise ValueError("A2 tangent block cache is not positive definite")
        result = _a_immutable(blocks)
        self._tangent_block_diagonal_cache = result
        return result

    def _get_tangent_diagonal(self) -> np.ndarray:
        cached = getattr(self, "_tangent_diagonal_cache", None)
        if cached is not None:
            return cached
        blocks = self._get_tangent_block_diagonal()
        diagonal = np.empty(self.n_dofs, dtype=np.float64)
        diagonal[0::2] = blocks[:, 0, 0]
        diagonal[1::2] = blocks[:, 1, 1]
        result = _a_immutable(diagonal)
        self._tangent_diagonal_cache = result
        return result

    def energy(self, displacement: Any) -> float:
        q = _a_f64(displacement, "A2 displacement", (self.n_dofs,))
        strain, _ = self._strain_stress(self._deformation_gradient(q))
        value = float(np.sum(self.center_factor * self._density(strain)))
        if not np.isfinite(value):
            raise FloatingPointError("A2 mechanics energy is non-finite")
        return value

    def residual(self, displacement: Any) -> np.ndarray:
        q = _a_f64(displacement, "A2 displacement", (self.n_dofs,))
        _, stress = self._strain_stress(self._deformation_gradient(q))
        result = self._endpoint_residual(stress)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("A2 mechanics residual is non-finite")
        return result

    def internal_force(self, displacement: Any) -> np.ndarray:
        return -self.residual(displacement)

    def tangent_action(self, direction: Any) -> np.ndarray:
        values = _a_f64(direction, "A2 direction", (self.n_dofs,))
        result = self._endpoint_residual(
            self._constitutive_tangent(self._delta_gradient(values))
        )
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("A2 tangent action is non-finite")
        return result

    def evaluate(self, displacement: Any) -> MechanicsEvaluation:
        q = _a_f64(displacement, "A2 displacement", (self.n_dofs,))
        deformation_gradient = self._deformation_gradient(q)
        strain, stress = self._strain_stress(deformation_gradient)
        density = self._density(strain)
        residual = self._endpoint_residual(stress)
        particle_strain = np.zeros((self.n_particles, 2, 2), dtype=np.float64)
        particle_stress = np.zeros((self.n_particles, 2, 2), dtype=np.float64)
        np.add.at(particle_strain, self.static.source, self.static.eta[:, None, None] * strain)
        np.add.at(particle_stress, self.static.source, self.static.eta[:, None, None] * stress)
        edge_effective = self.edge_degradation[:, None, None] * stress
        particle_effective = self.effective_center_degradation[:, None, None] * particle_stress
        energy = float(np.sum(self.center_factor * density))
        if not np.isfinite(energy) or not np.all(np.isfinite(residual)):
            raise FloatingPointError("A2 mechanics evaluation is non-finite")
        return MechanicsEvaluation(
            energy=energy,
            residual=residual,
            internal_force=-residual,
            edge_deformation_gradient=deformation_gradient,
            edge_strain=strain,
            edge_stress=stress,
            particle_strain=particle_strain,
            particle_stress=particle_stress,
            edge_effective_stress=edge_effective,
            particle_effective_stress=particle_effective,
            edge_energy_density=density,
        )

@dataclass(frozen=True)
class A2SeedProvenance:
    """Auditable normalized mechanics seed provenance."""

    kind: str
    source_state: str
    source_step: int
    source_sweep: int
    source_damage_digest: str
    target_damage_digest: str
    unit_system_digest: str
    seed_digest: str
    initial_residual_norm: float
    final_residual_norm: float
    early_exit_eligible: bool
    early_exit_used: bool

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("A2SeedProvenance.kind must be non-empty")
        if not isinstance(self.source_state, str) or not self.source_state:
            raise ValueError("A2SeedProvenance.source_state must be non-empty")
        for name in (
            "source_damage_digest", "target_damage_digest", "unit_system_digest", "seed_digest"
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"A2SeedProvenance.{name} must be a SHA256 digest")
        for name in ("source_step", "source_sweep"):
            object.__setattr__(self, name, _a_integer(getattr(self, name), f"A2SeedProvenance.{name}"))
        for name in ("initial_residual_norm", "final_residual_norm"):
            value = _a_scalar(getattr(self, name), f"A2SeedProvenance.{name}")
            if value < 0.0:
                raise ValueError(f"A2SeedProvenance.{name} must be non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "early_exit_eligible", bool(self.early_exit_eligible))
        object.__setattr__(self, "early_exit_used", bool(self.early_exit_used))
        if self.early_exit_used and not self.early_exit_eligible:
            raise ValueError("A2 seed early exit cannot be used when it is ineligible")

@dataclass(frozen=True)
class A2MechanicsSolveResult:
    """One exact-constraint fixed-damage mechanics solve."""

    displacement: np.ndarray
    evaluation: MechanicsEvaluation
    constraints: AffineConstraints
    outer_displacement: float
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
    static_runtime_identity: object
    static_digest: str
    damage_runtime_identity: object
    damage_digest: str
    unit_system_digest: str
    seed_provenance: A2SeedProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation, MechanicsEvaluation):
            raise TypeError("A2MechanicsSolveResult.evaluation must be MechanicsEvaluation")
        if not isinstance(self.constraints, AffineConstraints):
            raise TypeError("A2MechanicsSolveResult.constraints must be AffineConstraints")
        n = self.constraints.n_dofs
        for name, shape in (
            ("displacement", (n,)),
            ("free_residual", (len(self.constraints.free_dofs),)),
            ("total_force", (2,)),
        ):
            object.__setattr__(self, name, _a_immutable(_a_f64(getattr(self, name), name, shape)))
        for name in (
            "outer_displacement", "free_residual_norm", "free_rhs_norm",
            "free_residual_relative_norm", "cg_convergence_limit", "outer_reaction",
            "bottom_reaction", "anchor_reaction", "total_moment",
        ):
            value = _a_scalar(getattr(self, name), name)
            if name.endswith("norm") and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.free_rhs_norm < 0.0 or self.cg_convergence_limit <= 0.0:
            raise ValueError("A2 mechanics residual scales are invalid")
        if abs(self.free_residual_norm - np.linalg.norm(self.free_residual)) > _a_REPLAY_BOUND * max(1.0, self.free_residual_norm):
            raise ValueError("A2 free residual norm is inconsistent")
        expected_relative = self.free_residual_norm / max(self.free_rhs_norm, 1.0e-300)
        if abs(self.free_residual_relative_norm - expected_relative) > _a_REPLAY_BOUND * max(1.0, abs(expected_relative)):
            raise ValueError("A2 free residual relative norm is inconsistent")
        if self.free_residual_norm >= self.cg_convergence_limit:
            raise ValueError("A2 mechanics solve failed its authoritative free residual gate")
        for name in ("cg_iterations", "cg_info"):
            object.__setattr__(self, name, _a_integer(getattr(self, name), name))
        if not isinstance(self.static_digest, str) or len(self.static_digest) != 64:
            raise ValueError("A2 mechanics static digest is invalid")
        if not isinstance(self.damage_digest, str) or len(self.damage_digest) != 64:
            raise ValueError("A2 mechanics damage digest is invalid")
        if not isinstance(self.unit_system_digest, str) or len(self.unit_system_digest) != 64:
            raise ValueError("A2 mechanics unit-system digest is invalid")
        expected_unit_digest = _a_unit_system_digest(self.static_digest, self.damage_digest)
        if self.unit_system_digest != expected_unit_digest:
            raise ValueError("A2 mechanics unit-system digest does not match static/damage digests")
        if not isinstance(self.seed_provenance, A2SeedProvenance):
            raise TypeError("A2 mechanics seed provenance is invalid")
        if self.seed_provenance.target_damage_digest != self.damage_digest:
            raise ValueError("A2 mechanics seed target damage digest is not paired")
        if self.seed_provenance.unit_system_digest != self.unit_system_digest:
            raise ValueError("A2 mechanics seed unit-system digest is not paired")

    @property
    def U_ext(self) -> float:
        return self.outer_displacement

    @property
    def energy(self) -> float:
        return self.evaluation.energy

    @property
    def residual(self) -> np.ndarray:
        return self.evaluation.residual

    @property
    def R_ext(self) -> float:
        return self.outer_reaction

def _a_seed_record(
    kind: str,
    seed: np.ndarray,
    view: A2FixedDamageView,
    constraints: AffineConstraints,
    initial_residual_norm: float,
    final_residual_norm: float,
    *,
    source_state: str,
    source_step: int,
    source_sweep: int,
    source_damage_digest: str,
) -> A2SeedProvenance:
    return A2SeedProvenance(
        kind=kind,
        source_state=source_state,
        source_step=source_step,
        source_sweep=source_sweep,
        source_damage_digest=source_damage_digest,
        target_damage_digest=view.damage_digest,
        unit_system_digest=_a_unit_system_digest(view.static.deterministic_digest, view.damage_digest),
        seed_digest=_a_digest_primitives((("seed", seed),)),
        initial_residual_norm=initial_residual_norm,
        final_residual_norm=final_residual_norm,
        early_exit_eligible=False,
        early_exit_used=False,
    )

def _a_validate_view_constraints(view: A2FixedDamageView, constraints: AffineConstraints) -> None:
    if not isinstance(constraints, AffineConstraints):
        raise TypeError("constraints must be an AffineConstraints instance")
    static = view.static
    if constraints.n_particles != static.n_particles:
        raise ValueError("A2 view and constraints have different particle counts")
    for name in (
        "constrained_dofs", "free_dofs", "q_zero", "load_direction",
        "outer_top_selector", "outer_bottom_selector", "anchor_selector",
        "outer_bottom_mask", "outer_top_mask",
    ):
        if not np.array_equal(getattr(constraints, name), getattr(static, name)):
            raise ValueError(f"A2 constraints do not match the static identity field {name}")
    if constraints.anchor_particle != static.anchor_particle:
        raise ValueError("A2 constraints anchor does not match the static identity")



@dataclass(frozen=True)
class A2AmorHistoryResult:
    edge_positive_energy: np.ndarray
    particle_driver: np.ndarray
    previous_history: np.ndarray
    trial_history: np.ndarray

    def __post_init__(self) -> None:
        edge = _a_f64(self.edge_positive_energy, "edge_positive_energy")
        driver = _a_f64(self.particle_driver, "particle_driver")
        previous = _a_f64(self.previous_history, "previous_history")
        trial = _a_f64(self.trial_history, "trial_history")
        if edge.ndim != 1 or driver.ndim != 1 or previous.shape != driver.shape or trial.shape != driver.shape:
            raise ValueError("A2 Amor arrays have incompatible shapes")
        if np.any(driver < 0.0) or np.any(previous < 0.0) or np.any(trial < previous):
            raise ValueError("A2 Amor history maximum contract is violated")
        for name, value in (("edge_positive_energy", edge), ("particle_driver", driver), ("previous_history", previous), ("trial_history", trial)):
            object.__setattr__(self, name, _a_immutable(value))

    @property
    def history_previous(self) -> np.ndarray:
        return self.previous_history

    @property
    def history_trial(self) -> np.ndarray:
        return self.trial_history

def _a_amor_edge_energy(edge_strain: np.ndarray, lame_lambda: float, shear_modulus: float) -> np.ndarray:
    strain = _a_f64(edge_strain, "edge_strain")
    if strain.ndim != 3 or strain.shape[1:] != (2, 2):
        raise ValueError("edge_strain must have shape (E, 2, 2)")
    if _a_max_abs(strain - np.swapaxes(strain, 1, 2)) > 32.0 * _a_EPS64 * max(1.0, _a_max_abs(strain)):
        raise ValueError("edge_strain must be symmetric")
    lam = _a_scalar(lame_lambda, "lame_lambda")
    mu = _a_scalar(shear_modulus, "shear_modulus")
    if mu <= 0.0 or lam + 2.0 * mu / 3.0 <= 0.0:
        raise ValueError("Amor constitutive values are invalid")
    embedded = np.zeros((len(strain), 3, 3), dtype=np.float64)
    embedded[:, :2, :2] = strain
    trace = np.trace(embedded, axis1=1, axis2=2)
    deviatoric = embedded - (trace / 3.0)[:, None, None] * _a_IDENTITY_3
    result = 0.5 * (lam + 2.0 * mu / 3.0) * np.maximum(trace, 0.0) ** 2 + mu * np.sum(
        deviatoric * deviatoric, axis=(1, 2)
    )
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("A2 Amor edge energy is invalid")
    return _a_immutable(result)

def compute_a2_amor_history_from_edge_strain(
    edge_strain: Any,
    source: Any,
    target: Any,
    eta: Any,
    n_material: Any,
    lame_lambda: Any,
    shear_modulus: Any,
    previous_history: Any,
) -> A2AmorHistoryResult:
    """Accumulate Amor only from material source centers, including all targets."""

    strain = _a_f64(edge_strain, "edge_strain")
    if strain.ndim != 3 or strain.shape[1:] != (2, 2):
        raise ValueError("edge_strain must have shape (E, 2, 2)")
    E = len(strain)
    source_array = np.asarray(source)
    target_array = np.asarray(target)
    if source_array.dtype != _a_INT64 or target_array.dtype != _a_INT64 or source_array.shape != (E,) or target_array.shape != (E,):
        raise ValueError("A2 Amor source/target must be int64 shape (E,)")
    eta_array = _a_f64(eta, "eta", (E,))
    material_count = _a_integer(n_material, "n_material", positive=True)
    previous = _a_f64(previous_history, "previous_history", (material_count,))
    if np.any(source_array < 0) or np.any(target_array < 0):
        raise ValueError("A2 Amor source/target indices must be non-negative")
    retained = source_array < material_count
    if not np.any(retained):
        raise ValueError("A2 Amor union graph has no material source centers")
    source_material = source_array[retained]
    eta_material = eta_array[retained]
    if np.any(eta_material <= 0.0):
        raise ValueError("A2 Amor eta must be strictly positive")
    eta_sum = np.bincount(source_material, weights=eta_material, minlength=material_count)
    if not np.all(np.isfinite(eta_sum)) or np.max(np.abs(eta_sum - 1.0)) > _a_REPLAY_BOUND:
        raise ValueError("A2 Amor material-source eta must sum to one without renormalization")
    edge_positive_all = _a_amor_edge_energy(strain, _a_scalar(lame_lambda, "lame_lambda"), _a_scalar(shear_modulus, "shear_modulus"))
    edge_positive = edge_positive_all[retained]
    driver = np.zeros(material_count, dtype=np.float64)
    np.add.at(driver, source_material, eta_material * edge_positive)
    trial = np.maximum(previous, driver)
    return A2AmorHistoryResult(edge_positive_all, driver, previous, trial)

def compute_a2_amor_history(
    view: A2FixedDamageView,
    displacement: Any,
    previous_history: Any,
) -> A2AmorHistoryResult:
    if not isinstance(view, A2FixedDamageView):
        raise TypeError("view must be an A2FixedDamageView")
    evaluation = view.evaluate(displacement)
    return compute_a2_amor_history_from_edge_strain(
        evaluation.edge_strain,
        view.static.source,
        view.static.target,
        view.static.eta,
        view.static.material_count,
        view.lame_lambda,
        view.shear_modulus,
        previous_history,
    )
# ===== nonlocal_engineering_grip_jax =====








_j_FLOAT64 = np.dtype(np.float64)
_j_INT64 = np.dtype(np.int64)

_j_STATUS_RUNNING = np.int32(-1)
_j_STATUS_CONVERGED = np.int32(0)
_j_STATUS_MAXITER = np.int32(1)
_j_STATUS_BREAKDOWN = np.int32(2)
_j_STATUS_NONFINITE = np.int32(3)
_j_STATUS_EXPLICIT_RESIDUAL = np.int32(4)
_j_STATUS_NAMES = {
    int(_j_STATUS_CONVERGED): "converged",
    int(_j_STATUS_MAXITER): "maxiter",
    int(_j_STATUS_BREAKDOWN): "breakdown",
    int(_j_STATUS_NONFINITE): "nonfinite",
    int(_j_STATUS_EXPLICIT_RESIDUAL): "explicit_residual_failure",
}

def _j_require_x64_enabled() -> None:
    """Require the caller to have enabled JAX x64 before construction/use."""

    if not bool(jax.config.x64_enabled):
        raise RuntimeError(
            "fracture2d JAX A2 mechanics requires jax_enable_x64=True before "
            "constructing or using the backend; it will not enable or downcast"
        )

def _j_require_static_float64(
    values: Any,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != _j_FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array

def _j_require_static_int64(
    values: Any,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != _j_INT64:
        raise ValueError(f"{name} must use exact int64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array

def _j_require_dynamic_float64(
    values: Any,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray | jax.Array:
    """Validate a dynamic float input without silently downcasting it."""

    array = values if isinstance(values, jax.Array) else np.asarray(values)
    if array.dtype != _j_FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    finite = jnp.all(jnp.isfinite(array)) if isinstance(array, jax.Array) else np.all(np.isfinite(array))
    if not bool(np.asarray(jax.device_get(finite))):
        raise ValueError(f"{name} must contain only finite values")
    return array

def _j_require_finite_device_result(values: Any, name: str) -> None:
    leaves = jax.tree_util.tree_leaves(values)
    finite = jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves])
    )
    if not bool(np.asarray(jax.device_get(finite))):
        raise FloatingPointError(f"{name} contains non-finite values")

def _j_dynamic_float_scalar(
    value: Any,
    name: str,
    *,
    allow_zero: bool,
    nonnegative: bool = True,
) -> np.float64:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar, got shape {array.shape}")
    if array.dtype.kind == "f" and array.dtype != _j_FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a scalar number")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and allow_zero and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    if nonnegative and not allow_zero and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return np.float64(result)

def _j_dynamic_maxiter(value: Any) -> np.int32:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or not np.issubdtype(array.dtype, np.integer)
        or isinstance(value, (bool, np.bool_))
    ):
        raise ValueError("cg_maxiter must be a positive integer scalar")
    result = int(array)
    if result <= 0 or result > np.iinfo(np.int32).max:
        raise ValueError("cg_maxiter must be a positive int32-range integer")
    return np.int32(result)

def _j_segment_sum(values: jax.Array, segment_ids: jax.Array, n_segments: int) -> jax.Array:
    """Fixed-size segment reduction used for all edge/endpoint assembly."""

    return jax.ops.segment_sum(values, segment_ids, num_segments=n_segments)

def _j_closed_form_h_components(
    coefficient: np.ndarray,
    lame_lambda: float,
    shear_modulus: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact ``H(c)`` entries for flattened support occurrences."""

    cx = coefficient[:, 0]
    cy = coefficient[:, 1]
    hxx = (lame_lambda + 2.0 * shear_modulus) * cx * cx + shear_modulus * cy * cy
    hxy = (lame_lambda + shear_modulus) * cx * cy
    hyy = shear_modulus * cx * cx + (lame_lambda + 2.0 * shear_modulus) * cy * cy
    return (
        np.asarray(hxx, dtype=np.float64),
        np.asarray(hxy, dtype=np.float64),
        np.asarray(hyy, dtype=np.float64),
    )

@dataclass(frozen=True)
class JaxA2SolveResult:
    """Immutable device solve result with explicit numerical diagnostics.

    The numerical fields remain JAX arrays on the backend device.  Accessing
    ``status`` or ``converged`` intentionally performs the corresponding small
    host read; the PCG loop itself performs no host synchronization.  These
    values are execution diagnostics only; production acceptance is decided by
    the CPU-authoritative replay.
    """

    displacement: jax.Array
    damage: jax.Array
    u_bar: jax.Array
    cg_rtol: jax.Array
    cg_atol: jax.Array
    cg_maxiter: jax.Array
    free_rhs: jax.Array
    free_residual: jax.Array
    recursive_free_residual: jax.Array
    initial_displacement: jax.Array
    initial_free_residual: jax.Array
    free_rhs_norm: jax.Array
    free_residual_norm: jax.Array
    recursive_residual_norm: jax.Array
    free_residual_relative_norm: jax.Array
    initial_free_residual_norm: jax.Array
    cg_convergence_limit: jax.Array
    initial_guess_supplied: jax.Array
    cg_iterations: jax.Array
    cg_info: jax.Array
    status_code: jax.Array
    explicit_converged: jax.Array
    converged_flag: jax.Array
    energy: jax.Array
    top_reaction: jax.Array
    bottom_reaction: jax.Array
    anchor_reaction: jax.Array
    total_force: jax.Array
    total_moment: jax.Array

    @property
    def status(self) -> str:
        code = int(np.asarray(self.status_code))
        return _j_STATUS_NAMES.get(code, f"unknown({code})")

    @property
    def converged(self) -> bool:
        return bool(np.asarray(self.converged_flag))

    def require_converged(self) -> "JaxA2SolveResult":
        """Raise a strict numerical error unless the solve passed every gate."""

        if self.converged:
            return self
        raise RuntimeError(
            "JAX A2 hard-constraint PCG did not converge: "
            f"status={self.status}, iterations={int(np.asarray(self.cg_iterations))}, "
            f"recursive_norm={float(np.asarray(self.recursive_residual_norm)):.17g}, "
            f"explicit_norm={float(np.asarray(self.free_residual_norm)):.17g}, "
            f"limit={float(np.asarray(self.cg_convergence_limit)):.17g}"
        )

@dataclass(frozen=True)
class JaxA2CPUReplayResult:
    """Pair GPU execution diagnostics with the CPU-authoritative A2 result."""

    jax_result: JaxA2SolveResult
    cpu_result: A2MechanicsSolveResult

    @property
    def gpu_result(self) -> JaxA2SolveResult:
        return self.jax_result

    @property
    def result(self) -> A2MechanicsSolveResult:
        return self.cpu_result

class JaxA2UnionMechanics:
    """Device-resident float64 backend for A2 phase-measured union mechanics.

    ``static`` must be an accepted ``A2StaticMechanics`` union model and
    ``constraints`` its exact ``AffineConstraints`` identity.  All graph,
    support, and constraint arrays are validated then copied once to the
    explicit device; only the material damage field is dynamic per solve.
    """

    def __init__(
        self,
        static: A2StaticMechanics,
        constraints: AffineConstraints,
        device: jax.Device,
    ) -> None:
        _j_require_x64_enabled()
        if not isinstance(static, A2StaticMechanics):
            raise TypeError("static must be an accepted A2StaticMechanics instance")
        if not isinstance(constraints, AffineConstraints):
            raise TypeError("constraints must be an AffineConstraints instance")
        if constraints.n_particles != static.n_particles:
            raise ValueError("constraints have a different particle count")
        for name in (
            "constrained_dofs",
            "free_dofs",
            "q_zero",
            "load_direction",
            "outer_top_selector",
            "outer_bottom_selector",
            "anchor_selector",
            "outer_bottom_mask",
            "outer_top_mask",
        ):
            if not np.array_equal(getattr(constraints, name), getattr(static, name)):
                raise ValueError(
                    f"constraints do not match the static identity field {name}"
                )
        if constraints.anchor_particle != static.anchor_particle:
            raise ValueError("constraints anchor does not match the static identity")
        if not isinstance(device, jax.Device):
            raise TypeError("device must be an explicit jax.Device instance")
        if device not in tuple(jax.devices(device.platform)):
            raise ValueError(f"explicit JAX device is not available: {device!r}")

        self.static = static
        self.constraints = constraints
        self.device = device
        self.n_particles = int(static.n_particles)
        self.n_material = int(static.material_count)
        self.n_edges = int(static.n_edges)
        self.n_dofs = int(static.n_dofs)
        self.n_free = int(len(constraints.free_dofs))
        if self.n_free == 0:
            raise ValueError("affine constraints leave no free DOFs")
        if max(self.n_particles, self.n_edges, self.n_dofs) > np.iinfo(np.int32).max:
            raise ValueError("the validated union graph is too large for the int32 JAX index contract")

        coordinates = _j_require_static_float64(
            static.coordinates, "static.coordinates", (self.n_particles, 2)
        )
        material_phase = _j_require_static_float64(
            static.union_material_phase_volumes,
            "static.union_material_phase_volumes",
            (self.n_particles,),
        )
        grip_phase = _j_require_static_float64(
            static.grip_phase_volumes, "static.grip_phase_volumes", (self.n_particles,)
        )
        union_volumes = _j_require_static_float64(
            static.union_volumes, "static.union_volumes", (self.n_particles,)
        )
        if (
            np.any(material_phase < 0.0)
            or np.any(grip_phase < 0.0)
            or np.any(union_volumes <= 0.0)
        ):
            raise ValueError("static phase/union volumes are invalid")

        source = _j_require_static_int64(static.source, "static.source", (self.n_edges,))
        target = _j_require_static_int64(static.target, "static.target", (self.n_edges,))
        for name, values in (("source", source), ("target", target)):
            if np.any(values < 0) or np.any(values >= self.n_particles):
                raise ValueError(f"static.{name} contains an out-of-range index")

        support_indices = _j_require_static_int64(
            static.support_indices, "static.support_indices"
        )
        support_mask = np.asarray(static.support_mask)
        beta = _j_require_static_float64(static.beta, "static.beta")
        xi = _j_require_static_float64(static.xi, "static.xi")
        shape_inverse = _j_require_static_float64(
            static.shape_inverse, "static.shape_inverse", (self.n_edges, 2, 2)
        )
        if support_indices.ndim != 2 or support_indices.shape[0] != self.n_edges:
            raise ValueError(
                "static.support_indices must have shape (E, S) with the "
                f"accepted edge count E={self.n_edges}"
            )
        if support_mask.dtype != np.dtype(bool) or support_mask.shape != support_indices.shape:
            raise ValueError(
                "static.support_mask must be boolean with the same shape as support_indices"
            )
        if beta.shape != support_indices.shape:
            raise ValueError("static.beta must have the padded support shape")
        if xi.shape != support_indices.shape + (2,):
            raise ValueError("static.xi must have the padded support shape (E, S, 2)")
        if not np.any(support_mask):
            raise ValueError("static.support_mask must contain valid occurrences")

        valid_flat = np.flatnonzero(support_mask.reshape(-1))
        support_flat = support_indices.reshape(-1)[valid_flat]
        if np.any(support_flat < 0) or np.any(support_flat >= self.n_particles):
            raise ValueError(
                "valid static support indices contain an out-of-range particle index"
            )
        occurrence_edge64 = valid_flat // support_indices.shape[1]
        occurrence_particle64 = support_flat
        occurrence_source64 = source[occurrence_edge64]
        occurrence_beta = beta.reshape(-1)[valid_flat]
        occurrence_xi = xi.reshape(-1, 2)[valid_flat]
        if not np.all(np.isfinite(occurrence_beta)) or np.any(occurrence_beta <= 0.0):
            raise ValueError("valid support beta coefficients must be finite and positive")
        if not np.all(np.isfinite(occurrence_xi)):
            raise ValueError("valid support xi coefficients must be finite")

        # The first valid occurrence is the target endpoint in the accepted
        # target-inclusive construction.  Check this here so flattening cannot
        # silently change endpoint ownership.
        if not np.all(support_mask[:, 0]):
            raise ValueError("accepted support_mask must include the target slot")
        if not np.array_equal(support_indices[:, 0], target):
            raise ValueError("accepted support_indices first slot must equal target")

        occurrence_inverse = shape_inverse[occurrence_edge64]
        gradient_coefficient = np.einsum(
            "o,oa,oab->ob",
            occurrence_beta,
            occurrence_xi,
            occurrence_inverse,
            optimize=True,
        )

        lame_lambda = float(static.lame_lambda)
        shear_modulus = float(static.shear_modulus)
        hxx_occ, hxy_occ, hyy_occ = _j_closed_form_h_components(
            gradient_coefficient, lame_lambda, shear_modulus
        )
        source_coefficient = np.zeros((self.n_edges, 2), dtype=np.float64)
        np.add.at(source_coefficient, occurrence_edge64, gradient_coefficient)
        hxx_source, hxy_source, hyy_source = _j_closed_form_h_components(
            source_coefficient, lame_lambda, shear_modulus
        )
        eta = _j_require_static_float64(static.eta, "static.eta", (self.n_edges,))
        if np.any(eta <= 0.0):
            raise ValueError("static.eta must contain strictly positive weights")

        free_dofs = _j_require_static_int64(constraints.free_dofs, "constraints.free_dofs")
        constrained_dofs = _j_require_static_int64(
            constraints.constrained_dofs, "constraints.constrained_dofs"
        )
        free_mask = np.zeros((self.n_particles, 2), dtype=bool)
        free_mask.reshape(-1)[free_dofs] = True

        # Every index used by a device gather is range checked as int64 first,
        # then converted to int32.  The valid-occurrence arrays are the only
        # support arrays copied to the backend; padded invalid slots are absent.
        self.coordinates = self._put(coordinates)
        self.material_phase_volumes = self._put(material_phase)
        self.grip_phase_volumes = self._put(grip_phase)
        self.source = self._put(source.astype(np.int32, copy=False))
        self.target = self._put(target.astype(np.int32, copy=False))
        self.eta = self._put(eta)
        self.shape_inverse = self._put(shape_inverse)
        self.occurrence_edge = self._put(occurrence_edge64.astype(np.int32, copy=False))
        self.occurrence_particle = self._put(
            occurrence_particle64.astype(np.int32, copy=False)
        )
        self.occurrence_source = self._put(
            occurrence_source64.astype(np.int32, copy=False)
        )
        self.occurrence_beta = self._put(occurrence_beta)
        self.occurrence_xi = self._put(occurrence_xi)
        self.gradient_coefficient = self._put(gradient_coefficient)
        self.occurrence_hxx = self._put(hxx_occ)
        self.occurrence_hxy = self._put(hxy_occ)
        self.occurrence_hyy = self._put(hyy_occ)
        self.source_hxx = self._put(hxx_source)
        self.source_hxy = self._put(hxy_source)
        self.source_hyy = self._put(hyy_source)
        self.free_dofs = self._put(free_dofs.astype(np.int32, copy=False))
        self.constrained_dofs = self._put(constrained_dofs.astype(np.int32, copy=False))
        self.free_mask = self._put(free_mask)
        self.q_zero = self._put(
            _j_require_static_float64(constraints.q_zero, "constraints.q_zero", (self.n_dofs,))
        )
        self.load_direction = self._put(
            _j_require_static_float64(
                constraints.load_direction, "constraints.load_direction", (self.n_dofs,)
            )
        )
        self.outer_top_selector = self._put(
            _j_require_static_float64(
                constraints.outer_top_selector,
                "constraints.outer_top_selector",
                (self.n_dofs,),
            )
        )
        self.outer_bottom_selector = self._put(
            _j_require_static_float64(
                constraints.outer_bottom_selector,
                "constraints.outer_bottom_selector",
                (self.n_dofs,),
            )
        )
        self.anchor_selector = self._put(
            _j_require_static_float64(
                constraints.anchor_selector, "constraints.anchor_selector", (self.n_dofs,)
            )
        )
        self._identity_2 = self._put(np.eye(2, dtype=np.float64))
        self._lame_lambda = self._put(np.asarray(lame_lambda, dtype=np.float64))
        self._shear_modulus = self._put(np.asarray(shear_modulus, dtype=np.float64))
        self._k_res = self._put(np.asarray(float(static.k_res), dtype=np.float64))
        self._grip_degradation = self._put(
            np.asarray(1.0 + float(static.k_res), dtype=np.float64)
        )
        self._material_zeros = self._put(
            np.zeros(self.n_particles - self.n_material, dtype=np.float64)
        )

        self._trace_counts = {"solve": 0}
        self._compiled_solve = jax.jit(self._traced_solve)
        self._compiled_residual = jax.jit(self._residual_public)
        self._compiled_energy = jax.jit(self._energy_public)
        self._compiled_tangent = jax.jit(self._tangent_public)
        self._compiled_blocks = jax.jit(self._block_diagonal_public)
        self._compiled_inverse_blocks = jax.jit(self._block_inverse_public)
        self._compiled_preconditioner = jax.jit(self._preconditioner_public)
        self._compiled_center_weights = jax.jit(self._center_weights)

    def _put(self, values: Any) -> jax.Array:
        return jax.device_put(values, self.device)

    @property
    def trace_count(self) -> int:
        """Number of times the solve function has been traced (cache diagnostic)."""

        return int(self._trace_counts["solve"])

    @property
    def flattened_support_count(self) -> int:
        return int(self.occurrence_edge.shape[0])

    def _prepare_damage(self, damage: Any) -> jax.Array:
        """Validate material damage and its device center weights loudly."""

        values = _j_require_dynamic_float64(damage, "material damage", (self.n_material,))
        device_values = self._put(values)
        center_weights = self._compiled_center_weights(device_values)
        valid = jnp.all(jnp.isfinite(center_weights)) & jnp.all(center_weights > 0.0)
        if not bool(np.asarray(jax.device_get(valid))):
            raise FloatingPointError(
                "A2 phase center weights are non-finite or non-positive"
            )
        return device_values

    def _center_weights(self, damage: jax.Array) -> jax.Array:
        """Exactly ``Vm*((1-d)^2+k_res)+Vg*(1+k_res)`` on the padded prefix."""

        damage_full = jnp.concatenate((damage, self._material_zeros))
        material_degradation = (1.0 - damage_full) ** 2 + self._k_res
        return (
            self.material_phase_volumes * material_degradation
            + self.grip_phase_volumes * self._grip_degradation
        )

    def _edge_factor(self, damage: jax.Array) -> jax.Array:
        center_weights = self._center_weights(damage)
        return center_weights[self.source] * self.eta

    def _deformation_gradient(
        self, edge_factor: jax.Array, displacement: jax.Array
    ) -> jax.Array:
        del edge_factor
        values = displacement.reshape(self.n_particles, 2)
        displacement_difference = (
            values[self.occurrence_particle] - values[self.occurrence_source]
        )
        current_bond = self.occurrence_xi + displacement_difference
        moment_occurrence = (
            current_bond[:, :, None]
            * self.occurrence_xi[:, None, :]
            * self.occurrence_beta[:, None, None]
        )
        moment = _j_segment_sum(
            moment_occurrence, self.occurrence_edge, self.n_edges
        )
        return jnp.einsum("eab,ebc->eac", moment, self.shape_inverse)

    def _strain_stress(
        self, deformation_gradient: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        increment = deformation_gradient - self._identity_2
        strain = 0.5 * (increment + jnp.swapaxes(increment, 1, 2))
        trace = strain[:, 0, 0] + strain[:, 1, 1]
        stress = (
            self._lame_lambda * trace[:, None, None] * self._identity_2[None, :, :]
            + 2.0 * self._shear_modulus * strain
        )
        return strain, stress

    def _assemble_residual(
        self, stress: jax.Array, edge_factor: jax.Array
    ) -> jax.Array:
        stress_occurrence = stress[self.occurrence_edge]
        inverse_transpose = jnp.swapaxes(
            self.shape_inverse[self.occurrence_edge], 1, 2
        )
        endpoint = jnp.einsum(
            "oab,obc,oc->oa",
            stress_occurrence,
            inverse_transpose,
            self.occurrence_xi,
        )
        endpoint = (
            endpoint
            * self.occurrence_beta[:, None]
            * edge_factor[self.occurrence_edge, None]
        )
        support_result = _j_segment_sum(
            endpoint, self.occurrence_particle, self.n_particles
        )
        source_result = _j_segment_sum(
            -endpoint, self.occurrence_source, self.n_particles
        )
        return (support_result + source_result).reshape(self.n_dofs)

    def _residual_from_factor(
        self, edge_factor: jax.Array, displacement: jax.Array
    ) -> jax.Array:
        deformation_gradient = self._deformation_gradient(edge_factor, displacement)
        _, stress = self._strain_stress(deformation_gradient)
        return self._assemble_residual(stress, edge_factor)

    def _energy_from_factor(
        self, edge_factor: jax.Array, displacement: jax.Array
    ) -> jax.Array:
        deformation_gradient = self._deformation_gradient(edge_factor, displacement)
        strain, _ = self._strain_stress(deformation_gradient)
        trace = strain[:, 0, 0] + strain[:, 1, 1]
        density = (
            0.5 * self._lame_lambda * trace**2
            + self._shear_modulus * jnp.sum(strain * strain, axis=(1, 2))
        )
        return jnp.sum(edge_factor * density)

    def _tangent_from_factor(
        self, edge_factor: jax.Array, direction: jax.Array
    ) -> jax.Array:
        values = direction.reshape(self.n_particles, 2)
        displacement_difference = (
            values[self.occurrence_particle] - values[self.occurrence_source]
        )
        delta_f_occurrence = (
            displacement_difference[:, :, None]
            * self.gradient_coefficient[:, None, :]
        )
        delta_f = _j_segment_sum(
            delta_f_occurrence, self.occurrence_edge, self.n_edges
        )
        delta_strain = 0.5 * (delta_f + jnp.swapaxes(delta_f, 1, 2))
        trace = delta_strain[:, 0, 0] + delta_strain[:, 1, 1]
        delta_stress = (
            self._lame_lambda * trace[:, None, None] * self._identity_2[None, :, :]
            + 2.0 * self._shear_modulus * delta_strain
        )
        return self._assemble_residual(delta_stress, edge_factor)

    def _evaluate_from_factor(
        self, edge_factor: jax.Array, displacement: jax.Array
    ) -> dict[str, jax.Array]:
        energy = self._energy_from_factor(edge_factor, displacement)
        residual = self._residual_from_factor(edge_factor, displacement)
        residual_matrix = residual.reshape(self.n_particles, 2)
        return {
            "energy": energy,
            "residual": residual,
            "internal_force": -residual,
            "top_reaction": jnp.dot(self.outer_top_selector, residual),
            "bottom_reaction": jnp.dot(self.outer_bottom_selector, residual),
            "anchor_reaction": jnp.dot(self.anchor_selector, residual),
            "total_force": jnp.sum(residual_matrix, axis=0),
            "total_moment": jnp.sum(
                self.coordinates[:, 0] * residual_matrix[:, 1]
                - self.coordinates[:, 1] * residual_matrix[:, 0]
            ),
        }

    def _block_diagonal_from_factor(self, edge_factor: jax.Array) -> jax.Array:
        support_hxx = _j_segment_sum(
            edge_factor[self.occurrence_edge] * self.occurrence_hxx,
            self.occurrence_particle,
            self.n_particles,
        )
        support_hxy = _j_segment_sum(
            edge_factor[self.occurrence_edge] * self.occurrence_hxy,
            self.occurrence_particle,
            self.n_particles,
        )
        support_hyy = _j_segment_sum(
            edge_factor[self.occurrence_edge] * self.occurrence_hyy,
            self.occurrence_particle,
            self.n_particles,
        )
        source_hxx = _j_segment_sum(
            edge_factor * self.source_hxx, self.source, self.n_particles
        )
        source_hxy = _j_segment_sum(
            edge_factor * self.source_hxy, self.source, self.n_particles
        )
        source_hyy = _j_segment_sum(
            edge_factor * self.source_hyy, self.source, self.n_particles
        )
        hxx = support_hxx + source_hxx
        hxy = support_hxy + source_hxy
        hyy = support_hyy + source_hyy
        return jnp.stack(
            (
                jnp.stack((hxx, hxy), axis=-1),
                jnp.stack((hxy, hyy), axis=-1),
            ),
            axis=-2,
        )

    def _block_inverse_from_factor(self, edge_factor: jax.Array) -> jax.Array:
        blocks = self._block_diagonal_from_factor(edge_factor)
        hxx = blocks[:, 0, 0]
        hxy = blocks[:, 0, 1]
        hyy = blocks[:, 1, 1]
        determinant = hxx * hyy - hxy * hxy
        free_x = self.free_mask[:, 0]
        free_y = self.free_mask[:, 1]
        both_free = free_x & free_y
        only_x_free = free_x & ~free_y
        only_y_free = ~free_x & free_y

        # The full two-free inverse is the closed form adjugate/determinant.
        # One-free particles use the exact scalar diagonal inverse of the
        # eliminated sub-block, and no-free particles contribute zero.  The
        # masks are structural and never introduce a shift or clipping.
        inverse_xx = jnp.where(
            both_free,
            hyy / determinant,
            jnp.where(only_x_free, 1.0 / hxx, 0.0),
        )
        inverse_xy = jnp.where(both_free, -hxy / determinant, 0.0)
        inverse_yy = jnp.where(
            both_free,
            hxx / determinant,
            jnp.where(only_y_free, 1.0 / hyy, 0.0),
        )
        return jnp.stack(
            (
                jnp.stack((inverse_xx, inverse_xy), axis=-1),
                jnp.stack((inverse_xy, inverse_yy), axis=-1),
            ),
            axis=-2,
        )

    def _preconditioner_from_inverse(
        self, vector: jax.Array, inverse_blocks: jax.Array
    ) -> jax.Array:
        full = jnp.zeros((self.n_dofs,), dtype=jnp.float64).at[self.free_dofs].set(
            vector
        )
        particle = full.reshape(self.n_particles, 2)
        result = jnp.einsum("nij,nj->ni", inverse_blocks, particle)
        return result.reshape(self.n_dofs)[self.free_dofs]

    def _free_tangent_action(
        self, edge_factor: jax.Array, vector: jax.Array
    ) -> jax.Array:
        full = jnp.zeros((self.n_dofs,), dtype=jnp.float64).at[self.free_dofs].set(
            vector
        )
        return self._tangent_from_factor(edge_factor, full)[self.free_dofs]

    def _pcg(
        self,
        edge_factor: jax.Array,
        rhs: jax.Array,
        initial_values: jax.Array,
        initial_residual: jax.Array,
        convergence_limit: jax.Array,
        maxiter: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        inverse_blocks = self._block_inverse_from_factor(edge_factor)

        def precondition(vector: jax.Array) -> jax.Array:
            return self._preconditioner_from_inverse(vector, inverse_blocks)

        residual = initial_residual
        z = precondition(residual)
        direction = z
        rho = jnp.dot(residual, z)
        residual_norm = jnp.linalg.norm(residual)
        finite_initial = (
            jnp.all(jnp.isfinite(residual))
            & jnp.all(jnp.isfinite(z))
            & jnp.isfinite(rho)
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(convergence_limit)
            & jnp.all(jnp.isfinite(inverse_blocks))
        )
        initially_converged = finite_initial & (residual_norm < convergence_limit)
        initially_breakdown = finite_initial & (~initially_converged) & (rho <= 0.0)
        initial_status = jnp.where(
            ~finite_initial,
            _j_STATUS_NONFINITE,
            jnp.where(
                initially_converged,
                _j_STATUS_CONVERGED,
                jnp.where(initially_breakdown, _j_STATUS_BREAKDOWN, _j_STATUS_RUNNING),
            ),
        ).astype(jnp.int32)

        def condition(state: tuple[jax.Array, ...]) -> jax.Array:
            _, _, _, _, _, iteration, status = state
            return (status == _j_STATUS_RUNNING) & (iteration < maxiter)

        def body(state: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
            values, residual, z, direction, rho, iteration, status = state
            action = self._free_tangent_action(edge_factor, direction)
            denominator = jnp.dot(direction, action)
            finite_step = (
                jnp.all(jnp.isfinite(action))
                & jnp.isfinite(denominator)
                & jnp.isfinite(rho)
                & (denominator > 0.0)
                & (rho > 0.0)
            )

            def invalid_step(_: None) -> tuple[jax.Array, ...]:
                finite_values = (
                    jnp.all(jnp.isfinite(action))
                    & jnp.isfinite(denominator)
                    & jnp.isfinite(rho)
                )
                invalid_status = jnp.where(
                    finite_values, _j_STATUS_BREAKDOWN, _j_STATUS_NONFINITE
                ).astype(jnp.int32)
                return values, residual, z, direction, rho, iteration, invalid_status

            def valid_step(_: None) -> tuple[jax.Array, ...]:
                alpha = rho / denominator
                new_values = values + alpha * direction
                new_residual = residual - alpha * action
                new_z = precondition(new_residual)
                new_rho = jnp.dot(new_residual, new_z)
                new_norm = jnp.linalg.norm(new_residual)
                finite_update = (
                    jnp.all(jnp.isfinite(new_values))
                    & jnp.all(jnp.isfinite(new_residual))
                    & jnp.all(jnp.isfinite(new_z))
                    & jnp.isfinite(new_rho)
                    & jnp.isfinite(new_norm)
                )
                converged = finite_update & (new_norm < convergence_limit)
                positive_next = finite_update & (new_rho > 0.0)
                beta = new_rho / rho
                candidate_direction = new_z + beta * direction
                continue_update = finite_update & (~converged) & positive_next
                next_direction = jnp.where(
                    continue_update, candidate_direction, direction
                )
                next_z = jnp.where(continue_update, new_z, z)
                next_rho = jnp.where(continue_update, new_rho, rho)
                next_status = jnp.where(
                    ~finite_update,
                    _j_STATUS_NONFINITE,
                    jnp.where(
                        converged,
                        _j_STATUS_CONVERGED,
                        jnp.where(positive_next, _j_STATUS_RUNNING, _j_STATUS_BREAKDOWN),
                    ),
                ).astype(jnp.int32)
                return (
                    new_values,
                    new_residual,
                    next_z,
                    next_direction,
                    next_rho,
                    iteration + jnp.int32(1),
                    next_status,
                )

            return jax.lax.cond(finite_step, valid_step, invalid_step, operand=None)

        final_state = jax.lax.while_loop(
            condition,
            body,
            (
                initial_values,
                residual,
                z,
                direction,
                rho,
                jnp.int32(0),
                initial_status,
            ),
        )
        values, residual, _, _, _, iterations, status = final_state
        status = jnp.where(
            status == _j_STATUS_RUNNING, _j_STATUS_MAXITER, status
        ).astype(jnp.int32)
        return values, residual, iterations, status

    def _traced_solve(
        self,
        damage: jax.Array,
        load: jax.Array,
        relative_tolerance: jax.Array,
        absolute_tolerance: jax.Array,
        maxiter: jax.Array,
        supplied: jax.Array,
        guess: jax.Array,
    ) -> dict[str, jax.Array]:
        self._trace_counts["solve"] += 1
        edge_factor = self._edge_factor(damage)
        prescribed = self.q_zero + load * self.load_direction
        prescribed_residual = self._residual_from_factor(edge_factor, prescribed)
        free_rhs = -prescribed_residual[self.free_dofs]
        free_rhs_norm = jnp.linalg.norm(free_rhs)
        convergence_limit = jnp.maximum(
            absolute_tolerance, relative_tolerance * free_rhs_norm
        )

        initial_full = jnp.where(supplied, guess, prescribed)
        initial_free_residual = self._residual_from_factor(edge_factor, initial_full)[
            self.free_dofs
        ]
        # The public residual follows the CPU energy-gradient sign.  PCG's
        # Krylov residual is ``rhs - A @ x`` and therefore has the opposite
        # sign for a supplied nonzero initial free field.
        initial_krylov_residual = jnp.where(
            supplied, -initial_free_residual, free_rhs
        )
        initial_free_residual_norm = jnp.linalg.norm(initial_free_residual)
        initial_values = jnp.where(
            supplied,
            initial_full[self.free_dofs],
            jnp.zeros((self.n_free,), dtype=jnp.float64),
        )

        free_values, recursive_residual, iterations, solver_status = self._pcg(
            edge_factor,
            free_rhs,
            initial_values,
            initial_krylov_residual,
            convergence_limit,
            maxiter,
        )
        displacement = prescribed.at[self.free_dofs].set(free_values)
        evaluation = self._evaluate_from_factor(edge_factor, displacement)
        explicit_free_residual = evaluation["residual"][self.free_dofs]
        explicit_norm = jnp.linalg.norm(explicit_free_residual)
        recursive_norm = jnp.linalg.norm(recursive_residual)
        finite_final = (
            jnp.all(jnp.isfinite(displacement))
            & jnp.all(jnp.isfinite(recursive_residual))
            & jnp.all(jnp.isfinite(explicit_free_residual))
            & jnp.isfinite(explicit_norm)
            & jnp.all(
                jnp.stack(
                    [
                        jnp.all(jnp.isfinite(value))
                        for value in evaluation.values()
                    ]
                )
            )
            & jnp.isfinite(convergence_limit)
        )
        explicit_converged = finite_final & (explicit_norm < convergence_limit)
        final_status = jnp.where(
            ~finite_final,
            _j_STATUS_NONFINITE,
            jnp.where(
                (solver_status == _j_STATUS_CONVERGED) & (~explicit_converged),
                _j_STATUS_EXPLICIT_RESIDUAL,
                solver_status,
            ),
        ).astype(jnp.int32)
        converged = final_status == _j_STATUS_CONVERGED
        relative_norm = explicit_norm / jnp.maximum(free_rhs_norm, 1.0e-300)
        cg_info = jnp.where(
            final_status == _j_STATUS_CONVERGED,
            jnp.int32(0),
            jnp.where(final_status == _j_STATUS_MAXITER, iterations, jnp.int32(-1)),
        ).astype(jnp.int32)
        return {
            "displacement": displacement,
            "free_rhs": free_rhs,
            "free_residual": explicit_free_residual,
            "recursive_free_residual": recursive_residual,
            "initial_displacement": initial_full,
            "initial_free_residual": initial_free_residual,
            "free_rhs_norm": free_rhs_norm,
            "free_residual_norm": explicit_norm,
            "recursive_residual_norm": recursive_norm,
            "free_residual_relative_norm": relative_norm,
            "initial_free_residual_norm": initial_free_residual_norm,
            "cg_convergence_limit": convergence_limit,
            "initial_guess_supplied": supplied,
            "cg_iterations": iterations,
            "cg_info": cg_info,
            "status_code": final_status,
            "explicit_converged": explicit_converged,
            "converged": converged,
            "u_bar": load,
            "cg_rtol": relative_tolerance,
            "cg_atol": absolute_tolerance,
            "cg_maxiter": maxiter,
            **evaluation,
        }

    def _residual_public(self, damage: jax.Array, displacement: jax.Array) -> jax.Array:
        return self._residual_from_factor(self._edge_factor(damage), displacement)

    def _energy_public(self, damage: jax.Array, displacement: jax.Array) -> jax.Array:
        return self._energy_from_factor(self._edge_factor(damage), displacement)

    def _tangent_public(self, damage: jax.Array, direction: jax.Array) -> jax.Array:
        return self._tangent_from_factor(self._edge_factor(damage), direction)

    def _block_diagonal_public(self, damage: jax.Array) -> jax.Array:
        return self._block_diagonal_from_factor(self._edge_factor(damage))

    def _block_inverse_public(self, damage: jax.Array) -> jax.Array:
        return self._block_inverse_from_factor(self._edge_factor(damage))

    def _preconditioner_public(self, damage: jax.Array, vector: jax.Array) -> jax.Array:
        inverse_blocks = self._block_inverse_from_factor(self._edge_factor(damage))
        return self._preconditioner_from_inverse(vector, inverse_blocks)

    def center_weights(self, damage: Any) -> jax.Array:
        """Return the exact phase center weights for a material damage field."""

        _j_require_x64_enabled()
        result = self._compiled_center_weights(
            self._put(
                _j_require_dynamic_float64(damage, "material damage", (self.n_material,))
            )
        )
        _j_require_finite_device_result(result, "JAX A2 center weights")
        return result

    def residual(self, damage: Any, displacement: Any) -> jax.Array:
        _j_require_x64_enabled()
        damage_device = self._prepare_damage(damage)
        displacement_array = _j_require_dynamic_float64(
            displacement, "displacement", (self.n_dofs,)
        )
        result = self._compiled_residual(damage_device, self._put(displacement_array))
        _j_require_finite_device_result(result, "JAX A2 residual")
        return result

    def energy(self, damage: Any, displacement: Any) -> jax.Array:
        _j_require_x64_enabled()
        damage_device = self._prepare_damage(damage)
        displacement_array = _j_require_dynamic_float64(
            displacement, "displacement", (self.n_dofs,)
        )
        result = self._compiled_energy(damage_device, self._put(displacement_array))
        _j_require_finite_device_result(result, "JAX A2 energy")
        return result

    def tangent_action(self, damage: Any, direction: Any) -> jax.Array:
        """Apply the analytic flattened plane-strain Hessian action."""

        _j_require_x64_enabled()
        damage_device = self._prepare_damage(damage)
        direction_array = _j_require_dynamic_float64(direction, "direction", (self.n_dofs,))
        result = self._compiled_tangent(damage_device, self._put(direction_array))
        _j_require_finite_device_result(result, "JAX A2 tangent action")
        return result

    def tangent_block_diagonal(self, damage: Any) -> jax.Array:
        """Return exact particle-local tangent blocks for fixed damage."""

        _j_require_x64_enabled()
        result = self._compiled_blocks(self._prepare_damage(damage))
        _j_require_finite_device_result(result, "JAX A2 tangent block diagonal")
        return result

    def block_jacobi_inverse(self, damage: Any) -> jax.Array:
        """Return the exact masked closed-form block-Jacobi inverse."""

        _j_require_x64_enabled()
        result = self._compiled_inverse_blocks(self._prepare_damage(damage))
        _j_require_finite_device_result(result, "JAX A2 block-Jacobi inverse")
        return result

    def apply_block_jacobi(self, damage: Any, vector: Any) -> jax.Array:
        """Apply the exact masked block-Jacobi preconditioner to a free vector."""

        _j_require_x64_enabled()
        damage_device = self._prepare_damage(damage)
        vector_array = _j_require_dynamic_float64(vector, "free vector", (self.n_free,))
        result = self._compiled_preconditioner(damage_device, self._put(vector_array))
        _j_require_finite_device_result(result, "JAX A2 block-Jacobi application")
        return result

    def solve(
        self,
        damage: Any,
        u_bar: Any,
        cg_rtol: Any,
        cg_atol: Any,
        cg_maxiter: Any,
        *,
        initial_guess: Any | None = None,
    ) -> JaxA2SolveResult:
        """Run device-resident fixed-damage affine-constraint PCG.

        A non-converged numerical solve is returned with an explicit status so
        callers can inspect all device diagnostics.  ``require_converged`` or
        ``solve_strict`` supplies the raising policy when that is desired.
        """

        _j_require_x64_enabled()
        damage_device = self._prepare_damage(damage)
        load = _j_dynamic_float_scalar(u_bar, "u_bar", allow_zero=True, nonnegative=False)
        relative_tolerance = _j_dynamic_float_scalar(
            cg_rtol, "cg_rtol", allow_zero=True
        )
        absolute_tolerance = _j_dynamic_float_scalar(
            cg_atol, "cg_atol", allow_zero=True
        )
        if relative_tolerance == 0.0 and absolute_tolerance == 0.0:
            raise ValueError("cg_rtol and cg_atol cannot both be zero")
        maxiter = _j_dynamic_maxiter(cg_maxiter)

        prescribed = np.asarray(
            self.constraints.prescribed_vector(load), dtype=np.float64
        )
        if initial_guess is None:
            supplied = False
            guess_array = prescribed
        else:
            supplied = True
            guess_array = np.asarray(
                _j_require_dynamic_float64(initial_guess, "initial_guess", (self.n_dofs,)),
                dtype=np.float64,
            )
            if not np.array_equal(
                guess_array[self.constraints.constrained_dofs],
                prescribed[self.constraints.constrained_dofs],
            ):
                raise ValueError(
                    "initial_guess constrained values must exactly match the "
                    "prescribed displacement at u_bar"
                )

        values = self._compiled_solve(
            damage_device,
            self._put(np.asarray(load, dtype=np.float64)),
            self._put(np.asarray(relative_tolerance, dtype=np.float64)),
            self._put(np.asarray(absolute_tolerance, dtype=np.float64)),
            self._put(np.asarray(maxiter, dtype=np.int32)),
            self._put(np.asarray(supplied, dtype=bool)),
            self._put(guess_array),
        )
        return JaxA2SolveResult(
            displacement=values["displacement"],
            damage=damage_device,
            u_bar=values["u_bar"],
            cg_rtol=values["cg_rtol"],
            cg_atol=values["cg_atol"],
            cg_maxiter=values["cg_maxiter"],
            free_rhs=values["free_rhs"],
            free_residual=values["free_residual"],
            recursive_free_residual=values["recursive_free_residual"],
            initial_displacement=values["initial_displacement"],
            initial_free_residual=values["initial_free_residual"],
            free_rhs_norm=values["free_rhs_norm"],
            free_residual_norm=values["free_residual_norm"],
            recursive_residual_norm=values["recursive_residual_norm"],
            free_residual_relative_norm=values["free_residual_relative_norm"],
            initial_free_residual_norm=values["initial_free_residual_norm"],
            cg_convergence_limit=values["cg_convergence_limit"],
            initial_guess_supplied=values["initial_guess_supplied"],
            cg_iterations=values["cg_iterations"],
            cg_info=values["cg_info"],
            status_code=values["status_code"],
            explicit_converged=values["explicit_converged"],
            converged_flag=values["converged"],
            energy=values["energy"],
            top_reaction=values["top_reaction"],
            bottom_reaction=values["bottom_reaction"],
            anchor_reaction=values["anchor_reaction"],
            total_force=values["total_force"],
            total_moment=values["total_moment"],
        )

    def solve_strict(self, *args: Any, **kwargs: Any) -> JaxA2SolveResult:
        result = self.solve(*args, **kwargs)
        return result.require_converged()

    def replay_cpu(
        self,
        jax_result: JaxA2SolveResult,
        damage: Any,
        u_bar: Any,
        cg_rtol: Any,
        cg_atol: Any,
        cg_maxiter: Any,
        *,
        initial_guess: Any | None = None,
        seed_kind: str = "direct_load",
        source_state: str = "direct",
        source_step: int = 0,
        source_sweep: int = 0,
        source_damage_digest: str | None = None,
    ) -> JaxA2CPUReplayResult:
        """Replay one device displacement through a fresh CPU A2 view.

        The transferred displacement is the only GPU-produced field.  Energy,
        residual, reactions, force, moment, the explicit free-residual gate,
        and the seed provenance are recomputed independently on the CPU
        without rerunning CPU CG.  Only ``cpu_result`` is authoritative.
        """

        if not isinstance(jax_result, JaxA2SolveResult):
            raise TypeError("jax_result must be a JaxA2SolveResult instance")
        _j_require_x64_enabled()
        damage_array = np.asarray(
            _j_require_dynamic_float64(damage, "material damage", (self.n_material,)),
            dtype=np.float64,
        )
        load = float(
            _j_dynamic_float_scalar(u_bar, "u_bar", allow_zero=True, nonnegative=False)
        )
        relative_tolerance = float(
            _j_dynamic_float_scalar(cg_rtol, "cg_rtol", allow_zero=True)
        )
        absolute_tolerance = float(
            _j_dynamic_float_scalar(cg_atol, "cg_atol", allow_zero=True)
        )
        if relative_tolerance == 0.0 and absolute_tolerance == 0.0:
            raise ValueError("cg_rtol and cg_atol cannot both be zero")
        maxiter = int(_j_dynamic_maxiter(cg_maxiter))

        (
            status_code,
            displacement,
            result_damage,
            result_load,
            result_rtol,
            result_atol,
            result_maxiter,
            result_initial_displacement,
            gpu_iterations,
            result_initial_guess_supplied,
            result_cg_info,
        ) = jax.device_get(
            (
                jax_result.status_code,
                jax_result.displacement,
                jax_result.damage,
                jax_result.u_bar,
                jax_result.cg_rtol,
                jax_result.cg_atol,
                jax_result.cg_maxiter,
                jax_result.initial_displacement,
                jax_result.cg_iterations,
                jax_result.initial_guess_supplied,
                jax_result.cg_info,
            )
        )
        result_damage = np.asarray(
            _j_require_dynamic_float64(result_damage, "jax_result.damage", (self.n_material,)),
            dtype=np.float64,
        )
        displacement = np.asarray(
            _j_require_dynamic_float64(
                displacement, "jax_result.displacement", (self.n_dofs,)
            ),
            dtype=np.float64,
        )
        result_initial_displacement = np.asarray(
            _j_require_dynamic_float64(
                result_initial_displacement,
                "jax_result.initial_displacement",
                (self.n_dofs,),
            ),
            dtype=np.float64,
        )
        for name, value in (
            ("u_bar", result_load),
            ("cg_rtol", result_rtol),
            ("cg_atol", result_atol),
        ):
            _j_require_dynamic_float64(value, f"jax_result.{name}", ())
        if not np.array_equal(result_damage, damage_array):
            raise ValueError("jax_result damage does not match the CPU replay damage")
        for name, actual, expected in (
            ("u_bar", float(result_load), load),
            ("cg_rtol", float(result_rtol), relative_tolerance),
            ("cg_atol", float(result_atol), absolute_tolerance),
        ):
            if actual != expected:
                raise ValueError(f"jax_result {name} does not match the CPU replay value")
        if int(result_maxiter) != maxiter:
            raise ValueError(
                "jax_result cg_maxiter does not match the CPU replay value"
            )
        if int(status_code) != int(_j_STATUS_CONVERGED):
            raise RuntimeError(
                "cannot replay a non-converged JAX A2 mechanics result: "
                f"status={_j_STATUS_NAMES.get(int(status_code), str(int(status_code)))}"
            )
        if int(gpu_iterations) > maxiter:
            raise RuntimeError(
                "GPU iteration count exceeds the configured CG limit: "
                f"iterations={int(gpu_iterations)}, maxiter={maxiter}"
            )
        if int(result_cg_info) != 0:
            raise RuntimeError("converged jax_result must have cg_info=0")

        # This is intentionally a fresh CPU view.  The immutable fixed-damage
        # reference owns its own phase weights and tangent caches; nothing is
        # reused from the device execution.
        view = A2FixedDamageView(self.static, damage_array)
        constraints = self.constraints
        prescribed = constraints.prescribed_vector(load)
        if not np.array_equal(
            displacement[constraints.constrained_dofs],
            prescribed[constraints.constrained_dofs],
        ):
            raise RuntimeError(
                "GPU displacement violates the exact CPU affine-constraint values"
            )
        free = constraints.free_dofs
        evaluation = view.evaluate(displacement)
        prescribed_residual = view.residual(prescribed)
        rhs = -prescribed_residual[free]
        rhs_norm = float(np.linalg.norm(rhs))
        convergence_limit = max(absolute_tolerance, relative_tolerance * rhs_norm)
        free_residual = np.asarray(evaluation.residual[free], dtype=np.float64)
        free_residual_norm = float(np.linalg.norm(free_residual))
        if not np.isfinite(free_residual_norm) or free_residual_norm >= convergence_limit:
            raise RuntimeError(
                "GPU displacement fails the strict CPU CG gate: "
                f"norm={free_residual_norm:.17g}, allowed={convergence_limit:.17g}"
            )

        if initial_guess is None:
            if bool(result_initial_guess_supplied):
                raise ValueError(
                    "jax_result initial_guess_supplied does not match the CPU replay seed policy"
                )
            initial_guess_supplied = False
            seed = _a_immutable(prescribed)
        else:
            if not bool(result_initial_guess_supplied):
                raise ValueError(
                    "jax_result initial_guess_supplied does not match the CPU replay seed policy"
                )
            initial_guess_supplied = True
            seed = np.asarray(
                _j_require_dynamic_float64(initial_guess, "initial_guess", (self.n_dofs,)),
                dtype=np.float64,
            )
            _a_check_exact_constraints(seed, constraints, load, "A2 initial_guess")
            seed = _a_immutable(seed)
        if not np.array_equal(result_initial_displacement, np.asarray(seed)):
            raise ValueError(
                "jax_result initial_displacement does not match the CPU replay seed"
            )
        initial_residual_norm = float(np.linalg.norm(view.residual(np.asarray(seed))[free]))
        residual_matrix = evaluation.residual.reshape(view.n_particles, 2)
        total_force = np.sum(residual_matrix, axis=0)
        total_moment = float(
            np.sum(
                view.coordinates[:, 0] * residual_matrix[:, 1]
                - view.coordinates[:, 1] * residual_matrix[:, 0]
            )
        )
        if not np.all(np.isfinite(total_force)) or not np.isfinite(total_moment):
            raise FloatingPointError("A2 force/moment diagnostics are non-finite")
        if source_damage_digest is None:
            source_damage_digest = view.damage_digest
        seed_provenance = _a_seed_record(
            seed_kind,
            np.asarray(seed),
            view,
            constraints,
            initial_residual_norm,
            free_residual_norm,
            source_state=source_state,
            source_step=source_step,
            source_sweep=source_sweep,
            source_damage_digest=source_damage_digest,
        )
        cpu_result = A2MechanicsSolveResult(
            displacement=displacement,
            evaluation=evaluation,
            constraints=constraints,
            outer_displacement=load,
            free_residual=free_residual,
            free_residual_norm=free_residual_norm,
            free_rhs_norm=rhs_norm,
            free_residual_relative_norm=free_residual_norm / max(rhs_norm, 1.0e-300),
            cg_convergence_limit=convergence_limit,
            cg_iterations=int(gpu_iterations),
            cg_info=0,
            outer_reaction=float(constraints.outer_top_selector @ evaluation.residual),
            bottom_reaction=float(constraints.outer_bottom_selector @ evaluation.residual),
            anchor_reaction=float(constraints.anchor_selector @ evaluation.residual),
            total_force=total_force,
            total_moment=total_moment,
            static_runtime_identity=view.static.runtime_identity,
            static_digest=view.static.deterministic_digest,
            damage_runtime_identity=view.material_damage_runtime_identity,
            damage_digest=view.damage_digest,
            unit_system_digest=_a_unit_system_digest(
                view.static.deterministic_digest, view.damage_digest
            ),
            seed_provenance=seed_provenance,
        )
        return JaxA2CPUReplayResult(jax_result=jax_result, cpu_result=cpu_result)

@dataclass(frozen=True)
class JaxA2CPUReplayAdapter:
    """Return CPU-authoritative A2 mechanics results from the JAX backend.

    This is the only restricted adapter accepted by the engineering-grip
    controller.  It serves the work-check center/plus/minus solves and every
    stagger solve from one device backend, so a single fixed-shape compiled
    solve is reused for all changing damage/load/seed/tolerance inputs.
    """

    backend: JaxA2UnionMechanics

    def __post_init__(self) -> None:
        if not isinstance(self.backend, JaxA2UnionMechanics):
            raise TypeError("backend must be a JaxA2UnionMechanics instance")

    @classmethod
    def from_static(
        cls,
        static: A2StaticMechanics,
        constraints: AffineConstraints,
        device: jax.Device,
    ) -> "JaxA2CPUReplayAdapter":
        """Build the adapter from an accepted static model and constraints."""

        return cls(JaxA2UnionMechanics(static, constraints, device))

    @property
    def cpu_static(self) -> A2StaticMechanics:
        return self.backend.static

    @property
    def constraints(self) -> AffineConstraints:
        return self.backend.constraints

    @property
    def mechanics_backend(self) -> str:
        return f"jax_{self.backend.device.platform}_cpu_replay"

    @property
    def device(self) -> jax.Device:
        return self.backend.device

    def ensure_view(self, view: A2FixedDamageView, constraints: AffineConstraints) -> None:
        """Require exact static/constraint identity before any adapter solve."""

        if not isinstance(view, A2FixedDamageView):
            raise TypeError("view must be an A2FixedDamageView")
        if not isinstance(constraints, AffineConstraints):
            raise TypeError("constraints must be an AffineConstraints")
        if view.static is not self.backend.static:
            raise ValueError(
                "mechanics adapter static identity does not match the solve view"
            )
        _a_validate_view_constraints(view, self.backend.constraints)
        if constraints is not self.backend.constraints:
            raise ValueError(
                "mechanics adapter constraints are not the exact bound object"
            )

    def solve_pair(
        self,
        view: A2FixedDamageView,
        constraints: AffineConstraints,
        u_bar: Any,
        cg_rtol: Any,
        cg_atol: Any,
        cg_maxiter: Any,
        *,
        initial_guess: Any | None = None,
        seed_kind: str = "direct_load",
        source_state: str = "direct",
        source_step: int = 0,
        source_sweep: int = 0,
        source_damage_digest: str | None = None,
    ) -> JaxA2CPUReplayResult:
        """Run the device solve and return the CPU-authoritative replay pair."""

        self.ensure_view(view, constraints)
        load = _a_scalar(u_bar, "u_bar")
        damage = np.asarray(view.material_damage, dtype=np.float64)
        jax_result = self.backend.solve_strict(
            damage,
            load,
            cg_rtol,
            cg_atol,
            cg_maxiter,
            initial_guess=initial_guess,
        )
        replay = self.backend.replay_cpu(
            jax_result,
            damage,
            load,
            cg_rtol,
            cg_atol,
            cg_maxiter,
            initial_guess=initial_guess,
            seed_kind=seed_kind,
            source_state=source_state,
            source_step=source_step,
            source_sweep=source_sweep,
            source_damage_digest=source_damage_digest,
        )
        _a_check_exact_constraints(
            replay.cpu_result.displacement, constraints, load, "direct solve"
        )
        if replay.cpu_result.U_ext != load:
            raise ValueError("direct mechanics solve changed u_bar")
        return replay

    def solve_fixed_outer_displacement(
        self,
        view: A2FixedDamageView,
        constraints: AffineConstraints,
        u_bar: Any,
        cg_rtol: Any,
        cg_atol: Any,
        cg_maxiter: Any,
        *,
        initial_guess: Any | None = None,
        seed_kind: str = "direct_load",
        source_state: str = "direct",
        source_step: int = 0,
        source_sweep: int = 0,
        source_damage_digest: str | None = None,
    ) -> A2MechanicsSolveResult:
        """CPU-authoritative drop-in for the direct engineering-grip solve."""

        return self.solve_pair(
            view,
            constraints,
            u_bar,
            cg_rtol,
            cg_atol,
            cg_maxiter,
            initial_guess=initial_guess,
            seed_kind=seed_kind,
            source_state=source_state,
            source_step=source_step,
            source_sweep=source_sweep,
            source_damage_digest=source_damage_digest,
        ).cpu_result
# ===== nonlocal_engineering_grip =====











_e_FLOAT64 = np.dtype(np.float64)
_e_INT64 = np.dtype(np.int64)
_e_BOOL = np.dtype(bool)
_e_EPS64 = np.finfo(np.float64).eps
_e_RELATIVE_FLOOR = float(np.finfo(np.float64).tiny)
_e_REPLAY_BOUND = 512.0 * _e_EPS64

_e_KERNEL = "cubic"

_e_ACCEPTANCE_MODE = "strict"

WARNING_GATE_NAMES = (
    "displacement",
    "damage",
    "history",
    "reaction",
    "damage_residual",
    "damage_range",
    "healing",
    "history_monotonic",
    "force",
    "moment",
)

_e_WARNING_HARD_GATE_NAMES = ("finite", "mechanics_before", "mechanics_after")

_e_CRACK_TIP = np.asarray([0.5, 0.5], dtype=np.float64)

def _e_immutable(values: Any, dtype: Any | None = None) -> np.ndarray:
    """Make a bytes-backed immutable numerical array."""

    source = np.array(values, dtype=dtype, copy=True, order="C")
    result = np.frombuffer(source.tobytes(order="C"), dtype=source.dtype)
    result = result.reshape(source.shape)
    result.setflags(write=False)
    return result

def _e_scalar(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar, got shape {array.shape}")
    try:
        result = float(array)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a scalar number") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result

def _e_integer(value: Any, name: str, *, positive: bool = False) -> int:
    array = np.asarray(value)
    if array.ndim != 0 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be an integer")
    result = int(array)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result

def _e_f64(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != _e_FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array

def _e_bool(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != _e_BOOL:
        raise ValueError(f"{name} must use exact boolean dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array

def _e_max_abs(value: Any) -> float:
    array = np.asarray(value)
    return float(np.max(np.abs(array))) if array.size else 0.0

def _e_relative(first: float, second: float) -> float:
    return abs(float(first) - float(second)) / max(
        abs(float(first)), abs(float(second)), _e_RELATIVE_FLOOR
    )

def _e_validate_warning_safety_limit(value: Any) -> float:
    """Return the explicit warning range/healing safety limit."""

    if value is None:
        raise ValueError(
            "warning_safety_limit is required for warning acceptance mode"
        )
    limit = _e_scalar(value, "warning_safety_limit")
    if limit <= 0.0:
        raise ValueError("warning_safety_limit must be positive")
    return limit

def _e_compare_damage_residual_reconstruction(
    strong_actual: Any,
    volume_actual: Any,
    damage: Any,
    history: Any,
    matrix: Any,
    volumes: Any,
    Gc: Any,
    l0: Any,
    name: str,
) -> None:
    """Compare damage residuals using the scale of their signed constituents.

    The AT2 strong residual is the cancellation-prone sum

    ``Gc/l0*d + Gc*l0*(D @ d) + 2*H*d - 2*H``.

    A reconstruction error is therefore bounded componentwise by 512 float64
    epsilons times the sum of the absolute constituent terms.  The Euclidean
    residual is the same strong residual multiplied by the strictly positive
    particle volume, so its bound is multiplied by that volume as well.  No
    unit scale floor is appropriate for either comparison.
    """

    d = _e_f64(damage, f"{name} damage")
    if d.ndim != 1 or d.size == 0:
        raise ValueError(f"{name} damage must be a nonempty one-dimensional array")
    H = _e_f64(history, f"{name} history", d.shape)
    V = _e_f64(volumes, f"{name} volumes", d.shape)
    if np.any(V <= 0.0):
        raise ValueError(f"{name} volumes must be strictly positive")

    if issparse(matrix):
        M = matrix.tocsr()
        if M.dtype != _e_FLOAT64:
            raise ValueError(
                f"{name} damage matrix must use exact float64 dtype, got {M.dtype}"
            )
        if M.shape != (d.size, d.size):
            raise ValueError(
                f"{name} damage matrix must have shape {(d.size, d.size)}, "
                f"got {M.shape}"
            )
        if M.nnz and not np.all(np.isfinite(M.data)):
            raise ValueError(f"{name} damage matrix must contain only finite values")
    else:
        M = _e_f64(matrix, f"{name} damage matrix", (d.size, d.size))

    gc_value = _e_scalar(Gc, f"{name} Gc")
    l0_value = _e_scalar(l0, f"{name} l0")
    if gc_value <= 0.0 or l0_value <= 0.0:
        raise ValueError(f"{name} Gc/l0 must be positive")

    graph = np.asarray(M @ d, dtype=np.float64)
    if graph.shape != d.shape or not np.all(np.isfinite(graph)):
        raise ValueError(f"{name} damage graph product must be finite with damage shape")
    local = np.asarray(gc_value / l0_value * d, dtype=np.float64)
    graph_term = np.asarray(gc_value * l0_value * graph, dtype=np.float64)
    history_damage = np.asarray(2.0 * H * d, dtype=np.float64)
    history_constant = np.asarray(2.0 * H, dtype=np.float64)
    terms = (local, graph_term, history_damage, history_constant)
    if not all(np.all(np.isfinite(term)) for term in terms):
        raise ValueError(f"{name} damage residual constituents must be finite")
    term_scale = np.zeros(d.shape, dtype=np.float64)
    for term in terms:
        term_scale += np.abs(term)
    bound = np.asarray(_e_REPLAY_BOUND * term_scale, dtype=np.float64)
    expected = np.asarray(
        local + graph_term + history_damage - history_constant,
        dtype=np.float64,
    )
    expected_volume = np.asarray(V * expected, dtype=np.float64)
    volume_bound = np.asarray(V * bound, dtype=np.float64)
    if not all(
        np.all(np.isfinite(value))
        for value in (term_scale, bound, expected, expected_volume, volume_bound)
    ):
        raise ValueError(f"{name} damage residual reconstruction bound is non-finite")

    actual = _e_f64(strong_actual, f"{name} strong residual", d.shape)
    actual_volume = _e_f64(volume_actual, f"{name} volume residual", d.shape)
    strong_defect = np.abs(actual - expected)
    volume_defect = np.abs(actual_volume - expected_volume)
    if not np.all(np.isfinite(strong_defect)) or not np.all(
        np.isfinite(volume_defect)
    ):
        raise ValueError(f"{name} residual reconstruction defect is non-finite")
    if np.any(strong_defect > bound) or np.any(volume_defect > volume_bound):
        raise ValueError(
            f"{name} residual reconstruction exceeds 512eps constituent bound: "
            f"strong_defect={_e_max_abs(strong_defect):.17g}, "
            f"strong_tolerance={_e_max_abs(bound):.17g}, "
            f"volume_defect={_e_max_abs(volume_defect):.17g}, "
            f"volume_tolerance={_e_max_abs(volume_bound):.17g}"
        )

def _e_hash_named(values: Iterable[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, value in values:
        if not isinstance(name, str) or not name:
            raise ValueError("identity field names must be nonempty strings")
        if isinstance(value, str):
            encoded = value.encode("utf-8")
            digest.update(name.encode("utf-8"))
            digest.update(b"|str|")
            digest.update(np.asarray([len(encoded)], dtype=_e_INT64).tobytes())
            digest.update(encoded)
            continue
        array = np.asarray(value)
        if array.dtype.kind == "O":
            raise ValueError(f"identity field {name!r} has object dtype")
        digest.update(name.encode("utf-8"))
        digest.update(b"|array|")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=_e_INT64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()

def _e_state_digest(
    u_bar: float,
    union_displacement: np.ndarray,
    material_damage: np.ndarray,
    material_history: np.ndarray,
) -> str:
    return _e_hash_named(
        (
            ("u_bar", np.asarray(u_bar, dtype=np.float64)),
            ("union_displacement", union_displacement),
            ("material_damage", material_damage),
            ("material_history", material_history),
        )
    )

def _e_damage_digest(damage: np.ndarray) -> str:
    return _e_hash_named((("material_damage", damage),))

def _e_optional_integer(value: Any | None, name: str) -> int | None:
    if value is None:
        return None
    return _e_integer(value, name)

def _e_source_digest(model: "EngineeringGripModel") -> tuple[str, str]:
    geometry = model.geometry
    material_hash = _e_hash_named(
        (
            ("coordinates", geometry.material_reference_coordinates),
            ("volumes", geometry.material_reference_volumes),
            ("triangles", geometry.material_reference_triangles),
            ("edges", geometry.material_reference_edges),
            ("bottom", geometry.material_bottom_mask[: geometry.n_material]),
            ("top", geometry.material_top_mask[: geometry.n_material]),
            ("anchor", 0),
        )
    )
    union_hash = _e_hash_named(
        (
            ("coordinates", geometry.coordinates),
            ("material_phase_volumes", geometry.material_volumes),
            ("grip_phase_volumes", geometry.grip_volumes),
            ("union_volumes", geometry.union_volumes),
            ("union_triangles", geometry.union_triangles),
            ("candidate_edges", geometry.candidate_edges),
            ("visible_edges", geometry.directed_edges),
            ("removed_edges", geometry.removed_edges),
            ("volume_bottom_mask", model.constraints.outer_bottom_mask),
            ("volume_top_mask", model.constraints.outer_top_mask),
            ("visibility_tolerance", np.asarray(geometry.visibility_tolerance, dtype=np.float64)),
        )
    )
    return material_hash, union_hash

def _e_engineering_model_digest(model: "EngineeringGripModel") -> str:
    """Return the durable value identity of one engineering-grip model.

    The digest intentionally contains no runtime/cache token.  It is used by
    owner-backed validation when the authoritative replay is built from fresh
    fixed-damage views and therefore has different Python identities.
    """

    if not isinstance(model, EngineeringGripModel):
        raise TypeError("engineering model digest requires an EngineeringGripModel")
    geometry = model.geometry
    operator = model.material_operator
    return _e_hash_named(
        (
            ("static_digest", model.static_digest),
            ("material_hash", model.material_hash),
            ("union_hash", model.union_hash),
            ("Gc", model.Gc),
            ("l0", model.l0),
            ("crack_region_mask", model.crack_region_mask),
            ("material_coordinates", geometry.material_reference_coordinates),
            ("material_volumes", geometry.material_reference_volumes),
            ("material_edges", geometry.material_reference_edges),
            ("material_delta", geometry.delta),
            ("material_operator_coordinates", operator.coordinates),
            ("material_operator_volumes", operator.volumes),
            ("material_operator_edges", operator.directed_edges),
            ("material_operator_delta", operator.delta),
            ("material_operator_kernel", operator.kernel),
        )
    )

def _e_engineering_config_digest(config: "EngineeringGripConfig") -> str:
    """Return the durable value identity of an engineering-grip config."""

    if not isinstance(config, EngineeringGripConfig):
        raise TypeError("engineering config digest requires an EngineeringGripConfig")
    values: list[tuple[str, Any]] = [
        ("load_start", config.load_start),
        ("load_final", config.load_final),
        ("du_initial", config.du_initial),
        ("du_min", config.du_min),
        ("cutback_factor", config.cutback_factor),
        ("max_cutbacks", config.max_cutbacks),
        ("max_stagger_iterations", config.max_stagger_iterations),
        ("cg_rtol", config.cg_rtol),
        ("cg_atol", config.cg_atol),
        ("cg_maxiter", config.cg_maxiter),
        ("healing_atol", config.healing_atol),
        ("damage_range_atol", config.damage_range_atol),
        ("work_check_load", config.work_check_load),
        ("work_epsilon", config.work_epsilon),
        ("work_relative_limit", config.work_relative_limit),
    ]
    for name in (
        "displacement",
        "damage",
        "history",
        "reaction",
        "damage_residual",
        "force",
        "moment",
    ):
        tolerance = getattr(config, name)
        values.extend(
            (
                (f"{name}.atol", tolerance.atol),
                (f"{name}.rtol", tolerance.rtol),
                (f"{name}.scale_floor", tolerance.scale_floor),
            )
        )
    return _e_hash_named(values)

def build_crack_region_mask(coordinates: Any, delta: Any) -> np.ndarray:
    """Return material particles within one horizon of the unique notch tip."""

    values = _e_f64(coordinates, "crack-region coordinates")
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError("crack-region coordinates must have shape (N, 2)")
    horizon = _e_scalar(delta, "crack-region delta")
    if horizon <= 0.0:
        raise ValueError("crack-region delta must be positive")
    tip_matches = np.flatnonzero(np.all(values == _e_CRACK_TIP[None, :], axis=1))
    if len(tip_matches) != 1:
        raise ValueError(
            "crack-region coordinates must contain exactly one tip particle (0.5, 0.5)"
        )
    mask = np.linalg.norm(values - _e_CRACK_TIP[None, :], axis=1) <= horizon
    if not bool(mask[int(tip_matches[0])]):
        raise ValueError("the exact crack tip is outside its own crack region")
    return _e_immutable(np.asarray(mask, dtype=bool))

@dataclass(frozen=True)
class EngineeringGripModel:
    """Unequal union/material model used by the direct engineering controller."""

    geometry: UnionGripGeometry
    static: A2StaticMechanics
    constraints: AffineConstraints
    top_line: LineQuadrature
    bottom_line: LineQuadrature
    material_operator: ExplicitVolumeSparseD
    Gc: float
    l0: float
    crack_region_mask: np.ndarray
    material_hash: str
    union_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, UnionGripGeometry):
            raise TypeError("engineering-grip geometry must be UnionGripGeometry")
        if not isinstance(self.static, A2StaticMechanics):
            raise TypeError("engineering-grip static must be A2StaticMechanics")
        if not isinstance(self.constraints, AffineConstraints):
            raise TypeError("engineering-grip constraints must be AffineConstraints")
        if self.constraints.n_particles != self.geometry.n_particles:
            raise ValueError("engineering-grip constraint/model particle counts differ")
        expected = build_volume_grip_constraints(self.geometry, 0)
        for name in (
            "constrained_dofs",
            "free_dofs",
            "outer_bottom_mask",
            "outer_top_mask",
            "load_direction",
        ):
            if not np.array_equal(getattr(self.constraints, name), getattr(expected, name)):
                raise ValueError(f"engineering-grip constraint field {name} is not canonical")
        if not isinstance(self.top_line, LineQuadrature) or not isinstance(
            self.bottom_line, LineQuadrature
        ):
            raise TypeError("engineering-grip interface line data are invalid")
        if not isinstance(self.material_operator, ExplicitVolumeSparseD):
            raise TypeError("material_operator must be ExplicitVolumeSparseD")
        n_material = self.geometry.n_material
        if self.material_operator.shape != (n_material, n_material):
            raise ValueError("material operator must remain material-only")
        if self.material_operator.kernel != _e_KERNEL:
            raise ValueError("material operator must use the frozen cubic kernel")
        if not np.array_equal(
            np.asarray(self.material_operator.coordinates),
            self.geometry.coordinates[:n_material],
        ) or not np.array_equal(
            np.asarray(self.material_operator.volumes),
            self.geometry.material_reference_volumes,
        ) or not np.array_equal(
            np.asarray(self.material_operator.directed_edges, dtype=np.int64),
            self.geometry.material_reference_edges,
        ):
            raise ValueError("material operator is not the exact original material operator")
        Gc = _e_scalar(self.Gc, "engineering-grip Gc")
        l0 = _e_scalar(self.l0, "engineering-grip l0")
        if Gc <= 0.0 or l0 <= 0.0:
            raise ValueError("engineering-grip Gc/l0 must be positive")
        object.__setattr__(self, "Gc", Gc)
        object.__setattr__(self, "l0", l0)
        mask = _e_bool(self.crack_region_mask, "engineering-grip crack_region_mask", (n_material,))
        object.__setattr__(self, "crack_region_mask", _e_immutable(mask))
        if not isinstance(self.material_hash, str) or len(self.material_hash) != 64:
            raise ValueError("engineering-grip material_hash must be SHA256")
        if not isinstance(self.union_hash, str) or len(self.union_hash) != 64:
            raise ValueError("engineering-grip union_hash must be SHA256")
        expected_material, expected_union = _e_source_digest(self)
        if self.material_hash != expected_material or self.union_hash != expected_union:
            raise ValueError("engineering-grip cloud identity hash mismatch")

    @property
    def n_material(self) -> int:
        return int(self.geometry.n_material)

    @property
    def n_union(self) -> int:
        return int(self.geometry.n_particles)

    @property
    def n_dofs(self) -> int:
        return 2 * self.n_union

    @property
    def static_digest(self) -> str:
        return self.static.deterministic_digest

    @property
    def model_digest(self) -> str:
        """Durable model identity without Python runtime/cache identities."""

        return _e_engineering_model_digest(self)

    @property
    def material_phase_volumes(self) -> np.ndarray:
        return self.static.union_material_phase_volumes

    @property
    def grip_phase_volumes(self) -> np.ndarray:
        return self.static.grip_phase_volumes

    def fixed_damage_view(self, material_damage: Any) -> A2FixedDamageView:
        return A2FixedDamageView(self.static, material_damage)

def build_engineering_grip_model(
    geometry: UnionGripGeometry,
    youngs_modulus: Any,
    poisson_ratio: Any,
    k_res: Any,
    rank_rtol: Any,
    Gc: Any,
    l0: Any,
) -> EngineeringGripModel:
    """Build the fixed union mechanics/material-only damage model."""

    if not isinstance(geometry, UnionGripGeometry):
        raise TypeError("geometry must be a UnionGripGeometry")
    if not np.array_equal(
        geometry.material_reference_coordinates[0], np.asarray([0.0, 0.0], dtype=np.float64)
    ):
        raise ValueError("the original material particle zero must be the anchor (0, 0)")
    constraints = build_volume_grip_constraints(geometry, 0)
    top_line = physical_interface_line_weights(geometry, "top")
    bottom_line = physical_interface_line_weights(geometry, "bottom")
    static = A2StaticMechanics.from_geometry(
        geometry,
        constraints,
        top_line,
        bottom_line,
        youngs_modulus,
        poisson_ratio,
        k_res,
        rank_rtol,
    )
    material_operator = ExplicitVolumeSparseD(
        geometry.coordinates[: geometry.n_material],
        geometry.material_reference_volumes,
        geometry.material_reference_edges,
        geometry.delta,
        _e_KERNEL,
    )
    mask = build_crack_region_mask(
        geometry.material_reference_coordinates, geometry.delta
    )
    # Compute the two hashes from the fully declared constructor inputs,
    # keeping the final dataclass validation authoritative.
    material_hash = _e_hash_named(
        (
            ("coordinates", geometry.material_reference_coordinates),
            ("volumes", geometry.material_reference_volumes),
            ("triangles", geometry.material_reference_triangles),
            ("edges", geometry.material_reference_edges),
            ("bottom", geometry.material_bottom_mask[: geometry.n_material]),
            ("top", geometry.material_top_mask[: geometry.n_material]),
            ("anchor", 0),
        )
    )
    union_hash = _e_hash_named(
        (
            ("coordinates", geometry.coordinates),
            ("material_phase_volumes", geometry.material_volumes),
            ("grip_phase_volumes", geometry.grip_volumes),
            ("union_volumes", geometry.union_volumes),
            ("union_triangles", geometry.union_triangles),
            ("candidate_edges", geometry.candidate_edges),
            ("visible_edges", geometry.directed_edges),
            ("removed_edges", geometry.removed_edges),
            ("volume_bottom_mask", constraints.outer_bottom_mask),
            ("volume_top_mask", constraints.outer_top_mask),
            (
                "visibility_tolerance",
                np.asarray(geometry.visibility_tolerance, dtype=np.float64),
            ),
        )
    )
    return EngineeringGripModel(
        geometry=geometry,
        static=static,
        constraints=constraints,
        top_line=top_line,
        bottom_line=bottom_line,
        material_operator=material_operator,
        Gc=Gc,
        l0=l0,
        crack_region_mask=mask,
        material_hash=material_hash,
        union_hash=union_hash,
    )

@dataclass(frozen=True)
class EngineeringGripConfig:
    """Explicit strict continuation and one-time work-check settings."""

    load_start: float
    load_final: float
    du_initial: float
    du_min: float
    cutback_factor: float
    max_cutbacks: int
    max_stagger_iterations: int
    cg_rtol: float
    cg_atol: float
    cg_maxiter: int
    displacement: GateTolerance
    damage: GateTolerance
    history: GateTolerance
    reaction: GateTolerance
    damage_residual: GateTolerance
    force: GateTolerance
    moment: GateTolerance
    healing_atol: float
    damage_range_atol: float
    work_check_load: float
    work_epsilon: float
    work_relative_limit: float

    def __post_init__(self) -> None:
        start = _e_scalar(self.load_start, "load_start")
        final = _e_scalar(self.load_final, "load_final")
        du = _e_scalar(self.du_initial, "du_initial")
        du_min = _e_scalar(self.du_min, "du_min")
        factor = _e_scalar(self.cutback_factor, "cutback_factor")
        if start != 0.0 or final <= start:
            raise ValueError("engineering-grip load interval must start at exact zero and increase")
        if du <= 0.0 or du_min <= 0.0 or du_min > du or du > final:
            raise ValueError("engineering-grip increments are invalid")
        if not 0.0 < factor < 1.0:
            raise ValueError("cutback_factor must lie strictly between zero and one")
        max_cutbacks = _e_integer(self.max_cutbacks, "max_cutbacks", positive=True)
        max_stagger = _e_integer(
            self.max_stagger_iterations, "max_stagger_iterations", positive=True
        )
        rtol = _e_scalar(self.cg_rtol, "cg_rtol")
        atol = _e_scalar(self.cg_atol, "cg_atol")
        maxiter = _e_integer(self.cg_maxiter, "cg_maxiter", positive=True)
        if rtol < 0.0 or atol < 0.0 or (rtol == 0.0 and atol == 0.0):
            raise ValueError("CG tolerances must be non-negative and not both zero")
        for name in (
            "displacement",
            "damage",
            "history",
            "reaction",
            "damage_residual",
            "force",
            "moment",
        ):
            if not isinstance(getattr(self, name), GateTolerance):
                raise TypeError(f"{name} must be GateTolerance")
        healing = _e_scalar(self.healing_atol, "healing_atol")
        range_atol = _e_scalar(self.damage_range_atol, "damage_range_atol")
        check_load = _e_scalar(self.work_check_load, "work_check_load")
        epsilon = _e_scalar(self.work_epsilon, "work_epsilon")
        relative_limit = _e_scalar(self.work_relative_limit, "work_relative_limit")
        if healing < 0.0 or range_atol < 0.0:
            raise ValueError("healing/range tolerances must be non-negative")
        if check_load <= 0.0 or check_load > final:
            raise ValueError("work_check_load must lie in (0, load_final]")
        if epsilon <= 0.0 or epsilon >= check_load:
            raise ValueError("work_epsilon must lie strictly between zero and work_check_load")
        if relative_limit < 0.0:
            raise ValueError("work_relative_limit must be non-negative")
        for name, value in (
            ("load_start", start),
            ("load_final", final),
            ("du_initial", du),
            ("du_min", du_min),
            ("cutback_factor", factor),
            ("cg_rtol", rtol),
            ("cg_atol", atol),
            ("healing_atol", healing),
            ("damage_range_atol", range_atol),
            ("work_check_load", check_load),
            ("work_epsilon", epsilon),
            ("work_relative_limit", relative_limit),
        ):
            object.__setattr__(self, name, value)
        for name, value in (
            ("max_cutbacks", max_cutbacks),
            ("max_stagger_iterations", max_stagger),
            ("cg_maxiter", maxiter),
        ):
            object.__setattr__(self, name, value)

    @classmethod
    def smoke(cls) -> "EngineeringGripConfig":
        """Return the frozen short checkpoint configuration.

        This constructor is only a settings helper; it does not run the
        rounded cloud or select a production horizon.
        """

        gate = GateTolerance(1.0e-9, 1.0e-8, 1.0)
        residual = GateTolerance(1.0e-10, 1.0e-8, 1.0)
        return cls(
            load_start=0.0,
            load_final=1.0e-4,
            du_initial=1.0e-4,
            du_min=5.0e-5,
            cutback_factor=0.5,
            max_cutbacks=8,
            max_stagger_iterations=20,
            cg_rtol=1.0e-10,
            cg_atol=1.0e-13,
            cg_maxiter=3000,
            displacement=gate,
            damage=gate,
            history=gate,
            reaction=gate,
            damage_residual=residual,
            force=gate,
            moment=gate,
            healing_atol=1.0e-10,
            damage_range_atol=1.0e-10,
            work_check_load=1.0e-4,
            work_epsilon=1.0e-5,
            work_relative_limit=1.0e-7,
        )

    @property
    def u_final(self) -> float:
        return self.load_final

    @property
    def du(self) -> float:
        return self.du_initial

    @property
    def acceptance_mode(self) -> str:
        return _e_ACCEPTANCE_MODE

    @property
    def config_digest(self) -> str:
        """Durable configuration identity used by owner-backed replay."""

        return _e_engineering_config_digest(self)

@dataclass(frozen=True)
class EngineeringGripState:
    """Immutable committed direct-load state."""

    u_bar: float
    union_displacement: np.ndarray
    material_damage: np.ndarray
    material_history: np.ndarray

    def __post_init__(self) -> None:
        load = _e_scalar(self.u_bar, "state.u_bar")
        if load < 0.0:
            raise ValueError("state.u_bar must be non-negative")
        q = _e_f64(self.union_displacement, "state.union_displacement")
        d = _e_f64(self.material_damage, "state.material_damage")
        history = _e_f64(self.material_history, "state.material_history")
        if q.ndim != 1 or q.size == 0 or q.size % 2:
            raise ValueError("state.union_displacement must be a nonempty flat vector")
        if d.ndim != 1 or d.size == 0 or history.shape != d.shape:
            raise ValueError("state damage/history shapes are incompatible")
        if np.any(history < 0.0):
            raise ValueError("state history must be non-negative")
        object.__setattr__(self, "u_bar", load)
        object.__setattr__(self, "union_displacement", _e_immutable(q))
        object.__setattr__(self, "material_damage", _e_immutable(d))
        object.__setattr__(self, "material_history", _e_immutable(history))

    @property
    def q(self) -> np.ndarray:
        return self.union_displacement

    @property
    def d(self) -> np.ndarray:
        return self.material_damage

    @property
    def H(self) -> np.ndarray:
        return self.material_history

    @property
    def digest(self) -> str:
        return _e_state_digest(
            self.u_bar,
            self.union_displacement,
            self.material_damage,
            self.material_history,
        )

def _e_check_exact_constraints(
    displacement: np.ndarray,
    constraints: AffineConstraints,
    u_bar: float,
    name: str,
) -> None:
    prescribed = constraints.prescribed_vector(u_bar)
    if not np.array_equal(
        displacement[constraints.constrained_dofs],
        prescribed[constraints.constrained_dofs],
    ):
        raise ValueError(f"{name} does not satisfy exact volume-grip constraints")



def _e_validated_mechanics_adapter(
    model: EngineeringGripModel,
    mechanics_adapter: Any,
) -> JaxA2CPUReplayAdapter:
    """Validate the restricted mechanics adapter against the model identity.

    The adopted scheme is GPU-only: ``mechanics_adapter`` must be an explicit
    ``JaxA2CPUReplayAdapter`` bound to this exact model and constraint
    identity.  Generic callables and the removed CPU reference solver are not
    accepted because acceptance must be backed by the validated device path.
    """

    if mechanics_adapter is None:
        raise TypeError(
            "a JaxA2CPUReplayAdapter is required; the CPU reference solver is "
            "not part of the adopted GPU scheme"
        )
    if not isinstance(mechanics_adapter, JaxA2CPUReplayAdapter):
        raise TypeError(
            "mechanics_adapter must be a JaxA2CPUReplayAdapter instance; "
            "generic callables are not accepted"
        )
    if model.static is not mechanics_adapter.cpu_static:
        raise ValueError(
            "mechanics_adapter static identity does not match the engineering-grip model"
        )
    if model.constraints is not mechanics_adapter.constraints:
        raise ValueError(
            "mechanics_adapter constraints are not the exact model constraint object"
        )
    return mechanics_adapter

def _e_solve_direct_mechanics(
    mechanics_adapter: JaxA2CPUReplayAdapter,
    view: A2FixedDamageView,
    constraints: AffineConstraints,
    u_bar: Any,
    cg_rtol: Any,
    cg_atol: Any,
    cg_maxiter: Any,
    *,
    initial_guess: np.ndarray | None,
    seed_kind: str,
    source_state: str,
    source_step: int,
    source_sweep: int,
    source_damage_digest: str | None,
) -> A2MechanicsSolveResult:
    """Run one direct fixed-damage solve through the required GPU adapter."""

    return mechanics_adapter.solve_fixed_outer_displacement(
        view,
        constraints,
        u_bar,
        cg_rtol,
        cg_atol,
        cg_maxiter,
        initial_guess=initial_guess,
        seed_kind=seed_kind,
        source_state=source_state,
        source_step=source_step,
        source_sweep=source_sweep,
        source_damage_digest=source_damage_digest,
    )

def compute_material_source_amor(
    view: A2FixedDamageView,
    displacement: Any,
    previous_history: Any,
) -> Any:
    """Return the material-only Amor driver from all union targets.

    Every union edge with a material source is retained, including edges whose
    target is a collar particle.  Grip-source centers are not assigned a
    history field.
    """

    if not isinstance(view, A2FixedDamageView):
        raise TypeError("view must be an A2FixedDamageView")
    previous = _e_f64(
        previous_history, "material history", (view.static.material_count,)
    )
    if np.any(previous < 0.0):
        raise ValueError("material history must be non-negative")
    source = view.static.source
    retained = source < view.static.material_count
    if not np.any(retained):
        raise ValueError("union mechanics graph has no material source centers")
    eta_sum = np.bincount(
        source[retained],
        weights=view.static.eta[retained],
        minlength=view.static.material_count,
    )
    if np.max(np.abs(eta_sum - 1.0)) > _e_REPLAY_BOUND:
        raise ValueError("material-source eta does not sum to one")
    result = compute_a2_amor_history(view, displacement, previous)
    if result.particle_driver.shape != (view.static.material_count,):
        raise ValueError("material-source Amor driver has a nonmaterial shape")
    if result.trial_history.shape != result.particle_driver.shape:
        raise ValueError("material-source trial history has an incompatible shape")
    return result

@dataclass(frozen=True)
class EngineeringGripWorkCheck:
    """One exact-zero-damage centered mechanics energy/reaction check."""

    u_bar: float
    epsilon: float
    reaction: float
    reaction_fd: float
    relative_error: float
    relative_limit: float
    passed: bool
    center_result: A2MechanicsSolveResult
    plus_result: A2MechanicsSolveResult
    minus_result: A2MechanicsSolveResult
    cg_rtol: float
    cg_atol: float
    cg_maxiter: int
    center_result_digest: str
    plus_result_digest: str
    minus_result_digest: str

    def __post_init__(self) -> None:
        load = _e_scalar(self.u_bar, "work_check.u_bar")
        epsilon = _e_scalar(self.epsilon, "work_check.epsilon")
        reaction = _e_scalar(self.reaction, "work_check.reaction")
        fd = _e_scalar(self.reaction_fd, "work_check.reaction_fd")
        error = _e_scalar(self.relative_error, "work_check.relative_error")
        limit = _e_scalar(self.relative_limit, "work_check.relative_limit")
        cg_rtol = _e_scalar(self.cg_rtol, "work_check.cg_rtol")
        cg_atol = _e_scalar(self.cg_atol, "work_check.cg_atol")
        cg_maxiter = _e_integer(self.cg_maxiter, "work_check.cg_maxiter", positive=True)
        if load <= 0.0 or epsilon <= 0.0 or epsilon >= load or limit < 0.0:
            raise ValueError("work-check scalar settings are invalid")
        if cg_rtol < 0.0 or cg_atol < 0.0 or (cg_rtol == 0.0 and cg_atol == 0.0):
            raise ValueError("work-check CG tolerances are invalid")
        if error < 0.0:
            raise ValueError("work_check.relative_error must be non-negative")
        if not isinstance(self.passed, (bool, np.bool_)):
            raise TypeError("work_check.passed must be boolean")
        if not isinstance(self.center_result, A2MechanicsSolveResult):
            raise TypeError("work_check.center_result has the wrong type")
        if not isinstance(self.plus_result, A2MechanicsSolveResult) or not isinstance(
            self.minus_result, A2MechanicsSolveResult
        ):
            raise TypeError("work_check side results have the wrong type")
        object.__setattr__(self, "u_bar", load)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "reaction", reaction)
        object.__setattr__(self, "reaction_fd", fd)
        object.__setattr__(self, "relative_error", error)
        object.__setattr__(self, "relative_limit", limit)
        object.__setattr__(self, "cg_rtol", cg_rtol)
        object.__setattr__(self, "cg_atol", cg_atol)
        object.__setattr__(self, "cg_maxiter", cg_maxiter)
        object.__setattr__(self, "passed", bool(self.passed))
        for name in (
            "center_result_digest",
            "plus_result_digest",
            "minus_result_digest",
        ):
            object.__setattr__(
                self,
                name,
                _e_valid_digest(self.__dict__[name], f"work_check.{name}"),
            )
        self.validate()

    @property
    def top_reaction(self) -> float:
        return self.reaction

    def validate(self) -> None:
        """Recompute the shared work-check arithmetic and nested solve audit.

        The intrinsic portion deliberately uses only fields retained by this
        contract.  Model/config binding is performed by
        ``_validate_work_check_against_model_config`` when a parent result has
        those authoritative identities available.
        """

        results = (self.center_result, self.plus_result, self.minus_result)
        if len({id(result) for result in results}) != 3:
            raise ValueError("work-check center/plus/minus solves must be distinct objects")
        if not (
            self.center_result.outer_displacement == self.u_bar
            and self.plus_result.outer_displacement == self.u_bar + self.epsilon
            and self.minus_result.outer_displacement == self.u_bar - self.epsilon
        ):
            raise ValueError("work-check solve loads are not exactly centered on u_bar")
        constraints = self.center_result.constraints
        if self.plus_result.constraints is not constraints or self.minus_result.constraints is not constraints:
            raise ValueError("work-check mechanics solves do not share constraints")
        for name in (
            "static_runtime_identity",
            "static_digest",
            "damage_digest",
            "unit_system_digest",
        ):
            # A replay adapter may construct a fresh fixed-damage view for each
            # load.  Its cache/runtime token is intentionally not a
            # cross-solve identity contract; the zero material damage digest
            # and unit-system digest below bind the material/configuration
            # value semantics instead.
            values = tuple(getattr(result, name) for result in results)
            if name == "static_runtime_identity":
                same = all(value is values[0] for value in values[1:])
            else:
                same = all(value == values[0] for value in values[1:])
            if not same:
                raise ValueError(f"work-check solves do not share {name}")
        for result, expected_load, digest_name in zip(
            results,
            (self.u_bar, self.u_bar + self.epsilon, self.u_bar - self.epsilon),
            ("center_result_digest", "plus_result_digest", "minus_result_digest"),
            strict=True,
        ):
            _e_validate_work_check_mechanics_result(
                result,
                expected_load=expected_load,
                expected_cg_rtol=self.cg_rtol,
                expected_cg_atol=self.cg_atol,
                expected_cg_maxiter=self.cg_maxiter,
                expected_digest=getattr(self, digest_name),
                name=f"work_check.{digest_name.removesuffix('_digest')}",
            )
        reaction = _e_authoritative_work_check_reaction(self.center_result)
        reaction_fd = (
            self.plus_result.energy - self.minus_result.energy
        ) / (2.0 * self.epsilon)
        relative_error = _e_relative(reaction_fd, reaction)
        if self.reaction != reaction:
            raise ValueError("work-check reaction is not recomputed from center result")
        if self.reaction_fd != reaction_fd:
            raise ValueError("work-check reaction_fd is not the centered energy formula")
        if self.relative_error != relative_error:
            raise ValueError("work-check relative_error is not the authoritative denominator formula")
        if self.passed != bool(relative_error <= self.relative_limit):
            raise ValueError("work-check passed flag is not exactly relative_error <= relative_limit")

    def validate_with_owner(
        self,
        model: "EngineeringGripModel",
        config: "EngineeringGripConfig",
    ) -> None:
        return self._validate_with_owner_runner(
            model,
            config,
            _e_AUTHORITATIVE_RUN_ZERO_DAMAGE_WORK_CHECK,
        )

    def _validate_with_owner_runner(
        self,
        model: "EngineeringGripModel",
        config: "EngineeringGripConfig",
        runner: Any,
    ) -> None:
        """Validate this check against a fresh authoritative zero-damage replay.

        ``validate`` is deliberately value-only and keeps the in-memory
        cross-solve identity checks needed by the live contract.  This method
        first performs that structural validation and the model/config binding,
        then reruns the three centered mechanics solves with the exact stored
        load, epsilon, limit, and CG settings.  Fresh fixed-damage views have
        new runtime/cache identities, so only durable values and digests are
        compared.
        """

        self.validate()
        _e_validate_work_check_against_model_config(
            self, model, config, require_model_identity=False
        )
        if self.center_result.static_digest != model.static_digest:
            raise ValueError("work-check model static digest is not owner-paired")
        if _e_engineering_model_digest(model) != model.model_digest:
            raise ValueError("work-check owner model digest is not self-consistent")
        if _e_engineering_config_digest(config) != config.config_digest:
            raise ValueError("work-check owner config digest is not self-consistent")

        try:
            replay = runner(
                model,
                self.u_bar,
                self.epsilon,
                self.relative_limit,
                self.cg_rtol,
                self.cg_atol,
                self.cg_maxiter,
                mechanics_adapter=None,
            )
        except EngineeringGripWorkCheckFailure as error:
            # A failed work identity is still a durable numerical result.  Use
            # its immutable check for comparison rather than confusing a
            # legitimate pass/fail mismatch with a structural exception.
            replay = error.check
        if not isinstance(replay, EngineeringGripWorkCheck):
            raise TypeError("authoritative work-check replay returned the wrong type")

        for field in (
            "u_bar",
            "epsilon",
            "reaction",
            "reaction_fd",
            "relative_error",
            "relative_limit",
            "cg_rtol",
            "cg_atol",
            "cg_maxiter",
            "passed",
        ):
            if getattr(self, field) != getattr(replay, field):
                raise ValueError(f"work-check durable field {field} differs from owner replay")
        for field in ("center_result", "plus_result", "minus_result"):
            _e_compare_work_check_mechanics_result_durable(
                getattr(self, field), getattr(replay, field), f"work-check.{field}"
            )
        for field in ("center_result_digest", "plus_result_digest", "minus_result_digest"):
            if getattr(self, field) != getattr(replay, field):
                raise ValueError(f"work-check nested digest {field} differs from owner replay")

def _e_valid_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value

def _e_mechanics_result_digest(result: A2MechanicsSolveResult) -> str:
    """Return a deterministic digest of every retained mechanics result field."""

    provenance = result.seed_provenance
    return _e_hash_named(
        (
            ("displacement", result.displacement),
            ("evaluation_energy", result.energy),
            ("evaluation_residual", result.evaluation.residual),
            ("evaluation_internal_force", result.evaluation.internal_force),
            ("evaluation_edge_deformation_gradient", result.evaluation.edge_deformation_gradient),
            ("evaluation_edge_strain", result.evaluation.edge_strain),
            ("evaluation_edge_stress", result.evaluation.edge_stress),
            ("evaluation_particle_strain", result.evaluation.particle_strain),
            ("evaluation_particle_stress", result.evaluation.particle_stress),
            ("evaluation_edge_effective_stress", result.evaluation.edge_effective_stress),
            ("evaluation_particle_effective_stress", result.evaluation.particle_effective_stress),
            ("evaluation_edge_energy_density", result.evaluation.edge_energy_density),
            ("free_residual", result.free_residual),
            ("free_residual_norm", result.free_residual_norm),
            ("free_rhs_norm", result.free_rhs_norm),
            ("free_residual_relative_norm", result.free_residual_relative_norm),
            ("cg_convergence_limit", result.cg_convergence_limit),
            ("cg_iterations", result.cg_iterations),
            ("cg_info", result.cg_info),
            ("outer_displacement", result.outer_displacement),
            ("outer_reaction", result.outer_reaction),
            ("bottom_reaction", result.bottom_reaction),
            ("anchor_reaction", result.anchor_reaction),
            ("total_force", result.total_force),
            ("total_moment", result.total_moment),
            ("static_digest", result.static_digest),
            ("damage_digest", result.damage_digest),
            ("unit_system_digest", result.unit_system_digest),
            ("seed_kind", provenance.kind),
            ("seed_source_state", provenance.source_state),
            ("seed_source_step", provenance.source_step),
            ("seed_source_sweep", provenance.source_sweep),
            ("seed_source_damage_digest", provenance.source_damage_digest),
            ("seed_target_damage_digest", provenance.target_damage_digest),
            ("seed_unit_system_digest", provenance.unit_system_digest),
            ("seed_digest", provenance.seed_digest),
            ("seed_initial_residual_norm", provenance.initial_residual_norm),
            ("seed_final_residual_norm", provenance.final_residual_norm),
            ("seed_early_exit_eligible", provenance.early_exit_eligible),
            ("seed_early_exit_used", provenance.early_exit_used),
        )
    )

def _e_compare_work_check_mechanics_result_durable(
    actual: A2MechanicsSolveResult,
    expected: A2MechanicsSolveResult,
    name: str,
) -> None:
    """Compare one work-check solve without comparing runtime object tokens."""

    if not isinstance(actual, A2MechanicsSolveResult) or not isinstance(
        expected, A2MechanicsSolveResult
    ):
        raise TypeError(f"{name} work-check solve has the wrong type")
    scalar_names = (
        "outer_displacement",
        "free_residual_norm",
        "free_rhs_norm",
        "free_residual_relative_norm",
        "cg_convergence_limit",
        "cg_iterations",
        "cg_info",
        "outer_reaction",
        "bottom_reaction",
        "anchor_reaction",
        "total_moment",
        "static_digest",
        "damage_digest",
        "unit_system_digest",
    )
    for field in scalar_names:
        if getattr(actual, field) != getattr(expected, field):
            raise ValueError(f"{name} durable scalar {field} differs from owner replay")
    array_names = ("displacement", "free_residual", "total_force")
    evaluation_array_names = (
        "residual",
        "internal_force",
        "edge_deformation_gradient",
        "edge_strain",
        "edge_stress",
        "particle_strain",
        "particle_stress",
        "edge_effective_stress",
        "particle_effective_stress",
        "edge_energy_density",
    )
    for field in array_names:
        left = np.asarray(getattr(actual, field))
        right = np.asarray(getattr(expected, field))
        if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(f"{name} durable array {field} differs from owner replay")
    if actual.evaluation.energy != expected.evaluation.energy:
        raise ValueError(f"{name} energy differs from owner replay")
    for field in evaluation_array_names:
        left = np.asarray(getattr(actual.evaluation, field))
        right = np.asarray(getattr(expected.evaluation, field))
        if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(f"{name} evaluation array {field} differs from owner replay")
    actual_seed = actual.seed_provenance
    expected_seed = expected.seed_provenance
    for field in (
        "kind",
        "source_state",
        "source_step",
        "source_sweep",
        "source_damage_digest",
        "target_damage_digest",
        "unit_system_digest",
        "seed_digest",
        "initial_residual_norm",
        "final_residual_norm",
        "early_exit_eligible",
        "early_exit_used",
    ):
        if getattr(actual_seed, field) != getattr(expected_seed, field):
            raise ValueError(f"{name} seed provenance {field} differs from owner replay")
    actual_digest = _e_mechanics_result_digest(actual)
    expected_digest = _e_mechanics_result_digest(expected)
    if actual_digest != expected_digest:
        raise ValueError(f"{name} mechanics result digest differs from owner replay")

def _e_authoritative_work_check_reaction(result: A2MechanicsSolveResult) -> float:
    """Use the exact top-selector reaction expression used by the solve."""

    return float(result.constraints.outer_top_selector @ result.evaluation.residual)

def _e_validate_work_check_mechanics_result(
    result: A2MechanicsSolveResult,
    *,
    expected_load: float,
    expected_cg_rtol: float,
    expected_cg_atol: float,
    expected_cg_maxiter: int,
    expected_digest: str,
    name: str,
    model: "EngineeringGripModel | None" = None,
    require_model_identity: bool = True,
) -> None:
    if result.outer_displacement != expected_load:
        raise ValueError(f"{name} load is not exactly paired")
    if result.cg_info != 0:
        raise ValueError(f"{name} mechanics solve did not converge")
    if result.free_residual_norm >= result.cg_convergence_limit:
        raise ValueError(f"{name} mechanics solve failed its convergence gate")
    if result.cg_iterations > expected_cg_maxiter:
        raise ValueError(f"{name} mechanics solve exceeded cg_maxiter")
    residual = np.asarray(result.evaluation.residual)
    if residual.shape != (result.constraints.n_dofs,):
        raise ValueError(f"{name} evaluation residual has the wrong shape")
    expected_free = residual[result.constraints.free_dofs]
    if not np.array_equal(result.free_residual, expected_free):
        raise ValueError(f"{name} free residual is not paired with its evaluation")
    expected_norm = float(np.linalg.norm(expected_free))
    if result.free_residual_norm != expected_norm:
        raise ValueError(f"{name} free residual norm is not recomputed")
    expected_relative = expected_norm / max(result.free_rhs_norm, 1.0e-300)
    if result.free_residual_relative_norm != expected_relative:
        raise ValueError(f"{name} free residual relative norm is not recomputed")
    expected_limit = max(expected_cg_atol, expected_cg_rtol * result.free_rhs_norm)
    if result.cg_convergence_limit != expected_limit:
        raise ValueError(f"{name} convergence limit is not paired with CG settings")
    prescribed = result.constraints.prescribed_vector(expected_load)
    if not np.array_equal(
        result.displacement[result.constraints.constrained_dofs],
        prescribed[result.constraints.constrained_dofs],
    ):
        raise ValueError(f"{name} displacement violates exact constraints")
    actual_digest = _e_mechanics_result_digest(result)
    if actual_digest != expected_digest:
        raise ValueError(f"{name} result digest is not paired")
    if model is not None:
        if require_model_identity and result.constraints is not model.constraints:
            raise ValueError(f"{name} constraints are foreign to the model")
        if not require_model_identity:
            for field in (
                "n_particles",
                "constrained_dofs",
                "free_dofs",
                "q_zero",
                "load_direction",
                "outer_bottom_mask",
                "outer_top_mask",
                "anchor_particle",
                "anchor_coordinate",
                "outer_top_selector",
                "outer_bottom_selector",
                "anchor_selector",
            ):
                actual = getattr(result.constraints, field)
                expected = getattr(model.constraints, field)
                if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
                    if not isinstance(actual, np.ndarray) or not isinstance(expected, np.ndarray):
                        raise ValueError(f"{name} constraints field {field} has the wrong kind")
                    if actual.dtype != expected.dtype or actual.shape != expected.shape or not np.array_equal(actual, expected):
                        raise ValueError(f"{name} constraints field {field} differs from the owner")
                elif actual != expected:
                    raise ValueError(f"{name} constraints field {field} differs from the owner")
        if require_model_identity and result.static_runtime_identity is not model.static.runtime_identity:
            raise ValueError(f"{name} static runtime identity is foreign to the model")
        if result.static_digest != model.static_digest:
            raise ValueError(f"{name} static digest is foreign to the model")
        expected_reaction = _e_authoritative_work_check_reaction(result)
        if result.outer_reaction != expected_reaction:
            raise ValueError(f"{name} outer reaction is not recomputed from its residual")
        expected_bottom = float(
            result.constraints.outer_bottom_selector @ result.evaluation.residual
        )
        expected_anchor = float(
            result.constraints.anchor_selector @ result.evaluation.residual
        )
        if result.bottom_reaction != expected_bottom or result.anchor_reaction != expected_anchor:
            raise ValueError(f"{name} constrained reactions are not recomputed")
        residual_matrix = residual.reshape(model.n_union, 2)
        expected_force = np.sum(residual_matrix, axis=0)
        expected_moment = float(
            np.sum(
                model.static.coordinates[:, 0] * residual_matrix[:, 1]
                - model.static.coordinates[:, 1] * residual_matrix[:, 0]
            )
        )
        if not np.array_equal(result.total_force, expected_force):
            raise ValueError(f"{name} total force is not recomputed")
        if result.total_moment != expected_moment:
            raise ValueError(f"{name} total moment is not recomputed")

def _e_validate_work_check_against_model_config(
    check: EngineeringGripWorkCheck,
    model: "EngineeringGripModel",
    config: "EngineeringGripConfig",
    *,
    require_model_identity: bool = True,
) -> None:
    if not isinstance(model, EngineeringGripModel):
        raise TypeError("work-check model has the wrong type")
    if not isinstance(config, EngineeringGripConfig):
        raise TypeError("work-check config has the wrong type")
    check.validate()
    for work_name, config_name in (
        ("u_bar", "work_check_load"),
        ("epsilon", "work_epsilon"),
        ("relative_limit", "work_relative_limit"),
        ("cg_rtol", "cg_rtol"),
        ("cg_atol", "cg_atol"),
        ("cg_maxiter", "cg_maxiter"),
    ):
        if getattr(check, work_name) != getattr(config, config_name):
            raise ValueError(
                f"work-check {work_name} is not paired with engineering config {config_name}"
            )
    zero_digest = _e_damage_digest(np.zeros(model.n_material, dtype=np.float64))
    expected_unit_digest = _a_unit_system_digest(model.static_digest, zero_digest)
    for result, expected_load, digest_name in zip(
        (check.center_result, check.plus_result, check.minus_result),
        (check.u_bar, check.u_bar + check.epsilon, check.u_bar - check.epsilon),
        ("center_result_digest", "plus_result_digest", "minus_result_digest"),
        strict=True,
    ):
        _e_validate_work_check_mechanics_result(
            result,
            expected_load=expected_load,
            expected_cg_rtol=config.cg_rtol,
            expected_cg_atol=config.cg_atol,
            expected_cg_maxiter=config.cg_maxiter,
            expected_digest=getattr(check, digest_name),
            name=f"work-check {digest_name.removesuffix('_digest')}",
            model=model,
            require_model_identity=require_model_identity,
        )
        if result.damage_digest != zero_digest:
            raise ValueError("work-check mechanics result does not use exact zero damage")
        if result.unit_system_digest != expected_unit_digest:
            raise ValueError("work-check mechanics unit-system digest is not model-paired")
        provenance = result.seed_provenance
        if provenance.kind != "zero_work_check" or provenance.source_state != "zero_work_check":
            raise ValueError("work-check mechanics seed provenance is not canonical")
        if provenance.source_step != 0 or provenance.source_sweep != 0:
            raise ValueError("work-check mechanics seed ordering is not canonical")
        if provenance.source_damage_digest != zero_digest:
            raise ValueError("work-check mechanics source damage digest is not zero-paired")
        expected_seed_digest = _a_digest_primitives(
            (("seed", result.constraints.prescribed_vector(expected_load)),)
        )
        if provenance.seed_digest != expected_seed_digest:
            raise ValueError("work-check mechanics seed digest is not load-paired")

class EngineeringGripWorkCheckFailure(RuntimeError):
    """Raised before continuation when the one-time work identity fails."""

    def __init__(self, check: EngineeringGripWorkCheck):
        super().__init__(
            "exact-zero-damage engineering-grip work check failed: "
            f"error={check.relative_error:.17g}, limit={check.relative_limit:.17g}"
        )
        self.check = check

def run_zero_damage_work_check(
    model: EngineeringGripModel,
    u_bar: Any,
    epsilon: Any,
    relative_limit: Any,
    cg_rtol: Any,
    cg_atol: Any,
    cg_maxiter: Any,
    mechanics_adapter: JaxA2CPUReplayAdapter,
) -> EngineeringGripWorkCheck:
    """Run the single centered zero-damage reduced-energy/top-reaction check.

    ``mechanics_adapter`` is the required restricted JAX adapter; the
    center/plus/minus solves all run on that one device backend and the
    acceptance values remain CPU-authoritative.
    """

    if not isinstance(model, EngineeringGripModel):
        raise TypeError("model must be an EngineeringGripModel")
    adapter = _e_validated_mechanics_adapter(model, mechanics_adapter)
    load = _e_scalar(u_bar, "work_check.u_bar")
    perturbation = _e_scalar(epsilon, "work_check.epsilon")
    limit = _e_scalar(relative_limit, "work_check.relative_limit")
    if load <= 0.0 or perturbation <= 0.0 or perturbation >= load:
        raise ValueError("work-check load/epsilon must satisfy 0 < epsilon < u_bar")
    if limit < 0.0:
        raise ValueError("work-check relative limit must be non-negative")
    damage = np.zeros(model.n_material, dtype=np.float64)
    view = model.fixed_damage_view(damage)
    digest = view.damage_digest
    center = _e_solve_direct_mechanics(
        adapter,
        view,
        model.constraints,
        load,
        cg_rtol,
        cg_atol,
        cg_maxiter,
        initial_guess=None,
        seed_kind="zero_work_check",
        source_state="zero_work_check",
        source_step=0,
        source_sweep=0,
        source_damage_digest=digest,
    )
    plus = _e_solve_direct_mechanics(
        adapter,
        view,
        model.constraints,
        load + perturbation,
        cg_rtol,
        cg_atol,
        cg_maxiter,
        initial_guess=None,
        seed_kind="zero_work_check",
        source_state="zero_work_check",
        source_step=0,
        source_sweep=0,
        source_damage_digest=digest,
    )
    minus = _e_solve_direct_mechanics(
        adapter,
        view,
        model.constraints,
        load - perturbation,
        cg_rtol,
        cg_atol,
        cg_maxiter,
        initial_guess=None,
        seed_kind="zero_work_check",
        source_state="zero_work_check",
        source_step=0,
        source_sweep=0,
        source_damage_digest=digest,
    )
    reaction_fd = (plus.energy - minus.energy) / (2.0 * perturbation)
    relative_error = _e_relative(reaction_fd, center.outer_reaction)
    check = EngineeringGripWorkCheck(
        u_bar=load,
        epsilon=perturbation,
        reaction=center.outer_reaction,
        reaction_fd=reaction_fd,
        relative_error=relative_error,
        relative_limit=limit,
        passed=bool(relative_error <= limit),
        center_result=center,
        plus_result=plus,
        minus_result=minus,
        cg_rtol=_e_scalar(cg_rtol, "work_check.cg_rtol"),
        cg_atol=_e_scalar(cg_atol, "work_check.cg_atol"),
        cg_maxiter=_e_integer(cg_maxiter, "work_check.cg_maxiter", positive=True),
        center_result_digest=_e_mechanics_result_digest(center),
        plus_result_digest=_e_mechanics_result_digest(plus),
        minus_result_digest=_e_mechanics_result_digest(minus),
    )
    if not check.passed:
        raise EngineeringGripWorkCheckFailure(check)
    return check

_e_AUTHORITATIVE_RUN_ZERO_DAMAGE_WORK_CHECK = run_zero_damage_work_check

def _e_optional_scalar(value: Any | None, name: str) -> float | None:
    if value is None:
        return None
    return _e_scalar(value, name)

@dataclass(frozen=True)
class EngineeringGripDiagnostics:
    """Primitive strict-sweep diagnostics, complete or explicitly incomplete."""

    sweep_index: int
    target_u_bar: float
    complete_sweep: bool
    passed: bool
    failure_stage: str
    failure_reason: str
    damage_before: np.ndarray
    d_iter: np.ndarray
    d_next: np.ndarray | None
    q_before: np.ndarray | None
    q_after: np.ndarray | None
    psi_before: np.ndarray | None
    psi_after: np.ndarray | None
    H_used: np.ndarray | None
    H_next: np.ndarray | None
    rho_used: np.ndarray | None
    rho_next: np.ndarray | None
    rho_used_volume: np.ndarray | None
    rho_next_volume: np.ndarray | None
    energy_before: float | None
    energy_after: float | None
    top_reaction_before: float | None
    top_reaction_after: float | None
    bottom_reaction_before: float | None
    bottom_reaction_after: float | None
    anchor_reaction_before: float | None
    anchor_reaction_after: float | None
    force_before: np.ndarray | None
    force_after: np.ndarray | None
    moment_before: float | None
    moment_after: float | None
    free_residual_norm_before: float | None
    free_residual_norm_after: float | None
    free_residual_relative_before: float | None
    free_residual_relative_after: float | None
    cg_info_before: int | None
    cg_info_after: int | None
    e_u_material: float | None
    e_u_union: float | None
    e_damage: float | None
    e_history: float | None
    e_reaction: float | None
    damage_min: float | None
    damage_max: float | None
    healing_min: float | None
    history_min_gap: float | None
    rho_used_inf: float | None
    rho_next_inf: float | None
    displacement_scale: float | None
    damage_scale: float | None
    history_scale: float | None
    reaction_scale: float | None
    damage_residual_scale: float | None
    force_error: float | None
    force_scale: float | None
    moment_error: float | None
    moment_scale: float | None
    finite_pass: bool
    mechanics_before_pass: bool
    mechanics_after_pass: bool
    displacement_pass: bool
    damage_pass: bool
    history_pass: bool
    reaction_pass: bool
    damage_residual_pass: bool
    damage_range_pass: bool
    healing_pass: bool
    history_monotonic_pass: bool
    force_pass: bool
    moment_pass: bool
    available_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sweep_index", _e_integer(self.sweep_index, "sweep_index"))
        object.__setattr__(self, "target_u_bar", _e_scalar(self.target_u_bar, "target_u_bar"))
        if self.target_u_bar <= 0.0:
            raise ValueError("diagnostic target_u_bar must be positive")
        object.__setattr__(self, "complete_sweep", bool(self.complete_sweep))
        object.__setattr__(self, "passed", bool(self.passed))
        if not isinstance(self.failure_stage, str) or not isinstance(
            self.failure_reason, str
        ):
            raise TypeError("diagnostic failure fields must be strings")
        damage_before = _e_f64(self.damage_before, "diagnostic.damage_before")
        d_iter = _e_f64(self.d_iter, "diagnostic.d_iter", damage_before.shape)
        object.__setattr__(self, "damage_before", _e_immutable(damage_before))
        object.__setattr__(self, "d_iter", _e_immutable(d_iter))
        array_shapes = {
            "d_next": damage_before.shape,
            "psi_before": damage_before.shape,
            "psi_after": damage_before.shape,
            "H_used": damage_before.shape,
            "H_next": damage_before.shape,
            "rho_used": damage_before.shape,
            "rho_next": damage_before.shape,
            "rho_used_volume": damage_before.shape,
            "rho_next_volume": damage_before.shape,
            "q_before": (2 * len(damage_before),),
            "q_after": (2 * len(damage_before),),
            "force_before": (2,),
            "force_after": (2,),
        }
        # The displacement arrays contain the union, not the material prefix;
        # their exact length is checked by the controller/result pairing.
        for name, shape in array_shapes.items():
            value = getattr(self, name)
            if value is None:
                continue
            if name.startswith("q_"):
                array = _e_f64(value, f"diagnostic.{name}")
                if array.ndim != 1 or array.size == 0 or array.size % 2:
                    raise ValueError(f"diagnostic.{name} must be flat union displacement")
                object.__setattr__(self, name, _e_immutable(array))
            else:
                object.__setattr__(self, name, _e_immutable(_e_f64(value, f"diagnostic.{name}", shape)))
        for name in (
            "energy_before",
            "energy_after",
            "top_reaction_before",
            "top_reaction_after",
            "bottom_reaction_before",
            "bottom_reaction_after",
            "anchor_reaction_before",
            "anchor_reaction_after",
            "moment_before",
            "moment_after",
            "free_residual_norm_before",
            "free_residual_norm_after",
            "free_residual_relative_before",
            "free_residual_relative_after",
            "e_u_material",
            "e_u_union",
            "e_damage",
            "e_history",
            "e_reaction",
            "damage_min",
            "damage_max",
            "healing_min",
            "history_min_gap",
            "rho_used_inf",
            "rho_next_inf",
            "displacement_scale",
            "damage_scale",
            "history_scale",
            "reaction_scale",
            "damage_residual_scale",
            "force_error",
            "force_scale",
            "moment_error",
            "moment_scale",
        ):
            object.__setattr__(
                self,
                name,
                _e_optional_scalar(getattr(self, name), f"diagnostic.{name}"),
            )
        for name in ("cg_info_before", "cg_info_after"):
            object.__setattr__(
                self,
                name,
                _e_optional_integer(getattr(self, name), f"diagnostic.{name}"),
            )
        for name in (
            "finite_pass",
            "mechanics_before_pass",
            "mechanics_after_pass",
            "displacement_pass",
            "damage_pass",
            "history_pass",
            "reaction_pass",
            "damage_residual_pass",
            "damage_range_pass",
            "healing_pass",
            "history_monotonic_pass",
            "force_pass",
            "moment_pass",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        if self.complete_sweep and not self.failure_reason and not self.passed:
            raise ValueError("complete failed sweep must retain a failure reason")
        if not self.complete_sweep and (
            not self.failure_stage or not self.failure_reason
        ):
            raise ValueError("incomplete sweep must retain failure stage and reason")
        if self.passed and not self.complete_sweep:
            raise ValueError("incomplete sweep cannot pass")
        object.__setattr__(self, "available_fields", tuple(sorted(set(self.available_fields))))

    @property
    def gate_passes(self) -> dict[str, bool]:
        return {
            "finite": self.finite_pass,
            "mechanics_before": self.mechanics_before_pass,
            "mechanics_after": self.mechanics_after_pass,
            "displacement": self.displacement_pass,
            "damage": self.damage_pass,
            "history": self.history_pass,
            "reaction": self.reaction_pass,
            "damage_residual": self.damage_residual_pass,
            "damage_range": self.damage_range_pass,
            "healing": self.healing_pass,
            "history_monotonic": self.history_monotonic_pass,
            "force": self.force_pass,
            "moment": self.moment_pass,
        }

    @property
    def u_bar(self) -> float:
        return self.target_u_bar

    @property
    def damage_residual_inf(self) -> float | None:
        return self.rho_next_inf

def _e_mechanics_fields(
    result: A2MechanicsSolveResult | None,
) -> dict[str, Any]:
    if result is None:
        return {
            "energy": None,
            "top": None,
            "bottom": None,
            "anchor": None,
            "force": None,
            "moment": None,
            "free_norm": None,
            "free_relative": None,
        }
    return {
        "energy": result.energy,
        "top": result.outer_reaction,
        "bottom": result.bottom_reaction,
        "anchor": result.anchor_reaction,
        "force": result.total_force,
        "moment": result.total_moment,
        "free_norm": result.free_residual_norm,
        "free_relative": result.free_residual_relative_norm,
    }

def _e_geometry_length(model: EngineeringGripModel) -> float:
    return max(_e_max_abs(model.geometry.coordinates), _e_max_abs(np.ptp(model.geometry.coordinates, axis=0)), 1.0)

def _e_mechanics_gate(
    result: A2MechanicsSolveResult | None,
    config: EngineeringGripConfig,
    geometry_length: float,
) -> tuple[bool, bool, float, float, bool, bool]:
    if result is None:
        return False, False, math.inf, math.inf, False, False
    finite = bool(
        np.all(np.isfinite(result.displacement))
        and np.all(np.isfinite(result.residual))
        and np.all(np.isfinite(result.total_force))
        and np.isfinite(result.total_moment)
    )
    mechanics = bool(
        finite
        and result.cg_info == 0
        and result.free_residual_norm < result.cg_convergence_limit
    )
    force_error = _e_max_abs(result.total_force)
    force_scale = max(
        abs(result.outer_reaction),
        abs(result.bottom_reaction),
        abs(result.anchor_reaction),
        1.0,
    )
    force_pass = bool(force_error <= config.force.limit(force_scale))
    moment_error = abs(result.total_moment)
    moment_scale = force_scale * geometry_length
    moment_pass = bool(moment_error <= config.moment.limit(moment_scale))
    return (
        mechanics,
        finite,
        force_error,
        moment_error,
        force_pass,
        moment_pass,
    )

def _e_damage_residuals(
    model: EngineeringGripModel,
    damage: np.ndarray,
    H_used: np.ndarray,
    H_next: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    d = _e_f64(damage, "damage residual damage", (model.n_material,))
    used = _e_f64(H_used, "damage residual H_used", d.shape)
    next_history = _e_f64(H_next, "damage residual H_next", d.shape)
    D = model.material_operator.to_scipy().tocsr()
    graph = np.asarray(D @ d, dtype=np.float64)
    local = model.Gc / model.l0 * d
    graph_term = model.Gc * model.l0 * graph
    rho_used = local + graph_term + 2.0 * used * d - 2.0 * used
    rho_next = local + graph_term + 2.0 * next_history * d - 2.0 * next_history
    volume = np.asarray(model.material_operator.volumes, dtype=np.float64)
    terms = (
        local,
        graph_term,
        2.0 * used * d,
        2.0 * used,
        2.0 * next_history * d,
        2.0 * next_history,
    )
    if not all(np.all(np.isfinite(term)) for term in terms):
        raise FloatingPointError("engineering-grip damage residual terms are non-finite")
    scale = max((_e_max_abs(term) for term in terms), default=0.0)
    return (
        np.asarray(rho_used, dtype=np.float64),
        np.asarray(rho_next, dtype=np.float64),
        np.asarray(volume * rho_used, dtype=np.float64),
        np.asarray(volume * rho_next, dtype=np.float64),
        scale,
    )

def _e_incomplete_diagnostics(
    config: EngineeringGripConfig,
    committed: EngineeringGripState,
    d_iter: np.ndarray,
    target: float,
    stage: str,
    reason: str,
    *,
    before: A2MechanicsSolveResult | None = None,
    after: A2MechanicsSolveResult | None = None,
    psi_before: np.ndarray | None = None,
    psi_after: np.ndarray | None = None,
    H_used: np.ndarray | None = None,
    d_next: np.ndarray | None = None,
    H_next: np.ndarray | None = None,
    rho_used: np.ndarray | None = None,
    rho_next: np.ndarray | None = None,
    rho_used_volume: np.ndarray | None = None,
    rho_next_volume: np.ndarray | None = None,
    model: EngineeringGripModel,
    sweep_index: int,
) -> EngineeringGripDiagnostics:
    before_fields = _e_mechanics_fields(before)
    after_fields = _e_mechanics_fields(after)
    available: list[str] = ["damage_before", "d_iter"]
    for name, value in (
        ("q_before", None if before is None else before.displacement),
        ("q_after", None if after is None else after.displacement),
        ("psi_before", psi_before),
        ("psi_after", psi_after),
        ("H_used", H_used),
        ("d_next", d_next),
        ("H_next", H_next),
        ("rho_used", rho_used),
        ("rho_next", rho_next),
        ("rho_used_volume", rho_used_volume),
        ("rho_next_volume", rho_next_volume),
    ):
        if value is not None:
            available.append(name)
    (
        before_mechanics,
        before_finite,
        before_force,
        before_moment,
        before_force_pass,
        before_moment_pass,
    ) = _e_mechanics_gate(before, config, _e_geometry_length(model))
    (
        after_mechanics,
        after_finite,
        after_force,
        after_moment,
        after_force_pass,
        after_moment_pass,
    ) = _e_mechanics_gate(after, config, _e_geometry_length(model))
    force_error = max(
        before_force if before is not None else 0.0,
        after_force if after is not None else 0.0,
    )
    moment_error = max(
        before_moment if before is not None else 0.0,
        after_moment if after is not None else 0.0,
    )
    force_scale = max(
        value
        for value in (
            1.0,
            abs(before_fields["top"] or 0.0),
            abs(after_fields["top"] or 0.0),
            abs(before_fields["bottom"] or 0.0),
            abs(after_fields["bottom"] or 0.0),
            abs(before_fields["anchor"] or 0.0),
            abs(after_fields["anchor"] or 0.0),
        )
    )
    moment_scale = force_scale * _e_geometry_length(model)
    current = d_next if d_next is not None else d_iter
    return EngineeringGripDiagnostics(
        sweep_index=sweep_index,
        target_u_bar=target,
        complete_sweep=False,
        passed=False,
        failure_stage=stage,
        failure_reason=reason,
        damage_before=committed.material_damage,
        d_iter=d_iter,
        d_next=d_next,
        q_before=None if before is None else before.displacement,
        q_after=None if after is None else after.displacement,
        psi_before=psi_before,
        psi_after=psi_after,
        H_used=H_used,
        H_next=H_next,
        rho_used=rho_used,
        rho_next=rho_next,
        rho_used_volume=rho_used_volume,
        rho_next_volume=rho_next_volume,
        energy_before=before_fields["energy"],
        energy_after=after_fields["energy"],
        top_reaction_before=before_fields["top"],
        top_reaction_after=after_fields["top"],
        bottom_reaction_before=before_fields["bottom"],
        bottom_reaction_after=after_fields["bottom"],
        anchor_reaction_before=before_fields["anchor"],
        anchor_reaction_after=after_fields["anchor"],
        force_before=before_fields["force"],
        force_after=after_fields["force"],
        moment_before=before_fields["moment"],
        moment_after=after_fields["moment"],
        free_residual_norm_before=before_fields["free_norm"],
        free_residual_norm_after=after_fields["free_norm"],
        free_residual_relative_before=before_fields["free_relative"],
        free_residual_relative_after=after_fields["free_relative"],
        cg_info_before=None if before is None else before.cg_info,
        cg_info_after=None if after is None else after.cg_info,
        e_u_material=None,
        e_u_union=None,
        e_damage=None if d_next is None else _e_max_abs(d_next - d_iter),
        e_history=None,
        e_reaction=None,
        damage_min=float(np.min(current)),
        damage_max=float(np.max(current)),
        healing_min=None if d_next is None else float(np.min(d_next - committed.material_damage)),
        history_min_gap=(
            None
            if H_next is None
            else float(np.min(H_next - committed.material_history))
        ),
        rho_used_inf=None if rho_used is None else _e_max_abs(rho_used),
        rho_next_inf=None if rho_next is None else _e_max_abs(rho_next),
        displacement_scale=None,
        damage_scale=None if d_next is None else max(_e_max_abs(d_iter), _e_max_abs(d_next)),
        history_scale=None,
        reaction_scale=None,
        damage_residual_scale=None,
        force_error=force_error if before is not None or after is not None else None,
        force_scale=force_scale if before is not None or after is not None else None,
        moment_error=moment_error if before is not None or after is not None else None,
        moment_scale=moment_scale if before is not None or after is not None else None,
        finite_pass=bool(before_finite and after_finite),
        mechanics_before_pass=before_mechanics,
        mechanics_after_pass=after_mechanics,
        displacement_pass=False,
        damage_pass=False,
        history_pass=False,
        reaction_pass=False,
        damage_residual_pass=False,
        damage_range_pass=(
            d_next is not None
            and float(np.min(d_next)) >= -config.damage_range_atol
            and float(np.max(d_next)) <= 1.0 + config.damage_range_atol
        ),
        healing_pass=(
            d_next is not None
            and float(np.min(d_next - committed.material_damage)) >= -config.healing_atol
        ),
        history_monotonic_pass=False,
        # Preserve the independent status of every mechanics conservation
        # gate that was actually evaluated.  An incomplete before-stage
        # failure has no ``after`` solve; treating that absent result as a
        # failed force *and* moment gate forges the unavailable moment
        # provenance and changes the warning bypass audit.  With neither
        # solve available, both aggregate gates remain false.
        force_pass=bool(
            (before is not None or after is not None)
            and (before is None or before_force_pass)
            and (after is None or after_force_pass)
        ),
        moment_pass=bool(
            (before is not None or after is not None)
            and (before is None or before_moment_pass)
            and (after is None or after_moment_pass)
        ),
        available_fields=tuple(available),
    )

def _e_full_diagnostics(
    config: EngineeringGripConfig,
    model: EngineeringGripModel,
    committed: EngineeringGripState,
    d_iter: np.ndarray,
    d_next: np.ndarray,
    H_used: np.ndarray,
    H_next: np.ndarray,
    before: A2MechanicsSolveResult,
    after: A2MechanicsSolveResult,
    psi_before: np.ndarray,
    psi_after: np.ndarray,
    rho_used: np.ndarray,
    rho_next: np.ndarray,
    rho_used_volume: np.ndarray,
    rho_next_volume: np.ndarray,
    residual_scale: float,
    sweep_index: int,
    target: float,
    warning_safety_limit: float | None = None,
) -> EngineeringGripDiagnostics:
    geometry_length = _e_geometry_length(model)
    before_fields = _e_mechanics_fields(before)
    after_fields = _e_mechanics_fields(after)
    (
        mechanics_before,
        finite_before,
        force_before,
        moment_before,
        force_before_pass,
        moment_before_pass,
    ) = _e_mechanics_gate(before, config, geometry_length)
    (
        mechanics_after,
        finite_after,
        force_after,
        moment_after,
        force_after_pass,
        moment_after_pass,
    ) = _e_mechanics_gate(after, config, geometry_length)
    q_before = before.displacement
    q_after = after.displacement
    e_u_material = _e_max_abs(
        q_after[: 2 * model.n_material] - q_before[: 2 * model.n_material]
    )
    displacement_scale = max(
        _e_max_abs(q_before[: 2 * model.n_material]),
        _e_max_abs(q_after[: 2 * model.n_material]),
        abs(target),
    )
    e_u_union = _e_max_abs(q_after - q_before)
    union_scale = max(
        _e_max_abs(q_before),
        _e_max_abs(q_after),
        abs(target),
    )
    e_damage = _e_max_abs(d_next - d_iter)
    damage_scale = max(_e_max_abs(d_iter), _e_max_abs(d_next))
    e_history = _e_max_abs(H_next - H_used)
    history_scale = max(
        _e_max_abs(committed.material_history), _e_max_abs(H_used), _e_max_abs(H_next)
    )
    e_reaction = abs(after.outer_reaction - before.outer_reaction)
    reaction_scale = max(abs(before.outer_reaction), abs(after.outer_reaction))
    damage_min = float(np.min(d_next))
    damage_max = float(np.max(d_next))
    healing_min = float(np.min(d_next - committed.material_damage))
    # The warning limit is a final acceptance-safety value, not a gate
    # tolerance.  Keep validating it at this boundary for callers that pass
    # the warning-mode setting, but never use it to change the truthful strict
    # range/healing flags below.
    if warning_safety_limit is not None:
        _e_validate_warning_safety_limit(warning_safety_limit)
    history_min_gap = float(np.min(H_next - committed.material_history))
    force_error = max(force_before, force_after)
    force_scale = max(
        1.0,
        abs(before.outer_reaction),
        abs(after.outer_reaction),
        abs(before.bottom_reaction),
        abs(after.bottom_reaction),
        abs(before.anchor_reaction),
        abs(after.anchor_reaction),
    )
    moment_error = max(moment_before, moment_after)
    moment_scale = force_scale * geometry_length
    finite_pass = bool(
        finite_before
        and finite_after
        and all(
            np.all(np.isfinite(array))
            for array in (
                d_iter,
                d_next,
                psi_before,
                psi_after,
                H_used,
                H_next,
                rho_used,
                rho_next,
                rho_used_volume,
                rho_next_volume,
            )
        )
    )
    passes = {
        "finite": finite_pass,
        "mechanics_before": mechanics_before,
        "mechanics_after": mechanics_after,
        "displacement": bool(
            e_u_material <= config.displacement.limit(displacement_scale)
            and e_u_union <= config.displacement.limit(union_scale)
        ),
        "damage": e_damage <= config.damage.limit(damage_scale),
        "history": e_history <= config.history.limit(history_scale),
        "reaction": e_reaction <= config.reaction.limit(reaction_scale),
        "damage_residual": _e_max_abs(rho_next)
        <= config.damage_residual.limit(residual_scale),
        "damage_range": bool(
            damage_min >= -config.damage_range_atol
            and damage_max <= 1.0 + config.damage_range_atol
        ),
        "healing": bool(healing_min >= -config.healing_atol),
        "history_monotonic": history_min_gap >= 0.0,
        "force": bool(force_before_pass and force_after_pass),
        "moment": bool(moment_before_pass and moment_after_pass),
    }
    passed = bool(all(passes.values()))
    reason = "" if passed else ", ".join(name for name, value in passes.items() if not value)
    return EngineeringGripDiagnostics(
        sweep_index=sweep_index,
        target_u_bar=target,
        complete_sweep=True,
        passed=passed,
        failure_stage="" if passed else "gates",
        failure_reason=reason,
        damage_before=committed.material_damage,
        d_iter=d_iter,
        d_next=d_next,
        q_before=q_before,
        q_after=q_after,
        psi_before=psi_before,
        psi_after=psi_after,
        H_used=H_used,
        H_next=H_next,
        rho_used=rho_used,
        rho_next=rho_next,
        rho_used_volume=rho_used_volume,
        rho_next_volume=rho_next_volume,
        energy_before=before.energy,
        energy_after=after.energy,
        top_reaction_before=before.outer_reaction,
        top_reaction_after=after.outer_reaction,
        bottom_reaction_before=before.bottom_reaction,
        bottom_reaction_after=after.bottom_reaction,
        anchor_reaction_before=before.anchor_reaction,
        anchor_reaction_after=after.anchor_reaction,
        force_before=before.total_force,
        force_after=after.total_force,
        moment_before=before.total_moment,
        moment_after=after.total_moment,
        free_residual_norm_before=before.free_residual_norm,
        free_residual_norm_after=after.free_residual_norm,
        free_residual_relative_before=before.free_residual_relative_norm,
        free_residual_relative_after=after.free_residual_relative_norm,
        cg_info_before=before.cg_info,
        cg_info_after=after.cg_info,
        e_u_material=e_u_material,
        e_u_union=e_u_union,
        e_damage=e_damage,
        e_history=e_history,
        e_reaction=e_reaction,
        damage_min=damage_min,
        damage_max=damage_max,
        healing_min=healing_min,
        history_min_gap=history_min_gap,
        rho_used_inf=_e_max_abs(rho_used),
        rho_next_inf=_e_max_abs(rho_next),
        displacement_scale=union_scale,
        damage_scale=damage_scale,
        history_scale=history_scale,
        reaction_scale=reaction_scale,
        damage_residual_scale=residual_scale,
        force_error=force_error,
        force_scale=force_scale,
        moment_error=moment_error,
        moment_scale=moment_scale,
        finite_pass=finite_pass,
        mechanics_before_pass=mechanics_before,
        mechanics_after_pass=mechanics_after,
        displacement_pass=bool(passes["displacement"]),
        damage_pass=bool(passes["damage"]),
        history_pass=bool(passes["history"]),
        reaction_pass=bool(passes["reaction"]),
        damage_residual_pass=bool(passes["damage_residual"]),
        damage_range_pass=bool(passes["damage_range"]),
        healing_pass=bool(passes["healing"]),
        history_monotonic_pass=bool(passes["history_monotonic"]),
        force_pass=bool(passes["force"]),
        moment_pass=bool(passes["moment"]),
        available_fields=(
            "damage_before",
            "d_iter",
            "d_next",
            "q_before",
            "q_after",
            "psi_before",
            "psi_after",
            "H_used",
            "H_next",
            "rho_used",
            "rho_next",
            "rho_used_volume",
            "rho_next_volume",
        ),
    )

def _e_direct_seed(
    previous_q: np.ndarray,
    previous_load: float,
    target_load: float,
    constraints: AffineConstraints,
) -> np.ndarray:
    """Transfer a prior direct-load state without normalized loading algebra."""

    q = _e_f64(previous_q, "direct mechanics seed", (constraints.n_dofs,)).copy()
    _e_check_exact_constraints(q, constraints, previous_load, "source direct seed")
    prescribed = constraints.prescribed_vector(target_load)
    q[constraints.constrained_dofs] = prescribed[constraints.constrained_dofs]
    _e_check_exact_constraints(q, constraints, target_load, "target direct seed")
    if not np.all(np.isfinite(q)):
        raise FloatingPointError("direct mechanics seed is non-finite")
    return _e_immutable(q)

def _e_next_target(current: float, increment: float, final: float) -> float:
    remaining = final - current
    target = current + min(increment, remaining)
    if final - target <= 8.0 * abs(np.spacing(np.float64(final))):
        return final
    return target

def _e_is_minimum_increment(
    trial_increment: float,
    du_min: float,
    load_start: float,
    load_target: float,
) -> bool:
    """Treat a float64 representational excess over ``du_min`` as minimum.

    The subtraction is classified with eight ULPs at the largest participating
    load/increment scale, matching the endpoint-snap roundoff budget.
    """

    trial = _e_scalar(trial_increment, "trial_increment")
    minimum = _e_scalar(du_min, "du_min")
    start = _e_scalar(load_start, "load_start")
    target = _e_scalar(load_target, "load_target")
    scale = max(abs(trial), abs(minimum), abs(start), abs(target))
    ulp = abs(float(np.spacing(np.float64(scale))))
    if not np.isfinite(ulp):
        # Keep the bound finite at the largest representable endpoint.
        ulp = _e_EPS64 * scale
    roundoff_bound = 8.0 * ulp
    return bool(trial <= minimum + roundoff_bound)

def _e_diagnostic_payload(diagnostic: EngineeringGripDiagnostics) -> dict[str, Any]:
    array_names = (
        "damage_before",
        "d_iter",
        "d_next",
        "q_before",
        "q_after",
        "psi_before",
        "psi_after",
        "H_used",
        "H_next",
        "rho_used",
        "rho_next",
        "rho_used_volume",
        "rho_next_volume",
        "force_before",
        "force_after",
    )
    scalars = {
        name: getattr(diagnostic, name)
        for name in (
            "energy_before",
            "energy_after",
            "top_reaction_before",
            "top_reaction_after",
            "bottom_reaction_before",
            "bottom_reaction_after",
            "anchor_reaction_before",
            "anchor_reaction_after",
            "moment_before",
            "moment_after",
            "free_residual_norm_before",
            "free_residual_norm_after",
            "free_residual_relative_before",
            "free_residual_relative_after",
            "cg_info_before",
            "cg_info_after",
            "e_u_material",
            "e_u_union",
            "e_damage",
            "e_history",
            "e_reaction",
            "damage_min",
            "damage_max",
            "healing_min",
            "history_min_gap",
            "rho_used_inf",
            "rho_next_inf",
            "displacement_scale",
            "damage_scale",
            "history_scale",
            "reaction_scale",
            "damage_residual_scale",
            "force_error",
            "force_scale",
            "moment_error",
            "moment_scale",
        )
    }
    return {
        "sweep_index": diagnostic.sweep_index,
        "target_u_bar": diagnostic.target_u_bar,
        "complete_sweep": diagnostic.complete_sweep,
        "passed": diagnostic.passed,
        "failure_stage": diagnostic.failure_stage,
        "failure_reason": diagnostic.failure_reason,
        "gate_passes": diagnostic.gate_passes,
        **{
            name: bool(getattr(diagnostic, name))
            for name in _e_DIAGNOSTIC_BOOL_FIELDS
        },
        "available_fields": list(diagnostic.available_fields),
        **{
            name: (
                None
                if getattr(diagnostic, name) is None
                else np.asarray(getattr(diagnostic, name)).tolist()
            )
            for name in array_names
        },
        **scalars,
    }

_e_DIAGNOSTIC_ARRAY_FIELDS = (
    "damage_before",
    "d_iter",
    "d_next",
    "q_before",
    "q_after",
    "psi_before",
    "psi_after",
    "H_used",
    "H_next",
    "rho_used",
    "rho_next",
    "rho_used_volume",
    "rho_next_volume",
    "force_before",
    "force_after",
)

_e_DIAGNOSTIC_BOOL_FIELDS = (
    "finite_pass",
    "mechanics_before_pass",
    "mechanics_after_pass",
    "displacement_pass",
    "damage_pass",
    "history_pass",
    "reaction_pass",
    "damage_residual_pass",
    "damage_range_pass",
    "healing_pass",
    "history_monotonic_pass",
    "force_pass",
    "moment_pass",
)











_e_FLOAT64 = np.dtype(np.float64)

_e_OPERATOR_RTOL = 5.0e-13

def _e_bytes_backed_copy(
    values: Any, dtype: np.dtype[Any] | None = None
) -> np.ndarray:
    """Return a read-only array whose ultimate storage object is ``bytes``."""

    source = np.array(values, dtype=dtype, copy=True, order="C")
    raw = source.tobytes(order="C")
    result = np.frombuffer(raw, dtype=source.dtype, count=source.size)
    result = result.reshape(source.shape)
    result.setflags(write=False)
    return result

def _e_readonly_copy(values: Any, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    """Copy fixed numerical data into storage that cannot be made writable."""

    return _e_bytes_backed_copy(values, dtype)

def _e_require_float64_array(
    values: Any,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != _e_FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array

def _e_scalar_float(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar, got shape {array.shape}")
    try:
        result = float(array)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a scalar number") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result

def _e_validate_material_parameter(value: Any, name: str) -> float:
    result = _e_scalar_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return result

def _e_validate_operator(
    operator: ExplicitVolumeSparseD,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(operator, ExplicitVolumeSparseD):
        raise TypeError("operator must be an ExplicitVolumeSparseD instance")

    coordinates = _e_require_float64_array(
        np.asarray(operator.coordinates), "operator coordinates"
    )
    if coordinates.shape != (operator.shape[0], 2):
        raise ValueError(
            "fracture2d operator coordinates must have exact float64 shape "
            f"({operator.shape[0]}, 2), got {coordinates.shape}"
        )
    if operator.kernel != "cubic":
        raise ValueError(
            "fracture2d fixed-history damage requires the frozen cubic kernel, "
            f"got {operator.kernel!r}"
        )
    volumes = _e_require_float64_array(
        np.asarray(operator.volumes), "operator volumes"
    )
    if volumes.ndim != 1 or volumes.shape != (operator.shape[0],):
        raise ValueError(
            "operator volumes must have exact float64 shape "
            f"({operator.shape[0]},), got {volumes.shape}"
        )
    if np.any(volumes <= 0.0):
        raise ValueError("operator volumes must be strictly positive")

    edges = np.asarray(operator.directed_edges)
    if edges.dtype != np.dtype(np.int32) and edges.dtype != np.dtype(np.int64):
        raise ValueError(
            "operator directed_edges must contain integer indices, "
            f"got {edges.dtype}"
        )
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("operator directed_edges must have shape (E, 2)")
    edges = np.asarray(edges, dtype=np.int64)
    if np.any(edges < 0) or np.any(edges >= operator.shape[0]):
        raise ValueError("operator directed_edges contain an out-of-range index")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("operator directed_edges must not contain self edges")
    if len(np.unique(edges, axis=0)) != len(edges):
        raise ValueError("operator directed_edges contain duplicate edges")
    edge_set = {tuple(edge) for edge in edges.tolist()}
    if any((j, i) not in edge_set for i, j in edge_set):
        raise ValueError("operator directed_edges must be reciprocal")

    scipy_operator = operator.to_scipy().tocsr(copy=True)
    scipy_operator.sort_indices()
    if scipy_operator.shape != operator.shape:
        raise ValueError(
            "operator.to_scipy() shape does not match the operator shape: "
            f"{scipy_operator.shape} versus {operator.shape}"
        )
    if scipy_operator.dtype != _e_FLOAT64:
        raise ValueError(
            "operator.to_scipy() must preserve exact float64 coefficients, "
            f"got {scipy_operator.dtype}"
        )
    if scipy_operator.nnz and not np.all(np.isfinite(scipy_operator.data)):
        raise ValueError("operator.to_scipy() must contain only finite values")

    row = edges[:, 0]
    col = edges[:, 1]
    delta = _e_validate_material_parameter(operator.delta, "operator delta")
    distances = np.linalg.norm(coordinates[col] - coordinates[row], axis=1)
    if np.any(distances <= 0.0) or np.any(distances >= delta):
        raise ValueError(
            "operator directed_edges must satisfy the strict "
            "0 < distance < delta contract"
        )
    kernel = 120.0 / (np.pi * delta**4) * (1.0 - distances / delta) ** 2
    positive_weights = kernel * volumes[col]
    diagonal = np.bincount(
        row, weights=positive_weights, minlength=operator.shape[0]
    )
    expected = (
        coo_matrix(
            (-positive_weights, (row, col)),
            shape=operator.shape,
            dtype=np.float64,
        ).tocsr()
        + diags(diagonal, format="csr")
    )
    expected.sort_indices()
    if not np.array_equal(
        scipy_operator.indptr, expected.indptr
    ) or not np.array_equal(scipy_operator.indices, expected.indices):
        raise ValueError(
            "operator.to_scipy() sparsity does not match the frozen directed graph"
        )
    coefficient_scale = max(
        float(np.max(np.abs(expected.data))), np.finfo(float).tiny
    )
    coefficient_defect = float(
        np.max(np.abs(scipy_operator.data - expected.data))
    )
    if coefficient_defect > _e_OPERATOR_RTOL * coefficient_scale:
        raise ValueError(
            "operator.to_scipy() coefficients do not match the frozen 2D cubic "
            f"operator: defect={coefficient_defect}, scale={coefficient_scale}"
        )

    weighted = (diags(volumes, format="csr") @ scipy_operator).tocsr()
    weighted_skew = weighted - weighted.T
    weighted_scale = max(
        float(np.max(np.abs(weighted.data))) if weighted.nnz else 0.0,
        np.finfo(float).tiny,
    )
    weighted_defect = (
        float(np.max(np.abs(weighted_skew.data))) if weighted_skew.nnz else 0.0
    )
    if weighted_defect > _e_OPERATOR_RTOL * weighted_scale:
        raise ValueError(
            "operator.to_scipy() violates volume self-adjointness: "
            f"defect={weighted_defect}, scale={weighted_scale}"
        )
    constant_defect = np.asarray(
        scipy_operator @ np.ones(operator.shape[0], dtype=np.float64)
    )
    row_scale = max(
        float(np.max(np.asarray(np.abs(scipy_operator).sum(axis=1)))),
        np.finfo(float).tiny,
    )
    if float(np.max(np.abs(constant_defect))) > _e_OPERATOR_RTOL * row_scale:
        raise ValueError("operator.to_scipy() does not annihilate constants")
    return scipy_operator, volumes.copy(), edges, coordinates.copy()

def _e_validate_history(history: Any, n_particles: int) -> np.ndarray:
    values = _e_require_float64_array(
        history, "history", shape=(n_particles,)
    )
    if np.any(values < 0.0):
        raise ValueError("history must be non-negative")
    return values

@dataclass(frozen=True)
class FixedHistoryDamageResult:
    """Immutable direct solution and diagnostics for the fixed-history AT2 solve."""

    damage: np.ndarray
    history: np.ndarray
    crack_local_energy: float
    history_energy: float
    graph_energy: float
    total_energy: float
    strong_residual: np.ndarray
    euclidean_residual: np.ndarray
    strong_residual_norm: float
    strong_residual_inf: float
    euclidean_residual_norm: float
    euclidean_residual_inf: float
    damage_min: float
    damage_max: float
    below_zero_count: int
    above_one_count: int
    symmetry_defect: float
    symmetry_relative_defect: float

    def __post_init__(self) -> None:
        damage = np.asarray(self.damage)
        history = np.asarray(self.history)
        if damage.dtype != _e_FLOAT64 or damage.ndim != 1:
            raise ValueError("FixedHistoryDamageResult.damage must be float64 (N,)")
        if history.dtype != _e_FLOAT64 or history.shape != damage.shape:
            raise ValueError(
                "FixedHistoryDamageResult.history must be float64 with damage shape"
            )
        residual_specs = (
            ("strong_residual", self.strong_residual),
            ("euclidean_residual", self.euclidean_residual),
        )
        for name, values in residual_specs:
            array = np.asarray(values)
            if array.dtype != _e_FLOAT64 or array.shape != damage.shape:
                raise ValueError(
                    f"FixedHistoryDamageResult.{name} must be float64 with damage shape"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"FixedHistoryDamageResult.{name} must be finite")
            object.__setattr__(self, name, _e_readonly_copy(array))
        for name, values in (("damage", damage), ("history", history)):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"FixedHistoryDamageResult.{name} must be finite")
            object.__setattr__(self, name, _e_readonly_copy(values))

        scalar_names = (
            "crack_local_energy",
            "history_energy",
            "graph_energy",
            "total_energy",
            "strong_residual_norm",
            "strong_residual_inf",
            "euclidean_residual_norm",
            "euclidean_residual_inf",
            "damage_min",
            "damage_max",
            "symmetry_defect",
            "symmetry_relative_defect",
        )
        for name in scalar_names:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"FixedHistoryDamageResult.{name} must be finite")
            object.__setattr__(self, name, value)
        for name in ("below_zero_count", "above_one_count"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
                raise ValueError(f"FixedHistoryDamageResult.{name} must be an integer")
            value = int(value)
            if value < 0:
                raise ValueError(
                    f"FixedHistoryDamageResult.{name} must be non-negative"
                )
            object.__setattr__(self, name, value)

    @property
    def crack_energy(self) -> float:
        """Alias for the local crack-density energy component."""

        return self.crack_local_energy

    @property
    def local_energy(self) -> float:
        """Alias for the local crack-density energy component."""

        return self.crack_local_energy

    @property
    def crack_local(self) -> float:
        return self.crack_local_energy

    @property
    def energy(self) -> float:
        """Alias for the total fixed-history damage energy."""

        return self.total_energy

    @property
    def min_damage(self) -> float:
        return self.damage_min

    @property
    def max_damage(self) -> float:
        return self.damage_max

    @property
    def symmetry_error(self) -> float:
        return self.symmetry_defect

    @property
    def residual(self) -> np.ndarray:
        """Alias for the strong-form residual."""

        return self.strong_residual

    @property
    def system_residual(self) -> np.ndarray:
        """Alias for the Euclidean volume-multiplied residual."""

        return self.euclidean_residual

def solve_fixed_history_damage(
    operator: ExplicitVolumeSparseD,
    Gc: Any,
    l0: Any,
    history: Any,
) -> FixedHistoryDamageResult:
    """Solve the dimensional fixed-history nonlocal AT2 system directly.

    The returned system is the Euclidean volume-multiplied system

    ``S = M @ (diag(Gc/l0 + 2*H) + Gc*l0*D)``, ``b = 2*M@H``.

    The directed graph energy is evaluated as
    ``Gc*l0/4 * sum_(i,j) Vi*Vj*Jij*(dj-di)**2``; no damage clipping or hidden
    bound projection is performed.
    """

    gc_value = _e_validate_material_parameter(Gc, "Gc")
    l0_value = _e_validate_material_parameter(l0, "l0")
    D, volumes, edges, _ = _e_validate_operator(operator)
    n_particles = int(operator.shape[0])
    history_values = _e_validate_history(history, n_particles)

    mass = diags(volumes, format="csr")
    strong_matrix = diags(
        gc_value / l0_value + 2.0 * history_values,
        format="csr",
    ) + (gc_value * l0_value) * D
    system = (mass @ strong_matrix).tocsr()
    rhs = np.asarray(2.0 * volumes * history_values, dtype=np.float64)
    if system.shape != (n_particles, n_particles) or rhs.shape != (n_particles,):
        raise ValueError("fixed-history system has an incompatible shape")
    if system.nnz and not np.all(np.isfinite(system.data)):
        raise FloatingPointError("fixed-history system contains non-finite values")
    if not np.all(np.isfinite(rhs)):
        raise FloatingPointError("fixed-history right-hand side is non-finite")

    skew = system - system.T
    symmetry_defect = float(np.linalg.norm(skew.data, ord=np.inf)) if skew.nnz else 0.0
    system_norm = float(np.linalg.norm(system.data, ord=np.inf)) if system.nnz else 0.0
    symmetry_relative_defect = symmetry_defect / max(system_norm, 1.0e-300)
    if not np.isfinite(symmetry_defect) or not np.isfinite(symmetry_relative_defect):
        raise FloatingPointError("fixed-history system symmetry diagnostic is non-finite")
    if symmetry_relative_defect > _e_OPERATOR_RTOL:
        raise ValueError(
            "fixed-history Euclidean system is not symmetric within the frozen "
            f"float64 contract: relative defect={symmetry_relative_defect}"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            damage = np.asarray(spsolve(system, rhs), dtype=np.float64)
        except (MatrixRankWarning, RuntimeError, np.linalg.LinAlgError) as error:
            raise RuntimeError("fixed-history sparse direct solve failed or is singular") from error
    if damage.shape != (n_particles,) or not np.all(np.isfinite(damage)):
        raise FloatingPointError("fixed-history sparse direct solve returned non-finite damage")

    # Reconstruct both residual representations from the same signed
    # constituents used by the strong-form audit.  ``system @ damage - rhs``
    # is mathematically identical, but its separate sparse mass/strong-matrix
    # matvec can introduce an avoidable cancellation defect when the solved
    # residual is near machine precision.  Keeping the operation order here
    # makes the returned strong and volume residuals satisfy the declared
    # constituent-scale contract without changing the fixed-history equation.
    graph_damage = np.asarray(D @ damage, dtype=np.float64)
    strong_residual = np.asarray(
        gc_value / l0_value * damage
        + gc_value * l0_value * graph_damage
        + 2.0 * history_values * damage
        - 2.0 * history_values,
        dtype=np.float64,
    )
    euclidean_residual = np.asarray(volumes * strong_residual, dtype=np.float64)
    if not np.all(np.isfinite(strong_residual)) or not np.all(
        np.isfinite(euclidean_residual)
    ):
        raise FloatingPointError("fixed-history solve residual is non-finite")

    crack_local_energy = float(
        np.sum(volumes * (gc_value / (2.0 * l0_value)) * damage**2)
    )
    history_energy = float(np.sum(volumes * history_values * (1.0 - damage) ** 2))
    row = edges[:, 0]
    col = edges[:, 1]
    directed_coefficient = np.asarray(D[row, col]).reshape(-1)
    directed_weight = -directed_coefficient
    if not np.all(np.isfinite(directed_weight)) or np.any(directed_weight <= 0.0):
        raise ValueError("fixed-history graph coefficients must be finite and positive")
    graph_energy = float(
        gc_value
        * l0_value
        / 4.0
        * np.sum(
            volumes[row]
            * directed_weight
            * (damage[col] - damage[row]) ** 2
        )
    )
    total_energy = crack_local_energy + history_energy + graph_energy
    energy_values = (
        crack_local_energy,
        history_energy,
        graph_energy,
        total_energy,
    )
    if not all(np.isfinite(value) for value in energy_values):
        raise FloatingPointError("fixed-history damage energy is non-finite")

    return FixedHistoryDamageResult(
        damage=damage,
        history=history_values,
        crack_local_energy=crack_local_energy,
        history_energy=history_energy,
        graph_energy=graph_energy,
        total_energy=total_energy,
        strong_residual=strong_residual,
        euclidean_residual=euclidean_residual,
        strong_residual_norm=float(np.linalg.norm(strong_residual)),
        strong_residual_inf=float(np.max(np.abs(strong_residual))),
        euclidean_residual_norm=float(np.linalg.norm(euclidean_residual)),
        euclidean_residual_inf=float(np.max(np.abs(euclidean_residual))),
        damage_min=float(np.min(damage)),
        damage_max=float(np.max(damage)),
        below_zero_count=int(np.count_nonzero(damage < 0.0)),
        above_one_count=int(np.count_nonzero(damage > 1.0)),
        symmetry_defect=symmetry_defect,
        symmetry_relative_defect=symmetry_relative_defect,
    )

def _e_validate_nonnegative_scalar(value: Any, name: str) -> float:
    result = _e_scalar_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}")
    return result

@dataclass(frozen=True)
class GateTolerance:
    """Absolute/relative gate with an explicit scale floor.

    The gate limit is
    ``max(atol, rtol * max(error_scale, scale_floor))``.  All three
    constructor arguments are deliberately required; continuation tolerances
    must not be inherited from an unrelated solver configuration.
    """

    atol: float
    rtol: float
    scale_floor: float

    def __post_init__(self) -> None:
        atol = _e_validate_nonnegative_scalar(self.atol, "GateTolerance.atol")
        rtol = _e_validate_nonnegative_scalar(self.rtol, "GateTolerance.rtol")
        scale_floor = _e_scalar_float(self.scale_floor, "GateTolerance.scale_floor")
        if scale_floor <= 0.0:
            raise ValueError("GateTolerance.scale_floor must be positive and finite")
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "rtol", rtol)
        object.__setattr__(self, "scale_floor", scale_floor)

    def limit(self, error_scale: Any) -> float:
        """Return the explicit acceptance limit for one non-negative scale."""

        scale = _e_scalar_float(error_scale, "error_scale")
        if scale < 0.0:
            raise ValueError("error_scale must be non-negative and finite")
        return max(self.atol, self.rtol * max(scale, self.scale_floor))
# ===== nonlocal_engineering_grip_stabilized =====















STABILIZED_ACCEPTANCE_MODE = "stabilized"
STABILIZED_ACCEPTANCE_STATUS = "stabilized_converged"
STABILIZED_COMPLETION_STATUS = "stabilized_completed"
STABILIZED_FAILURE_STATUS = "stabilized_continuation_failure"
STABILIZED_FAILURE_CODE = "stabilized_continuation_failure"


STABILIZED_ARCHIVE_SCHEMA = "fracture2d-engineering-grip-stabilized-v5"
STABILIZED_FAILURE_ARCHIVE_SCHEMA = (
    "fracture2d-engineering-grip-stabilized-failure-v5"
)
STABILIZED_REPLAY_CONTRACT = "fracture2d-engineering-grip-stabilized-replay-v5"
STABILIZED_ARCHIVE_LAYOUT = "accepted-state-plus-final-solver-summary-v5"

STABILIZED_ANDERSON_POLICIES = (
    "anderson",
    "anderson_residual",
    "anderson_residual_backtracking",
    "anderson_residual_regularized_backtracking",
)





STABILIZED_ANDERSON_ALPHA_REDUCTION = 0.5
STABILIZED_ANDERSON_MAX_BACKTRACKS = 4



STABILIZED_ANDERSON_RESIDUAL_BACKTRACKING_MERIT_IDENTITY = (
    "committed_history_map_residual_backtracking_v1"
)
STABILIZED_ANDERSON_REGULARIZED_MERIT_IDENTITY = (
    "committed_history_map_residual_regularized_backtracking_v1"
)











_s_ANDERSON_SELECTIONS = ("accelerated", "picard_fallback")

STABILIZED_WARNING_ACCEPTANCE_MODE = "stabilized_warning"
STABILIZED_WARNING_ACCEPTANCE_STATUS = "stabilized_warning_accepted"
STABILIZED_WARNING_COMPLETION_STATUS = "stabilized_warning_completed"
STABILIZED_WARNING_FAILURE_STATUS = "stabilized_warning_continuation_failure"
STABILIZED_WARNING_FAILURE_CODE = "stabilized_warning_continuation_failure"
STABILIZED_WARNING_ARCHIVE_SCHEMA = (
    "fracture2d-engineering-grip-stabilized-warning-v1"
)
STABILIZED_WARNING_FAILURE_ARCHIVE_SCHEMA = (
    "fracture2d-engineering-grip-stabilized-warning-failure-v1"
)
STABILIZED_WARNING_REPLAY_CONTRACT = (
    "fracture2d-engineering-grip-stabilized-warning-replay-v1"
)
STABILIZED_WARNING_ARCHIVE_LAYOUT = (
    "accepted-state-plus-final-solver-summary-stabilized-warning-v1"
)
STABILIZED_ACCEPTANCE_MODES = (
    STABILIZED_ACCEPTANCE_MODE,
    STABILIZED_WARNING_ACCEPTANCE_MODE,
)







_s_TRIAL_Q_DIGEST_DECIMALS = 11

STABILIZED_COUPLED_GATE_NAMES = (
    "finite",
    "mechanics_before",
    "mechanics_after",
    "displacement",
    "damage",
    "history",
    "reaction",
    "damage_residual",
    "history_monotonic",
    "force",
    "moment",
    "damage_fixed_point",
    "damage_equation",
    "globalization",
)

STABILIZED_WARNING_HARD_GATE_NAMES = (
    "finite",
    "mechanics_before",
    "mechanics_after",
    "history_monotonic",
    "force",
    "moment",
    "damage_equation",
    "globalization",
)
STABILIZED_WARNING_GATE_NAMES = (
    "displacement",
    "damage",
    "history",
    "reaction",
    "damage_residual",
    "damage_fixed_point",
    "damage_range",
    "healing",
    "coupled",
)



_s_FLOAT64 = np.dtype(np.float64)




ANDERSON_RESIDUAL_HARD_GATE_ATTRIBUTION_SCHEMA = (
    "fracture2d-engineering-grip-anderson-residual-hard-gates-v3"
)

ANDERSON_RESIDUAL_HARD_GATE_NAMES = (
    "finite",
    "mechanics_before",
    "mechanics_after",
    "force",
    "moment",
    "damage_range_lower",
    "damage_range_upper",
    "healing",
    "history_ordering",
)
_s_ANDERSON_HARD_GATE_DETAIL_FIELDS = (
    "available",
    "passed",
    "raw",
    "scale",
    "limit",
    "normalized",
    "ratio_available",
    "metric_reason",
    "stage_reason",
    "cg_info",
    "cg_status_pass",
    "residual_limit_pass",
    "stage_metrics",
)
_s_ANDERSON_HARD_GATE_STAGE_NAMES = ("before", "after")
_s_ANDERSON_HARD_GATE_STATUSES = (
    "evaluated",
    "not_evaluated",
    "candidate_map_failure",
)
_s_ANDERSON_HARD_GATE_STAGE_METRIC_FIELDS = (
    "error",
    "scale",
    "limit",
    "normalized",
    "ratio_available",
    "passed",
)
_s_ANDERSON_HARD_GATE_METRIC_REASONS = {
    "finite": "boolean_gate",
    "mechanics_before": "cg_info_and_free_residual_limit",
    "mechanics_after": "cg_info_and_free_residual_limit",
    "force": "stage_worst_ratio_and_conjunction",
    "moment": "stage_worst_ratio_and_conjunction",
    "damage_range_lower": "damage_range_excess_and_limit",
    "damage_range_upper": "damage_range_excess_and_limit",
    "healing": "healing_excess_and_limit",
    "history_ordering": "history_ordering_excess_and_zero_limit",
}
_s_ANDERSON_HARD_GATE_UNAVAILABLE_REASONS = {
    "candidate_not_fully_evaluated",
    "mechanics_before_not_evaluated",
    "mechanics_after_not_evaluated",
    "damage_not_evaluated",
    "history_not_evaluated",
}

def _s_validate_stabilized_acceptance_mode(value: Any) -> str:
    if not isinstance(value, str) or value not in STABILIZED_ACCEPTANCE_MODES:
        raise ValueError(
            "stabilized acceptance mode must be one of "
            f"{STABILIZED_ACCEPTANCE_MODES}, got {value!r}"
        )
    return value

def _s_is_anderson_policy(policy: str) -> bool:
    return policy in STABILIZED_ANDERSON_POLICIES







def _s_anderson_backtracking_alpha_for_index(index: int, name: str) -> float:
    """Return the canonical alpha bound to one new-policy trial index."""

    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError(f"{name} must be a JSON integer")
    if index < 0 or index > STABILIZED_ANDERSON_MAX_BACKTRACKS:
        raise ValueError(f"{name} is outside the configured alpha schedule")
    alpha = float(STABILIZED_ANDERSON_ALPHA_REDUCTION**index)
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError(f"{name} produces an invalid alpha")
    return alpha



def _s_stabilized_contract(mode: str, *, failure: bool) -> dict[str, Any]:
    """Return the immutable archive identity for one stabilized mode."""

    validated_mode = _s_validate_stabilized_acceptance_mode(mode)
    if validated_mode == STABILIZED_ACCEPTANCE_MODE:
        return {
            "acceptance_mode": STABILIZED_ACCEPTANCE_MODE,
            "acceptance_status": STABILIZED_ACCEPTANCE_STATUS,
            "completion_status": (
                STABILIZED_FAILURE_STATUS
                if failure
                else STABILIZED_COMPLETION_STATUS
            ),
            "archive_schema": (
                STABILIZED_FAILURE_ARCHIVE_SCHEMA
                if failure
                else STABILIZED_ARCHIVE_SCHEMA
            ),
            "archive_layout": STABILIZED_ARCHIVE_LAYOUT,
            "replay_contract": STABILIZED_REPLAY_CONTRACT,
            "failure_code": (
                STABILIZED_FAILURE_CODE
                if failure
                else None
            ),
        }
    return {
        "acceptance_mode": STABILIZED_WARNING_ACCEPTANCE_MODE,
        "acceptance_status": STABILIZED_WARNING_ACCEPTANCE_STATUS,
        "completion_status": (
            STABILIZED_WARNING_FAILURE_STATUS
            if failure
            else STABILIZED_WARNING_COMPLETION_STATUS
        ),
        "archive_schema": (
            STABILIZED_WARNING_FAILURE_ARCHIVE_SCHEMA
            if failure
            else STABILIZED_WARNING_ARCHIVE_SCHEMA
        ),
        "archive_layout": STABILIZED_WARNING_ARCHIVE_LAYOUT,
        "replay_contract": STABILIZED_WARNING_REPLAY_CONTRACT,
    }


def _s_stabilized_failure_code(mode: str) -> str:
    """Return the failure identity paired with one stabilized mode."""

    validated_mode = _s_validate_stabilized_acceptance_mode(mode)
    return (
        STABILIZED_WARNING_FAILURE_CODE
        if validated_mode == STABILIZED_WARNING_ACCEPTANCE_MODE
        else STABILIZED_FAILURE_CODE
    )




def _s_immutable(values: Any, dtype: Any | None = None) -> np.ndarray:
    return _e_immutable(values, dtype)

def _s_f64(
    value: Any,
    name: str,
    shape: tuple[int, ...] | None = None,
    *,
    nonfinite_error: type[Exception] = ValueError,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != _s_FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise nonfinite_error(f"{name} must contain only finite values")
    return array

def _s_scalar(value: Any, name: str) -> float:
    return _e_scalar(value, name)

def _s_optional_scalar(value: Any | None, name: str) -> float | None:
    if value is None:
        return None
    return _s_scalar(value, name)

def _s_optional_nonnegative_integer(value: Any | None, name: str) -> int | None:
    if value is None:
        return None
    result = _e_integer(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result

def _s_optional_f64(
    value: Any | None, name: str, shape: tuple[int, ...] | None = None
) -> np.ndarray | None:
    if value is None:
        return None
    return _s_immutable(_s_f64(value, name, shape))

def _s_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a hexadecimal SHA256 digest") from error
    return value



def _s_array_close(
    actual: Any,
    expected: Any,
    name: str,
    relative_limit: float = 2.0e-9,
    absolute_limit: float | None = None,
) -> None:
    left = np.asarray(actual)
    right = np.asarray(expected)
    if left.dtype != right.dtype or left.shape != right.shape:
        raise ValueError(f"{name} shape/dtype does not match replay")
    defect = float(np.max(np.abs(left - right))) if left.size else 0.0
    scale = max(
        float(np.max(np.abs(left))) if left.size else 0.0,
        float(np.max(np.abs(right))) if right.size else 0.0,
        1.0,
    )
    limit = relative_limit * scale
    if absolute_limit is not None:
        limit = max(limit, float(absolute_limit))
    if not np.isfinite(defect) or defect > limit:
        raise ValueError(f"{name} does not match replay: defect={defect:.17g}")

def _s_diagnostic_from_payload(
    payload: Any,
    name: str,
    *,
    array_values: Mapping[str, np.ndarray] | None = None,
) -> EngineeringGripDiagnostics:
    """Rebuild one ordinary diagnostic from compact canonical JSON.

    The compact diagnostic JSON carries only scalar/boolean provenance.  The
    optional arrays are an in-memory/dataclass reconstruction seam; the public
    v5 archive does not serialize them and supplies no trial-array rows.
    """

    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must contain a diagnostic mapping")
    diagnostic_fields = {
        field.name for field in dataclass_fields(EngineeringGripDiagnostics)
    }
    required = diagnostic_fields | {"gate_passes"}
    missing = sorted(required - set(payload))
    unexpected = sorted(set(payload) - required)
    if missing:
        raise ValueError(f"{name} is missing diagnostic fields: {missing}")
    if unexpected:
        raise ValueError(f"{name} contains unknown diagnostic fields: {unexpected}")
    for field_name in (
        "complete_sweep",
        "passed",
        *_e_DIAGNOSTIC_BOOL_FIELDS,
    ):
        if type(payload[field_name]) is not bool:
            raise ValueError(f"{name}.{field_name} must be a JSON boolean")
    available = payload["available_fields"]
    if not isinstance(available, list) or not all(
        isinstance(item, str) for item in available
    ):
        raise ValueError(f"{name}.available_fields is invalid")
    gates = payload["gate_passes"]
    expected_gates = set(_e_WARNING_HARD_GATE_NAMES) | set(
        WARNING_GATE_NAMES
    )
    if not isinstance(gates, Mapping) or set(gates) != expected_gates or any(
        type(item) is not bool for item in gates.values()
    ):
        raise ValueError(f"{name}.gate_passes is invalid")
    diagnostic_payload = {
        field_name: payload[field_name] for field_name in diagnostic_fields
    }
    for field_name in _e_DIAGNOSTIC_ARRAY_FIELDS:
        if diagnostic_payload[field_name] is not None:
            raise ValueError(
                f"{name}.{field_name} must be null in compact stabilized JSON"
            )
        if array_values is not None and field_name in array_values:
            diagnostic_payload[field_name] = array_values[field_name]
    if diagnostic_payload["damage_before"] is None or diagnostic_payload["d_iter"] is None:
        raise ValueError(f"{name} is missing typed damage provenance arrays")
    diagnostic = EngineeringGripDiagnostics(**diagnostic_payload)
    if list(diagnostic.available_fields) != available:
        raise ValueError(f"{name}.available_fields are not canonical")
    if diagnostic.gate_passes != dict(gates):
        raise ValueError(f"{name}.gate_passes are forged")
    return diagnostic

def _s_compact_diagnostic_payload(
    diagnostic: EngineeringGripDiagnostics,
) -> dict[str, Any]:
    """Serialize diagnostic scalars/flags without duplicating numerical arrays.

    Runtime sweep diagnostics retain fixed-shape numerical primitives for the
    controller.  Complete relaxation candidates are represented in the archive
    by scalar/Boolean audit fields and explicit ``null`` array markers only.
    """

    payload = _e_diagnostic_payload(diagnostic)
    for name in _e_DIAGNOSTIC_ARRAY_FIELDS:
        payload[name] = None
    return payload

def _s_trial_q_after_digest(
    target: float,
    q_after: np.ndarray,
    d_trial: np.ndarray,
    H_next: np.ndarray,
) -> str:
    """Return the replay-stable fingerprint for one complete trial state."""

    q = _s_f64(q_after, "trial q-after digest displacement")
    decimals = _s_TRIAL_Q_DIGEST_DECIMALS
    quantized = np.round(q, decimals=decimals)
    damage = np.round(_s_f64(d_trial, "trial q-after digest damage"), decimals=decimals)
    history = np.round(_s_f64(H_next, "trial q-after digest history"), decimals=decimals)
    return _e_state_digest(target, quantized, damage, history)













@dataclass(frozen=True)
class StabilizedEngineeringGripConfig:
    """Adopted fixed damped-Picard configuration (strict or warning)."""

    base: EngineeringGripConfig
    omega_initial: float
    omega_min: float
    relaxation_policy: str = "fixed"
    acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE

    def __post_init__(self) -> None:
        if not isinstance(self.base, EngineeringGripConfig):
            raise TypeError("stabilized config base must be EngineeringGripConfig")
        if self.relaxation_policy != "fixed":
            raise ValueError(
                "the adopted scheme supports only the fixed damped-Picard policy"
            )
        mode = _s_validate_stabilized_acceptance_mode(self.acceptance_mode)
        omega_initial = _s_scalar(self.omega_initial, "omega_initial")
        omega_min = _s_scalar(self.omega_min, "omega_min")
        if not 0.0 < omega_min <= omega_initial <= 1.0:
            raise ValueError(
                "stabilized relaxation requires 0 < omega_min <= omega_initial <= 1"
            )
        if omega_min != omega_initial:
            raise ValueError("fixed relaxation requires omega_min == omega_initial")
        for name in STABILIZED_COUPLED_GATE_NAMES:
            if name in {
                "finite",
                "mechanics_before",
                "mechanics_after",
                "damage_fixed_point",
                "damage_equation",
                "globalization",
            }:
                continue
            if not hasattr(self.base, name):
                continue
            gate = getattr(self.base, name)
            if gate.atol == 0.0 and gate.rtol == 0.0:
                raise ValueError(
                    f"stabilized merit gate {name} needs a positive tolerance"
                )
        object.__setattr__(self, "omega_initial", omega_initial)
        object.__setattr__(self, "omega_min", omega_min)
        object.__setattr__(self, "relaxation_policy", "fixed")
        object.__setattr__(self, "acceptance_mode", mode)






def _s_anderson_scalar_relation_close(actual: float, expected: float) -> bool:
    """Compare derived Anderson scalars with a scale-aware float64 tolerance."""

    actual_value = float(actual)
    expected_value = float(expected)
    if not np.isfinite(actual_value) or not np.isfinite(expected_value):
        return False
    scale = max(
        abs(actual_value),
        abs(expected_value),
        np.finfo(np.float64).tiny,
    )
    tolerance = 256.0 * np.finfo(np.float64).eps * scale
    return abs(actual_value - expected_value) <= tolerance































@dataclass(frozen=True)
class StabilizedRelaxationTrial:
    """One explicit relaxation/backtracking candidate audit."""

    omega: float | None
    complete: bool
    hard_pass: bool
    globalization_pass: bool
    coupled_pass: bool
    selected: bool
    merit: float | None
    sufficient_decrease_limit: float | None
    failure_stage: str
    failure_reason: str
    candidate_diagnostics: EngineeringGripDiagnostics | None
    d_trial: np.ndarray | None
    q_after: np.ndarray | None
    d_trial_digest: str | None
    q_after_digest: str | None
    candidate_kind: str = "picard"
    # Historical accelerated rows intentionally retain ``None`` because their
    # raw proposal is not a line-search trial.  The distinct backtracking
    # policy records the actual alpha explicitly rather than overloading omega.
    trial_alpha: float | None = None
    candidate_fp_error: float | None = None
    candidate_fp_scale: float | None = None
    candidate_damage_residual_scale: float | None = None
    # The controller may replace an incomplete candidate diagnostic with a
    # stage-specific record after a hard rejection or candidate-map failure.
    # Retain only the already-computed scalar gate snapshot for the optional
    # residual-policy JSONL path; no numerical arrays are duplicated or added to
    # the lean v5 archive payload.
    candidate_hard_gate_attribution: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        omega = _s_optional_scalar(self.omega, "relaxation trial omega")
        if omega is not None and not 0.0 < omega <= 1.0:
            raise ValueError("relaxation trial omega must lie in (0, 1]")
        if self.candidate_kind not in ("picard", "picard_fallback", "anderson_accelerated"):
            raise ValueError(
                "relaxation trial candidate_kind is invalid: "
                f"{self.candidate_kind!r}"
            )
        if self.candidate_kind == "anderson_accelerated" and omega is not None:
            raise ValueError("Anderson accelerated trial cannot retain an omega")
        if self.candidate_kind != "anderson_accelerated" and omega is None:
            raise ValueError("Picard relaxation trial requires an omega")
        trial_alpha = _s_optional_scalar(self.trial_alpha, "relaxation trial alpha")
        if trial_alpha is not None and not 0.0 < trial_alpha <= 1.0:
            raise ValueError("relaxation trial alpha must lie in (0, 1]")
        if self.candidate_kind != "anderson_accelerated" and trial_alpha is not None:
            raise ValueError("Picard relaxation trial cannot retain trial_alpha")
        for name in ("failure_stage", "failure_reason"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"relaxation trial {name} must be a string")
        merit = _s_optional_scalar(self.merit, "relaxation trial merit")
        limit = _s_optional_scalar(
            self.sufficient_decrease_limit, "relaxation trial sufficient limit"
        )
        candidate_fp_error = _s_optional_scalar(
            self.candidate_fp_error, "relaxation trial candidate fixed-point error"
        )
        candidate_fp_scale = _s_optional_scalar(
            self.candidate_fp_scale, "relaxation trial candidate fixed-point scale"
        )
        candidate_residual_scale = _s_optional_scalar(
            self.candidate_damage_residual_scale,
            "relaxation trial candidate damage residual scale",
        )
        candidate = self.candidate_diagnostics
        if candidate is not None and not isinstance(
            candidate, EngineeringGripDiagnostics
        ):
            raise TypeError("relaxation trial candidate diagnostics have the wrong type")
        hard_gate_attribution = self.candidate_hard_gate_attribution
        if hard_gate_attribution is not None:
            if not isinstance(hard_gate_attribution, Mapping):
                raise TypeError(
                    "relaxation trial hard-gate attribution has the wrong type"
                )
            _s_validate_anderson_scalar_event(hard_gate_attribution)
        d_trial = _s_optional_f64(self.d_trial, "relaxation trial d_trial")
        q_after = _s_optional_f64(self.q_after, "relaxation trial q_after")
        if merit is not None and merit < 0.0:
            raise ValueError("relaxation trial merit must be non-negative")
        if limit is not None and limit < 0.0:
            raise ValueError("relaxation trial sufficient limit must be non-negative")
        for name, value in (
            ("candidate_fp_error", candidate_fp_error),
            ("candidate_fp_scale", candidate_fp_scale),
            ("candidate_damage_residual_scale", candidate_residual_scale),
        ):
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.candidate_kind != "anderson_accelerated" and any(
            value is not None
            for value in (
                candidate_fp_error,
                candidate_fp_scale,
                candidate_residual_scale,
            )
        ):
            raise ValueError(
                "Picard relaxation trial retains Anderson candidate metrics"
            )
        if self.candidate_kind == "anderson_accelerated" and self.complete and any(
            value is None
            for value in (
                candidate_fp_error,
                candidate_fp_scale,
                candidate_residual_scale,
            )
        ):
            raise ValueError("complete Anderson trial is missing candidate metrics")
        if candidate is None:
            raise ValueError("relaxation trial is missing candidate diagnostics")
        if bool(candidate.complete_sweep) != bool(self.complete):
            raise ValueError("relaxation trial candidate completion is not paired")
        if d_trial is None or candidate.d_next is None:
            if d_trial is not None or candidate.d_next is not None:
                raise ValueError("relaxation trial d_trial availability is not paired")
        elif not np.array_equal(d_trial, candidate.d_next):
            raise ValueError("relaxation trial d_trial is not paired with diagnostics")
        if q_after is None or candidate.q_after is None:
            if q_after is not None or candidate.q_after is not None:
                raise ValueError("relaxation trial q_after availability is not paired")
        elif not np.array_equal(q_after, candidate.q_after):
            _s_array_close(
                q_after,
                candidate.q_after,
                "relaxation trial q_after",
                relative_limit=2.0e-10,
                absolute_limit=2.0e-12,
            )
        if self.complete:
            if not self.hard_pass:
                raise ValueError("complete relaxation trial must pass hard gates")
            if merit is None and self.candidate_kind != "picard_fallback":
                raise ValueError("complete relaxation trial is missing merit fields")
            if (
                d_trial is None
                or q_after is None
                or self.d_trial_digest is None
                or self.q_after_digest is None
            ):
                raise ValueError("complete relaxation trial is missing state digests")
            if self.d_trial_digest != _e_damage_digest(d_trial):
                raise ValueError("complete relaxation trial damage digest is not paired")
            assert candidate.q_after is not None
            assert candidate.H_next is not None
            if self.q_after_digest != _s_trial_q_after_digest(
                candidate.target_u_bar,
                q_after,
                d_trial,
                candidate.H_next,
            ):
                raise ValueError("complete relaxation trial q-after digest is not paired")
        elif self.hard_pass or self.globalization_pass or self.coupled_pass or self.selected:
            raise ValueError("incomplete relaxation trial contains complete-only flags")
        if bool(self.selected) != bool(self.globalization_pass):
            raise ValueError("relaxation trial selection is not paired with globalization")
        if self.selected and not (
            self.complete and self.hard_pass and self.globalization_pass
        ):
            raise ValueError("selected relaxation trial did not pass globalization")
        for name in ("d_trial_digest", "q_after_digest"):
            value = getattr(self, name)
            if value is not None:
                _s_digest(value, f"relaxation trial {name}")
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "complete", bool(self.complete))
        object.__setattr__(self, "hard_pass", bool(self.hard_pass))
        object.__setattr__(self, "globalization_pass", bool(self.globalization_pass))
        object.__setattr__(self, "coupled_pass", bool(self.coupled_pass))
        object.__setattr__(self, "selected", bool(self.selected))
        object.__setattr__(self, "merit", merit)
        object.__setattr__(self, "sufficient_decrease_limit", limit)
        object.__setattr__(self, "candidate_diagnostics", candidate)
        object.__setattr__(self, "candidate_hard_gate_attribution", hard_gate_attribution)
        object.__setattr__(self, "d_trial", d_trial)
        object.__setattr__(self, "q_after", q_after)
        object.__setattr__(self, "candidate_kind", self.candidate_kind)
        object.__setattr__(self, "trial_alpha", trial_alpha)
        object.__setattr__(self, "candidate_fp_error", candidate_fp_error)
        object.__setattr__(self, "candidate_fp_scale", candidate_fp_scale)
        object.__setattr__(
            self,
            "candidate_damage_residual_scale",
            candidate_residual_scale,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "omega": self.omega,
            "complete": self.complete,
            "hard_pass": self.hard_pass,
            "globalization_pass": self.globalization_pass,
            "coupled_pass": self.coupled_pass,
            "selected": self.selected,
            "merit": self.merit,
            "sufficient_decrease_limit": self.sufficient_decrease_limit,
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            "candidate_diagnostics": (
                None
                if self.candidate_diagnostics is None
                else _s_compact_diagnostic_payload(self.candidate_diagnostics)
            ),
            # Keep null markers in JSON for explicit availability without
            # copying numerical trial vectors into the lean archive.
            "d_trial": None,
            "q_after": None,
            "d_trial_digest": self.d_trial_digest,
            "q_after_digest": self.q_after_digest,
            "candidate_kind": self.candidate_kind,
            "candidate_fp_error": self.candidate_fp_error,
            "candidate_fp_scale": self.candidate_fp_scale,
            "candidate_damage_residual_scale": self.candidate_damage_residual_scale,
        }
        # Historical trial payloads do not gain a null field merely because the
        # dataclass now understands the distinct backtracking policy. Every
        # accelerated row emitted by that policy carries a non-null alpha; the
        # compact v5 solver summary records the explicit null fallback.
        if self.trial_alpha is not None:
            payload["trial_alpha"] = self.trial_alpha
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        candidate_arrays: Mapping[str, np.ndarray] | None = None,
        trial_arrays: Mapping[str, np.ndarray] | None = None,
    ) -> "StabilizedRelaxationTrial":
        if not isinstance(payload, Mapping):
            raise ValueError("relaxation trial payload must be a mapping")
        required = {
            "omega",
            "complete",
            "hard_pass",
            "globalization_pass",
            "coupled_pass",
            "selected",
            "merit",
            "sufficient_decrease_limit",
            "failure_stage",
            "failure_reason",
            "candidate_diagnostics",
            "d_trial",
            "q_after",
            "d_trial_digest",
            "q_after_digest",
            "candidate_kind",
            "candidate_fp_error",
            "candidate_fp_scale",
            "candidate_damage_residual_scale",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"relaxation trial is missing fields: {missing}")
        optional = {"trial_alpha"}
        unexpected = sorted(set(payload) - required - optional)
        if unexpected:
            raise ValueError(f"relaxation trial contains unknown fields: {unexpected}")
        candidate_payload = payload["candidate_diagnostics"]
        candidate = (
            None
            if candidate_payload is None
            else _s_diagnostic_from_payload(
                candidate_payload,
                "relaxation trial candidate diagnostics",
                array_values=candidate_arrays,
            )
        )
        trial_d_trial = (
            None
            if trial_arrays is None
            else trial_arrays.get("d_trial")
        )
        trial_q_after = (
            None
            if trial_arrays is None
            else trial_arrays.get("q_after")
        )
        if payload["d_trial"] is not None or payload["q_after"] is not None:
            raise ValueError(
                "relaxation trial numerical arrays must be null in compact stabilized JSON"
            )
        return cls(
            omega=payload["omega"],
            complete=payload["complete"],
            hard_pass=payload["hard_pass"],
            globalization_pass=payload["globalization_pass"],
            coupled_pass=payload["coupled_pass"],
            selected=payload["selected"],
            merit=payload["merit"],
            sufficient_decrease_limit=payload["sufficient_decrease_limit"],
            failure_stage=payload["failure_stage"],
            failure_reason=payload["failure_reason"],
            candidate_diagnostics=candidate,
            d_trial=trial_d_trial,
            q_after=trial_q_after,
            d_trial_digest=payload["d_trial_digest"],
            q_after_digest=payload["q_after_digest"],
            candidate_kind=payload["candidate_kind"],
            trial_alpha=payload.get("trial_alpha"),
            candidate_fp_error=payload["candidate_fp_error"],
            candidate_fp_scale=payload["candidate_fp_scale"],
            candidate_damage_residual_scale=payload[
                "candidate_damage_residual_scale"
            ],
        )

@dataclass(frozen=True)
class StabilizedRelaxationSweepDiagnostics:
    """A complete or explicitly incomplete stabilized sweep."""

    base_diagnostics: EngineeringGripDiagnostics
    complete_sweep: bool
    passed: bool
    failure_stage: str
    failure_reason: str
    d_solved: np.ndarray | None
    d_trial: np.ndarray | None
    omega: float | None
    backtrack_count: int
    reference_merit: float | None
    merit: float | None
    sufficient_decrease_limit: float | None
    fixed_point_error: float | None
    fixed_point_scale: float | None
    fixed_point_pass: bool
    damage_equation_pass: bool
    globalization_pass: bool
    coupled_pass: bool
    trials: tuple[StabilizedRelaxationTrial, ...]
    anderson_history_size: int | None = None
    anderson_qr_rank: int | None = None
    anderson_coefficient_l2: float | None = None
    anderson_regularization_mu: float | None = None
    anderson_current_residual_l2: float | None = None
    anderson_regularization_lambda: float | None = None
    anderson_unregularized_ls_residual_l2: float | None = None
    anderson_selection: str | None = None
    trial_alpha: float | None = None
    anderson_fallback_reason: str | None = None
    anderson_restart_reason: str | None = None
    candidate_fp_error: float | None = None
    candidate_fp_scale: float | None = None
    candidate_damage_residual_scale: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_diagnostics, EngineeringGripDiagnostics):
            raise TypeError("stabilized sweep base diagnostics have the wrong type")
        d_shape = self.base_diagnostics.damage_before.shape
        for name in ("d_solved", "d_trial"):
            array = _s_optional_f64(
                getattr(self, name), f"stabilized sweep {name}", d_shape
            )
            object.__setattr__(self, name, array)
        if self.d_trial is not None and self.base_diagnostics.d_next is not None:
            if not np.array_equal(self.d_trial, self.base_diagnostics.d_next):
                raise ValueError("stabilized d_trial is not paired with base d_next")
        omega = _s_optional_scalar(self.omega, "stabilized sweep omega")
        if omega is not None and not 0.0 < omega <= 1.0:
            raise ValueError("stabilized sweep omega must lie in (0, 1]")
        reference = _s_optional_scalar(self.reference_merit, "stabilized reference merit")
        merit = _s_optional_scalar(self.merit, "stabilized merit")
        limit = _s_optional_scalar(
            self.sufficient_decrease_limit, "stabilized sufficient limit"
        )
        fixed_error = _s_optional_scalar(
            self.fixed_point_error, "stabilized fixed-point error"
        )
        fixed_scale = _s_optional_scalar(
            self.fixed_point_scale, "stabilized fixed-point scale"
        )
        candidate_fp_error = _s_optional_scalar(
            self.candidate_fp_error, "stabilized candidate fixed-point error"
        )
        candidate_fp_scale = _s_optional_scalar(
            self.candidate_fp_scale, "stabilized candidate fixed-point scale"
        )
        candidate_residual_scale = _s_optional_scalar(
            self.candidate_damage_residual_scale,
            "stabilized candidate damage residual scale",
        )
        coefficient_norm = _s_optional_scalar(
            self.anderson_coefficient_l2,
            "stabilized Anderson coefficient l2",
        )
        regularization_mu = _s_optional_scalar(
            self.anderson_regularization_mu,
            "stabilized Anderson regularization mu",
        )
        current_residual_l2 = _s_optional_scalar(
            self.anderson_current_residual_l2,
            "stabilized Anderson current residual l2",
        )
        regularization_lambda = _s_optional_scalar(
            self.anderson_regularization_lambda,
            "stabilized Anderson regularization lambda",
        )
        unregularized_ls_residual_l2 = _s_optional_scalar(
            self.anderson_unregularized_ls_residual_l2,
            "stabilized Anderson unregularized least-squares residual l2",
        )
        for name, value in (
            ("reference_merit", reference),
            ("merit", merit),
            ("sufficient_decrease_limit", limit),
            ("fixed_point_error", fixed_error),
            ("fixed_point_scale", fixed_scale),
            ("candidate_fp_error", candidate_fp_error),
            ("candidate_fp_scale", candidate_fp_scale),
            ("candidate_damage_residual_scale", candidate_residual_scale),
            ("anderson_coefficient_l2", coefficient_norm),
            ("anderson_current_residual_l2", current_residual_l2),
            ("regularization_lambda", regularization_lambda),
            (
                "anderson_unregularized_ls_residual_l2",
                unregularized_ls_residual_l2,
            ),
        ):
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if regularization_mu is not None and regularization_mu <= 0.0:
            raise ValueError("stabilized Anderson regularization mu must be positive")
        if regularization_lambda is not None and current_residual_l2 is None:
            raise ValueError(
                "stabilized Anderson lambda lacks its current residual l2"
            )
        if regularization_lambda is not None and regularization_mu is None:
            raise ValueError("stabilized Anderson lambda lacks its regularization mu")
        if (
            regularization_lambda is not None
            and regularization_mu is not None
            and current_residual_l2 is not None
            and not _s_anderson_scalar_relation_close(
                regularization_lambda,
                regularization_mu * current_residual_l2**2,
            )
        ):
            raise ValueError(
                "stabilized Anderson lambda is not paired with mu and current residual l2"
            )
        backtracks = _e_integer(
            self.backtrack_count, "stabilized backtrack_count"
        )
        trials = tuple(self.trials)
        if any(not isinstance(trial, StabilizedRelaxationTrial) for trial in trials):
            raise TypeError("stabilized sweep trials contain a wrong type")
        if trials:
            alpha_trials = sum(trial.trial_alpha is not None for trial in trials)
            expected_backtracks = (
                max(0, alpha_trials - 1)
                if alpha_trials
                else len(trials) - 1
            )
            if backtracks != expected_backtracks:
                raise ValueError("stabilized sweep backtrack count is not paired")
            alpha_positions = [
                index
                for index, trial in enumerate(trials)
                if trial.trial_alpha is not None
            ]
            if alpha_trials and alpha_positions != list(range(alpha_trials)):
                raise ValueError(
                    "stabilized sweep alpha trials are not a canonical prefix"
                )
            for index in alpha_positions:
                expected_alpha = _s_anderson_backtracking_alpha_for_index(
                    index, "stabilized sweep trial alpha index"
                )
                if trials[index].trial_alpha != expected_alpha:
                    raise ValueError(
                        "stabilized sweep trial alpha is not paired with its index"
                    )
        history_size = _s_optional_nonnegative_integer(
            self.anderson_history_size, "Anderson history size"
        )
        qr_rank = _s_optional_nonnegative_integer(
            self.anderson_qr_rank, "Anderson QR rank"
        )
        trial_alpha = _s_optional_scalar(self.trial_alpha, "Anderson trial alpha")
        if trial_alpha is not None and not 0.0 < trial_alpha <= 1.0:
            raise ValueError("Anderson trial alpha must lie in (0, 1]")
        if trial_alpha is not None:
            expected_alpha = _s_anderson_backtracking_alpha_for_index(
                backtracks, "Anderson sweep backtrack_count"
            )
            if trial_alpha != expected_alpha:
                raise ValueError(
                    "Anderson trial alpha is not paired with backtrack_count"
                )
        selection = self.anderson_selection
        if selection is not None and selection not in _s_ANDERSON_SELECTIONS:
            raise ValueError("stabilized Anderson selection is invalid")
        for name in ("anderson_fallback_reason", "anderson_restart_reason"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None")
        if selection == "accelerated" and self.anderson_fallback_reason is not None:
            raise ValueError("accelerated Anderson sweep retains a fallback reason")
        if selection == "picard_fallback" and self.anderson_fallback_reason is None:
            raise ValueError("Anderson Picard fallback lacks a fallback reason")
        if history_size is not None and qr_rank is not None and qr_rank > history_size:
            raise ValueError("Anderson QR rank exceeds the history size")
        if self.anderson_selection != "accelerated" and any(
            value is not None
            for value in (
                candidate_fp_error,
                candidate_fp_scale,
                candidate_residual_scale,
            )
        ):
            raise ValueError(
                "non-accelerated stabilized sweep retains Anderson candidate metrics"
            )
        selected = [trial for trial in trials if trial.selected]
        if len(selected) > 1:
            raise ValueError("stabilized sweep selects more than one trial")
        selected_trial = selected[0] if selected else None
        if self.complete_sweep:
            if not self.base_diagnostics.complete_sweep:
                raise ValueError("complete stabilized sweep needs complete base diagnostics")
            if self.d_solved is None or self.d_trial is None:
                raise ValueError("complete stabilized sweep is missing relaxation fields")
            if reference is None or fixed_error is None:
                raise ValueError("complete stabilized sweep is missing merit fields")
            if selected_trial is None:
                raise ValueError("complete stabilized sweep has no selected trial")
            if merit is None and selected_trial.candidate_kind != "picard_fallback":
                raise ValueError("complete accelerated sweep is missing selection merit")
            if selected_trial.omega != omega:
                raise ValueError("selected trial omega is not paired with sweep omega")
            if selected_trial.candidate_kind == "anderson_accelerated":
                if self.anderson_selection != "accelerated":
                    raise ValueError("accelerated trial is missing Anderson selection")
                if self.sufficient_decrease_limit is None:
                    raise ValueError("accelerated sweep is missing sufficient limit")
                if any(
                    value is None
                    for value in (
                        candidate_fp_error,
                        candidate_fp_scale,
                        candidate_residual_scale,
                    )
                ):
                    raise ValueError(
                        "accelerated sweep is missing candidate metrics"
                    )
                if (
                    candidate_fp_error != selected_trial.candidate_fp_error
                    or candidate_fp_scale != selected_trial.candidate_fp_scale
                    or candidate_residual_scale
                    != selected_trial.candidate_damage_residual_scale
                ):
                        raise ValueError(
                            "accelerated sweep candidate metrics are not paired"
                        )
                if trial_alpha != selected_trial.trial_alpha:
                    raise ValueError("accelerated sweep trial alpha is not paired")
            elif selected_trial.candidate_kind == "picard_fallback":
                if self.anderson_selection != "picard_fallback":
                    raise ValueError("Picard fallback is missing Anderson selection")
                if self.sufficient_decrease_limit is not None:
                    raise ValueError("Picard fallback retains a sufficient limit")
                if trial_alpha is not None:
                    raise ValueError("Picard fallback retains trial alpha")
        elif trial_alpha is not None:
            if not trials or trials[-1].trial_alpha != trial_alpha:
                raise ValueError("unselected stabilized sweep trial alpha is not paired")
        if self.passed and not (self.complete_sweep and self.coupled_pass and self.globalization_pass):
            raise ValueError("passed stabilized sweep does not pass all stabilized gates")
        if not self.complete_sweep and self.passed:
            raise ValueError("incomplete stabilized sweep cannot pass")
        if not isinstance(self.failure_stage, str) or not isinstance(
            self.failure_reason, str
        ):
            raise TypeError("stabilized sweep failure fields must be strings")
        if self.complete_sweep and not self.passed and not self.failure_reason:
            raise ValueError("failed complete stabilized sweep needs a failure reason")
        if not self.complete_sweep and (not self.failure_stage or not self.failure_reason):
            raise ValueError("incomplete stabilized sweep needs stage and reason")
        object.__setattr__(self, "complete_sweep", bool(self.complete_sweep))
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "backtrack_count", backtracks)
        object.__setattr__(self, "reference_merit", reference)
        object.__setattr__(self, "merit", merit)
        object.__setattr__(self, "sufficient_decrease_limit", limit)
        object.__setattr__(self, "fixed_point_error", fixed_error)
        object.__setattr__(self, "fixed_point_scale", fixed_scale)
        object.__setattr__(self, "anderson_history_size", history_size)
        object.__setattr__(self, "anderson_qr_rank", qr_rank)
        object.__setattr__(self, "anderson_coefficient_l2", coefficient_norm)
        object.__setattr__(self, "anderson_regularization_mu", regularization_mu)
        object.__setattr__(self, "anderson_current_residual_l2", current_residual_l2)
        object.__setattr__(self, "anderson_regularization_lambda", regularization_lambda)
        object.__setattr__(
            self,
            "anderson_unregularized_ls_residual_l2",
            unregularized_ls_residual_l2,
        )
        object.__setattr__(self, "anderson_selection", selection)
        object.__setattr__(self, "trial_alpha", trial_alpha)
        object.__setattr__(self, "candidate_fp_error", candidate_fp_error)
        object.__setattr__(self, "candidate_fp_scale", candidate_fp_scale)
        object.__setattr__(
            self,
            "candidate_damage_residual_scale",
            candidate_residual_scale,
        )
        for name in (
            "fixed_point_pass",
            "damage_equation_pass",
            "globalization_pass",
            "coupled_pass",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        object.__setattr__(self, "trials", trials)

    def __getattr__(self, name: str) -> Any:
        # Keep base diagnostic field names available to workflow/report code,
        # while the explicit stabilized fields above remain authoritative.
        base = object.__getattribute__(self, "base_diagnostics")
        return getattr(base, name)

    @property
    def d_next(self) -> np.ndarray | None:
        return self.d_trial

    @property
    def available_fields(self) -> tuple[str, ...]:
        fields = set(self.base_diagnostics.available_fields)
        if self.d_solved is not None:
            fields.add("d_solved")
        if self.d_trial is not None:
            fields.add("d_trial")
        return tuple(sorted(fields))

    @property
    def gate_passes(self) -> dict[str, bool]:
        return {
            **self.base_diagnostics.gate_passes,
            "damage_fixed_point": self.fixed_point_pass,
            "damage_equation": self.damage_equation_pass,
            "globalization": self.globalization_pass,
            "coupled": self.coupled_pass,
        }

    @property
    def sweep_index(self) -> int:
        return self.base_diagnostics.sweep_index

    @property
    def target_u_bar(self) -> float:
        return self.base_diagnostics.target_u_bar

    def payload(self) -> dict[str, Any]:
        payload = _s_compact_diagnostic_payload(self.base_diagnostics)
        payload.update(
            {
                "stabilized_diagnostic": True,
                "sweep_index": self.sweep_index,
                "target_u_bar": self.target_u_bar,
                "complete_sweep": self.complete_sweep,
                "passed": self.passed,
                "failure_stage": self.failure_stage,
                "failure_reason": self.failure_reason,
                "base_failure_stage": self.base_diagnostics.failure_stage,
                "base_failure_reason": self.base_diagnostics.failure_reason,
                "base_passed": self.base_diagnostics.passed,
                "base_complete_sweep": self.base_diagnostics.complete_sweep,
                "base_gate_passes": self.base_diagnostics.gate_passes,
                "base_available_fields": list(self.base_diagnostics.available_fields),
                "gate_passes": self.gate_passes,
                "available_fields": list(self.available_fields),
                # These arrays are stored once in the typed sweep arrays.
                "d_solved": None,
                "d_trial": None,
                "omega": self.omega,
                "backtrack_count": self.backtrack_count,
                "reference_merit": self.reference_merit,
                "merit": self.merit,
                "sufficient_decrease_limit": self.sufficient_decrease_limit,
                "fixed_point_error": self.fixed_point_error,
                "fixed_point_scale": self.fixed_point_scale,
                "fixed_point_pass": self.fixed_point_pass,
                "damage_equation_pass": self.damage_equation_pass,
                "globalization_pass": self.globalization_pass,
                "coupled_pass": self.coupled_pass,
                 "anderson_history_size": self.anderson_history_size,
                 "anderson_qr_rank": self.anderson_qr_rank,
                "anderson_selection": self.anderson_selection,
                "anderson_fallback_reason": self.anderson_fallback_reason,
                "anderson_restart_reason": self.anderson_restart_reason,
                "candidate_fp_error": self.candidate_fp_error,
                "candidate_fp_scale": self.candidate_fp_scale,
                 "candidate_damage_residual_scale": self.candidate_damage_residual_scale,
                 "relaxation_trials": [trial.payload() for trial in self.trials],
             }
         )
        if self.anderson_regularization_mu is not None:
            payload.update(
                {
                    "anderson_regularization_mu": self.anderson_regularization_mu,
                    "anderson_current_residual_l2": self.anderson_current_residual_l2,
                    "anderson_regularization_lambda": self.anderson_regularization_lambda,
                    "anderson_unregularized_ls_residual_l2": (
                        self.anderson_unregularized_ls_residual_l2
                    ),
                    "anderson_coefficient_l2": self.anderson_coefficient_l2,
                }
            )
        if self.trial_alpha is not None:
            payload["trial_alpha"] = self.trial_alpha
        return payload

@dataclass(frozen=True)
class StabilizedEngineeringGripAcceptedStep:
    """One committed stabilized load increment.

    Strict steps pass every stabilized gate.  Warning steps are committed only
    by the separate final-complete-sweep policy and retain the exact failed
    non-hard gate list and raw safety diagnostics.
    """

    load_start: float
    load_target: float
    increment: float
    state: EngineeringGripState
    diagnostics: StabilizedRelaxationSweepDiagnostics
    stagger_iterations: int
    cutback_count: int
    acceptance_status: str
    bypassed_gates: tuple[str, ...] = ()
    warning_range_excess: float | None = None
    warning_healing_magnitude: float | None = None
    acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE

    def __post_init__(self) -> None:
        start = _s_scalar(self.load_start, "stabilized accepted load_start")
        target = _s_scalar(self.load_target, "stabilized accepted load_target")
        increment = _s_scalar(self.increment, "stabilized accepted increment")
        if target <= start or target - start != increment:
            raise ValueError("stabilized accepted load interval is invalid")
        if not isinstance(self.state, EngineeringGripState):
            raise TypeError("stabilized accepted state has the wrong type")
        if not isinstance(self.diagnostics, StabilizedRelaxationSweepDiagnostics):
            raise TypeError("stabilized accepted diagnostics have the wrong type")
        if not self.diagnostics.passed and self.acceptance_status != STABILIZED_WARNING_ACCEPTANCE_STATUS:
            raise ValueError("stabilized accepted step needs a passing diagnostic")
        mode = _s_validate_stabilized_acceptance_mode(self.acceptance_mode)
        bypassed = tuple(self.bypassed_gates)
        if any(
            not isinstance(name, str) or name not in STABILIZED_WARNING_GATE_NAMES
            for name in bypassed
        ) or len(set(bypassed)) != len(bypassed):
            raise ValueError("stabilized accepted bypassed gates are invalid")
        canonical_bypassed = tuple(
            name for name in STABILIZED_WARNING_GATE_NAMES if name in bypassed
        )
        if bypassed != canonical_bypassed:
            raise ValueError("stabilized accepted bypassed gates are not canonical")
        if mode == STABILIZED_ACCEPTANCE_MODE:
            if self.acceptance_status != STABILIZED_ACCEPTANCE_STATUS:
                raise ValueError("strict stabilized accepted status is invalid")
            if (
                bypassed
                or self.warning_range_excess is not None
                or self.warning_healing_magnitude is not None
            ):
                raise ValueError("strict stabilized steps cannot retain warning fields")
        elif self.acceptance_status == STABILIZED_ACCEPTANCE_STATUS:
            if not self.diagnostics.passed or bypassed:
                raise ValueError("strict-passed warning-mode step has warning fields")
            if (
                self.warning_range_excess is not None
                or self.warning_healing_magnitude is not None
            ):
                raise ValueError("strict-passed warning-mode step cannot retain warning fields")
        elif self.acceptance_status == STABILIZED_WARNING_ACCEPTANCE_STATUS:
            if self.diagnostics.passed:
                raise ValueError("warning-accepted step must retain passed=False")
            if not bypassed:
                raise ValueError("warning-accepted step needs failed gates")
            expected = _s_stabilized_warning_failed_gates(self.diagnostics)
            if bypassed != expected:
                raise ValueError("warning-accepted bypassed gates are not exact")
            if self.warning_range_excess is None or self.warning_healing_magnitude is None:
                raise ValueError("warning-accepted step needs raw safety diagnostics")
            expected_range, expected_healing = _s_stabilized_warning_excesses(
                self.diagnostics
            )
            if (
                self.warning_range_excess != expected_range
                or self.warning_healing_magnitude != expected_healing
            ):
                raise ValueError("warning-accepted raw safety diagnostics are not paired")
        else:
            raise ValueError("stabilized accepted status is invalid")
        if self.state.u_bar != target:
            raise ValueError("stabilized accepted state load is not paired")
        if self.diagnostics.q_after is None or not np.array_equal(
            self.state.q, self.diagnostics.q_after
        ):
            raise ValueError("stabilized accepted state displacement is not paired")
        if self.diagnostics.d_trial is None or self.diagnostics.H_next is None:
            raise ValueError("stabilized accepted state fields are missing")
        if not np.array_equal(self.state.d, self.diagnostics.d_trial) or not np.array_equal(
            self.state.H, self.diagnostics.H_next
        ):
            raise ValueError("stabilized accepted damage/history fields are not paired")
        object.__setattr__(self, "load_start", start)
        object.__setattr__(self, "load_target", target)
        object.__setattr__(self, "increment", increment)
        object.__setattr__(
            self,
            "stagger_iterations",
            _e_integer(
                self.stagger_iterations, "stabilized stagger_iterations", positive=True
            ),
        )
        object.__setattr__(
            self, "cutback_count", _e_integer(self.cutback_count, "stabilized cutback_count")
        )
        object.__setattr__(self, "acceptance_mode", mode)
        object.__setattr__(self, "bypassed_gates", canonical_bypassed)
        object.__setattr__(
            self,
            "warning_range_excess",
            None
            if self.warning_range_excess is None
            else _s_scalar(
                self.warning_range_excess,
                "stabilized warning range excess",
            ),
        )
        object.__setattr__(
            self,
            "warning_healing_magnitude",
            None
            if self.warning_healing_magnitude is None
            else _s_scalar(
                self.warning_healing_magnitude,
                "stabilized warning healing magnitude",
            ),
        )

    @property
    def u_bar(self) -> float:
        return self.load_target

@dataclass(frozen=True)
class StabilizedEngineeringGripAttempt:
    """One load target, including relaxation trials and rollback identity."""

    cutback_count: int
    load_start: float
    load_target: float
    accepted: bool
    acceptance_status: str | None
    failure_stage: str | None
    failure_reason: str | None
    exception_type: str | None
    committed_before_digest: str
    committed_after_digest: str
    sweeps: tuple[StabilizedRelaxationSweepDiagnostics, ...]
    diagnostics: StabilizedRelaxationSweepDiagnostics
    bypassed_gates: tuple[str, ...] = ()
    warning_range_excess: float | None = None
    warning_healing_magnitude: float | None = None
    acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE

    def __post_init__(self) -> None:
        start = _s_scalar(self.load_start, "stabilized attempt load_start")
        target = _s_scalar(self.load_target, "stabilized attempt load_target")
        if target <= start:
            raise ValueError("stabilized attempt target must exceed load_start")
        before = _s_digest(self.committed_before_digest, "stabilized attempt before digest")
        after = _s_digest(self.committed_after_digest, "stabilized attempt after digest")
        sweeps = tuple(self.sweeps)
        if not sweeps or sweeps[-1] is not self.diagnostics:
            raise ValueError("stabilized attempt must retain its final sweep object")
        if any(not isinstance(item, StabilizedRelaxationSweepDiagnostics) for item in sweeps):
            raise TypeError("stabilized attempt sweeps contain a wrong type")
        mode = _s_validate_stabilized_acceptance_mode(self.acceptance_mode)
        bypassed = tuple(self.bypassed_gates)
        if any(
            not isinstance(name, str) or name not in STABILIZED_WARNING_GATE_NAMES
            for name in bypassed
        ) or len(set(bypassed)) != len(bypassed):
            raise ValueError("stabilized attempt bypassed gates are invalid")
        canonical_bypassed = tuple(
            name for name in STABILIZED_WARNING_GATE_NAMES if name in bypassed
        )
        if bypassed != canonical_bypassed:
            raise ValueError("stabilized attempt bypassed gates are not canonical")
        for index, sweep in enumerate(sweeps):
            if sweep.sweep_index != index or sweep.target_u_bar != target:
                raise ValueError("stabilized attempt sweep pairing is invalid")
            if not sweep.complete_sweep and index != len(sweeps) - 1:
                raise ValueError("incomplete stabilized sweep must be final")
            if sweep.passed and index != len(sweeps) - 1:
                raise ValueError("passing stabilized sweep must be final")
        if self.accepted:
            if self.failure_stage or self.failure_reason or self.exception_type:
                raise ValueError("accepted stabilized attempt retains failure fields")
            if before == after:
                raise ValueError("accepted stabilized attempt did not change state digest")
            if self.acceptance_status == STABILIZED_ACCEPTANCE_STATUS:
                if not self.diagnostics.passed or canonical_bypassed:
                    raise ValueError(
                        "strict-passed stabilized attempt has warning fields"
                    )
                if (
                    self.warning_range_excess is not None
                    or self.warning_healing_magnitude is not None
                ):
                    raise ValueError(
                        "strict-passed stabilized attempt has raw warning fields"
                    )
            elif self.acceptance_status == STABILIZED_WARNING_ACCEPTANCE_STATUS:
                if mode != STABILIZED_WARNING_ACCEPTANCE_MODE:
                    raise ValueError(
                        "warning-accepted stabilized attempt has strict mode"
                    )
                if self.diagnostics.passed or not canonical_bypassed:
                    raise ValueError(
                        "warning-accepted stabilized attempt has invalid gates"
                    )
                if canonical_bypassed != _s_stabilized_warning_failed_gates(
                    self.diagnostics
                ):
                    raise ValueError(
                        "warning-accepted stabilized attempt gates are not exact"
                    )
                if (
                    self.warning_range_excess is None
                    or self.warning_healing_magnitude is None
                ):
                    raise ValueError(
                        "warning-accepted stabilized attempt lacks raw excesses"
                    )
                expected_range, expected_healing = _s_stabilized_warning_excesses(
                    self.diagnostics
                )
                if (
                    self.warning_range_excess != expected_range
                    or self.warning_healing_magnitude != expected_healing
                ):
                    raise ValueError(
                        "warning-accepted stabilized attempt excesses are not paired"
                    )
            else:
                raise ValueError("accepted stabilized attempt status is invalid")
        else:
            if self.acceptance_status is not None:
                raise ValueError("rejected stabilized attempt has an acceptance status")
            if not self.failure_stage or not self.failure_reason:
                raise ValueError("rejected stabilized attempt needs failure fields")
            if self.exception_type != StabilizedEngineeringGripSweepFailure.__name__:
                raise ValueError("rejected stabilized attempt exception type is invalid")
            if self.diagnostics.passed:
                raise ValueError("rejected stabilized attempt diagnostic passes")
            if before != after:
                raise ValueError("rejected stabilized attempt changed committed state")
            if canonical_bypassed:
                raise ValueError("rejected stabilized attempt cannot bypass gates")
            if (
                self.warning_range_excess is not None
                or self.warning_healing_magnitude is not None
            ):
                raise ValueError("rejected stabilized attempt cannot retain warning fields")
        object.__setattr__(self, "cutback_count", _e_integer(self.cutback_count, "stabilized attempt cutback_count"))
        object.__setattr__(self, "load_start", start)
        object.__setattr__(self, "load_target", target)
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "committed_before_digest", before)
        object.__setattr__(self, "committed_after_digest", after)
        object.__setattr__(self, "sweeps", sweeps)
        object.__setattr__(self, "acceptance_mode", mode)
        object.__setattr__(self, "bypassed_gates", canonical_bypassed)
        object.__setattr__(
            self,
            "warning_range_excess",
            None
            if self.warning_range_excess is None
            else _s_scalar(
                self.warning_range_excess,
                "stabilized attempt warning range excess",
            ),
        )
        object.__setattr__(
            self,
            "warning_healing_magnitude",
            None
            if self.warning_healing_magnitude is None
            else _s_scalar(
                self.warning_healing_magnitude,
                "stabilized attempt warning healing magnitude",
            ),
        )

class StabilizedEngineeringGripSweepFailure(RuntimeError):
    """Typed numerical failure eligible for load cutback."""

    def __init__(self, stage: str, message: str, diagnostics: StabilizedRelaxationSweepDiagnostics):
        super().__init__(message)
        self.stage = stage
        self.diagnostics = diagnostics

class StabilizedEngineeringGripContinuationFailure(RuntimeError):
    """Typed minimum-increment/cutback failure with immutable audit records."""

    def __init__(
        self,
        message: str,
        last_committed_state: EngineeringGripState,
        attempts: tuple[StabilizedEngineeringGripAttempt, ...],
        work_check: EngineeringGripWorkCheck,
        *,
        failure_code: str = STABILIZED_FAILURE_CODE,
        acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE,
    ):
        super().__init__(message)
        if not isinstance(failure_code, str) or not failure_code:
            raise ValueError("stabilized continuation failure_code must be non-empty")
        self.last_committed_state = last_committed_state
        self.attempts = attempts
        self.work_check = work_check
        self.failure_code = failure_code
        self.acceptance_mode = _s_validate_stabilized_acceptance_mode(acceptance_mode)



@dataclass(frozen=True)
class StabilizedEngineeringGripResult:
    """Experimental result; never promoted to validated evidence."""

    model: EngineeringGripModel
    config: StabilizedEngineeringGripConfig
    work_check: EngineeringGripWorkCheck
    initial_state: EngineeringGripState
    states: tuple[EngineeringGripState, ...]
    accepted_steps: tuple[StabilizedEngineeringGripAcceptedStep, ...]
    attempts: tuple[StabilizedEngineeringGripAttempt, ...]
    completion_status: str = STABILIZED_COMPLETION_STATUS
    _acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE

    def __post_init__(self) -> None:
        self.validate()
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "accepted_steps", tuple(self.accepted_steps))
        object.__setattr__(self, "attempts", tuple(self.attempts))

    @property
    def acceptance_mode(self) -> str:
        return _s_validate_stabilized_acceptance_mode(self._acceptance_mode)

    @property
    def acceptance_status(self) -> str:
        return _s_stabilized_contract(
            self.acceptance_mode,
            failure=False,
                    )[
            "acceptance_status"
        ]

    @property
    def validated(self) -> bool:
        return False

    @property
    def experimental(self) -> bool:
        return True

    @property
    def final_state(self) -> EngineeringGripState:
        return self.states[-1]

    def validate(self) -> None:
        if not isinstance(self.model, EngineeringGripModel) or not isinstance(
            self.config, StabilizedEngineeringGripConfig
        ):
            raise TypeError("stabilized result model/config types are invalid")
        if not isinstance(self.work_check, EngineeringGripWorkCheck):
            raise TypeError("stabilized result work_check has the wrong type")
        _e_validate_work_check_against_model_config(
            self.work_check, self.model, self.config.base
        )
        if not self.work_check.passed:
            raise ValueError("stabilized result work check did not pass")
        mode = self.acceptance_mode
        expected_contract = _s_stabilized_contract(
            mode, failure=False
        )
        if self.completion_status != expected_contract["completion_status"]:
            raise ValueError("stabilized result completion status is invalid")
        states = tuple(self.states)
        steps = tuple(self.accepted_steps)
        attempts = tuple(self.attempts)
        if not states or states[0] is not self.initial_state:
            raise ValueError("stabilized result must begin with the exact initial state")
        if len(states) != len(steps) + 1:
            raise ValueError("stabilized state/step counts are inconsistent")
        initial = states[0]
        if (
            initial.u_bar != 0.0
            or not np.array_equal(initial.q, np.zeros_like(initial.q))
            or not np.array_equal(initial.d, np.zeros_like(initial.d))
            or not np.array_equal(initial.H, np.zeros_like(initial.H))
        ):
            raise ValueError("stabilized initial state must be exact zero")
        if initial.q.shape != (self.model.n_dofs,) or initial.d.shape != (self.model.n_material,):
            raise ValueError("stabilized initial state shape is inconsistent")
        if states[-1].u_bar != self.config.base.load_final:
            raise ValueError("stabilized result did not reach exact load_final")
        for state in states:
            if state.q.shape != (self.model.n_dofs,) or state.d.shape != (self.model.n_material,):
                raise ValueError("stabilized state shape does not match model")
            if not np.all(np.isfinite(state.q)) or not np.all(np.isfinite(state.d)):
                raise ValueError("stabilized committed state contains non-finite fields")
            if np.any(state.H < 0.0) or not np.all(np.isfinite(state.H)):
                raise ValueError("stabilized committed history is invalid")
            _e_check_exact_constraints(
                state.q, self.model.constraints, state.u_bar, "stabilized committed state"
            )
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, StabilizedEngineeringGripAcceptedStep):
                raise TypeError("stabilized accepted steps contain a wrong type")
            if step.state is not states[index]:
                raise ValueError("stabilized accepted step/state object pairing is invalid")
            if step.load_start != states[index - 1].u_bar or step.load_target != states[index].u_bar:
                raise ValueError("stabilized accepted load/state pairing is invalid")
            diag = step.diagnostics
            if step.acceptance_mode != mode:
                raise ValueError("stabilized accepted step mode is not paired")
            if (
                (step.acceptance_status == STABILIZED_ACCEPTANCE_STATUS and not diag.passed)
                or diag.d_trial is None
                or diag.d_solved is None
            ):
                raise ValueError("stabilized accepted step lacks a valid fixed-point audit")
            if (
                mode == STABILIZED_ACCEPTANCE_MODE
                and (not diag.damage_range_pass or not diag.healing_pass)
            ):
                raise ValueError(
                    "accepted step lacks strict range/healing gates"
                )
            selected_trials = [trial for trial in diag.trials if trial.selected]
            if len(selected_trials) != 1:
                raise ValueError("stabilized accepted step has an invalid selected trial")
            selected_trial = selected_trials[0]
            if selected_trial.candidate_kind == "anderson_accelerated":
                if selected_trial.d_trial is None or diag.d_trial is None:
                    raise ValueError("accelerated stabilized damage field is missing")
                if not np.array_equal(selected_trial.d_trial, diag.d_trial):
                    raise ValueError("accelerated stabilized damage field is forged")
            else:
                if diag.omega is None:
                    raise ValueError("Picard stabilized damage field is missing omega")
                expected_trial = diag.d_iter + diag.omega * (
                    diag.d_solved - diag.d_iter
                )
                if not np.array_equal(expected_trial, diag.d_trial):
                    raise ValueError("stabilized relaxed damage field is forged")
        if not attempts:
            raise ValueError("stabilized result must retain attempt audits")
        current = 0
        accepted_index = 0
        for attempt in attempts:
            if attempt.acceptance_mode != mode:
                raise ValueError("stabilized attempt mode is not paired")
            if attempt.load_start != states[current].u_bar:
                raise ValueError("stabilized attempt load start is not paired")
            if attempt.committed_before_digest != states[current].digest:
                raise ValueError("stabilized attempt before digest is not paired")
            if attempt.accepted:
                if accepted_index >= len(steps) or attempt.diagnostics is not steps[accepted_index].diagnostics:
                    raise ValueError("stabilized accepted attempt is not paired")
                current += 1
                if attempt.committed_after_digest != states[current].digest:
                    raise ValueError("stabilized accepted attempt after digest is not paired")
                accepted_index += 1
            elif attempt.committed_after_digest != states[current].digest:
                raise ValueError("stabilized rejected attempt changed committed state")
        if current != len(states) - 1 or accepted_index != len(steps):
            raise ValueError("stabilized attempt audit does not reach final state")
        object.__setattr__(self, "_acceptance_mode", mode)

def _s_normalized(error: float, gate: GateTolerance, scale: float, name: str) -> float:
    limit = gate.limit(scale)
    if limit <= 0.0:
        raise ValueError(f"{name} has a zero merit limit")
    value = abs(float(error)) / limit
    if not np.isfinite(value):
        raise FloatingPointError(f"{name} merit ratio is non-finite")
    return value

def _s_reference_merit_components(
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    d_iter: np.ndarray,
    d_solved: np.ndarray,
    H_used: np.ndarray,
) -> dict[str, float]:
    base = config.base
    rho_iter, _, _, _, residual_scale = _e_damage_residuals(
        model, d_iter, H_used, H_used
    )
    fixed_error = float(np.max(np.abs(d_solved - d_iter)))
    fixed_scale = max(float(np.max(np.abs(d_solved))), float(np.max(np.abs(d_iter))))
    components = {
        "reference_fixed_point": _s_normalized(
            fixed_error, base.damage, fixed_scale, "reference fixed-point"
        ),
        "reference_damage_residual": _s_normalized(
            float(np.max(np.abs(rho_iter))),
            base.damage_residual,
            residual_scale,
            "reference damage residual",
        ),
    }
    if not all(np.isfinite(value) for value in components.values()):
        raise FloatingPointError("reference merit component is non-finite")
    return components

def _s_reference_merit(
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    d_iter: np.ndarray,
    d_solved: np.ndarray,
    H_used: np.ndarray,
) -> float:
    components = _s_reference_merit_components(
        model, config, d_iter, d_solved, H_used
    )
    merit = max(components.values())
    if not np.isfinite(merit):
        raise FloatingPointError("reference merit is non-finite")
    return float(merit)

def _s_candidate_merit_components(
    diagnostics: EngineeringGripDiagnostics,
    fixed_point_error: float,
    fixed_point_scale: float,
    config: StabilizedEngineeringGripConfig,
) -> dict[str, float]:
    base = config.base
    fields = (
        (
            "material_displacement",
            diagnostics.e_u_material,
            base.displacement,
            diagnostics.displacement_scale,
            "material displacement",
        ),
        (
            "union_displacement",
            diagnostics.e_u_union,
            base.displacement,
            diagnostics.displacement_scale,
            "union displacement",
        ),
        ("damage_transition", diagnostics.e_damage, base.damage, diagnostics.damage_scale, "damage"),
        ("history_transition", diagnostics.e_history, base.history, diagnostics.history_scale, "history"),
        ("reaction_transition", diagnostics.e_reaction, base.reaction, diagnostics.reaction_scale, "reaction"),
        (
            "used_damage_residual",
            diagnostics.rho_used_inf,
            base.damage_residual,
            diagnostics.damage_residual_scale,
            "used damage residual",
        ),
        (
            "next_damage_residual",
            diagnostics.rho_next_inf,
            base.damage_residual,
            diagnostics.damage_residual_scale,
            "next damage residual",
        ),
        ("force", diagnostics.force_error, base.force, diagnostics.force_scale, "force"),
        ("moment", diagnostics.moment_error, base.moment, diagnostics.moment_scale, "moment"),
    )
    components = {
        key: _s_normalized(float(error), gate, float(scale), name)
        for key, error, gate, scale, name in fields
        if error is not None and scale is not None
    }
    components["fixed_point"] = _s_normalized(
        fixed_point_error, base.damage, fixed_point_scale, "fixed-point"
    )
    if not all(np.isfinite(value) for value in components.values()):
        raise FloatingPointError("candidate merit component is non-finite")
    return components

def _s_candidate_merit(
    diagnostics: EngineeringGripDiagnostics,
    fixed_point_error: float,
    fixed_point_scale: float,
    config: StabilizedEngineeringGripConfig,
) -> float:
    components = _s_candidate_merit_components(
        diagnostics, fixed_point_error, fixed_point_scale, config
    )
    merit = max(components.values(), default=0.0)
    if not np.isfinite(merit):
        raise FloatingPointError("candidate merit is non-finite")
    return float(merit)



























def _s_anderson_force_moment_stage_severity(
    metric: Mapping[str, Any],
) -> tuple[int, float]:
    """Return the deterministic severity key for one force/moment stage.

    A positive error against a zero limit is an infinite normalized violation,
    even though its ratio is intentionally unavailable.  A zero error against
    a zero limit is an exact pass and therefore has zero severity.  Positive
    limits use their available normalized ratio.  The canonical stage order is
    used by ``max`` for ties.
    """

    error = float(metric["error"])
    limit = float(metric["limit"])
    if limit == 0.0:
        return (2, error) if error > 0.0 else (0, 0.0)
    normalized = metric["normalized"]
    if normalized is None:
        raise ValueError("positive force/moment limit is missing its ratio")
    return (1, float(normalized))

def _s_anderson_force_moment_worst_stage(
    stage_metrics: Mapping[str, Mapping[str, Any]],
) -> str:
    """Return the canonical worst force/moment stage without new evaluation."""

    if tuple(stage_metrics) != _s_ANDERSON_HARD_GATE_STAGE_NAMES:
        raise ValueError("force/moment stage metrics are not canonical")
    return max(
        _s_ANDERSON_HARD_GATE_STAGE_NAMES,
        key=lambda name: _s_anderson_force_moment_stage_severity(stage_metrics[name]),
    )









def _s_fixed_point_pass(
    fixed_point_error: float,
    fixed_point_scale: float,
    config: StabilizedEngineeringGripConfig,
) -> bool:
    return bool(
        fixed_point_error
        <= config.base.damage.limit(fixed_point_scale)
    )

def _s_damage_equation_pass(
    rho_used: np.ndarray,
    residual_scale: float,
    config: StabilizedEngineeringGripConfig,
) -> bool:
    return bool(
        float(np.max(np.abs(rho_used)))
        <= config.base.damage_residual.limit(residual_scale)
    )

def _s_coupled_pass(
    base_diagnostics: EngineeringGripDiagnostics,
    fixed_point_pass: bool,
    damage_equation_pass: bool,
) -> bool:
    gates = base_diagnostics.gate_passes
    return bool(
        all(gates[name] for name in STABILIZED_COUPLED_GATE_NAMES if name in gates)
        and fixed_point_pass
        and damage_equation_pass
    )

def _s_stabilized_warning_failed_gates(
    diagnostic: "StabilizedRelaxationSweepDiagnostics",
) -> tuple[str, ...]:
    """Return the canonical non-hard gates failed by one complete sweep."""

    gates = diagnostic.gate_passes
    expected = set(STABILIZED_WARNING_HARD_GATE_NAMES) | set(
        STABILIZED_WARNING_GATE_NAMES
    )
    if set(gates) != expected:
        raise ValueError("stabilized warning diagnostic gate names are not canonical")
    if any(not gates[name] for name in STABILIZED_WARNING_HARD_GATE_NAMES):
        return ()
    return tuple(
        name for name in STABILIZED_WARNING_GATE_NAMES if not gates[name]
    )

def _s_stabilized_warning_excesses(
    diagnostic: "StabilizedRelaxationSweepDiagnostics",
) -> tuple[float, float]:
    """Return raw damage-range excess and healing magnitude for a trial."""

    damage = diagnostic.d_trial
    if damage is None:
        raise ValueError("stabilized warning excesses need trial damage")
    previous = diagnostic.base_diagnostics.damage_before
    if not np.all(np.isfinite(damage)) or not np.all(np.isfinite(previous)):
        raise FloatingPointError("stabilized warning excesses are non-finite")
    range_excess = max(
        0.0,
        -float(np.min(damage)),
        float(np.max(damage)) - 1.0,
    )
    healing_magnitude = max(0.0, -float(np.min(damage - previous)))
    if not np.isfinite(range_excess) or not np.isfinite(healing_magnitude):
        raise FloatingPointError("stabilized warning excesses are non-finite")
    return float(range_excess), float(healing_magnitude)

def _s_stabilized_warning_acceptance_decision(
    diagnostic: "StabilizedRelaxationSweepDiagnostics",
) -> tuple[bool, tuple[str, ...], float, float]:
    """Classify a final complete sweep under the isolated warning policy."""

    if not diagnostic.complete_sweep:
        return False, (), 0.0, 0.0
    if any(
        not diagnostic.gate_passes[name]
        for name in STABILIZED_WARNING_HARD_GATE_NAMES
    ):
        return False, (), 0.0, 0.0
    failed = _s_stabilized_warning_failed_gates(diagnostic)
    if not failed:
        return False, (), 0.0, 0.0
    range_excess, healing_magnitude = _s_stabilized_warning_excesses(diagnostic)
    return True, failed, range_excess, healing_magnitude

def _s_stabilized_failure_reason(
    base_diagnostics: EngineeringGripDiagnostics,
    fixed_point_pass: bool,
    damage_equation_pass: bool,
    coupled_pass: bool,
    *,
    include_raw_gates: bool,
) -> str:
    gates = {
        **base_diagnostics.gate_passes,
        "damage_fixed_point": fixed_point_pass,
        "damage_equation": damage_equation_pass,
        "globalization": True,
        "coupled": coupled_pass,
    }
    if include_raw_gates:
        names = (*STABILIZED_WARNING_HARD_GATE_NAMES, *STABILIZED_WARNING_GATE_NAMES)
    else:
        names = (*STABILIZED_COUPLED_GATE_NAMES, "coupled")
    return ", ".join(name for name in names if not gates[name])

def _s_wrap_sweep(
    base_diagnostics: EngineeringGripDiagnostics,
    *,
    complete_sweep: bool,
    passed: bool,
    failure_stage: str,
    failure_reason: str,
    d_solved: np.ndarray | None,
    d_trial: np.ndarray | None,
    omega: float | None,
    reference_merit: float | None,
    merit: float | None,
    sufficient_decrease_limit: float | None,
    fixed_point_error: float | None,
    fixed_point_scale: float | None,
    fixed_point_pass: bool,
    damage_equation_pass: bool,
    globalization_pass: bool,
    coupled_pass: bool,
    trials: Sequence[StabilizedRelaxationTrial],
    anderson_history_size: int | None = None,
    anderson_qr_rank: int | None = None,
    anderson_coefficient_l2: float | None = None,
    anderson_regularization_mu: float | None = None,
    anderson_current_residual_l2: float | None = None,
    anderson_regularization_lambda: float | None = None,
    anderson_unregularized_ls_residual_l2: float | None = None,
    anderson_selection: str | None = None,
    trial_alpha: float | None = None,
    anderson_fallback_reason: str | None = None,
    anderson_restart_reason: str | None = None,
    candidate_fp_error: float | None = None,
    candidate_fp_scale: float | None = None,
    candidate_damage_residual_scale: float | None = None,
) -> StabilizedRelaxationSweepDiagnostics:
    trial_values = tuple(trials)
    accelerated_alpha_trials = sum(
        trial.trial_alpha is not None for trial in trial_values
    )
    backtrack_count = (
        max(0, accelerated_alpha_trials - 1)
        if accelerated_alpha_trials
        else max(0, len(trial_values) - 1)
    )
    return StabilizedRelaxationSweepDiagnostics(
        base_diagnostics=base_diagnostics,
        complete_sweep=complete_sweep,
        passed=passed,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
        d_solved=d_solved,
        d_trial=d_trial,
        omega=omega,
        backtrack_count=backtrack_count,
        reference_merit=reference_merit,
        merit=merit,
        sufficient_decrease_limit=sufficient_decrease_limit,
        fixed_point_error=fixed_point_error,
        fixed_point_scale=fixed_point_scale,
        fixed_point_pass=fixed_point_pass,
        damage_equation_pass=damage_equation_pass,
        globalization_pass=globalization_pass,
        coupled_pass=coupled_pass,
        trials=trial_values,
        anderson_history_size=anderson_history_size,
        anderson_qr_rank=anderson_qr_rank,
        anderson_coefficient_l2=anderson_coefficient_l2,
        anderson_regularization_mu=anderson_regularization_mu,
        anderson_current_residual_l2=anderson_current_residual_l2,
        anderson_regularization_lambda=anderson_regularization_lambda,
        anderson_unregularized_ls_residual_l2=anderson_unregularized_ls_residual_l2,
        anderson_selection=anderson_selection,
        trial_alpha=trial_alpha,
        anderson_fallback_reason=anderson_fallback_reason,
        anderson_restart_reason=anderson_restart_reason,
        candidate_fp_error=candidate_fp_error,
        candidate_fp_scale=candidate_fp_scale,
        candidate_damage_residual_scale=candidate_damage_residual_scale,
    )

def _s_progress_json_safe(value: Any) -> Any:
    """Normalize one transient progress value to strict JSON primitives.

    Progress is intentionally a scalar-only interface.  In particular, arrays
    are rejected rather than converted to lists so a callback cannot
    accidentally retain a large trial field.  Non-finite optional values are
    represented by JSON ``null``; the controller's numerical diagnostics still
    retain their normal hard finite checks.
    """

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _s_progress_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_s_progress_json_safe(item) for item in value]
    raise TypeError(
        "stabilized progress events may contain only JSON scalars, strings, "
        "lists, and mappings; numerical arrays are not allowed"
    )



def _s_anderson_json_scalar(value: Any, name: str) -> float | None:
    """Validate one optional finite JSON number used by gate attribution."""

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite JSON number or null")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result

def _s_anderson_json_integer(value: Any, name: str) -> int:
    """Validate one non-negative JSON integer used by scalar provenance."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a JSON integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result

def _s_validate_anderson_scalar_event(event: Mapping[str, Any]) -> None:
    """Validate the optional current residual hard-gate fields.

    Historical scalar streams predate these fields, so absence is explicitly
    accepted.  Once present, however, the record is checked as a complete
    canonical attribution: every metric has one declared meaning, unavailable
    gates cannot retain numeric provenance, the ordered failed list must be
    derivable from the per-gate booleans, and numerical candidate-map failure
    is distinct from a hard-gate rejection.  This is scalar contract
    validation, not an additional physical solve.
    """

    if not isinstance(event, Mapping):
        raise TypeError("Anderson scalar diagnostic event must be a mapping")
    trial_alpha_present = "trial_alpha" in event
    trial_alpha_index_present = "trial_alpha_index" in event
    identity = event.get("selection_merit_identity")
    trial_record = event.get("record_type") in ("proposal", "sweep")
    record_type = event.get("record_type")
    backtracking_identity = identity in {
        STABILIZED_ANDERSON_RESIDUAL_BACKTRACKING_MERIT_IDENTITY,
        STABILIZED_ANDERSON_REGULARIZED_MERIT_IDENTITY,
    }
    if trial_alpha_index_present and (
        not backtracking_identity or record_type != "proposal"
    ):
        raise ValueError(
            "trial_alpha_index is valid only on backtracked Anderson trial records"
        )
    if trial_alpha_present or (backtracking_identity and trial_record):
        if not backtracking_identity or not trial_record:
            raise ValueError(
                "trial_alpha is valid only on backtracked Anderson trial records"
            )
        if not trial_alpha_present:
            raise ValueError("backtracked Anderson trial record is missing trial_alpha")
        alpha = _s_anderson_json_scalar(event.get("trial_alpha"), "trial_alpha")
        if record_type == "proposal":
            candidate_kind = event.get("candidate_kind")
            if candidate_kind not in ("anderson_accelerated", "picard_fallback"):
                raise ValueError("backtracked proposal candidate_kind is invalid")
            if candidate_kind == "anderson_accelerated" and alpha is None:
                raise ValueError("backtracked accelerated trial is missing trial_alpha")
            if candidate_kind == "picard_fallback" and alpha is not None:
                raise ValueError("backtracked Picard fallback retains trial_alpha")
            if candidate_kind == "anderson_accelerated":
                valid_alphas = tuple(
                    STABILIZED_ANDERSON_ALPHA_REDUCTION**index
                    for index in range(STABILIZED_ANDERSON_MAX_BACKTRACKS + 1)
                )
                if alpha not in valid_alphas:
                    raise ValueError("trial_alpha is not on the configured schedule")
                if not trial_alpha_index_present:
                    raise ValueError(
                        "backtracked accelerated trial is missing trial_alpha_index"
                    )
                alpha_index = _s_anderson_json_integer(
                    event["trial_alpha_index"], "trial_alpha_index"
                )
                expected_alpha = _s_anderson_backtracking_alpha_for_index(
                    alpha_index, "trial_alpha_index"
                )
                if alpha != expected_alpha:
                    raise ValueError(
                        "trial_alpha is not paired with trial_alpha_index"
                    )
            elif trial_alpha_index_present:
                raise ValueError("backtracked Picard fallback retains trial_alpha_index")
        else:
            if trial_alpha_index_present:
                raise ValueError("backtracked sweep retains trial_alpha_index")
            selection = event.get("anderson_selection")
            if selection not in ("accelerated", "picard_fallback"):
                raise ValueError("backtracked sweep selection is invalid")
            if selection == "accelerated" and alpha is None:
                raise ValueError("backtracked accelerated sweep is missing trial_alpha")
            if selection != "accelerated" and alpha is not None:
                raise ValueError("backtracked non-accelerated sweep retains trial_alpha")
            if "backtrack_count" not in event:
                raise ValueError("backtracked sweep is missing backtrack_count")
            backtrack_count = _s_anderson_json_integer(
                event["backtrack_count"], "backtrack_count"
            )
            if selection == "accelerated":
                valid_alphas = tuple(
                    STABILIZED_ANDERSON_ALPHA_REDUCTION**index
                    for index in range(STABILIZED_ANDERSON_MAX_BACKTRACKS + 1)
                )
                if alpha not in valid_alphas:
                    raise ValueError("trial_alpha is not on the configured schedule")
                expected_alpha = _s_anderson_backtracking_alpha_for_index(
                    backtrack_count, "backtrack_count"
                )
                if alpha != expected_alpha:
                    raise ValueError(
                        "trial_alpha is not paired with sweep backtrack_count"
                    )
    regularized_identity = (
        identity == STABILIZED_ANDERSON_REGULARIZED_MERIT_IDENTITY
    )
    regularization_fields = {
        "anderson_regularization_mu",
        "anderson_current_residual_l2",
        "anderson_regularization_lambda",
        "anderson_unregularized_ls_residual_l2",
        "anderson_coefficient_l2",
    }
    ambiguous_regularization_fields = {
        "regularization_mu",
        "regularization_lambda",
        "unregularized_residual_norm",
        "coefficient_norm",
    }
    ambiguous_present = ambiguous_regularization_fields.intersection(event)
    if ambiguous_present:
        raise ValueError(
            "ambiguous legacy regularization diagnostics are forbidden: "
            f"{sorted(ambiguous_present)}"
        )
    present_regularization_fields = regularization_fields.intersection(event)
    if present_regularization_fields and not regularized_identity:
        raise ValueError(
            "regularization diagnostics require the regularized Anderson merit identity"
        )
    if regularized_identity and record_type in (
        "anderson_proposal",
        "proposal",
        "sweep",
    ):
        if "anderson_coefficient_norm" in event:
            raise ValueError(
                "regularized Anderson scalar records must use "
                "anderson_coefficient_l2"
            )
        missing_regularization = sorted(regularization_fields - set(event))
        if missing_regularization:
            raise ValueError(
                "regularized Anderson scalar record is missing fields: "
                f"{missing_regularization}"
            )
        mu = _s_anderson_json_scalar(
            event["anderson_regularization_mu"], "anderson_regularization_mu"
        )
        if mu is None or mu <= 0.0:
            raise ValueError("anderson_regularization_mu must be finite and positive")
        current_residual_l2 = _s_anderson_json_scalar(
            event["anderson_current_residual_l2"],
            "anderson_current_residual_l2",
        )
        regularization_lambda = _s_anderson_json_scalar(
            event["anderson_regularization_lambda"],
            "anderson_regularization_lambda",
        )
        unregularized_ls_residual_l2 = _s_anderson_json_scalar(
            event["anderson_unregularized_ls_residual_l2"],
            "anderson_unregularized_ls_residual_l2",
        )
        coefficient_l2 = _s_anderson_json_scalar(
            event["anderson_coefficient_l2"], "anderson_coefficient_l2"
        )
        if regularization_lambda is not None and current_residual_l2 is None:
            raise ValueError(
                "anderson_regularization_lambda lacks current residual l2"
            )
        if regularization_lambda is not None and regularization_lambda < 0.0:
            raise ValueError("anderson_regularization_lambda must be non-negative")
        if current_residual_l2 is not None and current_residual_l2 < 0.0:
            raise ValueError("anderson_current_residual_l2 must be non-negative")
        if (
            unregularized_ls_residual_l2 is not None
            and unregularized_ls_residual_l2 < 0.0
        ):
            raise ValueError(
                "anderson_unregularized_ls_residual_l2 must be non-negative"
            )
        if coefficient_l2 is not None and coefficient_l2 < 0.0:
            raise ValueError("anderson_coefficient_l2 must be non-negative")
        if (
            regularization_lambda is not None
            and current_residual_l2 is not None
            and not _s_anderson_scalar_relation_close(
                regularization_lambda,
                mu * current_residual_l2**2,
            )
        ):
            raise ValueError(
                "anderson_regularization_lambda is not paired with mu and current residual l2"
            )
    detail_key = "hard_gate_diagnostics"
    failed_key = "failed_hard_gates"
    schema_key = "hard_gate_attribution_schema"
    present_details = detail_key in event
    present_failed = failed_key in event
    present_schema = schema_key in event
    present_status = "hard_gate_status" in event
    if not (present_details or present_failed or present_schema):
        if present_status:
            raise ValueError(
                "hard_gate_status cannot appear without hard-gate attribution"
            )
        return
    if present_details != present_failed:
        raise ValueError(
            "residual hard-gate attribution must contain both diagnostics and failed list"
        )
    if not present_details:
        raise ValueError(
            "hard_gate_attribution_schema cannot appear without hard-gate attribution"
        )
    if not present_schema:
        raise ValueError("hard-gate attribution is missing its schema")
    if event.get("record_type") not in (None, "proposal"):
        raise ValueError(
            "residual hard-gate attribution is valid only on proposal records"
        )
    if not present_status:
        raise ValueError("v3 hard-gate attribution is missing hard_gate_status")
    hard_gate_status = event["hard_gate_status"]
    if hard_gate_status not in _s_ANDERSON_HARD_GATE_STATUSES:
        raise ValueError("residual hard-gate attribution status is invalid")
    candidate_map_status = event.get("candidate_map_status")
    if "schema" in event and candidate_map_status is None:
        raise ValueError("v3 proposal attribution is missing candidate_map_status")
    if candidate_map_status is not None:
        if candidate_map_status not in {"completed", "failed", "not_requested"}:
            raise ValueError("hard-gate attribution has invalid candidate_map_status")
        candidate_map_failure = hard_gate_status == "candidate_map_failure"
        if candidate_map_failure != (candidate_map_status == "failed"):
            raise ValueError(
                "hard-gate status is inconsistent with candidate_map_status"
            )
        if candidate_map_failure and (
            event.get("candidate_kind") != "anderson_accelerated"
            or event.get("failure_stage") != "anderson_candidate_damage"
            or event.get("complete") is not False
        ):
            raise ValueError(
                "candidate-map failure has inconsistent candidate context"
            )
    if hard_gate_status == "not_evaluated" and (
        "schema" in event or candidate_map_status is not None
    ) and (
        event.get("candidate_kind") != "anderson_accelerated"
        or candidate_map_status != "not_requested"
        or event.get("complete") is not False
        or event.get("failure_stage")
        not in {"anderson_candidate_finite", "mechanics_after"}
    ):
        raise ValueError("not-evaluated hard-gate attribution has inconsistent context")
    if present_schema and event[schema_key] != ANDERSON_RESIDUAL_HARD_GATE_ATTRIBUTION_SCHEMA:
        raise ValueError("hard-gate attribution schema is invalid")
    details = event[detail_key]
    if not isinstance(details, Mapping) or tuple(details) != tuple(
        ANDERSON_RESIDUAL_HARD_GATE_NAMES
    ):
        raise ValueError("hard_gate_diagnostics do not contain canonical gate names")
    if "hard_pass" not in event or type(event["hard_pass"]) is not bool:
        raise ValueError("residual hard-gate attribution requires hard_pass")

    expected_detail_fields = tuple(_s_ANDERSON_HARD_GATE_DETAIL_FIELDS)

    def check_ratio(
        item: Mapping[str, Any],
        gate_name: str,
        *,
        strict: bool,
    ) -> bool:
        raw_value = _s_anderson_json_scalar(item["raw"], f"{gate_name}.raw")
        limit_value = _s_anderson_json_scalar(item["limit"], f"{gate_name}.limit")
        if raw_value is None or limit_value is None:
            raise ValueError(f"{gate_name} requires finite raw and limit metrics")
        if raw_value < 0.0 or limit_value < 0.0:
            raise ValueError(f"{gate_name} raw/limit metrics must be non-negative")
        expected_ratio_available = limit_value > 0.0
        if type(item["ratio_available"]) is not bool or item[
            "ratio_available"
        ] != expected_ratio_available:
            raise ValueError(f"{gate_name}.ratio_available is inconsistent")
        normalized = _s_anderson_json_scalar(
            item["normalized"], f"{gate_name}.normalized"
        )
        if expected_ratio_available:
            if normalized is None:
                raise ValueError(f"{gate_name}.normalized is missing")
            expected_normalized = raw_value / limit_value
            if not np.isclose(
                normalized,
                expected_normalized,
                rtol=2.0e-12,
                atol=2.0e-15,
            ):
                raise ValueError(f"{gate_name}.normalized is not raw/limit")
        elif normalized is not None:
            raise ValueError(
                f"{gate_name}.normalized must be null when its limit is zero"
            )
        return raw_value < limit_value if strict else raw_value <= limit_value

    def check_stage_metric(
        item: Mapping[str, Any], gate_name: str, stage_name: str
    ) -> dict[str, Any]:
        if not isinstance(item, Mapping) or tuple(item) != tuple(
            _s_ANDERSON_HARD_GATE_STAGE_METRIC_FIELDS
        ):
            raise ValueError(
                f"{gate_name}.stage_metrics[{stage_name}] is not canonical"
            )
        error = _s_anderson_json_scalar(
            item["error"], f"{gate_name}.{stage_name}.error"
        )
        scale = _s_anderson_json_scalar(
            item["scale"], f"{gate_name}.{stage_name}.scale"
        )
        limit = _s_anderson_json_scalar(
            item["limit"], f"{gate_name}.{stage_name}.limit"
        )
        normalized = _s_anderson_json_scalar(
            item["normalized"], f"{gate_name}.{stage_name}.normalized"
        )
        if (
            error is None
            or scale is None
            or limit is None
            or error < 0.0
            or scale < 0.0
            or limit < 0.0
        ):
            raise ValueError(f"{gate_name}.{stage_name} metric is invalid")
        expected_ratio_available = limit > 0.0
        if type(item["ratio_available"]) is not bool or item[
            "ratio_available"
        ] != expected_ratio_available:
            raise ValueError(
                f"{gate_name}.{stage_name}.ratio_available is inconsistent"
            )
        if expected_ratio_available:
            if normalized is None:
                raise ValueError(
                    f"{gate_name}.{stage_name}.normalized is missing"
                )
            expected_normalized = error / limit
            if not np.isclose(
                normalized,
                expected_normalized,
                rtol=2.0e-12,
                atol=2.0e-15,
            ):
                raise ValueError(
                    f"{gate_name}.{stage_name}.normalized is not error/limit"
                )
        elif normalized is not None:
            raise ValueError(
                f"{gate_name}.{stage_name}.normalized must be null when its limit is zero"
            )
        if type(item["passed"]) is not bool:
            raise ValueError(f"{gate_name}.{stage_name}.passed must be boolean")
        if item["passed"] != (error <= limit):
            raise ValueError(
                f"{gate_name}.{stage_name}.passed is inconsistent with error/limit"
            )
        return {
            "error": error,
            "scale": scale,
            "limit": limit,
            "normalized": normalized,
            "ratio_available": item["ratio_available"],
            "passed": item["passed"],
        }

    for gate_name in ANDERSON_RESIDUAL_HARD_GATE_NAMES:
        item = details[gate_name]
        if not isinstance(item, Mapping) or tuple(item) != expected_detail_fields:
            raise ValueError(f"hard_gate_diagnostics[{gate_name}] is not canonical")
        if (
            type(item["available"]) is not bool
            or type(item["passed"]) is not bool
            or type(item["ratio_available"]) is not bool
        ):
            raise ValueError(
                f"hard_gate_diagnostics[{gate_name}] availability/pass must be booleans"
            )
        stage_reason = item["stage_reason"]
        if not item["available"]:
            if type(stage_reason) is not str or stage_reason not in (
                _s_ANDERSON_HARD_GATE_UNAVAILABLE_REASONS
            ):
                raise ValueError(
                    f"hard_gate_diagnostics[{gate_name}] has a non-canonical unavailable reason"
                )
            if (
                item["passed"]
                or item["raw"] is not None
                or item["scale"] is not None
                or item["limit"] is not None
                or item["normalized"] is not None
                or item["ratio_available"]
                or item["cg_info"] is not None
                or item["cg_status_pass"] is not None
                or item["residual_limit_pass"] is not None
                or item["stage_metrics"] is not None
            ):
                raise ValueError(
                    f"unavailable hard gate {gate_name} retains a metric or pass"
                )
            if item["metric_reason"] != "unavailable":
                raise ValueError(
                    f"unavailable hard gate {gate_name} has a metric reason"
                )
            continue

        if stage_reason is not None:
            raise ValueError(
                f"hard_gate_diagnostics[{gate_name}] has an unexpected stage reason"
            )
        expected_reason = _s_ANDERSON_HARD_GATE_METRIC_REASONS[gate_name]
        if item["metric_reason"] != expected_reason:
            raise ValueError(
                f"hard_gate_diagnostics[{gate_name}] has a non-canonical metric reason"
            )

        if gate_name == "finite":
            if any(
                item[field] is not None
                for field in (
                    "raw",
                    "scale",
                    "limit",
                    "normalized",
                    "cg_info",
                    "cg_status_pass",
                    "residual_limit_pass",
                    "stage_metrics",
                )
            ) or item["ratio_available"]:
                raise ValueError("finite hard gate cannot retain a numeric metric")
            continue

        if gate_name in {"mechanics_before", "mechanics_after"}:
            cg_info = item["cg_info"]
            if type(cg_info) is not int or cg_info < 0:
                raise ValueError(f"{gate_name}.cg_info must be a non-negative integer")
            if (
                type(item["cg_status_pass"]) is not bool
                or type(item["residual_limit_pass"]) is not bool
            ):
                raise ValueError(f"{gate_name} CG status fields must be booleans")
            if item["scale"] is not None or item["stage_metrics"] is not None:
                raise ValueError(f"{gate_name} cannot retain force/moment metrics")
            raw_value = _s_anderson_json_scalar(item["raw"], f"{gate_name}.raw")
            limit_value = _s_anderson_json_scalar(item["limit"], f"{gate_name}.limit")
            if raw_value is None or limit_value is None or raw_value < 0.0 or limit_value < 0.0:
                raise ValueError(f"{gate_name} residual metrics are invalid")
            if item["cg_status_pass"] != (cg_info == 0):
                raise ValueError(f"{gate_name}.cg_status_pass is inconsistent")
            expected_residual_pass = limit_value > 0.0 and raw_value < limit_value
            if item["residual_limit_pass"] != expected_residual_pass:
                raise ValueError(f"{gate_name}.residual_limit_pass is inconsistent")
            expected_pass = bool(
                item["cg_status_pass"] and item["residual_limit_pass"]
            )
            if item["passed"] != expected_pass:
                raise ValueError(f"{gate_name}.passed is inconsistent with the CG gate")
            if check_ratio(item, gate_name, strict=True) != expected_residual_pass:
                raise ValueError(f"{gate_name} residual threshold is inconsistent")
            continue

        if gate_name in {"force", "moment"}:
            if any(
                item[field] is not None
                for field in ("cg_info", "cg_status_pass", "residual_limit_pass")
            ):
                raise ValueError(f"{gate_name} cannot retain CG-only fields")
            stage_metrics = item["stage_metrics"]
            if not isinstance(stage_metrics, Mapping) or tuple(stage_metrics) != tuple(
                _s_ANDERSON_HARD_GATE_STAGE_NAMES
            ):
                raise ValueError(f"{gate_name}.stage_metrics are not canonical")
            checked_stages = {
                stage_name: check_stage_metric(
                    stage_metrics[stage_name], gate_name, stage_name
                )
                for stage_name in _s_ANDERSON_HARD_GATE_STAGE_NAMES
            }
            raw_value = _s_anderson_json_scalar(item["raw"], f"{gate_name}.raw")
            scale_value = _s_anderson_json_scalar(item["scale"], f"{gate_name}.scale")
            limit_value = _s_anderson_json_scalar(item["limit"], f"{gate_name}.limit")
            normalized = _s_anderson_json_scalar(
                item["normalized"], f"{gate_name}.normalized"
            )
            if (
                raw_value is None
                or scale_value is None
                or limit_value is None
                or raw_value < 0.0
                or scale_value < 0.0
                or limit_value < 0.0
            ):
                raise ValueError(f"{gate_name} aggregate metric is invalid")
            expected_ratio_available = limit_value > 0.0
            if type(item["ratio_available"]) is not bool or item[
                "ratio_available"
            ] != expected_ratio_available:
                raise ValueError(
                    f"{gate_name}.ratio_available is inconsistent with its limit"
                )
            if expected_ratio_available:
                if normalized is None:
                    raise ValueError(f"{gate_name}.normalized is missing")
            elif normalized is not None:
                raise ValueError(
                    f"{gate_name}.normalized must be null when its limit is zero"
                )
            worst_name = _s_anderson_force_moment_worst_stage(checked_stages)
            worst = checked_stages[worst_name]
            if expected_ratio_available:
                if not np.isclose(
                    normalized,
                    worst["normalized"],
                    rtol=2.0e-12,
                    atol=2.0e-15,
                ):
                    raise ValueError(
                        f"{gate_name}.normalized is not the worst stage ratio"
                    )
            for field, stage_field, value in (
                ("raw", "error", raw_value),
                ("scale", "scale", scale_value),
                ("limit", "limit", limit_value),
            ):
                if not np.isclose(
                    value,
                    worst[stage_field],
                    rtol=2.0e-12,
                    atol=2.0e-15,
                ):
                    raise ValueError(
                        f"{gate_name}.{field} is not the worst-stage representative"
                    )
            if expected_ratio_available and not np.isclose(
                normalized,
                raw_value / limit_value,
                rtol=2.0e-12,
                atol=2.0e-15,
            ):
                raise ValueError(f"{gate_name}.normalized is not raw/limit")
            expected_pass = all(
                checked_stages[stage_name]["passed"]
                for stage_name in _s_ANDERSON_HARD_GATE_STAGE_NAMES
            )
            if item["passed"] != expected_pass:
                raise ValueError(
                    f"{gate_name}.passed is inconsistent with stage conjunction"
                )
            continue

        if gate_name in {
            "damage_range_lower",
            "damage_range_upper",
            "healing",
        }:
            if item["scale"] is not None or item["stage_metrics"] is not None:
                raise ValueError(f"{gate_name} cannot retain stage metrics")
            expected_pass = check_ratio(item, gate_name, strict=False)
            if item["passed"] != expected_pass:
                raise ValueError(f"{gate_name}.passed is inconsistent with raw/limit")
            if any(
                item[field] is not None
                for field in ("cg_info", "cg_status_pass", "residual_limit_pass")
            ):
                raise ValueError(f"{gate_name} cannot retain CG-only fields")
            continue

        if gate_name == "history_ordering":
            if any(
                item[field] is not None
                for field in (
                    "normalized",
                    "scale",
                    "cg_info",
                    "cg_status_pass",
                    "residual_limit_pass",
                    "stage_metrics",
                )
            ) or item["ratio_available"]:
                raise ValueError("history ordering cannot claim a ratio")
            raw_value = _s_anderson_json_scalar(item["raw"], "history_ordering.raw")
            limit_value = _s_anderson_json_scalar(
                item["limit"], "history_ordering.limit"
            )
            if raw_value is None or limit_value != 0.0 or raw_value < 0.0:
                raise ValueError("history ordering metric is invalid")
            if item["passed"] != (raw_value <= 0.0):
                raise ValueError("history ordering pass is inconsistent")
            continue

        raise ValueError(f"unknown residual hard gate {gate_name}")

    failed = event[failed_key]
    if not isinstance(failed, list) or any(
        type(name) is not str for name in failed
    ):
        raise ValueError("failed_hard_gates must be a string list")
    expected_failed = [
        name for name in ANDERSON_RESIDUAL_HARD_GATE_NAMES if not details[name]["passed"]
    ]
    if failed != expected_failed:
        raise ValueError("failed_hard_gates do not match per-gate pass flags")
    if hard_gate_status == "evaluated":
        if event["hard_pass"] != (not expected_failed):
            raise ValueError("hard_pass does not match failed_hard_gates")
    elif hard_gate_status == "candidate_map_failure" and (
        expected_failed or event["hard_pass"] is not False
    ):
        raise ValueError(
            "candidate-map failure attribution must have no failed hard gates "
            "and hard_pass=false"
        )
    elif hard_gate_status == "not_evaluated":
        if event["hard_pass"] is not False or not any(
            not details[name]["available"] for name in ANDERSON_RESIDUAL_HARD_GATE_NAMES
        ):
            raise ValueError(
                "not-evaluated attribution must retain unavailable hard gates "
                "and hard_pass=false"
            )









def _s_progress_ratio(
    error: float | None,
    gate: GateTolerance,
    scale: float | None,
) -> float | None:
    """Return one finite gate ratio, or ``None`` when its inputs are absent."""

    if error is None or scale is None:
        return None
    error_value = float(error)
    scale_value = float(scale)
    limit = float(gate.limit(scale_value))
    if not (
        np.isfinite(error_value)
        and np.isfinite(scale_value)
        and np.isfinite(limit)
        and limit > 0.0
    ):
        return None
    ratio = abs(error_value) / limit
    return float(ratio) if np.isfinite(ratio) else None

def _s_progress_sweep_is_due(
    sweep_index: int,
    progress_every_sweeps: int,
    max_sweeps: int,
    passed: bool,
) -> bool:
    """Return whether one zero-based sweep should cross the progress seam."""

    one_based = sweep_index + 1
    return bool(
        one_based == 1
        or one_based % progress_every_sweeps == 0
        or one_based == max_sweeps
        or passed
    )

def _s_progress_sweep_event(
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    diagnostic: StabilizedRelaxationSweepDiagnostics,
    *,
    cached_q_before: bool,
    max_sweeps: int,
) -> dict[str, Any]:
    """Build the scalar-only event for one selected outer stagger sweep."""

    damage = diagnostic.d_next
    if damage is None:
        damage = diagnostic.d_iter
    if damage is None or not np.all(np.isfinite(damage)):
        damage_min: float | None = None
        damage_max: float | None = None
    else:
        damage_min = float(np.min(damage))
        damage_max = float(np.max(damage))

    history = diagnostic.H_next
    if history is None:
        history = diagnostic.H_used
    history_max = (
        None
        if history is None or not np.all(np.isfinite(history))
        else float(np.max(history))
    )

    displacement = diagnostic.q_after
    if displacement is None:
        displacement = diagnostic.q_before
    max_material_displacement: float | None = None
    displacement_amplification: float | None = None
    if displacement is not None:
        values = np.asarray(displacement)
        required = 2 * model.n_material
        if values.ndim == 1 and values.size >= required and np.all(
            np.isfinite(values[:required])
        ):
            material = values[:required].reshape(model.n_material, 2)
            magnitude = np.linalg.norm(material, axis=1)
            if np.all(np.isfinite(magnitude)):
                max_material_displacement = float(np.max(magnitude))
                displacement_amplification = float(
                    max_material_displacement
                    / max(float(diagnostic.target_u_bar), np.finfo(np.float64).tiny)
                )

    reaction = diagnostic.top_reaction_after
    if reaction is None:
        reaction = diagnostic.top_reaction_before
    failing_gates = sorted(
        name for name, passed in diagnostic.gate_passes.items() if not passed
    )
    return {
        "event": "sweep",
        "target_load": diagnostic.target_u_bar,
        "sweep": diagnostic.sweep_index + 1,
        "max_sweeps": max_sweeps,
        "complete_sweep": diagnostic.complete_sweep,
        "passed": diagnostic.passed,
        "cached_q_before": bool(cached_q_before),
        "policy": config.relaxation_policy,
        "omega": diagnostic.omega,
        "anderson_history_size": diagnostic.anderson_history_size,
        "anderson_qr_rank": diagnostic.anderson_qr_rank,
        "anderson_selection": diagnostic.anderson_selection,
        "anderson_fallback_reason": diagnostic.anderson_fallback_reason,
        "anderson_restart_reason": diagnostic.anderson_restart_reason,
        "merit": diagnostic.merit,
        "d_max": damage_max,
        "d_min": damage_min,
        "H_max": history_max,
        "max_material_displacement": max_material_displacement,
        "displacement_amplification": displacement_amplification,
        "reaction": reaction,
        "fixed_point_error": diagnostic.fixed_point_error,
        "fixed_point_ratio": _s_progress_ratio(
            diagnostic.fixed_point_error,
            config.base.damage,
            diagnostic.fixed_point_scale,
        ),
        "history_ratio": _s_progress_ratio(
            diagnostic.e_history,
            config.base.history,
            diagnostic.history_scale,
        ),
        "damage_residual_ratio": _s_progress_ratio(
            diagnostic.rho_next_inf,
            config.base.damage_residual,
            diagnostic.damage_residual_scale,
        ),
        "failing_gates": failing_gates,
        "failure_stage": diagnostic.failure_stage or None,
        "failure_reason": diagnostic.failure_reason or None,
    }

def _s_progress_accepted_event(
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    step: StabilizedEngineeringGripAcceptedStep,
) -> dict[str, Any]:
    """Build the scalar-only event for one committed step."""

    sweep = _s_progress_sweep_event(
        model,
        config,
        step.diagnostics,
        cached_q_before=False,
        max_sweeps=config.base.max_stagger_iterations,
    )
    return {
        "event": "accepted",
        "load": step.load_target,
        "increment": step.increment,
        "sweeps": step.stagger_iterations,
        "policy": config.relaxation_policy,
        "reaction": sweep["reaction"],
        "d_max": sweep["d_max"],
        "H_max": sweep["H_max"],
        "max_material_displacement": sweep["max_material_displacement"],
        "displacement_amplification": sweep["displacement_amplification"],
    }

def _s_emit_progress_event(
    callback: Callable[[dict[str, Any]], None] | None,
    event: Mapping[str, Any],
    started: float,
) -> None:
    """Validate and deliver one transient progress event without printing."""

    if callback is None:
        return
    payload = _s_progress_json_safe(dict(event))
    if not isinstance(payload, dict):
        raise TypeError("stabilized progress event must be a mapping")
    payload["elapsed"] = float(time.perf_counter() - started)
    payload = _s_progress_json_safe(payload)
    # Validate the exact object delivered to the callback.  This both rejects
    # accidental arrays and guarantees that a workflow printer can use
    # allow_nan=False without changing the numerical path.
    json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    callback(dict(payload))

def _s_emit_failure_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    reason: str,
    committed_load: float,
    target_load: float | None,
    accepted_steps: int,
    attempts: int,
    failure_type: str,
    started: float,
) -> None:
    """Emit the scalar-only terminal failure event."""

    _s_emit_progress_event(
        callback,
        {
            "event": "failure",
            "reason": reason,
            "committed_load": committed_load,
            "target_load": target_load,
            "accepted_steps": accepted_steps,
            "attempts": attempts,
            "failure_type": failure_type,
        },
        started,
    )

def _s_incomplete_base(
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    committed: EngineeringGripState,
    d_iter: np.ndarray,
    target: float,
    stage: str,
    reason: str,
    sweep_index: int,
    *,
    before: A2MechanicsSolveResult | None = None,
    after: A2MechanicsSolveResult | None = None,
    psi_before: np.ndarray | None = None,
    psi_after: np.ndarray | None = None,
    H_used: np.ndarray | None = None,
    d_next: np.ndarray | None = None,
    rho_used: np.ndarray | None = None,
    rho_used_volume: np.ndarray | None = None,
) -> EngineeringGripDiagnostics:
    return _e_incomplete_diagnostics(
        config.base,
        committed,
        d_iter,
        target,
        stage,
        reason,
        before=before,
        after=after,
        psi_before=psi_before,
        psi_after=psi_after,
        H_used=H_used,
        d_next=d_next,
        rho_used=rho_used,
        rho_used_volume=rho_used_volume,
        model=model,
        sweep_index=sweep_index,
    )

def _s_relaxation_omega(config: StabilizedEngineeringGripConfig, index: int) -> float:
    if index != 0:
        raise ValueError("fixed relaxation has exactly one candidate per sweep")
    return float(config.omega_initial)


def _s_validate_reusable_mechanics_result(
    result: A2MechanicsSolveResult,
) -> None:
    """Require that a transient mechanics result owns immutable arrays.

    The stabilized cache retains the CPU-authoritative result returned by the
    selected mechanics solve.  In particular, it must not retain the JAX
    adapter's device result or an array view backed by mutable caller storage.
    ``A2MechanicsSolveResult`` and ``MechanicsEvaluation`` normally enforce
    this in their constructors; checking the flags here makes that ownership
    contract explicit at the cache boundary as well.
    """

    if not isinstance(result, A2MechanicsSolveResult):
        raise TypeError("stabilized mechanics reuse needs an A2 mechanics result")
    arrays = [result.displacement, result.free_residual, result.total_force]
    arrays.extend(
        getattr(result.evaluation, field.name)
        for field in dataclass_fields(result.evaluation)
        if isinstance(getattr(result.evaluation, field.name), np.ndarray)
    )
    if any(np.asarray(array).flags.writeable for array in arrays):
        raise ValueError(
            "stabilized mechanics reuse requires read-only mechanics arrays"
        )

@dataclass(frozen=True)
class _s_StabilizedMechanicsReuse:
    """One same-target/fixed-damage mechanics result held for one next sweep.

    This object is deliberately private and is never passed to an archive,
    result, or workflow layer.  The damage copy/digest and exact static,
    constraint, and load identities prevent a result from crossing a target or
    boundary-condition boundary.  The full result is retained rather than
    just ``q`` so all mechanics diagnostics used by the next ``q_before`` are
    the exact result of the selected candidate.
    """

    target: float
    static_runtime_identity: object
    static_digest: str
    constraints: object
    damage: np.ndarray
    damage_digest: str
    result: A2MechanicsSolveResult

    def __post_init__(self) -> None:
        target = _s_scalar(self.target, "stabilized mechanics reuse target")
        damage = _s_f64(self.damage, "stabilized mechanics reuse damage")
        digest = _s_digest(
            self.damage_digest, "stabilized mechanics reuse damage digest"
        )
        if digest != _e_damage_digest(damage):
            raise ValueError("stabilized mechanics reuse damage digest is not paired")
        static_digest = _s_digest(
            self.static_digest, "stabilized mechanics reuse static digest"
        )
        _s_validate_reusable_mechanics_result(self.result)
        if self.result.constraints is not self.constraints:
            raise ValueError(
                "stabilized mechanics reuse constraints are not the exact bound object"
            )
        if self.result.static_runtime_identity is not self.static_runtime_identity:
            raise ValueError(
                "stabilized mechanics reuse static identity is not paired"
            )
        if self.result.static_digest != static_digest:
            raise ValueError("stabilized mechanics reuse static digest is not paired")
        if self.result.outer_displacement != target:
            raise ValueError("stabilized mechanics reuse load is not paired")
        if self.result.damage_digest != digest:
            raise ValueError("stabilized mechanics reuse result damage is not paired")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "static_digest", static_digest)
        object.__setattr__(self, "damage", _s_immutable(damage))
        object.__setattr__(self, "damage_digest", digest)

    @classmethod
    def from_selected(
        cls,
        model: EngineeringGripModel,
        target: float,
        d_trial: np.ndarray,
        result: A2MechanicsSolveResult,
    ) -> "_StabilizedMechanicsReuse":
        """Capture one selected result without retaining mutable aliases."""

        damage = _s_immutable(_s_f64(d_trial, "selected stabilized trial damage"))
        digest = _e_damage_digest(damage)
        if result.constraints is not model.constraints:
            raise ValueError(
                "selected stabilized mechanics result has foreign constraints"
            )
        if result.static_runtime_identity is not model.static.runtime_identity:
            raise ValueError(
                "selected stabilized mechanics result has foreign static identity"
            )
        if result.static_digest != model.static.deterministic_digest:
            raise ValueError(
                "selected stabilized mechanics result has a foreign static digest"
            )
        if result.outer_displacement != _s_scalar(target, "selected stabilized target"):
            raise ValueError("selected stabilized mechanics result has a foreign load")
        if result.damage_digest != digest:
            raise ValueError(
                "selected stabilized mechanics result has a foreign damage digest"
            )
        _s_validate_reusable_mechanics_result(result)
        return cls(
            target=target,
            static_runtime_identity=model.static.runtime_identity,
            static_digest=model.static.deterministic_digest,
            constraints=model.constraints,
            damage=damage,
            damage_digest=digest,
            result=result,
        )

    def take(
        self,
        model: EngineeringGripModel,
        view: Any,
        target: float,
    ) -> A2MechanicsSolveResult | None:
        """Return the result only for an exact same-target/fixed-damage view."""

        if self.target != _s_scalar(target, "stabilized reuse target"):
            return None
        if self.static_runtime_identity is not model.static.runtime_identity:
            return None
        if self.static_digest != model.static.deterministic_digest:
            return None
        if self.constraints is not model.constraints:
            return None
        view_damage = _s_f64(
            view.material_damage,
            "stabilized reuse view damage",
            self.damage.shape,
        )
        if self.damage_digest != view.damage_digest:
            return None
        if not np.array_equal(self.damage, view_damage):
            return None
        result = self.result
        if result.constraints is not model.constraints:
            return None
        if result.static_runtime_identity is not model.static.runtime_identity:
            return None
        if result.static_digest != model.static.deterministic_digest:
            return None
        if result.outer_displacement != target:
            return None
        if result.damage_digest != view.damage_digest:
            return None
        _s_validate_reusable_mechanics_result(result)
        return result

def _s_run_stabilized_attempt(
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    committed: EngineeringGripState,
    target: float,
    cutback_count: int,
    step_index: int,
    mechanics_adapter: JaxA2CPUReplayAdapter | None,
    *,
    reuse_selected_mechanics: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_every_sweeps: int | None = None,
    progress_started: float | None = None,
    acceptance_mode: str | None = None,
    initial_damage: np.ndarray | None = None,
) -> tuple[StabilizedEngineeringGripAcceptedStep | None, StabilizedEngineeringGripAttempt]:
    """Solve one load; a noncommitted restart seed needs separate provenance.

    ``initial_damage`` is reserved for an isolated, strict fixed-omega
    continuation of an already audited complete trial at this same target.
    Legacy continuation/archives must not silently serialize a seeded attempt
    as a normal committed-damage predictor.
    """

    base = config.base
    mode = config.acceptance_mode if acceptance_mode is None else _s_validate_stabilized_acceptance_mode(acceptance_mode)
    anderson_policy = False
    residual_anderson_policy = False
    residual_backtracking_policy = False
    regularized_residual_backtracking_policy = False
    if mode != config.acceptance_mode:
        raise ValueError("stabilized attempt mode does not match configuration")
    
    if type(reuse_selected_mechanics) is not bool:
        raise TypeError("reuse_selected_mechanics must be a boolean")
    if initial_damage is None:
        d_iter = np.array(committed.d, dtype=np.float64, copy=True)
    else:
        if mode != STABILIZED_ACCEPTANCE_MODE or "fixed" != "fixed":
            raise ValueError("seeded stagger attempt requires strict fixed relaxation")
        seed = np.asarray(initial_damage)
        if (seed.dtype != np.float64
            or seed.shape != (model.n_material,)
            or not np.all(np.isfinite(seed))):
            raise ValueError("seeded stagger damage must be finite float64 material data")
        if (np.min(seed) < -base.damage_range_atol
            or np.max(seed) > 1.0 + base.damage_range_atol
            or np.min(seed - committed.d) < -base.healing_atol):
            raise ValueError("seeded stagger damage violates committed physical bounds")
        d_iter = np.array(seed, dtype=np.float64, copy=True)
    q_seed = _e_direct_seed(
        committed.q, committed.u_bar, target, model.constraints
    )
    sweeps: list[StabilizedRelaxationSweepDiagnostics] = []
    mechanics_reuse: _s_StabilizedMechanicsReuse | None = None
    anderson_history_x: list[np.ndarray] = []
    anderson_history_f: list[np.ndarray] = []
    sweep_started = 0.0
    anderson_coefficient_l2: float | None = None
    anderson_regularization_mu: float | None = (
        None
    )
    anderson_regularization_lambda: float | None = None
    anderson_current_residual_l2: float | None = None
    anderson_unregularized_ls_residual_l2: float | None = None
    anderson_proposal_candidate_available = False
    anderson_proposal_fallback_reason: str | None = None
    anderson_proposal_restart_reason: str | None = None

    def retain_sweep(
        diagnostic: StabilizedRelaxationSweepDiagnostics,
        stage_kind: str,
        *,
        cached_q_before: bool,
    ) -> None:
        """Retain a runtime sweep without charging it to archive storage."""

        del stage_kind
        sweeps.append(diagnostic)
        pass
        if progress_callback is None:
            return
        if progress_every_sweeps is None or progress_started is None:
            raise RuntimeError("stabilized progress settings were not initialized")
        if _s_progress_sweep_is_due(
            diagnostic.sweep_index,
            progress_every_sweeps,
            base.max_stagger_iterations,
            diagnostic.passed,
        ):
            _s_emit_progress_event(
                progress_callback,
                _s_progress_sweep_event(
                    model,
                    config,
                    diagnostic,
                    cached_q_before=cached_q_before,
                    max_sweeps=base.max_stagger_iterations,
                ),
                progress_started,
            )

    for sweep_index in range(base.max_stagger_iterations):
        sweep_started = time.perf_counter()
        anderson_coefficient_l2 = None
        anderson_regularization_lambda = None
        anderson_current_residual_l2 = None
        anderson_unregularized_ls_residual_l2 = None
        anderson_proposal_candidate_available = False
        anderson_proposal_fallback_reason = None
        anderson_proposal_restart_reason = None
        view_before = model.fixed_damage_view(d_iter)
        before: A2MechanicsSolveResult | None = None
        cached_q_before = False
        if reuse_selected_mechanics and mechanics_reuse is not None:
            candidate_reuse = mechanics_reuse
            # A cache is single-use.  A failure after this point must not leave
            # a result available to any later path, even though this attempt
            # returns immediately on all such failures.
            mechanics_reuse = None
            before = candidate_reuse.take(model, view_before, target)
            cached_q_before = before is not None
        if before is None:
            try:
                before = _e_solve_direct_mechanics(
                    mechanics_adapter,
                    view_before,
                    model.constraints,
                    target,
                    base.cg_rtol,
                    base.cg_atol,
                    base.cg_maxiter,
                    initial_guess=q_seed,
                    seed_kind="stabilized_previous_direct_solution",
                    source_state="stabilized_committed"
                    if sweep_index == 0
                    else "stabilized_complete_trial_sweep",
                    source_step=step_index,
                    source_sweep=sweep_index,
                    source_damage_digest=_e_damage_digest(committed.d)
                    if sweep_index == 0
                    else _e_damage_digest(d_iter),
                )
            except (RuntimeError, FloatingPointError) as error:
                mechanics_reuse = None
                base_diag = _s_incomplete_base(
                    model,
                    config,
                    committed,
                    d_iter,
                    target,
                    "mechanics_before",
                    str(error),
                    sweep_index,
                )
                diag = _s_wrap_sweep(
                    base_diag,
                    complete_sweep=False,
                    passed=False,
                    failure_stage="mechanics_before",
                    failure_reason=str(error),
                    d_solved=None,
                    d_trial=None,
                    omega=None,
                    reference_merit=None,
                    merit=None,
                    sufficient_decrease_limit=None,
                    fixed_point_error=None,
                    fixed_point_scale=None,
                    fixed_point_pass=False,
                    damage_equation_pass=False,
                    globalization_pass=False,
                    coupled_pass=False,
                    trials=(),
                )
                retain_sweep(
                    diag,
                    "incomplete_sweep",
                    cached_q_before=cached_q_before,
                )
                return _s_rejected_attempt(
                    committed,
                    target,
                    cutback_count,
                    sweeps,
                    diag,
                    acceptance_mode=mode,
                )

        mechanics_before, _, _, _, force_before_pass, moment_before_pass = _e_mechanics_gate(
            before, base, _e_geometry_length(model)
        )
        if not mechanics_before or not force_before_pass or not moment_before_pass:
            mechanics_reuse = None
            reason_parts = []
            if not mechanics_before:
                reason_parts.append("mechanics")
            if not force_before_pass:
                reason_parts.append("force")
            if not moment_before_pass:
                reason_parts.append("moment")
            reason = "mechanics-before " + "/".join(reason_parts) + " gate failed"
            base_diag = _s_incomplete_base(
                model,
                config,
                committed,
                d_iter,
                target,
                "mechanics_before_gate",
                reason,
                sweep_index,
                before=before,
            )
            diag = _s_wrap_sweep(
                base_diag,
                complete_sweep=False,
                passed=False,
                failure_stage="mechanics_before_gate",
                failure_reason=reason,
                d_solved=None,
                d_trial=None,
                omega=None,
                reference_merit=None,
                merit=None,
                sufficient_decrease_limit=None,
                fixed_point_error=None,
                fixed_point_scale=None,
                fixed_point_pass=False,
                damage_equation_pass=False,
                globalization_pass=False,
                coupled_pass=False,
                trials=(),
            )
            retain_sweep(
                diag,
                "incomplete_sweep",
                cached_q_before=cached_q_before,
            )
            return _s_rejected_attempt(
                committed,
                target,
                cutback_count,
                sweeps,
                diag,
                acceptance_mode=mode,
            )

        try:
            amor_before = compute_material_source_amor(
                view_before, before.displacement, committed.H
            )
            psi_before = _s_f64(
                amor_before.particle_driver,
                "stabilized psi_before",
                (model.n_material,),
                nonfinite_error=FloatingPointError,
            )
            H_used = _s_immutable(np.maximum(committed.H, psi_before))
        except (RuntimeError, FloatingPointError) as error:
            mechanics_reuse = None
            base_diag = _s_incomplete_base(
                model,
                config,
                committed,
                d_iter,
                target,
                "amor_before",
                str(error),
                sweep_index,
                before=before,
            )
            diag = _s_wrap_sweep(
                base_diag,
                complete_sweep=False,
                passed=False,
                failure_stage="amor_before",
                failure_reason=str(error),
                d_solved=None,
                d_trial=None,
                omega=None,
                reference_merit=None,
                merit=None,
                sufficient_decrease_limit=None,
                fixed_point_error=None,
                fixed_point_scale=None,
                fixed_point_pass=False,
                damage_equation_pass=False,
                globalization_pass=False,
                coupled_pass=False,
                trials=(),
            )
            retain_sweep(
                diag,
                "incomplete_sweep",
                cached_q_before=cached_q_before,
            )
            return _s_rejected_attempt(
                committed,
                target,
                cutback_count,
                sweeps,
                diag,
                acceptance_mode=mode,
            )

        try:
            damage_result = solve_fixed_history_damage(
                model.material_operator, model.Gc, model.l0, H_used
            )
            if not isinstance(damage_result, FixedHistoryDamageResult):
                raise TypeError("stabilized fixed-history solve returned the wrong type")
            d_solved = _s_f64(
                damage_result.damage,
                "stabilized fixed-history damage",
                (model.n_material,),
                nonfinite_error=FloatingPointError,
            )
            if not np.array_equal(damage_result.history, H_used):
                raise ValueError("stabilized fixed-history result has a different H_used")
            rho_used = _s_f64(
                damage_result.strong_residual,
                "stabilized fixed-history strong residual",
                d_solved.shape,
                nonfinite_error=FloatingPointError,
            )
            rho_used_volume = _s_f64(
                damage_result.euclidean_residual,
                "stabilized fixed-history volume residual",
                d_solved.shape,
                nonfinite_error=FloatingPointError,
            )
            _e_compare_damage_residual_reconstruction(
                rho_used,
                rho_used_volume,
                d_solved,
                H_used,
                model.material_operator.to_scipy().tocsr(),
                model.material_operator.volumes,
                model.Gc,
                model.l0,
                "stabilized fixed-history residual",
            )
            _, _, _, _, solved_scale = _e_damage_residuals(
                model, d_solved, H_used, H_used
            )
        except (RuntimeError, FloatingPointError) as error:
            mechanics_reuse = None
            base_diag = _s_incomplete_base(
                model,
                config,
                committed,
                d_iter,
                target,
                "damage",
                str(error),
                sweep_index,
                before=before,
                psi_before=psi_before,
                H_used=H_used,
            )
            diag = _s_wrap_sweep(
                base_diag,
                complete_sweep=False,
                passed=False,
                failure_stage="damage",
                failure_reason=str(error),
                d_solved=None,
                d_trial=None,
                omega=None,
                reference_merit=None,
                merit=None,
                sufficient_decrease_limit=None,
                fixed_point_error=None,
                fixed_point_scale=None,
                fixed_point_pass=False,
                damage_equation_pass=False,
                globalization_pass=False,
                coupled_pass=False,
                trials=(),
            )
            retain_sweep(
                diag,
                "incomplete_sweep",
                cached_q_before=cached_q_before,
            )
            return _s_rejected_attempt(
                committed,
                target,
                cutback_count,
                sweeps,
                diag,
                acceptance_mode=mode,
            )

        fixed_point_error = float(np.max(np.abs(d_solved - d_iter)))
        fixed_point_scale = max(float(np.max(np.abs(d_solved))), float(np.max(np.abs(d_iter))))
        fixed_pass = _s_fixed_point_pass(fixed_point_error, fixed_point_scale, config)
        damage_eq_pass = _s_damage_equation_pass(rho_used, solved_scale, config)
        reference_merit = (
            _s_reference_merit(model, config, d_iter, d_solved, H_used)
        )
        trials: list[StabilizedRelaxationTrial] = []
        selected_diag: StabilizedRelaxationSweepDiagnostics | None = None
        selected_after: A2MechanicsSolveResult | None = None
        last_base_diag: EngineeringGripDiagnostics | None = None
        last_candidate_after: A2MechanicsSolveResult | None = None
        last_candidate_damage: np.ndarray | None = None

        anderson_history_size: int | None = None
        anderson_qr_rank: int | None = None
        anderson_selection: str | None = None
        anderson_fallback_reason: str | None = None
        anderson_restart_reason: str | None = None
        candidate_fp_error: float | None = None
        candidate_fp_scale: float | None = None
        candidate_damage_residual_scale: float | None = None
        candidate_specs: list[
            tuple[str, np.ndarray, float | None, float | None]
        ] = []
        trial_count = (
            1
        )
        for backtrack_count in range(trial_count):
            candidate_fp_error = None
            candidate_fp_scale = None
            candidate_damage_residual_scale = None
            candidate_kind = "picard"
            omega = _s_relaxation_omega(config, backtrack_count)
            trial_alpha = None
            d_trial = _s_f64(
                d_iter + omega * (d_solved - d_iter),
                "stabilized relaxed damage",
                (model.n_material,),
                nonfinite_error=FloatingPointError,
            )
            sufficient_limit = None
            view_after = model.fixed_damage_view(d_trial)
            candidate_after: A2MechanicsSolveResult | None = None
            try:
                after = _e_solve_direct_mechanics(
                    mechanics_adapter,
                    view_after,
                    model.constraints,
                    target,
                    base.cg_rtol,
                    base.cg_atol,
                    base.cg_maxiter,
                    initial_guess=_e_direct_seed(
                        before.displacement, target, target, model.constraints
                    ),
                    seed_kind="stabilized_mechanics_before_solution",
                    source_state="stabilized_mechanics_before",
                    source_step=step_index,
                    source_sweep=sweep_index,
                    source_damage_digest=_e_damage_digest(d_iter),
                )
                candidate_after = after
                after_mechanics, _, _, _, _, _ = _e_mechanics_gate(
                    after, base, _e_geometry_length(model)
                )
                if not after_mechanics:
                    raise RuntimeError("stabilized mechanics-after gate failed")
                amor_after = compute_material_source_amor(
                    view_after, after.displacement, committed.H
                )
                psi_after = _s_f64(
                    amor_after.particle_driver,
                    "stabilized psi_after",
                    (model.n_material,),
                    nonfinite_error=FloatingPointError,
                )
                H_next = _s_immutable(np.maximum(committed.H, psi_after))
                _, rho_next, _, rho_next_volume, residual_scale = _e_damage_residuals(
                    model, d_trial, H_used, H_next
                )
                _e_compare_damage_residual_reconstruction(
                    rho_used,
                    rho_used_volume,
                    d_solved,
                    H_used,
                    model.material_operator.to_scipy().tocsr(),
                    model.material_operator.volumes,
                    model.Gc,
                    model.l0,
                    "stabilized rho_used reconstruction",
                )
                base_diag = _e_full_diagnostics(
                    base,
                    model,
                    committed,
                    d_iter,
                    d_trial,
                    H_used,
                    H_next,
                    before,
                    after,
                    psi_before,
                    psi_after,
                    rho_used,
                    rho_next,
                    rho_used_volume,
                    rho_next_volume,
                    residual_scale,
                    sweep_index,
                    target,
                )
                last_base_diag = base_diag
                last_candidate_after = after
                last_candidate_damage = np.array(d_trial, dtype=np.float64, copy=True)
                hard_pass = (
                    bool(
                        base_diag.finite_pass
                        and base_diag.mechanics_before_pass
                        and base_diag.mechanics_after_pass
                    )
                )
                
                merit = _s_candidate_merit(
                    base_diag, fixed_point_error, fixed_point_scale, config
                )
                globalization_pass = hard_pass
                coupled_pass = _s_coupled_pass(base_diag, fixed_pass, damage_eq_pass)
                trial = StabilizedRelaxationTrial(
                    omega=omega,
                    complete=True,
                    hard_pass=hard_pass,
                    globalization_pass=globalization_pass,
                    coupled_pass=coupled_pass,
                    selected=globalization_pass,
                    merit=merit,
                    sufficient_decrease_limit=sufficient_limit,
                    failure_stage=""
                    if globalization_pass
                    else "globalization",
                    failure_reason=""
                    if globalization_pass
                    else (
                        "accelerated candidate failed hard gates"
                        if candidate_kind == "anderson_accelerated" and not hard_pass
                        else "sufficient-decrease merit condition failed"
                    ),
                    candidate_diagnostics=base_diag,
                    d_trial=d_trial,
                    q_after=after.displacement,
                    d_trial_digest=_e_damage_digest(d_trial),
                    q_after_digest=_s_trial_q_after_digest(
                        target, after.displacement, d_trial, H_next
                    ),
                    candidate_kind=candidate_kind,
                    trial_alpha=trial_alpha,
                    candidate_fp_error=candidate_fp_error,
                    candidate_fp_scale=candidate_fp_scale,
                    candidate_damage_residual_scale=candidate_damage_residual_scale,
                    candidate_hard_gate_attribution=(
                        None
                    ),
                )
                
                trials.append(trial)
                if globalization_pass:
                    
                    candidate_pass = coupled_pass
                    anderson_strict = False
                    
                    if mode == STABILIZED_WARNING_ACCEPTANCE_MODE:
                        # Warning mode must not classify a raw range/healing
                        # defect as a strict pass.  The strict stabilized path
                        # intentionally keeps its historical coupled-gate
                        # definition unchanged.
                        candidate_pass = bool(
                            coupled_pass
                            and base_diag.damage_range_pass
                            and base_diag.healing_pass
                        )
                    failure_reason = _s_stabilized_failure_reason(
                        base_diag,
                        fixed_pass,
                        damage_eq_pass,
                        coupled_pass,
                        include_raw_gates=(
                            mode == STABILIZED_WARNING_ACCEPTANCE_MODE
                            or anderson_strict
                        ),
                    )
                    selected_diag = _s_wrap_sweep(
                        base_diag,
                        complete_sweep=True,
                        passed=candidate_pass,
                        failure_stage="" if candidate_pass else "gates",
                        failure_reason="" if candidate_pass else failure_reason,
                        d_solved=d_solved,
                        d_trial=d_trial,
                        omega=omega,
                        reference_merit=reference_merit,
                        merit=merit,
                        sufficient_decrease_limit=sufficient_limit,
                        fixed_point_error=fixed_point_error,
                        fixed_point_scale=fixed_point_scale,
                        fixed_point_pass=fixed_pass,
                        damage_equation_pass=damage_eq_pass,
                        globalization_pass=True,
                        coupled_pass=coupled_pass,
                        trials=trials,
                        anderson_history_size=anderson_history_size,
                        anderson_qr_rank=anderson_qr_rank,
                        anderson_coefficient_l2=anderson_coefficient_l2,
                        anderson_regularization_mu=anderson_regularization_mu,
                        anderson_current_residual_l2=anderson_current_residual_l2,
                        anderson_regularization_lambda=anderson_regularization_lambda,
                        anderson_unregularized_ls_residual_l2=(
                            anderson_unregularized_ls_residual_l2
                        ),
                        anderson_selection=anderson_selection,
                        trial_alpha=trial_alpha,
                        anderson_fallback_reason=anderson_fallback_reason,
                        anderson_restart_reason=anderson_restart_reason,
                        candidate_fp_error=candidate_fp_error,
                        candidate_fp_scale=candidate_fp_scale,
                        candidate_damage_residual_scale=(
                            candidate_damage_residual_scale
                        ),
                    )
                    selected_after = after
                    break
            except (RuntimeError, FloatingPointError) as error:
                mechanics_reuse = None
                last_base_diag = _s_incomplete_base(
                    model,
                    config,
                    committed,
                    d_iter,
                    target,
                    "mechanics_after",
                    str(error),
                    sweep_index,
                    before=before,
                    after=candidate_after,
                    psi_before=psi_before,
                    H_used=H_used,
                    d_next=d_trial,
                    rho_used=rho_used,
                    rho_used_volume=rho_used_volume,
                )
                last_candidate_after = candidate_after
                last_candidate_damage = np.array(d_trial, dtype=np.float64, copy=True)
                trial = StabilizedRelaxationTrial(
                    omega=omega,
                    complete=False,
                    hard_pass=False,
                    globalization_pass=False,
                    coupled_pass=False,
                    selected=False,
                    merit=None,
                    sufficient_decrease_limit=sufficient_limit,
                    failure_stage="mechanics_after",
                    failure_reason=str(error),
                    candidate_diagnostics=last_base_diag,
                    d_trial=d_trial,
                    q_after=None
                    if candidate_after is None
                    else candidate_after.displacement,
                    d_trial_digest=_e_damage_digest(d_trial),
                    q_after_digest=None,
                    candidate_kind=candidate_kind,
                    trial_alpha=trial_alpha,
                    candidate_hard_gate_attribution=(
                                None
                    ),
                )
                trials.append(trial)
                

            
            break
            next_omega = _s_relaxation_omega(config, backtrack_count + 1)
            if next_omega == omega:
                break

        if selected_diag is None:
            # No candidate satisfied the policy selection condition.  Preserve
            # the last finite candidate's base fields when available, but do
            # not update d_iter or the committed state.
            mechanics_reuse = None
            selection_failure_reason = (
                "fixed relaxation candidate failed hard gates"
            )
            if last_base_diag is None:
                base_diag = _s_incomplete_base(
                    model,
                    config,
                    committed,
                    d_iter,
                    target,
                    "globalization",
                    selection_failure_reason,
                    sweep_index,
                    before=before,
                    psi_before=psi_before,
                    H_used=H_used,
                    rho_used=rho_used,
                    rho_used_volume=rho_used_volume,
                )
                last_candidate_damage = None
            else:
                base_diag = last_base_diag
            diag = _s_wrap_sweep(
                base_diag,
                complete_sweep=False,
                passed=False,
                failure_stage="globalization",
                failure_reason=selection_failure_reason,
                d_solved=d_solved,
                d_trial=last_candidate_damage,
                omega=trials[-1].omega if trials else None,
                reference_merit=reference_merit,
                merit=trials[-1].merit if trials else None,
                sufficient_decrease_limit=(
                    trials[-1].sufficient_decrease_limit if trials else None
                ),
                fixed_point_error=fixed_point_error,
                fixed_point_scale=fixed_point_scale,
                fixed_point_pass=fixed_pass,
                damage_equation_pass=damage_eq_pass,
                globalization_pass=False,
                coupled_pass=False,
                trials=trials,
                anderson_history_size=anderson_history_size,
                anderson_qr_rank=anderson_qr_rank,
                anderson_coefficient_l2=anderson_coefficient_l2,
                anderson_regularization_mu=anderson_regularization_mu,
                anderson_current_residual_l2=anderson_current_residual_l2,
                anderson_regularization_lambda=anderson_regularization_lambda,
                anderson_unregularized_ls_residual_l2=(
                    anderson_unregularized_ls_residual_l2
                ),
                anderson_selection=(
                    anderson_selection
                ),
                trial_alpha=(
                    trials[-1].trial_alpha
                    if trials and trials[-1].candidate_kind == "anderson_accelerated"
                    else None
                ),
                anderson_fallback_reason=anderson_fallback_reason,
                anderson_restart_reason=anderson_restart_reason,
            )
            retain_sweep(
                diag,
                "incomplete_sweep",
                cached_q_before=cached_q_before,
            )
            return _s_rejected_attempt(
                committed,
                target,
                cutback_count,
                sweeps,
                diag,
                acceptance_mode=mode,
            )

        warning_acceptance = False
        warning_bypassed: tuple[str, ...] = ()
        warning_range_excess: float | None = None
        warning_healing_magnitude: float | None = None
        if (
            mode == STABILIZED_WARNING_ACCEPTANCE_MODE
            and not selected_diag.passed
            and sweep_index + 1 == base.max_stagger_iterations
        ):
            (
                warning_acceptance,
                warning_bypassed,
                warning_range_excess,
                warning_healing_magnitude,
            ) = _s_stabilized_warning_acceptance_decision(selected_diag)

        

        if selected_diag.passed or warning_acceptance:
            mechanics_reuse = None
            retain_sweep(
                selected_diag,
                "complete_sweep",
                cached_q_before=cached_q_before,
            )
            assert selected_after is not None
            assert selected_diag.d_trial is not None
            assert selected_diag.H_next is not None
            state = EngineeringGripState(
                u_bar=target,
                union_displacement=selected_after.displacement,
                material_damage=selected_diag.d_trial,
                material_history=selected_diag.H_next,
            )
            step = StabilizedEngineeringGripAcceptedStep(
                load_start=committed.u_bar,
                load_target=target,
                increment=target - committed.u_bar,
                state=state,
                diagnostics=selected_diag,
                stagger_iterations=sweep_index + 1,
                cutback_count=cutback_count,
                acceptance_status=(
                    STABILIZED_WARNING_ACCEPTANCE_STATUS
                    if warning_acceptance
                    else STABILIZED_ACCEPTANCE_STATUS
                ),
                bypassed_gates=warning_bypassed,
                warning_range_excess=warning_range_excess,
                warning_healing_magnitude=warning_healing_magnitude,
                acceptance_mode=mode,
            )
            attempt = StabilizedEngineeringGripAttempt(
                cutback_count=cutback_count,
                load_start=committed.u_bar,
                load_target=target,
                accepted=True,
                acceptance_status=(
                    STABILIZED_WARNING_ACCEPTANCE_STATUS
                    if warning_acceptance
                    else STABILIZED_ACCEPTANCE_STATUS
                ),
                failure_stage=None,
                failure_reason=None,
                exception_type=None,
                committed_before_digest=committed.digest,
                committed_after_digest=state.digest,
                sweeps=tuple(sweeps),
                diagnostics=selected_diag,
                bypassed_gates=warning_bypassed,
                warning_range_excess=warning_range_excess,
                warning_healing_magnitude=warning_healing_magnitude,
                acceptance_mode=mode,
            )
            return step, attempt

        retain_sweep(
            selected_diag,
            "complete_sweep",
            cached_q_before=cached_q_before,
        )
        if sweep_index + 1 == base.max_stagger_iterations:
            mechanics_reuse = None
            return _s_rejected_attempt(
                committed,
                target,
                cutback_count,
                sweeps,
                selected_diag,
                acceptance_mode=mode,
            )
        assert selected_after is not None
        d_iter = np.array(selected_diag.d_trial, dtype=np.float64, copy=True)
        q_seed = _e_direct_seed(
            selected_after.displacement, target, target, model.constraints
        )
        if reuse_selected_mechanics:
            mechanics_reuse = _s_StabilizedMechanicsReuse.from_selected(
                model,
                target,
                d_iter,
                selected_after,
            )
        else:
            mechanics_reuse = None

    raise RuntimeError("stabilized attempt produced no final diagnostic")

def _s_rejected_attempt(
    committed: EngineeringGripState,
    target: float,
    cutback_count: int,
    sweeps: Sequence[StabilizedRelaxationSweepDiagnostics],
    diagnostic: StabilizedRelaxationSweepDiagnostics,
    *,
    acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE,
) -> tuple[None, StabilizedEngineeringGripAttempt]:
    attempt = StabilizedEngineeringGripAttempt(
        cutback_count=cutback_count,
        load_start=committed.u_bar,
        load_target=target,
        accepted=False,
        acceptance_status=None,
        failure_stage=diagnostic.failure_stage or "gates",
        failure_reason=diagnostic.failure_reason or "stabilized sweep failed",
        exception_type=StabilizedEngineeringGripSweepFailure.__name__,
        committed_before_digest=committed.digest,
        committed_after_digest=committed.digest,
        sweeps=tuple(sweeps),
        diagnostics=diagnostic,
        acceptance_mode=_s_validate_stabilized_acceptance_mode(acceptance_mode),
    )
    return None, attempt

def _s_validate_progress_settings(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    progress_every_sweeps: int | None,
) -> int | None:
    """Validate the transient progress callback/cadence pair."""

    if progress_callback is None:
        if progress_every_sweeps is not None:
            raise ValueError(
                "progress_every_sweeps requires progress_callback"
            )
        return None
    if not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    if progress_every_sweeps is None:
        raise ValueError(
            "progress_every_sweeps is required when progress_callback is provided"
        )
    return _e_integer(
        progress_every_sweeps,
        "progress_every_sweeps",
        positive=True,
    )

def run_stabilized_engineering_grip(
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    mechanics_adapter: JaxA2CPUReplayAdapter,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_every_sweeps: int | None = None,
    _reuse_selected_mechanics: bool = True,
) -> StabilizedEngineeringGripResult:
    """Run the adopted fresh-zero fixed damped-Picard continuation."""

    if not isinstance(model, EngineeringGripModel) or not isinstance(
        config, StabilizedEngineeringGripConfig
    ):
        raise TypeError("run_stabilized_engineering_grip requires model/config types")
    progress_every = _s_validate_progress_settings(
        progress_callback, progress_every_sweeps
    )
    started = time.perf_counter()
    if type(_reuse_selected_mechanics) is not bool:
        raise TypeError("_reuse_selected_mechanics must be a boolean")
    adapter = _e_validated_mechanics_adapter(model, mechanics_adapter)
    try:
        work_check = run_zero_damage_work_check(
            model,
            config.base.work_check_load,
            config.base.work_epsilon,
            config.base.work_relative_limit,
            config.base.cg_rtol,
            config.base.cg_atol,
            config.base.cg_maxiter,
            mechanics_adapter=adapter,
        )
    except Exception as error:
        _s_emit_failure_progress(
            progress_callback,
            reason=str(error),
            committed_load=0.0,
            target_load=None,
            accepted_steps=0,
            attempts=0,
            failure_type=type(error).__name__,
            started=started,
        )
        raise
    initial = EngineeringGripState(
        u_bar=0.0,
        union_displacement=np.zeros(model.n_dofs, dtype=np.float64),
        material_damage=np.zeros(model.n_material, dtype=np.float64),
        material_history=np.zeros(model.n_material, dtype=np.float64),
    )
    states = [initial]
    steps: list[StabilizedEngineeringGripAcceptedStep] = []
    attempts: list[StabilizedEngineeringGripAttempt] = []
    committed = initial
    increment = config.base.du_initial
    cutbacks = 0
    while committed.u_bar < config.base.load_final:
        target = _e_next_target(
            committed.u_bar, increment, config.base.load_final
        )
        if target <= committed.u_bar:
            message = "stabilized continuation made no load progress"
            _s_emit_failure_progress(
                progress_callback,
                reason=message,
                committed_load=committed.u_bar,
                target_load=target,
                accepted_steps=len(steps),
                attempts=len(attempts),
                failure_type=StabilizedEngineeringGripContinuationFailure.__name__,
                started=started,
            )
            raise StabilizedEngineeringGripContinuationFailure(
                message,
                committed,
                tuple(attempts),
                work_check,
                failure_code=_s_stabilized_failure_code(config.acceptance_mode),
                acceptance_mode=config.acceptance_mode,
            )
        _s_emit_progress_event(
            progress_callback,
            {
                "event": "attempt_start",
                "step_index": len(steps) + 1,
                "committed_load": committed.u_bar,
                "target_load": target,
                "increment": target - committed.u_bar,
                "cutback_count": cutbacks,
                "max_sweeps": config.base.max_stagger_iterations,
                "omega": config.omega_initial,
                "policy": config.relaxation_policy,
            },
            started,
        )
        step, attempt = _s_run_stabilized_attempt(
            model,
            config,
            committed,
            target,
            cutbacks,
            len(steps) + 1,
            adapter,
            reuse_selected_mechanics=_reuse_selected_mechanics,
            progress_callback=progress_callback,
            progress_every_sweeps=progress_every,
            progress_started=started,
            acceptance_mode=config.acceptance_mode,
        )
        attempts.append(attempt)
        if step is not None:
            steps.append(step)
            committed = step.state
            states.append(committed)
            increment = step.increment
            cutbacks = 0
            if progress_callback is not None:
                _s_emit_progress_event(
                    progress_callback,
                    _s_progress_accepted_event(model, config, step),
                    started,
                )
            continue
        if attempt.committed_before_digest != attempt.committed_after_digest:
            raise RuntimeError("stabilized rejected attempt changed committed state")
        at_minimum = _e_is_minimum_increment(
            target - committed.u_bar,
            config.base.du_min,
            committed.u_bar,
            target,
        )
        at_cutback_limit = cutbacks >= config.base.max_cutbacks
        if at_minimum or at_cutback_limit:
            reasons = []
            if at_minimum:
                reasons.append("du_min")
            if at_cutback_limit:
                reasons.append("max_cutbacks")
            message = (
                "stabilized continuation failed at "
                f"u_bar={target:.17g} ({', '.join(reasons)}): {attempt.failure_reason}"
            )
            _s_emit_failure_progress(
                progress_callback,
                reason=message,
                committed_load=committed.u_bar,
                target_load=target,
                accepted_steps=len(steps),
                attempts=len(attempts),
                failure_type=StabilizedEngineeringGripContinuationFailure.__name__,
                started=started,
            )
            raise StabilizedEngineeringGripContinuationFailure(
                message,
                committed,
                tuple(attempts),
                work_check,
                failure_code=_s_stabilized_failure_code(config.acceptance_mode),
                acceptance_mode=config.acceptance_mode,
            )
        old_increment = target - committed.u_bar
        cutbacks += 1
        increment = max(
            config.base.du_min,
            old_increment * config.base.cutback_factor,
        )
        _s_emit_progress_event(
            progress_callback,
            {
                "event": "cutback",
                "rejected_target": target,
                "reason": attempt.failure_reason,
                "old_increment": old_increment,
                "new_increment": increment,
                "cutback_count": cutbacks,
                "committed_load": committed.u_bar,
            },
            started,
        )
    result = StabilizedEngineeringGripResult(
        model=model,
        config=config,
        work_check=work_check,
        initial_state=initial,
        states=tuple(states),
        accepted_steps=tuple(steps),
        attempts=tuple(attempts),
        _acceptance_mode=config.acceptance_mode,
        completion_status=_s_stabilized_contract(
            config.acceptance_mode,
            failure=False,
        )["completion_status"],
    )
    _s_emit_progress_event(
        progress_callback,
        {
            "event": "completed",
            "load": result.final_state.u_bar,
            "accepted_steps": len(result.accepted_steps),
            "attempts": len(result.attempts),
        },
        started,
    )
    return result





























SOURCE_CLOUD = Path(
    "/root/autodl-tmp/data/fracture2d/engineering_grip/source/material_cloud_d048.npz"
)
LEGACY_SOLUTION = Path(
    "/root/autodl-tmp/data/fracture2d/engineering_grip/"
    "engineering_grip_damped_warning_o5_u7p5e3_dumin6p25e6_s144.npz"
)
SCHEME = {
    "delta": 0.048,
    "h_g": 0.02,
    "youngs_modulus": 210.0e3,
    "poisson_ratio": 0.3,
    "k_res": 1.0e-8,
    "rank_rtol": 1.0e-12,
    "Gc": 2.7,
    "l0": 0.024,
    "visibility_tolerance": 1.0e-12,
    "du_initial": 1.0e-4,
    "du_min": 6.25e-6,
    "max_stagger_iterations": 144,
    "omega": 0.5,
    "acceptance_mode": "stabilized_warning",
    "load_final": 0.0075,
}
DEFAULT_PLOT_LOADS = (0.0058, 0.0060, 0.0062, 0.0065, 0.0070, 0.0075)
DRIFT_ENVELOPE = 1.08
SOLUTION_SCHEMA = "fracture2d-direct-nonlocal-warning-solution-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_cloud(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "coordinates", "triangles", "volumes", "bottom", "top",
            "anchor_particle", "d048",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"source cloud is missing fields: {missing}")
        return {name: np.array(archive[name], copy=True) for name in required}


def _build_model(args: argparse.Namespace):
    import jax

    jax.config.update("jax_enable_x64", True)
    pass
    pass

    cloud = _load_cloud(Path(args.source_cloud))
    geometry = build_union_geometry_from_arrays(
        cloud["coordinates"], cloud["triangles"], cloud["volumes"],
        cloud["bottom"], cloud["top"], int(cloud["anchor_particle"]),
        cloud["d048"], args.delta, args.h_g, args.visibility_tolerance,
    )
    model = build_engineering_grip_model(
        geometry, args.youngs_modulus, args.poisson_ratio, args.k_res,
        args.rank_rtol, args.gc, args.l0,
    )
    return model, cloud


def _build_warning_config(args: argparse.Namespace):
    pass
    pass

    gate = GateTolerance(1.0e-9, 1.0e-8, 1.0)
    residual = GateTolerance(1.0e-10, 1.0e-8, 1.0)
    base = EngineeringGripConfig(
        load_start=0.0, load_final=args.load_final,
        du_initial=args.du_initial, du_min=args.du_min,
        cutback_factor=0.5, max_cutbacks=8,
        max_stagger_iterations=args.max_stagger_iterations,
        cg_rtol=1.0e-10, cg_atol=1.0e-13, cg_maxiter=3000,
        displacement=gate, damage=gate, history=gate, reaction=gate,
        damage_residual=residual, force=gate, moment=gate,
        healing_atol=1.0e-10, damage_range_atol=1.0e-10,
        work_check_load=1.0e-4, work_epsilon=1.0e-5,
        work_relative_limit=1.0e-7,
    )
    return StabilizedEngineeringGripConfig(
        base=base, omega_initial=0.5, omega_min=0.5,
        acceptance_mode=STABILIZED_WARNING_ACCEPTANCE_MODE,
    )


def _build_adapter(args: argparse.Namespace, model):
    import jax

    pass

    gpu_devices = tuple(device for device in jax.devices() if device.platform == "gpu")
    if not gpu_devices:
        raise RuntimeError(
            "direct_nonlocal_solver run requires a visible JAX GPU; "
            "no CPU fallback is provided"
        )
    if args.gpu_device_index >= len(gpu_devices):
        raise ValueError(
            f"gpu-device-index {args.gpu_device_index} is out of range; "
            f"visible GPU count is {len(gpu_devices)}"
        )
    device = gpu_devices[args.gpu_device_index]
    adapter = JaxA2CPUReplayAdapter.from_static(
        model.static, model.constraints, device
    )
    return adapter, str(device)


def _status_is_warning(status: str) -> bool:
    return status.endswith("warning_accepted")


def _drift_mask(loads: np.ndarray, q: np.ndarray, envelope: float) -> np.ndarray:
    """Material-point mask: True where the field is retained for plotting."""

    if loads.ndim != 1 or q.ndim != 3 or q.shape[0] != len(loads):
        raise ValueError("drift mask requires loads (T,) and material q (T, N, 2)")
    speeds = np.linalg.norm(q, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        amplification = np.where(loads[:, None] > 0.0, speeds / loads[:, None], 0.0)
    return amplification <= envelope


def _retained_range(values: np.ndarray, retained: np.ndarray) -> tuple[float, float]:
    if values.shape != retained.shape:
        raise ValueError("retained range requires matching value/mask shapes")
    if not np.any(retained):
        raise ValueError("retained range removed every point")
    kept = values[retained]
    return float(np.min(kept)), float(np.max(kept))


def _save_compact_solution(path: Path, model, cloud, result, scheme: dict) -> None:
    states = result.states
    loads = np.asarray([state.u_bar for state in states], dtype=np.float64)
    union_q = np.stack([np.asarray(state.q, dtype=np.float64) for state in states])
    damage = np.stack([np.asarray(state.d, dtype=np.float64) for state in states])
    history = np.stack([np.asarray(state.H, dtype=np.float64) for state in states])
    reaction = np.concatenate((
        np.asarray([0.0], dtype=np.float64),
        np.asarray(
            [step.diagnostics.top_reaction_after for step in result.accepted_steps],
            dtype=np.float64,
        ),
    ))
    status = np.asarray(
        ["initial", *[step.acceptance_status for step in result.accepted_steps]],
        dtype=np.str_,
    )
    warning = np.asarray(
        [False, *[_status_is_warning(str(item)) for item in status[1:]]],
        dtype=bool,
    )
    metadata = {
        "schema": SOLUTION_SCHEMA,
        "source_cloud": str(Path(scheme["source_cloud"])),
        "source_cloud_sha256": scheme["source_cloud_sha256"],
        "model_digest": model.model_digest,
        "static_digest": model.static_digest,
        "acceptance_mode": scheme["acceptance_mode"],
        "scheme": {key: value for key, value in scheme.items() if key != "source_cloud_sha256"},
        "field_equations_validated": False,
        "validated": False,
        "experimental": True,
    }
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing solution: {path}")
    np.savez_compressed(
        path,
        loads=loads, union_q=union_q, damage=damage, history=history,
        top_reaction=reaction, acceptance_status=status, warning=warning,
        coordinates=cloud["coordinates"], triangles=cloud["triangles"],
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _load_solution(path: Path, source_cloud: Path | None) -> dict:
    """Normalize a compact solution or a legacy stabilized archive."""

    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        if "metadata_json" in names and "union_q" in names:
            metadata = json.loads(str(archive["metadata_json"]))
            if metadata.get("schema") != SOLUTION_SCHEMA:
                raise ValueError("compact solution schema mismatch")
            coordinates = np.asarray(archive["coordinates"], dtype=np.float64)
            triangles = np.asarray(archive["triangles"], dtype=np.int64)
            union_q = np.asarray(archive["union_q"], dtype=np.float64)
            damage = np.asarray(archive["damage"], dtype=np.float64)
            history = np.asarray(archive["history"], dtype=np.float64)
            loads = np.asarray(archive["loads"], dtype=np.float64)
            reaction = np.asarray(archive["top_reaction"], dtype=np.float64)
            status = np.asarray(archive["acceptance_status"], dtype=np.str_)
            source_sha = metadata.get("source_cloud_sha256")
        elif "state_u_bar" in names and "state_union_displacement" in names:
            if source_cloud is None:
                raise ValueError(
                    "legacy stabilized archives store no geometry; pass --source-cloud"
                )
            cloud = _load_cloud(source_cloud)
            coordinates = cloud["coordinates"]
            triangles = cloud["triangles"]
            loads = np.asarray(archive["state_u_bar"], dtype=np.float64)
            union_q = np.asarray(archive["state_union_displacement"], dtype=np.float64)
            damage = np.asarray(archive["state_material_damage"], dtype=np.float64)
            history = np.asarray(archive["state_material_history"], dtype=np.float64)
            reaction = np.concatenate((
                np.asarray([0.0], dtype=np.float64),
                np.asarray(archive["accepted_top_reaction_after"], dtype=np.float64),
            ))
            status = np.asarray(archive["state_acceptance_status"], dtype=np.str_)
            source_sha = _sha256(source_cloud)
        else:
            raise ValueError(f"unrecognized solution archive layout: {sorted(names)}")
    material_dofs = 2 * len(coordinates)
    if union_q.shape[1] < material_dofs:
        raise ValueError("solution union displacement is smaller than the material cloud")
    q = union_q[:, :material_dofs].reshape(len(loads), len(coordinates), 2)
    if damage.shape != (len(loads), len(coordinates)):
        raise ValueError("solution damage shape does not match the material cloud")
    if history.shape != damage.shape or reaction.shape != loads.shape:
        raise ValueError("solution history/reaction shape mismatch")
    if status.shape != loads.shape:
        raise ValueError("solution acceptance-status shape mismatch")
    return {
        "coordinates": coordinates,
        "triangles": triangles,
        "loads": loads,
        "q": q,
        "damage": damage,
        "history": history,
        "reaction": reaction,
        "status": status,
        "source_cloud_sha256": source_sha,
    }


def _solution_statistics(solution: dict, envelope: float) -> dict:
    loads, q, damage = solution["loads"], solution["q"], solution["damage"]
    retained = _drift_mask(loads, q, envelope)
    outliers = [int(np.count_nonzero(~retained[i])) for i in range(len(loads))]
    amplification = np.zeros(len(loads))
    for index in range(1, len(loads)):
        if loads[index] > 0.0:
            amplification[index] = float(
                np.max(np.linalg.norm(q[index], axis=1)) / loads[index]
            )
    warnings = [str(item) for item in solution["status"]]
    return {
        "states": int(len(loads)),
        "warning_states": int(sum(_status_is_warning(item) for item in warnings)),
        "final_load": float(loads[-1]),
        "final_reaction": float(solution["reaction"][-1]),
        "peak_reaction": float(np.max(solution["reaction"])),
        "peak_load": float(loads[int(np.argmax(solution["reaction"]))]),
        "final_max_amplification": float(amplification[-1]),
        "outlier_counts": {
            f"{float(loads[i]):.7f}": outliers[i] for i in range(len(loads))
        },
        "final_retained_range_ux": _retained_range(
            q[-1, :, 0], retained[-1]
        ),
        "final_retained_range_uy": _retained_range(
            q[-1, :, 1], retained[-1]
        ),
        "final_retained_range_damage": _retained_range(
            damage[-1], retained[-1]
        ),
    }


def _plot(solution: dict, output_dir: Path, loads_wanted: tuple[float, ...],
          envelope: float) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    output_dir.mkdir(parents=True, exist_ok=True)
    coordinates = solution["coordinates"]
    triangulation = mtri.Triangulation(
        coordinates[:, 0], coordinates[:, 1], solution["triangles"]
    )
    retained = _drift_mask(solution["loads"], solution["q"], envelope)

    selected: list[int] = []
    for wanted in loads_wanted:
        matches = np.flatnonzero(np.isclose(solution["loads"], wanted, rtol=0.0, atol=1e-12))
        if len(matches) == 1:
            selected.append(int(matches[0]))
    if not selected:
        raise ValueError("none of the requested plot loads exist in the solution")

    fields = (
        ("$u_x$", solution["q"][:, :, 0]),
        ("$u_y$", solution["q"][:, :, 1]),
        ("$|u|$", np.linalg.norm(solution["q"], axis=2)),
        ("$d$", solution["damage"]),
    )
    ranges: dict[str, dict] = {}
    figure, axes = plt.subplots(
        len(fields), len(selected),
        figsize=(3.1 * len(selected), 2.9 * len(fields)),
        constrained_layout=True, squeeze=False,
    )
    for column, index in enumerate(selected):
        load = float(solution["loads"][index])
        keep = retained[index]
        count = int(np.count_nonzero(~keep))
        for row, (label, values) in enumerate(fields):
            if row < 3:
                panel = values[index]
                if row < 2:
                    bound = float(np.max(np.abs(panel[keep])))
                    vmin, vmax = -bound, bound
                else:
                    vmin, vmax = _retained_range(panel, keep)
            else:
                vmin, vmax = _retained_range(values[index], keep)
            mappable = axes[row, column].tripcolor(
                triangulation, values[index], shading="gouraud",
                cmap="coolwarm", vmin=vmin, vmax=vmax,
            )
            colorbar = figure.colorbar(
                mappable, ax=axes[row, column], fraction=0.046, pad=0.03
            )
            colorbar.ax.tick_params(labelsize=5)
            if count:
                axes[row, column].plot(
                    coordinates[~keep, 0], coordinates[~keep, 1],
                    linestyle="none", marker="o", markersize=2.5,
                    markerfacecolor="none", markeredgecolor="black",
                    markeredgewidth=0.5,
                )
            if row == 0:
                axes[row, column].set_title(
                    f"u={load:.4f}  reaction={solution['reaction'][index]:.3f}\n"
                    f"drift points drawn: {count}", fontsize=7,
                )
            if column == 0:
                axes[row, column].set_ylabel(label, fontsize=8)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            key = f"{load:.7f}"
            ranges.setdefault(key, {})[label] = {
                "vmin": vmin, "vmax": vmax, "drift_points": count,
            }
    fields_path = output_dir / "fields_drift_excluded_ranges.png"
    figure.savefig(fields_path, dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    axis.plot(solution["loads"], solution["reaction"], color="darkorange",
              label="nonlocal warning path")
    warning_mask = np.asarray(
        [False, *[_status_is_warning(str(item)) for item in solution["status"][1:]]]
    )
    axis.scatter(solution["loads"][warning_mask], solution["reaction"][warning_mask],
                 marker="x", color="crimson", label="warning-accepted")
    axis.set_xlabel("imposed displacement")
    axis.set_ylabel("reaction")
    axis.legend(loc="best", fontsize=8)
    curve_path = output_dir / "response_curve.png"
    figure.savefig(curve_path, dpi=170)
    plt.close(figure)

    summary = {
        "fields_figure": str(fields_path),
        "curve_figure": str(curve_path),
        "selected_loads": [float(solution["loads"][i]) for i in selected],
        "drift_envelope": envelope,
        "ranges": ranges,
    }
    (output_dir / "plot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _run_command(args: argparse.Namespace) -> int:
    pass

    if Path(args.output).exists():
        raise FileExistsError(f"refusing to overwrite existing solution: {args.output}")
    model, cloud = _build_model(args)
    config = _build_warning_config(args)
    adapter, device = _build_adapter(args, model)
    started = time.perf_counter()

    def progress(event):
        if event["event"] == "attempt_start":
            print(json.dumps({
                "event": "attempt_start", "step": event["step_index"],
                "load": event["target_load"],
                "elapsed": time.perf_counter() - started,
            }, sort_keys=True), flush=True)

    result = run_stabilized_engineering_grip(
        model, config, mechanics_adapter=adapter,
        progress_callback=progress if args.progress else None,
        progress_every_sweeps=args.progress_every if args.progress else None,
    )
    scheme = dict(SCHEME)
    scheme["source_cloud"] = str(Path(args.source_cloud))
    scheme["source_cloud_sha256"] = _sha256(Path(args.source_cloud))
    scheme["load_final"] = float(args.load_final)
    _save_compact_solution(Path(args.output), model, cloud, result, scheme)
    stats = _solution_statistics(
        _load_solution(Path(args.output), None), DRIFT_ENVELOPE
    )
    print(json.dumps({
        "event": "run_complete", "device": device, "output": str(args.output),
        "sha256": _sha256(Path(args.output)), "statistics": stats,
        "elapsed": time.perf_counter() - started,
    }, sort_keys=True), flush=True)
    return 0


def _verify_command(args: argparse.Namespace) -> int:
    archive = Path(args.archive)
    if not archive.exists():
        raise FileNotFoundError(f"solution archive not found: {archive}")
    solution = _load_solution(
        archive, Path(args.source_cloud) if args.source_cloud else None
    )
    stats = _solution_statistics(solution, args.envelope)
    print(json.dumps({
        "event": "verify", "archive": str(archive),
        "sha256": _sha256(archive),
        "source_cloud_sha256": solution["source_cloud_sha256"],
        "statistics": stats,
    }, sort_keys=True, indent=2), flush=True)
    return 0


def _plot_command(args: argparse.Namespace) -> int:
    solution = _load_solution(
        Path(args.archive), Path(args.source_cloud) if args.source_cloud else None
    )
    summary = _plot(
        solution, Path(args.output_dir), tuple(args.loads), args.envelope
    )
    print(json.dumps({"event": "plot_complete", **summary}, sort_keys=True), flush=True)
    return 0


def _add_geometry_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-cloud", default=str(SOURCE_CLOUD))
    parser.add_argument("--delta", type=float, default=SCHEME["delta"])
    parser.add_argument("--h-g", type=float, default=SCHEME["h_g"])
    parser.add_argument("--visibility-tolerance", type=float,
                        default=SCHEME["visibility_tolerance"])
    parser.add_argument("--youngs-modulus", type=float, default=SCHEME["youngs_modulus"])
    parser.add_argument("--poisson-ratio", type=float, default=SCHEME["poisson_ratio"])
    parser.add_argument("--k-res", type=float, default=SCHEME["k_res"])
    parser.add_argument("--rank-rtol", type=float, default=SCHEME["rank_rtol"])
    parser.add_argument("--gc", type=float, default=SCHEME["Gc"])
    parser.add_argument("--l0", type=float, default=SCHEME["l0"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="fresh zero-start warning continuation")
    run.set_defaults(handler=_run_command)
    _add_geometry_arguments(run)
    run.add_argument("--output", required=True)
    run.add_argument("--gpu-device-index", type=int, default=0)
    run.add_argument("--load-final", type=float, default=SCHEME["load_final"])
    run.add_argument("--du-initial", type=float, default=SCHEME["du_initial"])
    run.add_argument("--du-min", type=float, default=SCHEME["du_min"])
    run.add_argument("--max-stagger-iterations", type=int,
                     default=SCHEME["max_stagger_iterations"])
    run.add_argument("--progress", action="store_true")
    run.add_argument("--progress-every", type=int, default=24)

    verify = subparsers.add_parser("verify", help="audit a solution archive")
    verify.set_defaults(handler=_verify_command)
    verify.add_argument("--archive", default=str(LEGACY_SOLUTION))
    verify.add_argument("--source-cloud", default=str(SOURCE_CLOUD))
    verify.add_argument("--envelope", type=float, default=DRIFT_ENVELOPE)

    plot = subparsers.add_parser("plot", help="drift-excluded field plots")
    plot.set_defaults(handler=_plot_command)
    plot.add_argument("--archive", default=str(LEGACY_SOLUTION))
    plot.add_argument("--source-cloud", default=str(SOURCE_CLOUD))
    plot.add_argument("--output-dir", required=True)
    plot.add_argument("--envelope", type=float, default=DRIFT_ENVELOPE)
    plot.add_argument("--loads", type=float, nargs="+", default=list(DEFAULT_PLOT_LOADS))

    arguments = parser.parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
