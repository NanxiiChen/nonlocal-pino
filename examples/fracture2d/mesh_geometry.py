"""Checked rounded-notch visibility and engineering volume-grip layout."""
from collections import defaultdict
from typing import Any
import numpy as np
from scipy.spatial import cKDTree

_g_FLOAT64 = np.dtype(np.float64)
_g_INT64 = np.dtype(np.int64)
_g_GEOMETRY_EPS_FACTOR = 64.0
_g_AREA_RECONSTRUCTION_EPS_FACTOR = 512.0
_g_INTERFACE_INPUT_TOLERANCE = 1e-12

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
