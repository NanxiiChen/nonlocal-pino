"""Stage 1 particles, quadrature, boundary tags, and slot visibility.

This module is deliberately independent of the local FEniCS reference solver.
It consumes the triangle and physical-group data in a Gmsh/meshio mesh and
returns the fixed data needed by later direct nonlocal operators.  It does not
assemble a mechanics or damage operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


__all__ = [
    "BoundaryMasks",
    "BoundaryTags",
    "ParticleCloud",
    "ROUNDED_SLOT_TAGS",
    "build_particle_cloud",
    "load_gmsh_particle_cloud",
    "segment_is_visible",
]


@dataclass(frozen=True)
class BoundaryTags:
    """Names of the required Gmsh physical groups.

    The names are part of the mesh-to-particle contract.  In particular, the
    lower-left anchor is a tagged physical point; it is not inferred from a
    coordinate tolerance.
    """

    domain: str
    bottom: str
    top: str
    outer_right: str
    outer_left_upper: str
    outer_left_lower: str
    slot_upper: str
    slot_lower: str
    slot_tip: str
    anchor: str

    @property
    def all_names(self) -> tuple[str, ...]:
        return (
            self.domain,
            self.bottom,
            self.top,
            self.outer_right,
            self.outer_left_upper,
            self.outer_left_lower,
            self.slot_upper,
            self.slot_lower,
            self.slot_tip,
            self.anchor,
        )

    @property
    def boundary_curve_names(self) -> tuple[str, ...]:
        return (
            self.bottom,
            self.top,
            self.outer_right,
            self.outer_left_upper,
            self.outer_left_lower,
            self.slot_upper,
            self.slot_lower,
            self.slot_tip,
        )


ROUNDED_SLOT_TAGS = BoundaryTags(
    domain="domain",
    bottom="bottom",
    top="top",
    outer_right="right",
    outer_left_upper="left_upper",
    outer_left_lower="left_lower",
    slot_upper="notch_upper",
    slot_lower="notch_lower",
    slot_tip="notch_tip",
    anchor="anchor",
)


# This is a numerical predicate floor, not a physical clearance.  It is
# multiplied by a geometry length scale before it is used as an absolute
# distance tolerance.
_GEOMETRY_EPS_FACTOR = 64.0
_DIRECT_BASELINE_DTYPE = np.dtype(np.float64)


def _readonly_array(values: Any, dtype: Any | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class BoundaryMasks:
    """Deterministic particle masks obtained from physical mesh entities."""

    bottom: np.ndarray
    top: np.ndarray
    outer_right: np.ndarray
    outer_left_upper: np.ndarray
    outer_left_lower: np.ndarray
    slot_upper: np.ndarray
    slot_lower: np.ndarray
    slot_tip: np.ndarray
    anchor_mask: np.ndarray
    anchor_particle: int

    def __post_init__(self) -> None:
        mask_names = (
            "bottom",
            "top",
            "outer_right",
            "outer_left_upper",
            "outer_left_lower",
            "slot_upper",
            "slot_lower",
            "slot_tip",
            "anchor_mask",
        )
        for name in mask_names:
            mask = np.asarray(getattr(self, name), dtype=bool)
            if mask.ndim != 1:
                raise ValueError(f"Boundary mask {name} must be one-dimensional")
            object.__setattr__(self, name, _readonly_array(mask, dtype=bool))
        mask_lengths = {len(getattr(self, name)) for name in mask_names}
        if len(mask_lengths) != 1:
            raise ValueError("All boundary masks must have the same particle length")
        if int(self.anchor_particle) != self.anchor_particle or self.anchor_particle < 0:
            raise ValueError("anchor_particle must be a non-negative integer")
        if self.anchor_particle >= len(self.anchor_mask):
            raise ValueError("anchor_particle is outside the boundary-mask length")
        if np.count_nonzero(self.anchor_mask) != 1:
            raise ValueError("anchor_mask must contain exactly one particle")
        if not self.anchor_mask[int(self.anchor_particle)]:
            raise ValueError("anchor_particle does not agree with anchor_mask")

    @property
    def outer_sides(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(
            {
                "right": self.outer_right,
                "left_upper": self.outer_left_upper,
                "left_lower": self.outer_left_lower,
            }
        )

    @property
    def slot_faces(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(
            {
                "upper": self.slot_upper,
                "lower": self.slot_lower,
            }
        )


@dataclass(frozen=True)
class ParticleCloud:
    """Fixed float64 Stage 1 data for a rounded-slot material particle cloud.

    ``visibility_tolerance`` stores the caller-supplied absolute input.  The
    visibility predicate applies a per-geometry effective tolerance that also
    includes its declared scale-aware machine floor.  The direct fracture2d
    baseline serializes geometry, quadrature, and graph data as float64; a
    float32 graph is a separate future discretization that must be rebuilt and
    revalidated rather than obtained by casting this cloud.
    """

    coordinates: np.ndarray
    triangles: np.ndarray
    triangle_areas: np.ndarray
    volumes: np.ndarray
    first_moments: np.ndarray
    boundary: BoundaryMasks
    candidate_edges: np.ndarray
    visible_edges: np.ndarray
    removed_edges: np.ndarray
    neighbor_counts: np.ndarray
    physical_tag_ids: Mapping[str, int]
    horizon: float
    visibility_tolerance: float

    def __post_init__(self) -> None:
        array_specs = (
            ("coordinates", _DIRECT_BASELINE_DTYPE),
            ("triangles", np.int64),
            ("triangle_areas", _DIRECT_BASELINE_DTYPE),
            ("volumes", _DIRECT_BASELINE_DTYPE),
            ("first_moments", _DIRECT_BASELINE_DTYPE),
            ("candidate_edges", np.int64),
            ("visible_edges", np.int64),
            ("removed_edges", np.int64),
            ("neighbor_counts", np.int64),
        )
        for name, dtype in array_specs:
            raw_array = np.asarray(getattr(self, name))
            if name in {
                "coordinates",
                "triangle_areas",
                "volumes",
                "first_moments",
            } and raw_array.dtype != _DIRECT_BASELINE_DTYPE:
                raise ValueError(
                    f"{name} must use float64 for the fracture2d direct "
                    f"baseline, got {raw_array.dtype}"
                )
            if name in {
                "triangles",
                "candidate_edges",
                "visible_edges",
                "removed_edges",
                "neighbor_counts",
            } and not np.issubdtype(raw_array.dtype, np.integer):
                raise ValueError(f"{name} must contain integer values")
            object.__setattr__(
                self,
                name,
                _readonly_array(raw_array, dtype=dtype),
            )
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ValueError(
                f"coordinates must have shape (N, 2), got {self.coordinates.shape}"
            )
        n_particles = len(self.coordinates)
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(
                f"triangles must have shape (N, 3), got {self.triangles.shape}"
            )
        if self.triangle_areas.shape != (len(self.triangles),):
            raise ValueError("triangle_areas must have one value per triangle")
        if self.volumes.shape != (n_particles,):
            raise ValueError("volumes must have one value per particle")
        if self.first_moments.shape != (2,):
            raise ValueError("first_moments must have shape (2,)")
        for name in ("candidate_edges", "visible_edges", "removed_edges"):
            edges = getattr(self, name)
            if edges.ndim != 2 or edges.shape[1] != 2:
                raise ValueError(f"{name} must have shape (N, 2)")
        if self.neighbor_counts.shape != (n_particles,):
            raise ValueError("neighbor_counts must have one value per particle")
        if len(self.triangles) and (
            np.min(self.triangles) < 0 or np.max(self.triangles) >= n_particles
        ):
            raise ValueError("triangles contain an invalid particle index")
        if np.any(self.neighbor_counts < 0):
            raise ValueError("neighbor_counts must be non-negative")
        if not np.all(np.isfinite(self.coordinates)):
            raise ValueError("coordinates must be finite")
        if not np.all(np.isfinite(self.triangle_areas)) or np.any(
            self.triangle_areas <= 0.0
        ):
            raise ValueError("triangle_areas must be finite and positive")
        if not np.all(np.isfinite(self.volumes)) or np.any(self.volumes <= 0.0):
            raise ValueError("volumes must be finite and positive")
        if not np.all(np.isfinite(self.first_moments)):
            raise ValueError("first_moments must be finite")
        object.__setattr__(
            self,
            "physical_tag_ids",
            MappingProxyType(
                {str(key): int(value) for key, value in self.physical_tag_ids.items()}
            ),
        )
        if not np.isfinite(self.horizon) or self.horizon <= 0.0:
            raise ValueError("horizon must be positive and finite")
        if not np.isfinite(self.visibility_tolerance) or self.visibility_tolerance < 0.0:
            raise ValueError("visibility_tolerance must be finite and non-negative")

    @property
    def material_area(self) -> float:
        return float(np.sum(self.volumes))

    @property
    def boundary_masks(self) -> BoundaryMasks:
        """Alias exposing the tag-derived masks for downstream operators."""
        return self.boundary

    @property
    def candidate_pair_count(self) -> int:
        return int(len(self.candidate_edges) // 2)

    @property
    def visible_pair_count(self) -> int:
        return int(len(self.visible_edges) // 2)

    @property
    def removed_pair_count(self) -> int:
        return int(len(self.removed_edges) // 2)

    @property
    def neighbor_min(self) -> int:
        return int(np.min(self.neighbor_counts)) if len(self.neighbor_counts) else 0

    @property
    def neighbor_mean(self) -> float:
        return float(np.mean(self.neighbor_counts)) if len(self.neighbor_counts) else 0.0

    @property
    def neighbor_max(self) -> int:
        return int(np.max(self.neighbor_counts)) if len(self.neighbor_counts) else 0


def _validate_visibility_options(
    horizon: float,
    visibility_tolerance: float,
) -> None:
    try:
        horizon_value = float(horizon)
        tolerance_value = float(visibility_tolerance)
    except (TypeError, ValueError) as error:
        raise ValueError("horizon and visibility_tolerance must be scalar numbers") from error
    if not np.isfinite(horizon_value) or horizon_value <= 0.0:
        raise ValueError(f"horizon must be positive and finite, got {horizon}")
    if not np.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise ValueError(
            "visibility_tolerance must be finite and non-negative, "
            f"got {visibility_tolerance}"
        )


def _polygon_length_scale(vertices: np.ndarray) -> float:
    edge_lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
    extent = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    scale = max(float(np.max(edge_lengths)), extent)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Polygon geometry must have a positive finite length scale")
    return scale


def _effective_visibility_tolerance(
    input_tolerance: float,
    geometry_length_scale: float,
) -> float:
    if not np.isfinite(geometry_length_scale) or geometry_length_scale <= 0.0:
        raise ValueError("Visibility geometry length scale must be positive and finite")
    machine_floor = _GEOMETRY_EPS_FACTOR * np.finfo(float).eps * geometry_length_scale
    return max(input_tolerance, machine_floor)


def _segment_geometry_length_scale(
    polygon_length_scale: float,
    opening_segment: np.ndarray,
    segment_direction: np.ndarray,
) -> float:
    return max(
        polygon_length_scale,
        float(np.linalg.norm(opening_segment[1] - opening_segment[0])),
        float(np.linalg.norm(segment_direction)),
    )


def _coerce_connectivity(values: Any, name: str, width: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width}), got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError(f"{name} must contain integer node indices")
    return np.asarray(array, dtype=np.int64)


def _coerce_tag_values(values: Any, name: str, expected_count: int) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if len(array) != expected_count:
        raise ValueError(
            f"Physical tag array for {name!r} has {len(array)} entries; "
            f"expected {expected_count}"
        )
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError(f"Physical tags for {name!r} must be integers")
    return np.asarray(array, dtype=np.int64)


def _mesh_cells(mesh: Any) -> dict[str, np.ndarray]:
    cells_dict = getattr(mesh, "cells_dict", None)
    if cells_dict is not None:
        return {str(name): np.asarray(values) for name, values in cells_dict.items()}

    cells = getattr(mesh, "cells", None)
    if cells is None:
        raise ValueError("mesh data must expose mesh.cells_dict or mesh.cells")
    if isinstance(cells, Mapping):
        return {str(name): np.asarray(values) for name, values in cells.items()}

    result: dict[str, list[np.ndarray]] = {}
    for block in cells:
        if hasattr(block, "type") and hasattr(block, "data"):
            cell_type = str(block.type)
            values = block.data
        elif isinstance(block, tuple) and len(block) == 2:
            cell_type = str(block[0])
            values = block[1]
        else:
            raise ValueError("Unsupported mesh cell block; expected meshio CellBlock")
        result.setdefault(cell_type, []).append(np.asarray(values))
    return {
        name: np.concatenate(values, axis=0) for name, values in result.items()
    }


def _mesh_physical_cell_data(mesh: Any) -> Mapping[str, Any]:
    cell_data_dict = getattr(mesh, "cell_data_dict", None)
    if cell_data_dict is not None:
        physical = cell_data_dict.get("gmsh:physical")
        if physical is not None:
            return physical
    raise ValueError(
        "Gmsh physical tags are required; mesh data has no "
        "cell_data_dict['gmsh:physical']"
    )


def _physical_group_ids(mesh: Any, tags: BoundaryTags) -> dict[str, int]:
    field_data = getattr(mesh, "field_data", None)
    if field_data is None:
        raise ValueError("Gmsh field_data with physical names is required")
    if len(set(tags.all_names)) != len(tags.all_names):
        raise ValueError(f"Physical tag names must be unique, got {tags.all_names}")

    expected_dimensions = {
        tags.domain: 2,
        tags.bottom: 1,
        tags.top: 1,
        tags.outer_right: 1,
        tags.outer_left_upper: 1,
        tags.outer_left_lower: 1,
        tags.slot_upper: 1,
        tags.slot_lower: 1,
        tags.slot_tip: 1,
        tags.anchor: 0,
    }
    result: dict[str, int] = {}
    for name, expected_dimension in expected_dimensions.items():
        if name not in field_data:
            raise ValueError(
                f"Required Gmsh physical name {name!r} is missing from field_data"
            )
        value = np.asarray(field_data[name]).reshape(-1)
        if len(value) != 2 or not np.all(np.isfinite(value)):
            raise ValueError(f"field_data[{name!r}] must contain [tag, dimension]")
        if value[0] != np.floor(value[0]) or value[1] != np.floor(value[1]):
            raise ValueError(f"field_data[{name!r}] must contain integer [tag, dimension]")
        group_id = int(value[0])
        dimension = int(value[1])
        if dimension != expected_dimension:
            raise ValueError(
                f"Physical group {name!r} has dimension {dimension}; "
                f"expected {expected_dimension}"
            )
        result[name] = group_id
    if len(set(result.values())) != len(result):
        raise ValueError(f"Required physical groups must have unique IDs, got {result}")
    return result


def _require_tagged_cells(
    cells: Mapping[str, np.ndarray],
    physical: Mapping[str, Any],
    cell_type: str,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    if cell_type not in cells:
        raise ValueError(f"Gmsh mesh has no {cell_type!r} cells")
    connectivity = _coerce_connectivity(cells[cell_type], cell_type, width)
    if cell_type not in physical:
        raise ValueError(f"Gmsh {cell_type} cells have no physical tags")
    tags = _coerce_tag_values(physical[cell_type], cell_type, len(connectivity))
    return connectivity, tags


def _validate_node_indices(connectivity: np.ndarray, n_points: int, name: str) -> None:
    if len(connectivity) and (
        np.min(connectivity) < 0 or np.max(connectivity) >= n_points
    ):
        raise ValueError(f"{name} contains node indices outside [0, {n_points})")


def _sorted_particle_data(
    points: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    material_nodes = np.unique(triangles)
    material_points = points[material_nodes]
    order = np.lexsort((material_points[:, 1], material_points[:, 0]))
    ordered_original_nodes = material_nodes[order]
    coordinates = points[ordered_original_nodes]
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError("Material particle coordinates must be unique")

    old_to_new = np.full(len(points), -1, dtype=np.int64)
    old_to_new[ordered_original_nodes] = np.arange(len(ordered_original_nodes))
    remapped_triangles = old_to_new[triangles]
    if np.any(remapped_triangles < 0):
        raise ValueError("Triangle connectivity references a non-material particle")
    if np.any(
        (remapped_triangles[:, 0] == remapped_triangles[:, 1])
        | (remapped_triangles[:, 0] == remapped_triangles[:, 2])
        | (remapped_triangles[:, 1] == remapped_triangles[:, 2])
    ):
        raise ValueError("Mesh contains a degenerate triangle with repeated nodes")
    remapped_triangles = np.sort(remapped_triangles, axis=1)
    triangle_order = np.lexsort(
        (
            remapped_triangles[:, 2],
            remapped_triangles[:, 1],
            remapped_triangles[:, 0],
        )
    )
    remapped_triangles = remapped_triangles[triangle_order]
    if len(np.unique(remapped_triangles, axis=0)) != len(remapped_triangles):
        raise ValueError("Mesh contains duplicate triangles")
    return coordinates, remapped_triangles, old_to_new


def _barycentric_quadrature(
    coordinates: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0 = coordinates[triangles[:, 0]]
    p1 = coordinates[triangles[:, 1]]
    p2 = coordinates[triangles[:, 2]]
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (
        p1[:, 1] - p0[:, 1]
    ) * (p2[:, 0] - p0[:, 0])
    areas = 0.5 * np.abs(cross)
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError("Mesh contains a non-positive or non-finite triangle area")

    volumes = np.zeros(len(coordinates), dtype=float)
    contribution = areas / 3.0
    np.add.at(volumes, triangles[:, 0], contribution)
    np.add.at(volumes, triangles[:, 1], contribution)
    np.add.at(volumes, triangles[:, 2], contribution)
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise ValueError("Barycentric nodal volumes must be finite and positive")

    area_sum = float(np.sum(areas))
    volume_sum = float(np.sum(volumes))
    if not np.isfinite(area_sum) or not np.isfinite(volume_sum):
        raise ValueError("Triangle area and nodal-volume sums must be finite")
    if not np.isclose(volume_sum, area_sum, rtol=0.0, atol=32.0 * np.finfo(float).eps):
        raise ValueError(
            f"Barycentric volume sum {volume_sum} disagrees with triangle area {area_sum}"
        )
    first_moments = np.sum(volumes[:, None] * coordinates, axis=0)
    if not np.all(np.isfinite(first_moments)):
        raise ValueError("Barycentric first moments must be finite")
    return areas, volumes, first_moments


def _sorted_directed_edges(pairs: np.ndarray) -> np.ndarray:
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    directed = np.concatenate((pairs, pairs[:, ::-1]), axis=0)
    order = np.lexsort((directed[:, 1], directed[:, 0]))
    return np.asarray(directed[order], dtype=np.int64)


def _validate_directed_edges(edges: np.ndarray, n_particles: int, name: str) -> None:
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {edges.shape}")
    if len(edges) and (
        np.min(edges) < 0 or np.max(edges) >= n_particles
    ):
        raise ValueError(f"{name} contains an invalid particle index")
    if len(edges) and np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError(f"{name} must not contain self edges")
    if len(edges) > 1:
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        if not np.array_equal(order, np.arange(len(edges))):
            raise ValueError(f"{name} is not deterministically sorted")
    edge_set = {tuple(edge) for edge in edges.tolist()}
    if len(edge_set) != len(edges):
        raise ValueError(f"{name} contains duplicate directed edges")
    if any((j, i) not in edge_set for i, j in edge_set):
        raise ValueError(f"{name} is not reciprocal")


def _edge_key(first: int, second: int) -> tuple[int, int]:
    first_value = int(first)
    second_value = int(second)
    if first_value == second_value:
        raise ValueError("Mesh edge has repeated endpoint indices")
    return (
        (first_value, second_value)
        if first_value < second_value
        else (second_value, first_value)
    )


def _triangle_edge_incidence(triangles: np.ndarray) -> dict[tuple[int, int], int]:
    incidence: dict[tuple[int, int], int] = {}
    for triangle in triangles:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            key = _edge_key(first, second)
            incidence[key] = incidence.get(key, 0) + 1
    nonmanifold = sorted(
        (edge, count) for edge, count in incidence.items() if count > 2
    )
    if nonmanifold:
        raise ValueError(
            "Material triangle edge has incidence greater than two: "
            f"{nonmanifold[0][0]} has incidence {nonmanifold[0][1]}"
        )
    return incidence


def _validate_physical_boundary_edges(
    triangles: np.ndarray,
    line_cells: np.ndarray,
    line_tags: np.ndarray,
    physical_ids: Mapping[str, int],
    tags: BoundaryTags,
) -> set[tuple[int, int]]:
    """Require physical line cells to be exactly the material mesh boundary."""
    incidence = _triangle_edge_incidence(triangles)
    line_name_by_id = {
        physical_ids[name]: name for name in tags.boundary_curve_names
    }
    tagged_edges: dict[tuple[int, int], tuple[int, str]] = {}
    for edge, group_id in zip(line_cells, line_tags):
        group_value = int(group_id)
        if group_value not in line_name_by_id:
            raise ValueError(
                f"Unknown physical line tag {group_value}; expected one of "
                f"{sorted(line_name_by_id)}"
            )
        key = _edge_key(edge[0], edge[1])
        edge_incidence = incidence.get(key, 0)
        if edge_incidence == 0:
            raise ValueError(
                f"Physical line edge {key} is not a material mesh edge"
            )
        if edge_incidence != 1:
            raise ValueError(
                f"Physical line edge {key} is internal; triangle incidence is "
                f"{edge_incidence}, not one"
            )
        if key in tagged_edges:
            previous_id, previous_name = tagged_edges[key]
            current_name = line_name_by_id[group_value]
            if previous_id != group_value:
                raise ValueError(
                    f"Physical boundary edge {key} has inconsistent tags "
                    f"{previous_name!r} and {current_name!r}"
                )
            raise ValueError(
                f"Physical boundary edge {key} is duplicated with tag "
                f"{current_name!r}"
            )
        tagged_edges[key] = (group_value, line_name_by_id[group_value])

    material_boundary = {
        edge for edge, edge_incidence in incidence.items() if edge_incidence == 1
    }
    tagged_boundary = set(tagged_edges)
    missing = sorted(material_boundary - tagged_boundary)
    extra = sorted(tagged_boundary - material_boundary)
    if missing:
        raise ValueError(
            "Material boundary edge(s) are untagged; first missing edge is "
            f"{missing[0]}"
        )
    if extra:
        raise ValueError(
            "Physical line edge(s) are not in the complete material boundary; "
            f"first extra edge is {extra[0]}"
        )
    return tagged_boundary


def _cross2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _point_on_segment(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    tolerance: float,
) -> bool:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length == 0.0:
        return float(np.linalg.norm(point - start)) <= tolerance
    distance = abs(_cross2d(point - start, direction)) / length
    if distance > tolerance:
        return False
    projection = float((point - start) @ direction)
    margin = tolerance * length
    return -margin <= projection <= float(direction @ direction) + margin


def _segments_intersect_or_are_ambiguous(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    tolerance: float,
) -> bool:
    """Conservatively detect intersection or tolerance-scale ambiguity."""
    if any(
        _point_on_segment(point, second_start, second_end, tolerance)
        for point in (first_start, first_end)
    ) or any(
        _point_on_segment(point, first_start, first_end, tolerance)
        for point in (second_start, second_end)
    ):
        return True

    first_direction = first_end - first_start
    second_direction = second_end - second_start
    orientation = (
        _cross2d(first_direction, second_start - first_start),
        _cross2d(first_direction, second_end - first_start),
        _cross2d(second_direction, first_start - second_start),
        _cross2d(second_direction, first_end - second_start),
    )
    scale = max(
        float(np.linalg.norm(first_direction)),
        float(np.linalg.norm(second_direction)),
    )
    if scale <= 0.0:
        raise ValueError("Polygon edges must have positive length")
    orientation_tolerance = tolerance * scale
    if all(abs(value) <= orientation_tolerance for value in orientation):
        axis = int(np.argmax(np.abs(first_direction)))
        first_interval = sorted(
            (float(first_start[axis]), float(first_end[axis]))
        )
        second_interval = sorted(
            (float(second_start[axis]), float(second_end[axis]))
        )
        return min(first_interval[1], second_interval[1]) + tolerance >= max(
            first_interval[0], second_interval[0]
        )
    first_opposite = (
        (orientation[0] > orientation_tolerance and orientation[1] < -orientation_tolerance)
        or (orientation[0] < -orientation_tolerance and orientation[1] > orientation_tolerance)
    )
    second_opposite = (
        (orientation[2] > orientation_tolerance and orientation[3] < -orientation_tolerance)
        or (orientation[2] < -orientation_tolerance and orientation[3] > orientation_tolerance)
    )
    return first_opposite and second_opposite


def _validate_simple_polygon(vertices: np.ndarray, tolerance: float) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError(
            f"Slot polygon must have shape (N, 2) with N >= 3, got {vertices.shape}"
        )
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Slot polygon vertices must be finite")
    scale = _polygon_length_scale(vertices)
    effective_tolerance = _effective_visibility_tolerance(tolerance, scale)
    edge_lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
    if np.any(edge_lengths <= effective_tolerance):
        raise ValueError("Slot polygon contains a zero-length or ambiguous edge")
    for index in range(len(vertices)):
        incoming = vertices[index] - vertices[index - 1]
        outgoing = vertices[(index + 1) % len(vertices)] - vertices[index]
        edge_scale = max(
            float(np.linalg.norm(incoming)), float(np.linalg.norm(outgoing))
        )
        if abs(_cross2d(incoming, outgoing)) <= effective_tolerance * edge_scale and (
            float(incoming @ outgoing) < -effective_tolerance * edge_scale
        ):
            raise ValueError(
                "Slot polygon contains overlapping adjacent edges at vertex "
                f"{index}"
            )
    twice_area = float(
        np.sum(
            vertices[:, 0] * np.roll(vertices[:, 1], -1)
            - vertices[:, 1] * np.roll(vertices[:, 0], -1)
        )
    )
    area_tolerance = max(
        effective_tolerance * scale,
        _GEOMETRY_EPS_FACTOR * np.finfo(float).eps * scale * scale,
    )
    if abs(twice_area) <= area_tolerance:
        raise ValueError("Slot polygon has zero or ambiguous area")

    edge_count = len(vertices)
    for first_index in range(edge_count):
        first_start = vertices[first_index]
        first_end = vertices[(first_index + 1) % edge_count]
        for second_index in range(first_index + 1, edge_count):
            if second_index == first_index + 1 or (
                first_index == 0 and second_index == edge_count - 1
            ):
                continue
            second_start = vertices[second_index]
            second_end = vertices[(second_index + 1) % edge_count]
            if _segments_intersect_or_are_ambiguous(
                first_start,
                first_end,
                second_start,
                second_end,
                effective_tolerance,
            ):
                raise ValueError(
                    "Slot polygon has intersecting or tolerance-ambiguous "
                    f"non-adjacent edges {first_index} and {second_index}"
                )


def _build_neighbor_graph(
    coordinates: np.ndarray,
    polygon: np.ndarray,
    opening_segment: np.ndarray,
    horizon: float,
    visibility_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - project dependency is SciPy.
        raise ImportError(
            "Stage 1 neighbor construction requires scipy.spatial.cKDTree"
        ) from error

    tree = cKDTree(coordinates)
    raw_pairs = np.asarray(
        tree.query_pairs(float(horizon), output_type="ndarray"), dtype=np.int64
    )
    if raw_pairs.size:
        pairs = raw_pairs.reshape((-1, 2))
        pair_order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        pairs = pairs[pair_order]
        distances = np.linalg.norm(
            coordinates[pairs[:, 1]] - coordinates[pairs[:, 0]], axis=1
        )
        pairs = pairs[(distances > 0.0) & (distances < horizon)]
    else:
        pairs = np.empty((0, 2), dtype=np.int64)

    polygon_length_scale = _polygon_length_scale(polygon)
    polygon_bounds = (np.min(polygon, axis=0), np.max(polygon, axis=0))
    input_tolerance = float(visibility_tolerance)
    visible_flags = np.asarray(
        [
            _segment_is_visible_arrays(
                coordinates[i],
                coordinates[j],
                polygon,
                opening_segment,
                input_tolerance,
                polygon_length_scale,
                polygon_bounds,
            )
            for i, j in pairs
        ],
        dtype=bool,
    )
    visible_pairs = pairs[visible_flags]
    removed_pairs = pairs[~visible_flags]
    candidate_edges = _sorted_directed_edges(pairs)
    visible_edges = _sorted_directed_edges(visible_pairs)
    removed_edges = _sorted_directed_edges(removed_pairs)
    _validate_directed_edges(candidate_edges, len(coordinates), "candidate_edges")
    _validate_directed_edges(visible_edges, len(coordinates), "visible_edges")
    _validate_directed_edges(removed_edges, len(coordinates), "removed_edges")
    if not set(map(tuple, visible_edges.tolist())).issubset(
        set(map(tuple, candidate_edges.tolist()))
    ):
        raise ValueError("Visible edges must be a subset of candidate edges")
    if not set(map(tuple, removed_edges.tolist())).isdisjoint(
        set(map(tuple, visible_edges.tolist()))
    ):
        raise ValueError("Visible and removed edge sets overlap")
    neighbor_counts = np.bincount(
        visible_edges[:, 0], minlength=len(coordinates)
    ).astype(np.int64, copy=False)
    return candidate_edges, visible_edges, removed_edges, neighbor_counts


def _build_slot_polygon(
    points: np.ndarray,
    line_cells: np.ndarray,
    line_tags: np.ndarray,
    physical_ids: Mapping[str, int],
    physical_boundary_edges: set[tuple[int, int]],
    tags: BoundaryTags,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Order the tagged slot chain and close it across the unmeshed opening."""
    slot_names = (tags.slot_upper, tags.slot_tip, tags.slot_lower)
    slot_ids = {physical_ids[name]: name for name in slot_names}
    outer_left_names = (tags.outer_left_upper, tags.outer_left_lower)
    outer_left_ids = {physical_ids[name]: name for name in outer_left_names}
    outer_left_nodes: dict[str, set[int]] = {
        name: set() for name in outer_left_names
    }
    group_edges: dict[str, list[tuple[int, int]]] = {name: [] for name in slot_names}
    adjacency: dict[int, list[tuple[int, str, tuple[int, int]]]] = {}
    for edge, group_id in zip(line_cells, line_tags):
        name = slot_ids.get(int(group_id))
        if name is not None:
            key = _edge_key(edge[0], edge[1])
            group_edges[name].append(key)
            adjacency.setdefault(key[0], []).append((key[1], name, key))
            adjacency.setdefault(key[1], []).append((key[0], name, key))
            continue
        outer_name = outer_left_ids.get(int(group_id))
        if outer_name is not None:
            outer_left_nodes[outer_name].update(map(int, edge))

    missing_groups = [name for name, edges in group_edges.items() if not edges]
    if missing_groups:
        raise ValueError(
            "Tagged slot topology is incomplete; missing line groups "
            f"{missing_groups}"
        )
    bad_degree = sorted(
        (node, len(neighbors))
        for node, neighbors in adjacency.items()
        if len(neighbors) not in (1, 2)
    )
    if bad_degree:
        raise ValueError(
            "Tagged slot topology is not a simple chain; node "
            f"{bad_degree[0][0]} has degree {bad_degree[0][1]}"
        )
    endpoints = sorted(node for node, neighbors in adjacency.items() if len(neighbors) == 1)
    if len(endpoints) != 2:
        raise ValueError(
            "Tagged slot topology must form one open chain with exactly two "
            f"opening endpoints, got {len(endpoints)}"
        )

    endpoint_groups = {
        node: {name for _, name, _ in adjacency[node]} for node in endpoints
    }
    upper_endpoints = [
        node for node in endpoints if endpoint_groups[node] == {tags.slot_upper}
    ]
    lower_endpoints = [
        node for node in endpoints if endpoint_groups[node] == {tags.slot_lower}
    ]
    if len(upper_endpoints) != 1 or len(lower_endpoints) != 1:
        raise ValueError(
            "Tagged slot topology must have one notch-upper and one notch-lower "
            "opening endpoint"
        )
    start = upper_endpoints[0]
    end = lower_endpoints[0]
    if start not in outer_left_nodes[tags.outer_left_upper] or end not in outer_left_nodes[
        tags.outer_left_lower
    ]:
        raise ValueError(
            "Tagged slot topology opening endpoints must connect to the tagged "
            "left-upper and left-lower outer boundaries"
        )

    ordered_nodes = [start]
    ordered_edge_names: list[str] = []
    visited_edges: set[tuple[int, int]] = set()
    current = start
    while current != end:
        choices = [
            entry for entry in adjacency[current] if entry[2] not in visited_edges
        ]
        if len(choices) != 1:
            raise ValueError(
                "Tagged slot topology cannot be walked as one open chain from "
                f"opening endpoint {start}"
            )
        next_node, edge_name, edge_key = choices[0]
        if next_node in ordered_nodes:
            raise ValueError("Tagged slot topology contains a cycle")
        visited_edges.add(edge_key)
        ordered_edge_names.append(edge_name)
        ordered_nodes.append(next_node)
        current = next_node
    if len(visited_edges) != sum(len(edges) for edges in group_edges.values()):
        raise ValueError("Tagged slot topology is disconnected from its opening chain")

    blocks: list[str] = []
    for name in ordered_edge_names:
        if not blocks or blocks[-1] != name:
            blocks.append(name)
    expected_blocks = [tags.slot_upper, tags.slot_tip, tags.slot_lower]
    if blocks != expected_blocks:
        raise ValueError(
            "Tagged slot topology must run notch-upper, notch-tip, notch-lower "
            f"between its opening endpoints, got {blocks}"
        )

    polygon = np.asarray(points[np.asarray(ordered_nodes, dtype=np.int64)], dtype=float)
    opening = np.asarray([polygon[0], polygon[-1]], dtype=float)
    opening_key = _edge_key(start, end)
    if opening_key in physical_boundary_edges:
        raise ValueError(
            "The slot opening closure is unexpectedly represented by a physical "
            "boundary edge"
        )
    _validate_simple_polygon(polygon, tolerance)
    return polygon, opening


def _point_location_in_polygon(
    point: np.ndarray,
    polygon: np.ndarray,
    tolerance: float,
) -> int:
    """Return 1 for strict interior, 0 for exterior, and -1 for closure."""
    for index in range(len(polygon)):
        if _point_on_segment(
            point,
            polygon[index],
            polygon[(index + 1) % len(polygon)],
            tolerance,
        ):
            return -1

    inside = False
    for index in range(len(polygon)):
        first = polygon[index]
        second = polygon[(index + 1) % len(polygon)]
        if (first[1] > point[1]) != (second[1] > point[1]):
            crossing_x = first[0] + (second[0] - first[0]) * (
                (point[1] - first[1]) / (second[1] - first[1])
            )
            if crossing_x > point[0]:
                inside = not inside
    return 1 if inside else 0


def _segment_boundary_parameters(
    start: np.ndarray,
    direction: np.ndarray,
    polygon: np.ndarray,
    tolerance: float,
) -> list[float]:
    direction_length = float(np.linalg.norm(direction))
    direction_squared = float(direction @ direction)
    parameter_tolerance = max(
        _GEOMETRY_EPS_FACTOR * np.finfo(float).eps,
        tolerance / direction_length,
    )
    edge_starts = polygon
    edge_directions = np.roll(polygon, -1, axis=0) - polygon
    edge_lengths = np.linalg.norm(edge_directions, axis=1)
    if np.any(edge_lengths == 0.0):
        raise ValueError(
            "segment_is_visible requires a validated polygon without "
            "zero-length edges"
        )
    offsets = edge_starts - start
    denominators = direction[0] * edge_directions[:, 1] - direction[1] * edge_directions[:, 0]
    determinant_tolerances = (
        _GEOMETRY_EPS_FACTOR
        * np.finfo(float).eps
        * direction_length
        * edge_lengths
    )
    nonparallel = np.abs(denominators) > determinant_tolerances
    parameters = [0.0, 1.0]
    if np.any(nonparallel):
        nonparallel_denominators = denominators[nonparallel]
        nonparallel_offsets = offsets[nonparallel]
        nonparallel_directions = edge_directions[nonparallel]
        segment_parameters = (
            nonparallel_offsets[:, 0] * nonparallel_directions[:, 1]
            - nonparallel_offsets[:, 1] * nonparallel_directions[:, 0]
        ) / nonparallel_denominators
        edge_parameters = (
            nonparallel_offsets[:, 0] * direction[1]
            - nonparallel_offsets[:, 1] * direction[0]
        ) / nonparallel_denominators
        edge_parameter_tolerances = np.maximum(
            _GEOMETRY_EPS_FACTOR * np.finfo(float).eps,
            tolerance / edge_lengths[nonparallel],
        )
        accepted = (
            (segment_parameters >= -parameter_tolerance)
            & (segment_parameters <= 1.0 + parameter_tolerance)
            & (edge_parameters >= -edge_parameter_tolerances)
            & (edge_parameters <= 1.0 + edge_parameter_tolerances)
        )
        parameters.extend(
            np.clip(segment_parameters[accepted], 0.0, 1.0).tolist()
        )

    parallel = ~nonparallel
    if np.any(parallel):
        parallel_offsets = offsets[parallel]
        collinear = np.abs(
            parallel_offsets[:, 0] * direction[1]
            - parallel_offsets[:, 1] * direction[0]
        ) <= tolerance * direction_length
        if np.any(collinear):
            collinear_starts = edge_starts[parallel][collinear]
            collinear_directions = edge_directions[parallel][collinear]
            edge_start_parameters = (
                (collinear_starts - start) @ direction
            ) / direction_squared
            edge_end_parameters = (
                (collinear_starts + collinear_directions - start) @ direction
            ) / direction_squared
            overlaps = (
                np.maximum(edge_start_parameters, edge_end_parameters)
                >= -parameter_tolerance
            ) & (
                np.minimum(edge_start_parameters, edge_end_parameters)
                <= 1.0 + parameter_tolerance
            )
            parameters.extend(
                np.clip(
                    np.column_stack(
                        (edge_start_parameters[overlaps], edge_end_parameters[overlaps])
                    ).reshape(-1),
                    0.0,
                    1.0,
                ).tolist()
            )
    return parameters


def _unique_parameters(parameters: list[float]) -> list[float]:
    ordered = sorted(float(np.clip(value, 0.0, 1.0)) for value in parameters)
    if not ordered:
        return [0.0, 1.0]
    unique = [ordered[0]]
    for value in ordered[1:]:
        if value != unique[-1]:
            unique.append(value)
    return unique


def _interval_is_polygon_boundary(
    start: np.ndarray,
    direction: np.ndarray,
    lower: float,
    upper: float,
    polygon: np.ndarray,
) -> bool:
    boundary_tolerance = _GEOMETRY_EPS_FACTOR * np.finfo(float).eps * _polygon_length_scale(
        polygon
    )
    sample_parameters = (
        lower + 0.25 * (upper - lower),
        lower + 0.5 * (upper - lower),
        lower + 0.75 * (upper - lower),
    )
    samples = [start + parameter * direction for parameter in sample_parameters]
    for index in range(len(polygon)):
        if all(
            _point_on_segment(
                sample,
                polygon[index],
                polygon[(index + 1) % len(polygon)],
                boundary_tolerance,
            )
            for sample in samples
        ):
            return True
    return False


def _segment_overlaps_opening(
    start: np.ndarray,
    end: np.ndarray,
    opening: np.ndarray,
    tolerance: float,
) -> bool:
    opening_start, opening_end = opening
    opening_direction = opening_end - opening_start
    opening_length = float(np.linalg.norm(opening_direction))
    if opening_length == 0.0:
        raise ValueError("Slot opening closure endpoints must be distinct")
    segment_direction = end - start
    segment_length = float(np.linalg.norm(segment_direction))
    if (
        abs(_cross2d(start - opening_start, opening_direction)) / opening_length
        > tolerance
        or abs(_cross2d(end - opening_start, opening_direction)) / opening_length
        > tolerance
    ):
        return False
    unit = opening_direction / opening_length
    segment_projection = np.asarray(
        [(start - opening_start) @ unit, (end - opening_start) @ unit],
        dtype=float,
    )
    overlap = min(float(np.max(segment_projection)), opening_length) - max(
        float(np.min(segment_projection)), 0.0
    )
    return overlap > max(
        tolerance,
        _GEOMETRY_EPS_FACTOR * np.finfo(float).eps * segment_length,
    )


def _segment_is_visible_arrays(
    start: np.ndarray,
    end: np.ndarray,
    polygon: np.ndarray,
    opening_segment: np.ndarray,
    input_tolerance: float,
    polygon_length_scale: float,
    polygon_bounds: tuple[np.ndarray, np.ndarray],
) -> bool:
    direction = end - start
    direction_length = float(np.linalg.norm(direction))
    if direction_length == 0.0:
        raise ValueError("segment endpoints must be distinct")
    geometry_length_scale = _segment_geometry_length_scale(
        polygon_length_scale,
        opening_segment,
        direction,
    )
    effective_tolerance = _effective_visibility_tolerance(
        input_tolerance,
        geometry_length_scale,
    )
    if _segment_overlaps_opening(
        start,
        end,
        opening_segment,
        effective_tolerance,
    ):
        return False
    polygon_min, polygon_max = polygon_bounds
    segment_min = np.minimum(start, end)
    segment_max = np.maximum(start, end)
    if np.any(segment_max < polygon_min - effective_tolerance) or np.any(
        segment_min > polygon_max + effective_tolerance
    ):
        return True

    parameters = _unique_parameters(
        _segment_boundary_parameters(
            start,
            direction,
            polygon,
            effective_tolerance,
        ),
    )
    for lower, upper in zip(parameters[:-1], parameters[1:]):
        if upper == lower:
            continue
        midpoint = start + 0.5 * (lower + upper) * direction
        location = _point_location_in_polygon(
            midpoint,
            polygon,
            effective_tolerance,
        )
        if location == 1:
            return False
        if location == 0:
            continue
        if _interval_is_polygon_boundary(
            start,
            direction,
            lower,
            upper,
            polygon,
        ):
            continue
        quarter = start + (lower + 0.25 * (upper - lower)) * direction
        three_quarter = start + (lower + 0.75 * (upper - lower)) * direction
        quarter_location = _point_location_in_polygon(
            quarter,
            polygon,
            effective_tolerance,
        )
        three_quarter_location = _point_location_in_polygon(
            three_quarter,
            polygon,
            effective_tolerance,
        )
        if quarter_location == 1 or three_quarter_location == 1:
            return False
        raise ValueError(
            "Segment visibility is ambiguous within the supplied absolute "
            "tolerance near a polygon boundary"
        )
    return True


def segment_is_visible(
    start: Any,
    end: Any,
    polygon: Any,
    opening_segment: Any,
    tolerance: float,
) -> bool:
    """Return visibility against one polygonized slot and its open closure.

    The candidate segment is split at polygon-edge intersections.  It is
    removed only when an open subinterval has a strict polygon-interior sample;
    segments on a polygon edge and tangent segments are retained.  The
    unmeshed opening closure is treated separately as void, so a segment with
    positive overlap along that line is removed.  ``tolerance`` is an explicit
    absolute distance tolerance.  A non-collinear segment whose classification
    remains within that tolerance of the polygon boundary raises instead of
    being silently classified.  ``polygon`` must be the ordered, simple,
    nondegenerate polygon produced by the validated mesh loader; this public
    predicate checks array shape and finiteness but does not revalidate polygon
    topology on every graph edge.
    """
    try:
        tolerance_value = float(tolerance)
    except (TypeError, ValueError) as error:
        raise ValueError("tolerance must be a scalar number") from error
    if not np.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise ValueError(
            f"tolerance must be finite and non-negative, got {tolerance}"
        )
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    polygon_array = np.asarray(polygon, dtype=float)
    opening_array = np.asarray(opening_segment, dtype=float)
    if start_array.shape != (2,) or end_array.shape != (2,):
        raise ValueError(
            f"segment endpoints must have shape (2,), got {start_array.shape} and "
            f"{end_array.shape}"
        )
    if polygon_array.ndim != 2 or polygon_array.shape[1:] != (2,) or len(polygon_array) < 3:
        raise ValueError(
            "polygon must have shape (N, 2) with N >= 3, "
            f"got {polygon_array.shape}"
        )
    if opening_array.shape != (2, 2):
        raise ValueError(
            f"opening_segment must have shape (2, 2), got {opening_array.shape}"
        )
    if not all(
        np.all(np.isfinite(array))
        for array in (start_array, end_array, polygon_array, opening_array)
    ):
        raise ValueError("Visibility geometry and segment endpoints must be finite")
    return _segment_is_visible_arrays(
        start_array,
        end_array,
        polygon_array,
        opening_array,
        tolerance_value,
        _polygon_length_scale(polygon_array),
        (np.min(polygon_array, axis=0), np.max(polygon_array, axis=0)),
    )


def _build_boundary_masks(
    material_original_nodes: np.ndarray,
    old_to_new: np.ndarray,
    line_cells: np.ndarray,
    line_tags: np.ndarray,
    vertex_cells: np.ndarray,
    vertex_tags: np.ndarray,
    physical_ids: Mapping[str, int],
    tags: BoundaryTags,
) -> BoundaryMasks:
    line_name_by_id = {
        physical_ids[name]: name for name in tags.boundary_curve_names
    }
    line_nodes: dict[str, set[int]] = {name: set() for name in line_name_by_id.values()}
    for edge, group_id in zip(line_cells, line_tags):
        if int(group_id) not in line_name_by_id:
            raise ValueError(
                f"Unknown physical line tag {int(group_id)}; expected one of "
                f"{sorted(line_name_by_id)}"
            )
        if edge[0] == edge[1]:
            raise ValueError("Physical boundary contains a degenerate line cell")
        line_nodes[line_name_by_id[int(group_id)]].update(map(int, edge))
    missing = [name for name, nodes in line_nodes.items() if not nodes]
    if missing:
        raise ValueError(f"Required physical boundary groups have no line cells: {missing}")

    material_set = set(map(int, material_original_nodes.tolist()))

    def mask_for(name: str) -> np.ndarray:
        original_nodes = line_nodes[name]
        if not original_nodes.issubset(material_set):
            raise ValueError(
                f"Physical group {name!r} references nodes outside material triangles"
            )
        mapped = old_to_new[np.asarray(sorted(original_nodes), dtype=np.int64)]
        if np.any(mapped < 0):
            raise ValueError(f"Physical group {name!r} has unmapped material nodes")
        mask = np.zeros(len(material_original_nodes), dtype=bool)
        mask[mapped] = True
        return mask

    vertex_name_by_id = {physical_ids[tags.anchor]: tags.anchor}
    anchor_original_nodes: set[int] = set()
    for vertex, group_id in zip(vertex_cells[:, 0], vertex_tags):
        if int(group_id) not in vertex_name_by_id:
            raise ValueError(
                f"Unknown physical point tag {int(group_id)}; only the tagged "
                f"anchor {tags.anchor!r} is accepted"
            )
        anchor_original_nodes.add(int(vertex))
    if len(anchor_original_nodes) != 1:
        raise ValueError(
            "The tagged lower-left anchor must contain exactly one mesh point"
        )
    anchor_original = next(iter(anchor_original_nodes))
    if anchor_original not in material_set:
        raise ValueError("The tagged anchor is not a material triangle vertex")

    bottom = mask_for(tags.bottom)
    top = mask_for(tags.top)
    outer_right = mask_for(tags.outer_right)
    outer_left_upper = mask_for(tags.outer_left_upper)
    outer_left_lower = mask_for(tags.outer_left_lower)
    slot_upper = mask_for(tags.slot_upper)
    slot_lower = mask_for(tags.slot_lower)
    slot_tip = mask_for(tags.slot_tip)
    anchor_particle = int(old_to_new[anchor_original])
    anchor_mask = np.zeros(len(material_original_nodes), dtype=bool)
    anchor_mask[anchor_particle] = True
    if not bottom[anchor_particle] or not outer_left_lower[anchor_particle]:
        raise ValueError(
            "The tagged anchor must lie in both the tagged bottom and lower-left "
            "outer boundary groups"
        )
    if np.any(bottom & top):
        raise ValueError("Top and bottom physical particle masks overlap")
    return BoundaryMasks(
        bottom=bottom,
        top=top,
        outer_right=outer_right,
        outer_left_upper=outer_left_upper,
        outer_left_lower=outer_left_lower,
        slot_upper=slot_upper,
        slot_lower=slot_lower,
        slot_tip=slot_tip,
        anchor_mask=anchor_mask,
        anchor_particle=anchor_particle,
    )


def build_particle_cloud(
    mesh: Any,
    *,
    horizon: float,
    boundary_tags: BoundaryTags,
    visibility_tolerance: float,
) -> ParticleCloud:
    """Build a deterministic Stage 1 cloud from meshio-compatible Gmsh data.

    Required mesh data are ``points``, triangle/line/vertex cells,
    ``field_data`` physical names, and ``cell_data_dict['gmsh:physical']``.
    Triangle cells must all carry the named material-surface tag. Physical line
    cells must exactly cover the triangle edges with incidence one. Boundary
    masks and the anchor are taken only from tagged mesh entities; visibility is
    defined by the polygonized tagged slot chain and its unmeshed opening
    closure. The returned ``visibility_tolerance`` is the caller input; the
    predicate adds its scale-aware machine floor internally.
    """
    if not isinstance(boundary_tags, BoundaryTags):
        raise TypeError(
            "boundary_tags must be a BoundaryTags instance; implicit tag names "
            "are not supported"
        )
    _validate_visibility_options(horizon, visibility_tolerance)

    raw_points = np.asarray(getattr(mesh, "points", None), dtype=float)
    if raw_points.ndim != 2 or raw_points.shape[1] < 2 or len(raw_points) == 0:
        raise ValueError(
            f"mesh.points must have shape (N, >=2), got {raw_points.shape}"
        )
    if not np.all(np.isfinite(raw_points)):
        raise ValueError("mesh.points must be finite")
    points = raw_points[:, :2]

    cells = _mesh_cells(mesh)
    physical = _mesh_physical_cell_data(mesh)
    physical_ids = _physical_group_ids(mesh, boundary_tags)
    triangle_cells, triangle_tags = _require_tagged_cells(
        cells, physical, "triangle", 3
    )
    line_cells, line_tags = _require_tagged_cells(cells, physical, "line", 2)
    vertex_cells, vertex_tags = _require_tagged_cells(cells, physical, "vertex", 1)
    _validate_node_indices(triangle_cells, len(points), "triangle cells")
    _validate_node_indices(line_cells, len(points), "line cells")
    _validate_node_indices(vertex_cells, len(points), "vertex cells")
    if np.any(
        (triangle_cells[:, 0] == triangle_cells[:, 1])
        | (triangle_cells[:, 0] == triangle_cells[:, 2])
        | (triangle_cells[:, 1] == triangle_cells[:, 2])
    ):
        raise ValueError("Mesh contains a degenerate triangle with repeated nodes")
    if np.any(triangle_tags != physical_ids[boundary_tags.domain]):
        raise ValueError(
            "Every triangle must carry the named material physical surface "
            f"{boundary_tags.domain!r}"
        )

    physical_boundary_edges = _validate_physical_boundary_edges(
        triangle_cells,
        line_cells,
        line_tags,
        physical_ids,
        boundary_tags,
    )
    slot_polygon, opening_segment = _build_slot_polygon(
        points,
        line_cells,
        line_tags,
        physical_ids,
        physical_boundary_edges,
        boundary_tags,
        float(visibility_tolerance),
    )

    material_original_nodes = np.unique(triangle_cells)
    coordinates, triangles, old_to_new = _sorted_particle_data(
        points, triangle_cells
    )
    triangle_areas, volumes, first_moments = _barycentric_quadrature(
        coordinates, triangles
    )
    boundary = _build_boundary_masks(
        material_original_nodes,
        old_to_new,
        line_cells,
        line_tags,
        vertex_cells,
        vertex_tags,
        physical_ids,
        boundary_tags,
    )
    candidate_edges, visible_edges, removed_edges, neighbor_counts = (
        _build_neighbor_graph(
            coordinates,
            slot_polygon,
            opening_segment,
            float(horizon),
            float(visibility_tolerance),
        )
    )
    return ParticleCloud(
        coordinates=coordinates,
        triangles=triangles,
        triangle_areas=triangle_areas,
        volumes=volumes,
        first_moments=first_moments,
        boundary=boundary,
        candidate_edges=candidate_edges,
        visible_edges=visible_edges,
        removed_edges=removed_edges,
        neighbor_counts=neighbor_counts,
        physical_tag_ids=physical_ids,
        horizon=float(horizon),
        visibility_tolerance=float(visibility_tolerance),
    )


def load_gmsh_particle_cloud(
    msh_path: str | Path,
    *,
    horizon: float,
    boundary_tags: BoundaryTags,
    visibility_tolerance: float,
) -> ParticleCloud:
    """Read a tagged Gmsh ``.msh`` file and build a Stage 1 cloud."""
    path = Path(msh_path)
    if not path.is_file():
        raise FileNotFoundError(f"Gmsh mesh file does not exist: {path}")
    try:
        import meshio
    except ImportError as error:  # pragma: no cover - depends on environment.
        raise ImportError(
            "Reading Gmsh files requires meshio; install/use the mesh environment"
        ) from error
    mesh = meshio.read(path)
    return build_particle_cloud(
        mesh,
        horizon=horizon,
        boundary_tags=boundary_tags,
        visibility_tolerance=visibility_tolerance,
    )
