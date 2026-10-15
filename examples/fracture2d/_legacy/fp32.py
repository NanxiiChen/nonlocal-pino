"""Separately rebuilt float32 graph and BA data for the fracture2d PINN.

The float64 direct model is used only for its mesh/collar topology and the
unchanged physical constraints. Every horizon edge, cubic weight, support,
shape tensor and damage operator is recomputed on rounded float32 coordinates.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from pinn.operators import ExplicitVolumeSparseD
from examples.fracture2d._legacy import nonlocal_baseline as direct


@dataclass(frozen=True)
class FP32Geometry:
    coordinates: np.ndarray
    n_material: int
    n_particles: int


@dataclass(frozen=True)
class FP32Static:
    source: np.ndarray
    target: np.ndarray
    eta: np.ndarray


@dataclass(frozen=True)
class _SegmentPack:
    """Static sort/offset data for one deterministic segmented reduction."""

    order: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    starts_flag: np.ndarray


def _segment_pack(ids, n_segments):
    """Group occurrence rows by segment id without any device-side atomics."""

    values = np.asarray(ids, dtype=np.int64)
    order = np.argsort(values, kind="stable").astype(np.int32)
    counts = np.bincount(values, minlength=n_segments)
    ends = np.cumsum(counts).astype(np.int32)
    starts = (ends - counts).astype(np.int32)
    starts_flag = np.zeros(len(values), dtype=bool)
    nonempty = starts < len(values)
    starts_flag[starts[nonempty]] = True
    return _SegmentPack(
        order=order, starts=starts, ends=ends, starts_flag=starts_flag
    )


@dataclass(frozen=True)
class FP32Model:
    geometry: FP32Geometry
    static: FP32Static
    constraints: direct.AffineConstraints
    material_operator: ExplicitVolumeSparseD
    Gc: float
    l0: float
    youngs_modulus: float
    poisson_ratio: float
    k_res: float
    n_material: int
    n_dofs: int
    candidate_added: int
    candidate_removed: int
    material_added: int
    material_removed: int
    max_condition: float
    support_indices: np.ndarray
    support_mask: np.ndarray
    beta: np.ndarray
    xi: np.ndarray
    shape_inverse: np.ndarray
    material_volumes: np.ndarray
    grip_volumes: np.ndarray


def _quadrature(coordinates, triangles, n_particles):
    p = coordinates[triangles]
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    areas = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) * np.float32(0.5)
    if np.any(areas <= 0) or not np.all(np.isfinite(areas)):
        raise ValueError("float32 triangles have zero/invalid area")
    volumes = np.zeros(n_particles, dtype=np.float32)
    np.add.at(volumes, triangles.ravel(), np.repeat(areas / np.float32(3), 3))
    return volumes


def _graph(geometry, coordinates, delta):
    delta32 = np.float32(delta)
    # The rounded float32 horizon, not the original float64 comparison, owns
    # graph membership on this independently named discretization.
    radius = float(delta32) + 8 * np.finfo(np.float32).eps
    pairs = np.array(list(cKDTree(coordinates).query_pairs(radius)), dtype=np.int64).reshape(-1, 2)
    if not len(pairs):
        raise ValueError("float32 graph has no horizon pairs")
    distance = np.sqrt(np.sum((coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]])**2,
                                     axis=1, dtype=np.float32))
    pairs = pairs[(distance > 0) & (distance < delta32)]
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    polygon = direct._g_boundary_polygon_from_triangles(
        coordinates.astype(np.float64), geometry.union_triangles
    )
    visible = direct._g_UnionPolygonPredicate(polygon, geometry.visibility_tolerance)
    kept = np.array([visible(coordinates[i].astype(np.float64),
                             coordinates[j].astype(np.float64)) for i, j in pairs], dtype=bool)
    visible_pairs = pairs[kept]
    edges = direct._g_directed_edges_from_pairs(visible_pairs)
    if not len(edges):
        raise ValueError("float32 union visibility graph is empty")
    canonical = np.sort(edges, axis=1)
    _, counts = np.unique(canonical, axis=0, return_counts=True)
    if np.any(counts != 2):
        raise ValueError("float32 union graph is not reciprocal")
    actual_distance = np.sqrt(np.sum((coordinates[edges[:, 0]] - coordinates[edges[:, 1]])**2,
                                          axis=1, dtype=np.float32))
    if not np.all((actual_distance > 0) & (actual_distance < delta32)):
        raise ValueError("float32 visible graph violates strict horizon")
    return pairs, edges


def build_fp32_model(reference: direct.EngineeringGripModel):
    if not isinstance(reference, direct.EngineeringGripModel):
        raise TypeError("float32 rebuild requires the validated float64 direct model")
    geometry = reference.geometry
    coordinates = np.asarray(geometry.coordinates, dtype=np.float32)
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError("float32 coordinate rounding merged distinct particles")
    n = len(coordinates)
    nm = reference.n_material
    material_volumes = _quadrature(coordinates, geometry.material_reference_triangles, n)
    grip_triangles = geometry.union_triangles[len(geometry.material_reference_triangles):]
    grip_volumes = _quadrature(coordinates, grip_triangles, n)
    if np.any(material_volumes[nm:] != 0):
        raise ValueError("float32 material phase leaks onto collar particles")
    volume = material_volumes + grip_volumes
    if np.any(volume <= 0) or not np.all(np.isfinite(volume)):
        raise ValueError("float32 union volumes must be finite and positive")
    pairs, edges = _graph(geometry, coordinates, geometry.delta)
    material_edges = edges[(edges[:, 0] < nm) & (edges[:, 1] < nm)]
    if not len(material_edges):
        raise ValueError("float32 material graph is empty")
    damage_operator = ExplicitVolumeSparseD(
        coordinates[:nm], material_volumes[:nm], material_edges, geometry.delta, "cubic"
    )

    source, target = edges[:, 0], edges[:, 1]
    delta32 = np.float32(geometry.delta)
    dist = np.sqrt(np.sum((coordinates[target] - coordinates[source])**2,
                          axis=1, dtype=np.float32))
    omega = (np.float32(120.0) / (np.float32(np.pi) * delta32**4)
             * (np.float32(1) - dist / delta32)**2)
    if np.any(omega <= 0) or not np.all(np.isfinite(omega)):
        raise ValueError("float32 cubic weights are not strictly positive")
    family = np.bincount(source, weights=(omega * volume[target]).astype(np.float64),
                         minlength=n).astype(np.float32)
    if np.any(family <= 0):
        raise ValueError("float32 BA family mass must be positive")
    adjacency = [set() for _ in range(n)]
    edge_weight = {}
    for (i, j), w in zip(edges, omega):
        adjacency[int(i)].add(int(j))
        edge_weight[int(i), int(j)] = w

    supports, betas, xis, inverses, conditions = [], [], [], [], []
    eta = np.empty(len(edges), dtype=np.float32)
    for edge_index, (i, j) in enumerate(edges):
        i, j = int(i), int(j)
        support = np.asarray([j, *sorted(adjacency[i] & adjacency[j] - {i, j})], dtype=np.int32)
        beta = np.asarray([edge_weight[i, int(k)] * volume[int(k)] for k in support],
                          dtype=np.float32)
        xi = coordinates[support] - coordinates[i]
        shape = np.einsum("s,sa,sb->ab", beta, xi, xi, optimize=True, dtype=np.float32)
        eig = np.linalg.eigvalsh(shape)
        if eig[0] <= 0 or eig[0] <= np.float32(reference.static.rank_rtol) * eig[-1]:
            raise ValueError(f"float32 BA support has nonpositive/deficient shape at {i}->{j}")
        cond = float(eig[-1] / eig[0])
        if not np.isfinite(cond):
            raise ValueError(f"float32 BA support condition is invalid at {i}->{j}")
        supports.append(support)
        betas.append(beta)
        xis.append(xi)
        inverses.append(np.linalg.inv(shape))
        conditions.append(cond)
        eta[edge_index] = beta[0] / family[i]
    if not np.all(np.isfinite(eta)) or np.any(eta <= 0):
        raise ValueError("float32 BA eta must be finite and positive")
    eta_sum = np.bincount(source, weights=eta.astype(np.float64), minlength=n)
    if np.max(np.abs(eta_sum - 1)) > 5e-5:
        raise ValueError("float32 BA eta does not sum to one")
    max_support = max(map(len, supports))
    indices = np.zeros((len(edges), max_support), dtype=np.int32)
    mask = np.zeros((len(edges), max_support), dtype=bool)
    padded_beta = np.zeros((len(edges), max_support), dtype=np.float32)
    padded_xi = np.zeros((len(edges), max_support, 2), dtype=np.float32)
    for k, (support, beta, xi) in enumerate(zip(supports, betas, xis)):
        count = len(support)
        indices[k, :count], mask[k, :count] = support, True
        padded_beta[k, :count], padded_xi[k, :count] = beta, xi
    inverse = np.stack(inverses).astype(np.float32)

    old_candidates = np.asarray(geometry.candidate_edges)
    old_set = {tuple(row) for row in old_candidates[old_candidates[:, 0] < old_candidates[:, 1]]}
    new_set = {tuple(row) for row in pairs}
    old_mat = {tuple(row) for row in geometry.material_reference_edges}
    new_mat = {tuple(row) for row in material_edges}
    model = FP32Model(
        geometry=FP32Geometry(coordinates, nm, n),
        static=FP32Static(source, target, eta),
        constraints=reference.constraints,
        material_operator=damage_operator,
        Gc=reference.Gc, l0=reference.l0,
        youngs_modulus=reference.static.youngs_modulus,
        poisson_ratio=reference.static.poisson_ratio, k_res=reference.static.k_res,
        n_material=nm, n_dofs=2*n,
        candidate_added=len(new_set - old_set), candidate_removed=len(old_set - new_set),
        material_added=len(new_mat - old_mat), material_removed=len(old_mat - new_mat),
        max_condition=max(conditions), support_indices=indices,
        support_mask=mask, beta=padded_beta, xi=padded_xi,
        shape_inverse=inverse, material_volumes=material_volumes,
        grip_volumes=grip_volumes,
    )
    return model


class FP32A2Backend(direct.JaxA2UnionMechanics):
    """Reuse the adopted A2 JAX residual methods on rebuilt float32 supports."""

    def __init__(self, model: FP32Model, device: jax.Device):
        if not isinstance(model, FP32Model):
            raise TypeError("FP32 A2 backend requires a separately built FP32Model")
        self.static = model.static
        self.constraints = model.constraints
        self.device = device
        self.n_particles = model.geometry.n_particles
        self.n_material = model.n_material
        self.n_edges = len(model.static.source)
        self.n_dofs = model.n_dofs
        self.free_dofs = jax.device_put(model.constraints.free_dofs.astype(np.int32), device)
        self.q_zero = jax.device_put(model.constraints.q_zero.astype(np.float32), device)
        self.load_direction = jax.device_put(model.constraints.load_direction.astype(np.float32), device)
        self.coordinates = jax.device_put(model.geometry.coordinates, device)
        self.material_phase_volumes = jax.device_put(model.material_volumes, device)
        self.grip_phase_volumes = jax.device_put(model.grip_volumes, device)
        self.source = jax.device_put(model.static.source.astype(np.int32), device)
        self.target = jax.device_put(model.static.target.astype(np.int32), device)
        self.eta = jax.device_put(model.static.eta, device)
        self.shape_inverse = jax.device_put(model.shape_inverse, device)
        valid = np.flatnonzero(model.support_mask.ravel())
        n_support = model.support_indices.shape[1]
        edge_ids = (valid // n_support).astype(np.int32)
        particle_ids = np.asarray(model.support_indices.ravel()[valid], dtype=np.int32)
        source_ids = np.asarray(model.static.source[edge_ids], dtype=np.int32)
        self.occurrence_edge = jax.device_put(edge_ids, device)
        self.occurrence_particle = jax.device_put(particle_ids, device)
        self.occurrence_source = jax.device_put(source_ids, device)
        # Deterministic replacements for the atomic scatter-adds in the adopted
        # assembly: pre-sorted occurrence groups summed with a cumsum.
        self._deformation_pack = self._device_pack(
            _segment_pack(edge_ids, self.n_edges), device
        )
        self._particle_pack = self._device_pack(
            _segment_pack(particle_ids, self.n_particles), device
        )
        self._source_pack = self._device_pack(
            _segment_pack(source_ids, self.n_particles), device
        )
        self.occurrence_beta = jax.device_put(model.beta.ravel()[valid], device)
        self.occurrence_xi = jax.device_put(model.xi.reshape(-1, 2)[valid], device)
        self._identity_2 = jax.device_put(np.eye(2, dtype=np.float32), device)
        E = np.float32(model.youngs_modulus)
        nu = np.float32(model.poisson_ratio)
        self._lame_lambda = jax.device_put(E * nu / ((1+nu)*(1-2*nu)), device)
        self._shear_modulus = jax.device_put(E / (2*(1+nu)), device)
        self._k_res = jax.device_put(np.float32(model.k_res), device)
        self._grip_degradation = jax.device_put(np.float32(1.0) + np.float32(model.k_res), device)
        self._material_zeros = jax.device_put(np.zeros(self.n_particles - self.n_material, dtype=np.float32), device)
        self.outer_top_selector = jax.device_put(model.constraints.outer_top_selector.astype(np.float32), device)
        self.outer_bottom_selector = jax.device_put(model.constraints.outer_bottom_selector.astype(np.float32), device)
        self.anchor_selector = jax.device_put(model.constraints.anchor_selector.astype(np.float32), device)

    @staticmethod
    def _device_pack(pack: _SegmentPack, device) -> _SegmentPack:
        return _SegmentPack(
            order=jax.device_put(pack.order, device),
            starts=jax.device_put(pack.starts, device),
            ends=jax.device_put(pack.ends, device),
            starts_flag=jax.device_put(pack.starts_flag, device),
        )

    @staticmethod
    def _segmented_sum(values, pack):
        """Run-reproducible segment sum with per-segment prefix resets.

        The cumulative scan restarts at every segment start, so no segment sum
        is ever cancelled against the global running total (which a plain
        global cumsum cannot avoid in float32).
        """
        values = values[pack.order]
        flag_shape = (pack.starts_flag.shape[0],) + (1,) * (values.ndim - 1)
        flags = jnp.broadcast_to(pack.starts_flag.reshape(flag_shape), values.shape)

        def combine(left, right):
            left_value, left_flag = left
            right_value, right_flag = right
            value = jnp.where(right_flag, right_value, left_value + right_value)
            return value, left_flag | right_flag

        scanned, _ = jax.lax.associative_scan(combine, (values, flags))
        last = jnp.maximum(pack.ends - 1, 0)
        result = scanned[last]
        result_shape = (pack.ends.shape[0],) + (1,) * (values.ndim - 1)
        nonempty = (pack.ends > pack.starts).reshape(result_shape)
        return jnp.where(nonempty, result, jnp.zeros_like(result))

    def _deformation_gradient(self, edge_factor, displacement):
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
        moment = self._segmented_sum(moment_occurrence, self._deformation_pack)
        return jnp.einsum("eab,ebc->eac", moment, self.shape_inverse)

    def _assemble_residual(self, stress, edge_factor):
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
        support_result = self._segmented_sum(endpoint, self._particle_pack)
        source_result = self._segmented_sum(-endpoint, self._source_pack)
        return (support_result + source_result).reshape(self.n_dofs)

    def solve(self, *args, **kwargs):
        raise RuntimeError("float32 PINN backend evaluates residuals only; direct GPU PCG remains float64")

    def solve_strict(self, *args, **kwargs):
        raise RuntimeError("float32 PINN backend evaluates residuals only; direct GPU PCG remains float64")

    def replay_cpu(self, *args, **kwargs):
        raise RuntimeError("float32 PINN fields require separate comparison with the float64 direct model")
