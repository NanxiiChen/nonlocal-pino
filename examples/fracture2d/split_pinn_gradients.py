"""Optional analytic reverse rules for the unchanged FP64 full-graph operators.

Primal values reuse the adopted implementations. A2 energy's displacement
gradient is its assembled force residual; edge-factor gradient is energy
density. Reciprocal scalar kernels satisfy V D = D^T V, giving the ordinary
Euclidean adjoint D^T z = V D(V^-1 z), including nonuniform nodal volumes.
These rules avoid generic reverse-mode through large deterministic scans.
"""

import jax
import jax.numpy as jnp
import numpy as np

from pinn.operators import ExplicitVolumeSparseD
from examples.fracture2d.fp64 import FP64A2Backend


def _energy_rule(backend, deformation, assemble):
    if not isinstance(backend, FP64A2Backend) or not jax.config.jax_enable_x64:
        raise ValueError("analytic energy requires the validated FP64 backend with x64 enabled")

    def validate(factor, q):
        if (factor.shape != (backend.n_edges,) or q.shape != (backend.n_dofs,)
                or factor.dtype != jnp.float64 or q.dtype != jnp.float64):
            raise ValueError("analytic energy requires FP64 model-shaped factor and displacement")

    @jax.custom_vjp
    def energy(factor, q):
        validate(factor, q)
        F = deformation(factor, q)
        strain, _ = backend._strain_stress(F)
        trace = strain[:, 0, 0] + strain[:, 1, 1]
        density = (0.5 * backend._lame_lambda * trace**2
                   + backend._shear_modulus * jnp.sum(strain * strain, axis=(1, 2)))
        return jnp.sum(factor * density)

    def forward(factor, q):
        validate(factor, q)
        F = deformation(factor, q)
        strain, stress = backend._strain_stress(F)
        trace = strain[:, 0, 0] + strain[:, 1, 1]
        density = (0.5 * backend._lame_lambda * trace**2
                   + backend._shear_modulus * jnp.sum(strain * strain, axis=(1, 2)))
        value = jnp.sum(factor * density)
        force = assemble(stress, factor)
        return value, (density, force)

    def backward(context, cotangent):
        density, force = context
        return cotangent * density, cotangent * force

    energy.defvjp(forward, backward)
    return energy


def make_analytic_energy(backend):
    return _energy_rule(backend, backend._deformation_gradient, backend._assemble_residual)


def make_analytic_matvec(operator):
    if (not isinstance(operator, ExplicitVolumeSparseD) or not jax.config.jax_enable_x64
            or np.asarray(operator.volumes).dtype != np.float64):
        raise ValueError("analytic scalar adjoint requires a validated FP64 reciprocal operator")
    volumes = operator.volumes

    @jax.custom_vjp
    def matvec(values):
        return operator.deterministic_matvec(values)

    def forward(values):
        return operator.deterministic_matvec(values), None

    def backward(_, cotangent):
        volume = volumes.reshape((-1,) + (1,) * (cotangent.ndim - 1))
        return (volume * operator.deterministic_matvec(cotangent / volume),)

    matvec.defvjp(forward, backward)
    return matvec


def _padded_occurrence_rows(ids, n_rows):
    counts = np.bincount(ids, minlength=n_rows)
    width = int(np.max(counts))
    if width < 1:
        raise ValueError("padded occurrence rows cannot be empty")
    order = np.argsort(ids, kind="stable").astype(np.int32)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    indices = np.zeros((n_rows, width), dtype=np.int32)
    mask = np.zeros((n_rows, width), dtype=bool)
    for row in range(n_rows):
        count = counts[row]
        indices[row, :count] = order[offsets[row]:offsets[row + 1]]
        mask[row, :count] = True
    return jnp.asarray(indices), jnp.asarray(mask), width


def make_padded_energy(backend):
    """Same A2 sums, reduced within fixed padded rows instead of global scans."""
    if not isinstance(backend, FP64A2Backend):
        raise TypeError("padded energy requires a validated FP64 backend")
    static = backend.static
    mask = np.asarray(static.support_mask)
    indices = jnp.asarray(np.where(mask, static.support_indices, 0), dtype=jnp.int32)
    beta = jnp.asarray(np.where(mask, static.beta, 0.))
    xi = jnp.asarray(np.where(mask[..., None], static.xi, 0.))
    particle_rows, particle_mask, particle_width = _padded_occurrence_rows(
        np.asarray(backend.occurrence_particle), backend.n_particles)
    source_rows, source_mask, source_width = _padded_occurrence_rows(
        np.asarray(backend.occurrence_source), backend.n_particles)

    def deformation(factor, q):
        del factor
        values = q.reshape(backend.n_particles, 2)
        current = xi + (values[indices] - values[backend.source, None, :])
        moment = jnp.sum(current[..., :, None] * xi[..., None, :] * beta[..., None, None], axis=1)
        return jnp.sum(moment[:, :, :, None] * backend.shape_inverse[:, None, :, :], axis=2)

    def assemble(stress, factor):
        inverse = jnp.swapaxes(backend.shape_inverse[backend.occurrence_edge], 1, 2)
        contracted = jnp.sum(stress[backend.occurrence_edge, :, :, None] * inverse[:, None, :, :], axis=2)
        endpoint = (jnp.sum(contracted * backend.occurrence_xi[:, None, :], axis=2)
                    * backend.occurrence_beta[:, None] * factor[backend.occurrence_edge, None])
        support = jnp.sum(jnp.where(particle_mask[..., None], endpoint[particle_rows], 0.), axis=1)
        source = jnp.sum(jnp.where(source_mask[..., None], -endpoint[source_rows], 0.), axis=1)
        return (support + source).reshape(backend.n_dofs)

    info = {"support_slots": int(mask.size), "support_occurrences": int(np.count_nonzero(mask)),
            "support_width": mask.shape[1], "particle_row_width": particle_width,
            "source_row_width": source_width}
    return _energy_rule(backend, deformation, assemble), info


def make_padded_matvec(operator):
    """Pack unchanged CSR off-diagonals once; use deterministic per-row sums."""
    if (not isinstance(operator, ExplicitVolumeSparseD) or not jax.config.jax_enable_x64
            or np.asarray(operator.volumes).dtype != np.float64):
        raise ValueError("padded scalar operator requires the validated FP64 reciprocal operator")
    matrix = operator.to_scipy()
    matrix.setdiag(0.)
    matrix.eliminate_zeros()
    matrix.sort_indices()
    counts = np.diff(matrix.indptr)
    n, width = len(counts), int(np.max(counts))
    columns = np.zeros((n, width), dtype=np.int32)
    weights = np.zeros((n, width), dtype=np.float64)
    for row in range(n):
        start, end = matrix.indptr[row:row + 2]
        columns[row, :end - start] = matrix.indices[start:end]
        weights[row, :end - start] = matrix.data[start:end]
    columns, weights = jnp.asarray(columns), jnp.asarray(weights)
    volumes, diagonal = operator.volumes, operator.diagonal

    def forward_matvec(values):
        if values.ndim not in (1, 2) or values.shape[0] != n or values.dtype != jnp.float64:
            raise ValueError("padded matvec requires FP64 model-shaped fields")
        trailing = (1,) * (values.ndim - 1)
        return (diagonal.reshape((n,) + trailing) * values
                + jnp.sum(weights.reshape(weights.shape + trailing) * values[columns], axis=1))

    @jax.custom_vjp
    def matvec(values):
        return forward_matvec(values)

    def forward(values):
        return forward_matvec(values), None

    def backward(_, cotangent):
        volume = volumes.reshape((-1,) + (1,) * (cotangent.ndim - 1))
        return (volume * forward_matvec(cotangent / volume),)

    matvec.defvjp(forward, backward)
    return matvec, {"row_width": width, "edge_count": int(np.sum(counts)),
                    "padded_slots": n * width, "padding_ratio": float(n * width / np.sum(counts))}
