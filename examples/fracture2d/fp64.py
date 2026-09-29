"""Deterministic float64 residual backend for the adopted reference graph.

``JaxA2UnionMechanics`` (the frozen direct backend) assembles with atomic
``jax.ops.segment_sum``, which is not run-reproducible on the GPU: identical
float64 PINN runs diverged (state-1 field error `0.027` vs `0.052` under the
same configuration). This subclass keeps every float64 method and replaces
the scatter reductions with the sorted, per-segment prefix-reset sums already
used by ``FP32A2Backend``. The frozen ``nonlocal_baseline`` module is not
modified.
"""

import jax.numpy as jnp
import numpy as np

from examples.fracture2d import nonlocal_baseline as direct
from examples.fracture2d.fp32 import FP32A2Backend, _segment_pack


class FP64A2Backend(direct.JaxA2UnionMechanics):
    """Float64 residual assembly with deterministic segmented reductions."""

    def __init__(self, static, constraints, device):
        super().__init__(static, constraints, device)
        valid = np.flatnonzero(np.asarray(static.support_mask).reshape(-1))
        n_support = np.asarray(static.support_indices).shape[1]
        edge_ids = (valid // n_support).astype(np.int32)
        particle_ids = (
            np.asarray(static.support_indices).reshape(-1)[valid].astype(np.int32)
        )
        source_ids = np.asarray(static.source)[edge_ids].astype(np.int32)
        self._deformation_pack = FP32A2Backend._device_pack(
            _segment_pack(edge_ids, self.n_edges), device
        )
        self._particle_pack = FP32A2Backend._device_pack(
            _segment_pack(particle_ids, self.n_particles), device
        )
        self._source_pack = FP32A2Backend._device_pack(
            _segment_pack(source_ids, self.n_particles), device
        )

    _segmented_sum = staticmethod(FP32A2Backend._segmented_sum)

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
        # Explicit broadcast contraction instead of a batched GEMM: the GEMM
        # backward is not run-reproducible on this GPU stack.
        return jnp.sum(
            moment[:, :, :, None] * self.shape_inverse[:, None, :, :], axis=2
        )

    def _assemble_residual(self, stress, edge_factor):
        stress_occurrence = stress[self.occurrence_edge]
        inverse_transpose = jnp.swapaxes(
            self.shape_inverse[self.occurrence_edge], 1, 2
        )
        contracted = jnp.sum(
            stress_occurrence[:, :, :, None] * inverse_transpose[:, None, :, :],
            axis=2,
        )
        endpoint = jnp.sum(contracted * self.occurrence_xi[:, None, :], axis=2)
        endpoint = (
            endpoint
            * self.occurrence_beta[:, None]
            * edge_factor[self.occurrence_edge, None]
        )
        support_result = self._segmented_sum(endpoint, self._particle_pack)
        source_result = self._segmented_sum(-endpoint, self._source_pack)
        return (support_result + source_result).reshape(self.n_dofs)

    def _tangent_from_factor(self, edge_factor, direction):
        values = direction.reshape(self.n_particles, 2)
        displacement_difference = (
            values[self.occurrence_particle] - values[self.occurrence_source]
        )
        delta_f_occurrence = (
            displacement_difference[:, :, None]
            * self.gradient_coefficient[:, None, :]
        )
        delta_f = self._segmented_sum(delta_f_occurrence, self._deformation_pack)
        delta_strain = 0.5 * (delta_f + jnp.swapaxes(delta_f, 1, 2))
        trace = delta_strain[:, 0, 0] + delta_strain[:, 1, 1]
        delta_stress = (
            self._lame_lambda * trace[:, None, None] * self._identity_2[None, :, :]
            + 2.0 * self._shear_modulus * delta_strain
        )
        return self._assemble_residual(delta_stress, edge_factor)

    def _block_diagonal_from_factor(self, edge_factor):
        support_hxx = self._segmented_sum(
            edge_factor[self.occurrence_edge] * self.occurrence_hxx,
            self._particle_pack,
        )
        support_hxy = self._segmented_sum(
            edge_factor[self.occurrence_edge] * self.occurrence_hxy,
            self._particle_pack,
        )
        support_hyy = self._segmented_sum(
            edge_factor[self.occurrence_edge] * self.occurrence_hyy,
            self._particle_pack,
        )
        source_hxx = self._segmented_sum(
            edge_factor[self.occurrence_edge] * self.source_hxx[self.occurrence_edge],
            self._source_pack,
        )
        source_hxy = self._segmented_sum(
            edge_factor[self.occurrence_edge] * self.source_hxy[self.occurrence_edge],
            self._source_pack,
        )
        source_hyy = self._segmented_sum(
            edge_factor[self.occurrence_edge] * self.source_hyy[self.occurrence_edge],
            self._source_pack,
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
