"""Differentiable full-graph residuals for the adopted 2D nonlocal fracture model.

The mechanics assembly and material operator are the direct solver's objects.
Only the plane-strain Amor accumulation is expressed here in JAX so gradients
can flow from history back to a neural displacement field.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d._legacy.nonlocal_baseline import EngineeringGripModel, JaxA2UnionMechanics
from examples.fracture2d._legacy.fp32 import FP32Model, FP32A2Backend


def update_amor_history(driver: jax.Array, H_committed: jax.Array) -> jax.Array:
    """Maximum against the fixed, previously accepted cumulative history."""
    return jnp.maximum(jax.lax.stop_gradient(H_committed), driver)


@dataclass(frozen=True)
class NonlocalResiduals:
    mechanics: jax.Array
    damage: jax.Array
    history: jax.Array
    driver: jax.Array
    trial_history: jax.Array
    reaction: jax.Array


class FractureResidual:
    """One fixed-shape A2/Amor/AT2 residual evaluator.

    Inputs are complete union displacement, material damage/history and the
    committed material history. Constraints are lifted before evaluation; no
    collocation subsampling or loss normalization changes the equations.
    """

    def __init__(self, model: EngineeringGripModel | FP32Model,
                 backend: JaxA2UnionMechanics):
        if not isinstance(model, (EngineeringGripModel, FP32Model)):
            raise TypeError("model must be a validated float64 or rebuilt float32 model")
        if isinstance(model, FP32Model):
            if not isinstance(backend, FP32A2Backend):
                raise TypeError("float32 model requires its rebuilt FP32A2Backend")
            dtype = np.float32
        else:
            if not isinstance(backend, JaxA2UnionMechanics) or isinstance(backend, FP32A2Backend):
                raise TypeError("direct model requires JaxA2UnionMechanics")
            dtype = np.float64
        if backend.static is not model.static or backend.constraints is not model.constraints:
            raise ValueError("backend must own the exact model static and constraints")
        if dtype == np.float64 and not jax.config.jax_enable_x64:
            raise ValueError("JAX x64 must be enabled before building residuals")
        self.dtype = dtype
        self.model = model
        self.backend = backend
        self.operator = model.material_operator
        self.n_material = model.n_material
        self.n_dofs = model.n_dofs
        source = np.asarray(model.static.source)
        material_edges = np.flatnonzero(source < self.n_material)
        if not len(material_edges):
            raise ValueError("Amor requires material-source edges")
        self.material_edges = jax.device_put(material_edges.astype(np.int32), backend.device)
        self.material_source = jax.device_put(source[material_edges].astype(np.int32), backend.device)
        self.material_eta = jax.device_put(
            np.asarray(model.static.eta)[material_edges], backend.device
        )
        self.gc_over_l0 = jax.device_put(dtype(model.Gc / model.l0), backend.device)
        self.gc_l0 = jax.device_put(dtype(model.Gc * model.l0), backend.device)
        free_mask = np.zeros(self.n_dofs, dtype=bool)
        free_mask[np.asarray(model.constraints.free_dofs)] = True
        self.free_mask = jax.device_put(free_mask, backend.device)

    def lift(self, raw_q: jax.Array, load: jax.Array) -> jax.Array:
        """Impose every collar/anchor displacement exactly via affine lifting.

        The select blend is used instead of an ``.at[free_dofs].set`` scatter:
        the scatter compiled to a different field when fused into a jitted
        objective (eager and jitted values differed by ``2e-5`` on the same
        inputs), while ``jnp.where`` is numerically identical eagerly and
        inside jit.
        """
        backend = self.backend
        prescribed = backend.q_zero + load * backend.load_direction
        return jnp.where(self.free_mask, raw_q, prescribed)

    def evaluate(
        self, q: jax.Array, d: jax.Array, H: jax.Array | None,
        H_committed: jax.Array,
    ) -> NonlocalResiduals:
        """Evaluate unscaled residuals; explicit ``H=None`` uses exact trial history."""
        backend = self.backend
        factor = backend._edge_factor(d)
        evaluation = backend._evaluate_from_factor(factor, q)
        deformation = backend._deformation_gradient(factor, q)
        strain, _ = backend._strain_stress(deformation)
        strain = strain[self.material_edges]
        trace = jnp.trace(strain, axis1=1, axis2=2)
        dev_2 = strain - trace[:, None, None] / 3.0 * backend._identity_2
        dev_norm_3 = jnp.sum(dev_2 * dev_2, axis=(1, 2)) + (trace / 3.0) ** 2
        bulk = backend._lame_lambda + 2.0 * backend._shear_modulus / 3.0
        positive = 0.5 * bulk * jnp.maximum(trace, 0.0) ** 2 + backend._shear_modulus * dev_norm_3
        driver = jax.ops.segment_sum(
            self.material_eta * positive, self.material_source,
            num_segments=self.n_material,
        )
        trial_history = update_amor_history(driver, H_committed)
        if H is None:
            H = trial_history
        damage = (self.gc_over_l0 * d
                  + self.gc_l0 * self.operator.deterministic_matvec(d)
                  + 2.0 * H * (d - 1.0))
        return NonlocalResiduals(
            mechanics=evaluation["residual"][backend.free_dofs],
            damage=damage,
            history=H - trial_history,
            driver=driver,
            trial_history=trial_history,
            reaction=evaluation["top_reaction"],
        )

    def validate_fields(self, q, d, H, H_committed, load):
        """Fail on malformed host fields before entering a compiled train step."""
        for name, value, shape in (
            ("q", q, (self.n_dofs,)),
            ("d", d, (self.n_material,)),
            ("H", H, (self.n_material,)),
            ("H_committed", H_committed, (self.n_material,)),
            ("load", load, ()),
        ):
            arr = np.asarray(value)
            if arr.shape != shape or arr.dtype != self.dtype or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite {np.dtype(self.dtype)} with shape {shape}")
        if np.any(np.asarray(H_committed) < 0.0):
            raise ValueError("committed history must be nonnegative")
        prescribed = self.model.constraints.prescribed_vector(float(load)).astype(self.dtype)
        fixed = self.model.constraints.constrained_dofs
        if not np.array_equal(np.asarray(q)[fixed], prescribed[fixed]):
            raise ValueError("q violates the exact affine constraints")
