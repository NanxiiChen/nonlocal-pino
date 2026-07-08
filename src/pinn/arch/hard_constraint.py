"""Hard constraint output layer for KKS phase-field model.

Transforms raw network output (φ_raw, c_L_raw) into physically bounded
quantities via the KKS mixture rule.

Outputs:
  φ = 0.5·tanh(raw[0]) + 0.5                    ∈ [0, 1]
  c_L = (1 - c_Se + c_Le) · (0.5·tanh(raw[1]) + 0.5)  ∈ [0, 1-c_Se+c_Le]
  c_S = c_L + (c_Se - c_Le)                       equal chemical potential
  c   = h(φ)·c_S + (1-h(φ))·c_L                   KKS mixture rule
"""

import jax.numpy as jnp
import equinox as eqx


def _h(phi):
    return -2.0 * phi**3 + 3.0 * phi**2


class TanhConstraint(eqx.Module):
    """Maps raw network output to (-1, 1) via tanh. No trainable params.

    Suitable for single-field phase-field problems where φ ∈ (-1, 1),
    e.g. Allen-Cahn with tanh-profile analytical solutions.
    """

    def __call__(self, raw):
        return jnp.tanh(raw)


class HardConstraint(eqx.Module):
    """Physical hard constraints in output layer.  No trainable params."""
    dc0: float = eqx.field(static=True)  # c_Se - c_Le
    cl_max: float = eqx.field(static=True)  # 1 - c_Se + c_Le

    def __init__(self, c_se, c_le):
        self.dc0 = c_se - c_le
        self.cl_max = 1.0 - c_se + c_le

    def __call__(self, raw):
        """raw: (..., 2) → (..., 2) = (φ, c)."""
        bounded = 0.5 * jnp.tanh(raw) + 0.5            # both in [0, 1]
        phi = bounded[..., 0]
        cl_raw = bounded[..., 1]
        c_l = self.cl_max * cl_raw                      # [0, cl_max]
        c_s = c_l + self.dc0                            # equal chemical potential
        c = _h(phi) * c_s + (1.0 - _h(phi)) * c_l       # mixture rule
        return jnp.stack([phi, c], axis=-1)
