"""Output constraints for physics-aware network wrapping.

These are NOT architectures — they are post-processing transforms applied
via ``ConstrainedForward`` in each example's ``_build_network``.
"""

import jax.numpy as jnp
import equinox as eqx


def kks_constraint(raw, c_se, c_le):
    """KKS mixture rule: raw (..., 2) → constrained (φ, c).

    φ  = 0.5·tanh(raw[0]) + 0.5                  ∈ [0, 1]
    c_L = (1 - c_Se + c_Le) · (0.5·tanh(raw[1]) + 0.5)
    c_S = c_L + (c_Se - c_Le)
    c   = h(φ)·c_S + (1 − h(φ))·c_L              KKS mixture rule
    """
    dc0 = c_se - c_le
    cl_max = 1.0 - c_se + c_le
    bounded = 0.5 * jnp.tanh(raw) + 0.5
    phi = bounded[..., 0]
    cl_raw = bounded[..., 1]
    c_l = cl_max * cl_raw
    c_s = c_l + dc0
    h_phi = -2.0 * phi**3 + 3.0 * phi**2
    c = h_phi * c_s + (1.0 - h_phi) * c_l
    return jnp.stack([phi, c], axis=-1)


def tanh_constraint(raw):
    """Single-field tanh: raw (...) → tanh(raw) ∈ (-1, 1)."""
    return jnp.tanh(raw)


class ConstrainedForward(eqx.Module):
    """Wrap a raw network with an output constraint function.

        net = ConstrainedForward(raw_net, lambda raw: kks_constraint(raw, c_se, c_le))
    """
    net: eqx.Module
    constraint: callable = eqx.field(static=True)

    def __call__(self, x, t):
        return self.constraint(self.net(x, t))
