"""Dirichlet BC hard constraint: modular, model-agnostic.

    output(x) = baseline(x) + mask(x) * raw_network_output(x)

baseline exactly satisfies the boundary conditions.
mask zeros at all Dirichlet boundaries, letting raw network output
freely modulate the interior without affecting BC.
"""

import equinox as eqx


class DirichletBC(eqx.Module):
    """Hard Dirichlet BC enforcement.  No trainable parameters."""
    baseline: callable = eqx.field(static=True)
    mask: callable = eqx.field(static=True)

    def __init__(self, baseline, mask):
        self.baseline = baseline
        self.mask = mask

    def __call__(self, x, raw):
        """x: (dim,) or (N, dim).  raw: (...) — network output at x."""
        return self.baseline(x) + self.mask(x) * raw


# ── 1D KKS corrosion presets ─────────────────────────────────────────────────

def make_1d_corrosion_bc(c_se=1.0):
    """Return (baseline, mask) callables for 1D KKS corrosion.

    BC:  φ(0)=1, φ(1)=0,  c(0)=c_se, c(1)=0.
    """
    import jax.numpy as jnp

    def baseline(x):
        x0 = x[..., 0]
        return jnp.stack([1.0 - x0, c_se * (1.0 - x0)], axis=-1)

    def mask(x):
        x0 = x[..., 0]
        m = x0 * (1.0 - x0)
        return jnp.stack([m, m], axis=-1)

    return DirichletBC(baseline, mask)
