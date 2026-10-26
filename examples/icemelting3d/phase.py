"""IC-only phase representation; the network learns all subsequent motion."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np


FORMULATIONS = ("raw_phase", "ic_phase", "ic_latent")


class InitialPhase(eqx.Module):
    raw: eqx.Module
    radius: float = eqx.field(static=True)
    centre: tuple = eqx.field(static=True)
    width: float = eqx.field(static=True)

    def latent(self, x, t):
        u0 = (self.radius - jnp.linalg.norm(x - jnp.asarray(self.centre))) / self.width
        return u0 + t * self.raw(x, t)

    def __call__(self, x, t):
        return jnp.tanh(self.latent(x, t))


def wrap_phase(raw, config):
    """Wrap a raw scalar model, preserving the legacy builder/checkpoint tree."""
    mode = config.phase.formulation
    if mode not in FORMULATIONS:
        raise ValueError(f"Unknown phase.formulation={mode!r}; expected {FORMULATIONS}")
    if mode == "raw_phase":
        return raw
    if config.arch.use_hard_constraint:
        raise ValueError("IC-only modes require arch.use_hard_constraint=False; phase.py handles the constraint")
    if config.arch.out_dim != 1 or config.arch.spatial_dim != 3:
        raise ValueError("IC-only melting requires a scalar model with spatial_dim=3")
    # Zero correction starts from the prescribed stationary IC, not an evolved profile.
    if config.arch.type == "film":
        raw = eqx.tree_at(lambda n: (n.trunk_output.weight, n.trunk_output.bias), raw,
                         (jnp.zeros_like(raw.trunk_output.weight), jnp.zeros_like(raw.trunk_output.bias)))
    else:
        raw = eqx.tree_at(lambda n: (n.output.weight, n.output.bias), raw,
                         (jnp.zeros_like(raw.output.weight), jnp.zeros_like(raw.output.bias)))
    return InitialPhase(raw, float(config.physical.R0 / config.nondim.L_c),
                        tuple(config.nondim.xc), float(np.sqrt(2.) * config.physical.eps_star))
