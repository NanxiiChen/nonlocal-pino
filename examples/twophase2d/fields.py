"""Physical full-grid fields, optionally projected onto conserved phase mass."""
import jax
import jax.numpy as jnp
from examples.twophase2d.problem import initial_field


def predict_fields(net, points, t, config):
    fields = jax.vmap(net, in_axes=(0, None))(points, t)
    if 'phase_mass' in config:
        if points.shape != (config.sample.Nx*config.sample.Ny, 2):
            raise ValueError('Exact grid mass requires the fixed full periodic quadrature grid')
        initial = initial_field(points, config, xp=jnp)
        shift = jnp.mean(fields[:, 0]-initial)
        fields = fields.at[:, 0].add(-shift)
    return fields
