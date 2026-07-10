"""FiLM-conditioned MLP: time modulates spatial features at every layer."""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from .embedding import FourierEmbedding


class _FiLMPINN(eqx.Module):
    spatial_emb: FourierEmbedding | None
    temporal_emb: FourierEmbedding | None
    film_mlp: eqx.nn.MLP
    trunk_layers: list
    trunk_output: eqx.nn.Linear
    hard_constraint: eqx.Module | None
    dirichlet_bc: eqx.Module | None
    hidden_dim: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    activation: callable = eqx.field(static=True)

    def __call__(self, x, t):
        if self.spatial_emb is not None:
            x = self.spatial_emb(jnp.atleast_1d(x))
        else:
            x = jnp.atleast_1d(x)
        if self.temporal_emb is not None:
            t = self.temporal_emb(jnp.atleast_1d(t))
        else:
            t = jnp.atleast_1d(t)

        film_out = self.film_mlp(t)
        film = film_out.reshape(self.num_layers, 2, self.hidden_dim)
        gamma = film[:, 0, :]
        beta  = film[:, 1, :]

        h = x
        for i, layer in enumerate(self.trunk_layers):
            h = layer(h)
            h = gamma[i] * h + beta[i]
            h = self.activation(h)

        out = self.trunk_output(h)
        if self.hard_constraint is not None:
            out = self.hard_constraint(out)
        if self.dirichlet_bc is not None:
            out = self.dirichlet_bc(jnp.atleast_1d(x), out)
        return out


def make_film_pinn(spatial_dim: int, out_dim: int, *,
                   use_spatial_emb: bool = True,
                   use_temporal_emb: bool = True,
                   spatial_emb_dim: int = 64, spatial_scale: float = 2.0,
                   temporal_emb_dim: int = 64, temporal_scale: float | tuple = 2.0,
                   hidden_dim: int = 64, num_layers: int = 4,
                   film_hidden_dim: int = 32, film_num_layers: int = 2,
                   activation=jnp.tanh,
                   hard_constraint: eqx.Module | None = None,
                   dirichlet_bc: eqx.Module | None = None,
                   key: jr.PRNGKey):
    if isinstance(activation, str):
        activation = _ACTIVATIONS.get(activation, jnp.tanh)
    k_sp, k_tp, k_film, k_trunk, k_out = jr.split(key, 5)

    if use_spatial_emb:
        n_spatial = len(spatial_scale) if isinstance(spatial_scale, (tuple, list)) else 1
        spatial_emb = FourierEmbedding(spatial_dim, spatial_emb_dim, spatial_scale, key=k_sp)
        x_in = 2 * spatial_emb_dim * n_spatial
    else:
        spatial_emb = None
        x_in = spatial_dim

    if use_temporal_emb:
        n_temporal = len(temporal_scale) if isinstance(temporal_scale, (tuple, list)) else 1
        temporal_emb = FourierEmbedding(1, temporal_emb_dim, temporal_scale, key=k_tp)
        t_in = 2 * temporal_emb_dim * n_temporal
    else:
        temporal_emb = None
        t_in = 1

    # FiLM MLP: t features → (2 * num_layers * hidden_dim)
    film_out_dim = 2 * num_layers * hidden_dim
    film_mlp = eqx.nn.MLP(t_in, film_out_dim, width_size=film_hidden_dim,
                          depth=film_num_layers, activation=activation, key=k_film)

    # Trunk: x features → hidden_dim → ... → hidden_dim (num_layers times)
    trunk_layers = []
    k_trunk_i = jr.split(k_trunk, num_layers)
    trunk_layers.append(eqx.nn.Linear(x_in, hidden_dim, key=k_trunk_i[0]))
    for i in range(1, num_layers):
        trunk_layers.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=k_trunk_i[i]))

    trunk_output = eqx.nn.Linear(hidden_dim, out_dim, key=k_out)

    return _FiLMPINN(spatial_emb, temporal_emb, film_mlp, trunk_layers,
                     trunk_output, hard_constraint, dirichlet_bc,
                     hidden_dim, num_layers, activation)


_ACTIVATIONS = {
    'tanh': jnp.tanh, 'relu': jax.nn.relu, 'gelu': jax.nn.gelu,
    'silu': jax.nn.silu, 'swish': jax.nn.swish,
}
