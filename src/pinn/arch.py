"""PINN architectures with built-in Fourier embeddings and a unified factory.

MLP / ModifiedMLP / FiLM — each handles its own embedding construction.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from .embedding import FourierEmbedding


# ═══════════════════════════════════════════════════════════════════════════════
# MLP trunk
# ═══════════════════════════════════════════════════════════════════════════════

class MLP(eqx.Module):
    """MLP trunk: embeds x and t internally, concatenates, feeds through layers.

    ``tensor_split`` (static): split first layer into x-only and t-only matmuls
    — exact same math, fewer FLOPs under vmap(x) × vmap(t) evaluation.
    """
    spatial_emb: FourierEmbedding | None
    temporal_emb: FourierEmbedding | None
    layers: list
    activation: callable = eqx.field(static=True)
    tensor_split: bool = eqx.field(static=True, default=False)

    def __init__(self, spatial_dim: int, out_dim: int, *,
                 hidden_dim: int = 64, num_layers: int = 4,
                 activation=jnp.tanh,
                 use_spatial_emb: bool = True,
                 use_temporal_emb: bool = True,
                 spatial_emb_dim: int = 64,
                 spatial_scale: float | tuple = 2.0,
                 temporal_emb_dim: int = 64,
                 temporal_scale: float | tuple = 2.0,
                 static: bool = False,
                 tensor_split: bool = False,
                 key: jr.PRNGKey):
        k_sp, k_tp, k_trunk = jr.split(key, 3)

        # ── spatial embedding ──
        if use_spatial_emb:
            self.spatial_emb = FourierEmbedding(
                spatial_dim, spatial_emb_dim, spatial_scale, key=k_sp)
            n_s = len(spatial_scale) if isinstance(spatial_scale, (tuple, list)) else 1
            s_dim = 2 * spatial_emb_dim * n_s
        else:
            self.spatial_emb = None
            s_dim = spatial_dim

        # ── temporal embedding ──
        if static:
            self.temporal_emb = None
            t_dim = 0
        elif use_temporal_emb:
            self.temporal_emb = FourierEmbedding(
                1, temporal_emb_dim, temporal_scale, key=k_tp)
            n_t = len(temporal_scale) if isinstance(temporal_scale, (tuple, list)) else 1
            t_dim = 2 * temporal_emb_dim * n_t
        else:
            self.temporal_emb = None
            t_dim = 1

        # ── trunk layers ──
        trunk_in = s_dim + t_dim
        keys = jr.split(k_trunk, num_layers + 1)
        self.layers = [eqx.nn.Linear(trunk_in, hidden_dim, key=keys[0])]
        for i in range(1, num_layers):
            self.layers.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[i]))
        self.layers.append(eqx.nn.Linear(hidden_dim, out_dim, key=keys[-1]))
        self.activation = activation
        self.tensor_split = tensor_split

    def __call__(self, x: jnp.ndarray, t=None) -> jnp.ndarray:
        ex = (self.spatial_emb(jnp.atleast_1d(x))
              if self.spatial_emb is not None else jnp.atleast_1d(x))
        if self.temporal_emb is not None:
            et = self.temporal_emb(jnp.atleast_1d(t))
            if self.tensor_split:
                dx = ex.shape[0]
                h = self.layers[0].weight[:, :dx]@ex + self.layers[0].weight[:, dx:]@et + self.layers[0].bias
            else:
                h = self.layers[0](jnp.concatenate([ex, et]))
        else:
            h = self.layers[0](ex)
        h = self.activation(h)
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        return self.layers[-1](h)


# ═══════════════════════════════════════════════════════════════════════════════
# ModifiedMLP trunk
# ═══════════════════════════════════════════════════════════════════════════════

class ModifiedMLP(eqx.Module):
    """Modified MLP trunk: embeds x and t internally, feeds through gated layers.

    u, v: fixed per-input gate vectors.
    Each hidden layer: z = tanh(layer(x)), x = z * u + (1-z) * v.
    """
    spatial_emb: FourierEmbedding | None
    temporal_emb: FourierEmbedding | None
    u_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    hidden_layers: list
    output: eqx.nn.Linear
    activation: callable = eqx.field(static=True)

    def __init__(self, spatial_dim: int, out_dim: int, *,
                 hidden_dim: int = 64, num_layers: int = 4,
                 activation=jnp.tanh,
                 use_spatial_emb: bool = True,
                 use_temporal_emb: bool = True,
                 spatial_emb_dim: int = 64,
                 spatial_scale: float | tuple = 2.0,
                 temporal_emb_dim: int = 64,
                 temporal_scale: float | tuple = 2.0,
                 static: bool = False,
                 key: jr.PRNGKey):
        k_sp, k_tp, k_u, k_v, k_out, *k_hidden = jr.split(key, 5 + num_layers)

        # ── spatial embedding ──
        if use_spatial_emb:
            self.spatial_emb = FourierEmbedding(
                spatial_dim, spatial_emb_dim, spatial_scale, key=k_sp)
            n_s = len(spatial_scale) if isinstance(spatial_scale, (tuple, list)) else 1
            s_dim = 2 * spatial_emb_dim * n_s
        else:
            self.spatial_emb = None
            s_dim = spatial_dim

        # ── temporal embedding ──
        if static:
            self.temporal_emb = None
            t_dim = 0
        elif use_temporal_emb:
            self.temporal_emb = FourierEmbedding(
                1, temporal_emb_dim, temporal_scale, key=k_tp)
            n_t = len(temporal_scale) if isinstance(temporal_scale, (tuple, list)) else 1
            t_dim = 2 * temporal_emb_dim * n_t
        else:
            self.temporal_emb = None
            t_dim = 1

        trunk_in = s_dim + t_dim
        self.u_proj = eqx.nn.Linear(trunk_in, hidden_dim, key=k_u)
        self.v_proj = eqx.nn.Linear(trunk_in, hidden_dim, key=k_v)
        self.hidden_layers = [eqx.nn.Linear(trunk_in, hidden_dim, key=k_hidden[0])]
        for ki in k_hidden[1:]:
            self.hidden_layers.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=ki))
        self.output = eqx.nn.Linear(hidden_dim, out_dim, key=k_out)
        self.activation = activation

    def __call__(self, x: jnp.ndarray, t=None) -> jnp.ndarray:
        ex = (self.spatial_emb(jnp.atleast_1d(x))
              if self.spatial_emb is not None else jnp.atleast_1d(x))
        if self.temporal_emb is not None:
            et = self.temporal_emb(jnp.atleast_1d(t))
            h = jnp.concatenate([ex, et])
        else:
            h = ex

        u = self.activation(self.u_proj(h))
        v = self.activation(self.v_proj(h))
        for layer in self.hidden_layers:
            z = jnp.tanh(layer(h))
            h = z * u + (1.0 - z) * v
        return self.output(h)


# ═══════════════════════════════════════════════════════════════════════════════
# FiLM trunk
# ═══════════════════════════════════════════════════════════════════════════════

class FiLM(eqx.Module):
    """FiLM trunk: embeds x and t internally; time modulates each spatial layer.

    spatial_emb(x) → trunk layers modulated by film_mlp(temporal_emb(t)).
    """
    spatial_emb: FourierEmbedding | None
    temporal_emb: FourierEmbedding | None
    film_mlp: eqx.nn.MLP
    trunk_layers: list
    trunk_output: eqx.nn.Linear
    hidden_dim: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    activation: callable = eqx.field(static=True)

    def __init__(self, spatial_dim: int, out_dim: int, *,
                 hidden_dim: int = 64, num_layers: int = 4,
                 activation=jnp.tanh,
                 use_spatial_emb: bool = True,
                 use_temporal_emb: bool = True,
                 spatial_emb_dim: int = 64,
                 spatial_scale: float | tuple = 2.0,
                 temporal_emb_dim: int = 64,
                 temporal_scale: float | tuple = 2.0,
                 film_hidden_dim: int = 32, film_num_layers: int = 2,
                 key: jr.PRNGKey):
        k_sp, k_tp, k_film, k_trunk, k_out = jr.split(key, 5)

        # ── spatial embedding ──
        if use_spatial_emb:
            self.spatial_emb = FourierEmbedding(
                spatial_dim, spatial_emb_dim, spatial_scale, key=k_sp)
            n_s = len(spatial_scale) if isinstance(spatial_scale, (tuple, list)) else 1
            s_dim = 2 * spatial_emb_dim * n_s
        else:
            self.spatial_emb = None
            s_dim = spatial_dim

        # ── temporal embedding ──
        if use_temporal_emb:
            self.temporal_emb = FourierEmbedding(
                1, temporal_emb_dim, temporal_scale, key=k_tp)
            n_t = len(temporal_scale) if isinstance(temporal_scale, (tuple, list)) else 1
            t_dim = 2 * temporal_emb_dim * n_t
        else:
            self.temporal_emb = None
            t_dim = 1

        # ── FiLM conditioning MLP ──
        film_out_dim = 2 * num_layers * hidden_dim
        self.film_mlp = eqx.nn.MLP(t_dim, film_out_dim, width_size=film_hidden_dim,
                                   depth=film_num_layers, activation=activation,
                                   key=k_film)

        # ── spatial trunk layers ──
        k_tr_i = jr.split(k_trunk, num_layers)
        self.trunk_layers = [eqx.nn.Linear(s_dim, hidden_dim, key=k_tr_i[0])]
        for i in range(1, num_layers):
            self.trunk_layers.append(eqx.nn.Linear(hidden_dim, hidden_dim,
                                                   key=k_tr_i[i]))
        self.trunk_output = eqx.nn.Linear(hidden_dim, out_dim, key=k_out)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation

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
        beta = film[:, 1, :]

        h = x
        for i, layer in enumerate(self.trunk_layers):
            h = layer(h)
            h = gamma[i] * h + beta[i]
            h = self.activation(h)

        return self.trunk_output(h)


# ═══════════════════════════════════════════════════════════════════════════════
# Unified factory
# ═══════════════════════════════════════════════════════════════════════════════

_ACTIVATIONS = {
    'tanh': jnp.tanh, 'relu': jax.nn.relu, 'gelu': jax.nn.gelu,
    'silu': jax.nn.silu, 'swish': jax.nn.swish,
}


def make_pinn(spatial_dim: int, out_dim: int, *,
              hidden_dim: int = 64, num_layers: int = 4,
              trunk_type: str = "mlp",
              activation=jnp.tanh,
              use_spatial_emb: bool = True,
              use_temporal_emb: bool = True,
              spatial_emb_dim: int = 64,
              spatial_scale: float | tuple = 2.0,
              temporal_emb_dim: int = 64,
              temporal_scale: float | tuple = 2.0,
              static: bool = False,
              tensor_split: bool = False,
              film_hidden_dim: int = 32, film_num_layers: int = 2,
              key: jr.PRNGKey) -> MLP | ModifiedMLP | FiLM:
    """Create a PINN model with built-in Fourier embeddings.

    ``trunk_type``: "mlp" | "modified_mlp" | "film".
    ``tensor_split``: mlp trunk only (exact math, fewer FLOPs).
    ``film_hidden_dim``, ``film_num_layers``: film trunk only.
    """
    if isinstance(activation, str):
        activation = _ACTIVATIONS.get(activation, jnp.tanh)

    if static and trunk_type == "film":
        raise ValueError("FiLM trunk requires temporal modulation; "
                         "cannot be used with static=True.")
    if trunk_type == "film":
        return FiLM(
            spatial_dim=spatial_dim, out_dim=out_dim,
            hidden_dim=hidden_dim, num_layers=num_layers,
            activation=activation,
            use_spatial_emb=use_spatial_emb,
            use_temporal_emb=use_temporal_emb,
            spatial_emb_dim=spatial_emb_dim, spatial_scale=spatial_scale,
            temporal_emb_dim=temporal_emb_dim, temporal_scale=temporal_scale,
            film_hidden_dim=film_hidden_dim, film_num_layers=film_num_layers,
            key=key)

    embedding_kw = dict(
        use_spatial_emb=use_spatial_emb,
        use_temporal_emb=use_temporal_emb,
        spatial_emb_dim=spatial_emb_dim, spatial_scale=spatial_scale,
        temporal_emb_dim=temporal_emb_dim, temporal_scale=temporal_scale,
        static=static)

    if trunk_type == "mlp":
        return MLP(
            spatial_dim=spatial_dim, out_dim=out_dim,
            hidden_dim=hidden_dim, num_layers=num_layers,
            activation=activation,
            tensor_split=tensor_split,
            **embedding_kw,
            key=key)
    elif trunk_type == "modified_mlp":
        return ModifiedMLP(
            spatial_dim=spatial_dim, out_dim=out_dim,
            hidden_dim=hidden_dim, num_layers=num_layers,
            activation=activation,
            **embedding_kw,
            key=key)
    else:
        raise ValueError(f"Unknown trunk_type: {trunk_type!r}. "
                         f"Expected 'mlp', 'modified_mlp', or 'film'.")
