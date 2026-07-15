"""PINN model composing Fourier embeddings with an MLP trunk."""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from .embedding import FourierEmbedding
from .mlp import MLP
from .modified_mlp import ModifiedMLP


class _PINN(eqx.Module):
    spatial_emb: FourierEmbedding | None
    temporal_emb: FourierEmbedding | None
    trunk: eqx.Module
    hard_constraint: eqx.Module | None
    dirichlet_bc: eqx.Module | None
    tensor_split: bool = eqx.field(static=True, default=False)

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        ex = (self.spatial_emb(jnp.atleast_1d(x))
              if self.spatial_emb is not None else jnp.atleast_1d(x))
        et = (self.temporal_emb(jnp.atleast_1d(t))
              if self.temporal_emb is not None else jnp.atleast_1d(t))

        if self.tensor_split and isinstance(self.trunk, MLP):
            # Exact split of the first trunk layer:
            #   W @ [ex; et] + b  =  (W_x @ ex) + (W_t @ et + b)
            # hx depends only on x and ht only on t, so under the losses'
            # vmap(x) x vmap(t) tensor-product evaluation each partial matmul
            # is hoisted out of the other batch dimension: the dominant
            # (N_x*N_t, in_dim, hidden) matmul becomes O(N_x) + O(N_t) work
            # plus an elementwise broadcast add.  Same math, fewer FLOPs.
            first = self.trunk.layers[0]
            dx = ex.shape[0]
            hx = first.weight[:, :dx] @ ex
            ht = first.weight[:, dx:] @ et + first.bias
            h = self.trunk.activation(hx + ht)
            for layer in self.trunk.layers[1:-1]:
                h = self.trunk.activation(layer(h))
            out = self.trunk.layers[-1](h)
        else:
            out = self.trunk(jnp.concatenate([ex, et]))

        if self.hard_constraint is not None:
            out = self.hard_constraint(out)
        if self.dirichlet_bc is not None:
            out = self.dirichlet_bc(jnp.atleast_1d(x), out)
        return out


def make_pinn(spatial_dim: int, out_dim: int, *,
              use_spatial_emb: bool = True,
              use_temporal_emb: bool = True,
              spatial_emb_dim: int = 64, spatial_scale: float = 2.0,
              temporal_emb_dim: int = 64, temporal_scale: float = 2.0,
              hidden_dim: int = 64, num_layers: int = 4,
              trunk_type: str = "mlp",
              activation=jnp.tanh,
              hard_constraint: eqx.Module | None = None,
              dirichlet_bc: eqx.Module | None = None,
              tensor_split: bool = False,
              key: jr.PRNGKey) -> _PINN:
    """Factory: create a PINN model.

    Fourier embeddings for x and t can be toggled independently via
    ``use_spatial_emb`` / ``use_temporal_emb``.

    ``trunk_type`` is "mlp" or "modified_mlp".
    ``tensor_split`` (mlp trunk only): split the first layer into x-only and
    t-only matmuls — exact same math, much cheaper under the tensor-product
    (vmap over x times vmap over t) evaluation used in training.
    Single-sample: x ∈ R^spatial_dim, t ∈ R → R^out_dim.
    """
    k_spatial, k_temporal, k_trunk = jr.split(key, 3)

    if use_spatial_emb:
        spatial_emb = FourierEmbedding(spatial_dim, spatial_emb_dim,
                                       spatial_scale, key=k_spatial)
        n_s = len(spatial_scale) if isinstance(spatial_scale, (tuple, list)) else 1
        trunk_in = 2 * spatial_emb_dim * n_s
    else:
        spatial_emb = None
        trunk_in = spatial_dim

    if use_temporal_emb:
        temporal_emb = FourierEmbedding(1, temporal_emb_dim,
                                        temporal_scale, key=k_temporal)
        n_t = len(temporal_scale) if isinstance(temporal_scale, (tuple, list)) else 1
        trunk_in += 2 * temporal_emb_dim * n_t
    else:
        temporal_emb = None
        trunk_in += 1

    if isinstance(activation, str):
        activation = _ACTIVATIONS.get(activation, jnp.tanh)
    if trunk_type == "mlp":
        trunk = MLP(trunk_in, hidden_dim, out_dim, num_layers, activation=activation, key=k_trunk)
    elif trunk_type == "modified_mlp":
        trunk = ModifiedMLP(trunk_in, hidden_dim, out_dim, num_layers, key=k_trunk)
        if tensor_split:
            raise ValueError("tensor_split requires trunk_type='mlp'")
    else:
        raise ValueError(f"Unknown trunk_type: {trunk_type}")

    return _PINN(spatial_emb, temporal_emb, trunk, hard_constraint,
                 dirichlet_bc, tensor_split)


_ACTIVATIONS = {
    'tanh': jnp.tanh, 'relu': jax.nn.relu, 'gelu': jax.nn.gelu,
    'silu': jax.nn.silu, 'swish': jax.nn.swish,
}


if __name__ == "__main__":
    import jax
    key = jr.PRNGKey(0)
    x, t = jnp.array([0.5]), jnp.array([0.1])

    for emb, trunk_type in [(True, "mlp"), (True, "modified_mlp"), (False, "mlp")]:
        net = make_pinn(spatial_dim=1, out_dim=2,
                        use_spatial_emb=emb, use_temporal_emb=emb,
                        trunk_type=trunk_type,
                        hidden_dim=32, num_layers=3, key=key)

        out = net(x, t)
        assert out.shape == (2,), f"shape: {out.shape}"

        batch_x, batch_t = jr.normal(key, (16, 1)), jr.normal(key, (16, 1))
        batch_out = jax.vmap(net)(batch_x, batch_t)
        assert batch_out.shape == (16, 2)

        grad = jax.grad(lambda u, v: net(u, v).sum())(x, t)
        print(f"make_pinn(emb={emb}, trunk={trunk_type}): "
              f"forward {out.shape}, vmap {batch_out.shape}, grad {grad.shape}  ✓")
