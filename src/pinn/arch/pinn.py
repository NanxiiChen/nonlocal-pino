"""PINN model composing Fourier embeddings with an MLP trunk."""

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

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        parts = []
        if self.spatial_emb is not None:
            parts.append(self.spatial_emb(x))
        else:
            parts.append(x)
        if self.temporal_emb is not None:
            parts.append(self.temporal_emb(t))
        else:
            parts.append(t)
        h = jnp.concatenate(parts)
        return self.trunk(h)


def make_pinn(spatial_dim: int, out_dim: int, *,
              use_fourier: bool = True,
              spatial_emb_dim: int = 64, spatial_scale: float = 2.0,
              temporal_emb_dim: int = 64, temporal_scale: float = 2.0,
              hidden_dim: int = 64, num_layers: int = 4,
              trunk_type: str = "mlp",
              key: jr.PRNGKey) -> _PINN:
    """Factory: create a PINN model.

    If `use_fourier`, applies Fourier embeddings to x and t separately
    before concatenation. Otherwise concatenates raw coordinates directly.

    `trunk_type` is "mlp" or "modified_mlp".
    Single-sample: x ∈ R^spatial_dim, t ∈ R → R^out_dim.
    """
    k_spatial, k_temporal, k_trunk = jr.split(key, 3)

    if use_fourier:
        spatial_emb = FourierEmbedding(spatial_dim, spatial_emb_dim,
                                       spatial_scale, key=k_spatial)
        temporal_emb = FourierEmbedding(1, temporal_emb_dim,
                                        temporal_scale, key=k_temporal)
        trunk_in = 2 * (spatial_emb_dim + temporal_emb_dim)
    else:
        spatial_emb = None
        temporal_emb = None
        trunk_in = spatial_dim + 1

    if trunk_type == "mlp":
        trunk = MLP(trunk_in, hidden_dim, out_dim, num_layers, key=k_trunk)
    elif trunk_type == "modified_mlp":
        trunk = ModifiedMLP(trunk_in, hidden_dim, out_dim, num_layers, key=k_trunk)
    else:
        raise ValueError(f"Unknown trunk_type: {trunk_type}")

    return _PINN(spatial_emb, temporal_emb, trunk)


if __name__ == "__main__":
    import jax
    key = jr.PRNGKey(0)
    x, t = jnp.array([0.5]), jnp.array([0.1])

    for use_fourier, trunk_type in [(True, "mlp"), (True, "modified_mlp"), (False, "mlp")]:
        net = make_pinn(spatial_dim=1, out_dim=2,
                        use_fourier=use_fourier, trunk_type=trunk_type,
                        hidden_dim=32, num_layers=3, key=key)

        out = net(x, t)
        assert out.shape == (2,), f"shape: {out.shape}"

        batch_x, batch_t = jr.normal(key, (16, 1)), jr.normal(key, (16, 1))
        batch_out = jax.vmap(net)(batch_x, batch_t)
        assert batch_out.shape == (16, 2)

        grad = jax.grad(lambda u, v: net(u, v).sum())(x, t)
        print(f"make_pinn(fourier={use_fourier}, trunk={trunk_type}): "
              f"forward {out.shape}, vmap {batch_out.shape}, grad {grad.shape}  ✓")
