"""Fourier coordinate embedding with learnable frequencies."""

import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class FourierEmbedding(eqx.Module):
    """Learnable Fourier feature embedding.

    Maps x ∈ R^d to [sin(π x @ W), cos(π x @ W)] ∈ R^{2*m}
    where W ~ N(0, scale²) is learnable with shape (d, m).

    Single-sample: (d,) → (2*m,).
    """
    kernel: jnp.ndarray

    def __init__(self, input_dim: int, emb_dim: int, scale: float = 2.0, *, key: jr.PRNGKey):
        self.kernel = jr.normal(key, (input_dim, emb_dim)) * scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = jnp.pi * (x @ self.kernel)
        return jnp.concatenate([jnp.sin(y), jnp.cos(y)])


if __name__ == "__main__":
    key = jr.PRNGKey(0)
    emb = FourierEmbedding(input_dim=1, emb_dim=8, scale=2.0, key=key)
    x = jnp.array([0.5])
    out = emb(x)
    assert out.shape == (16,), f"shape: {out.shape}"
    print(f"FourierEmbedding: (1,) → {out.shape}  ✓")
