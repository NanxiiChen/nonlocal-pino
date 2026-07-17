"""Fourier coordinate embedding with learnable frequencies."""

import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class FourierEmbedding(eqx.Module):
    """Learnable Fourier feature embedding.

    Maps x ∈ R^d to [sin(π x @ W), cos(π x @ W)] ∈ R^{2*m}
    where W ~ N(0, scale²) is learnable with shape (d, m).

    If scale is a tuple, uses multi-scale: concatenates kernels for each scale,
    giving output dimension 2 * emb_dim * len(scale).
    """
    kernel: jnp.ndarray

    def __init__(self, input_dim: int, emb_dim: int, scale: float | tuple = 2.0, *, key: jr.PRNGKey):
        if isinstance(scale, (tuple, list)):
            kernels = []
            for i, s in enumerate(scale):
                k_i = jr.fold_in(key, i)
                kernels.append(jr.normal(k_i, (input_dim, emb_dim)) * s)
            self.kernel = jnp.concatenate(kernels, axis=1)
        else:
            self.kernel = jr.normal(key, (input_dim, emb_dim)) * scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = jnp.pi * (x @ self.kernel)
        return jnp.concatenate([jnp.sin(y), jnp.cos(y)])
