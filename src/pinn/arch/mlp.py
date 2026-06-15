"""Standard multi-layer perceptron."""

from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class MLP(eqx.Module):
    """Standard MLP: Linear → act → ... → Linear (no activation on output).

    Single-sample: (in_dim,) → (out_dim,).
    """
    layers: list
    activation: Callable = eqx.field(static=True)

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int, activation: Callable = jnp.tanh, *, key: jr.PRNGKey):
        keys = jr.split(key, num_layers + 1)
        self.layers = [eqx.nn.Linear(in_dim, hidden_dim, key=keys[0])]
        for i in range(1, num_layers):
            self.layers.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[i]))
        self.layers.append(eqx.nn.Linear(hidden_dim, out_dim, key=keys[-1]))
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


if __name__ == "__main__":
    import jax
    key = jr.PRNGKey(0)
    mlp = MLP(in_dim=16, hidden_dim=32, out_dim=2, num_layers=3, key=key)
    x = jr.normal(key, (16,))
    out = mlp(x)
    assert out.shape == (2,), f"shape: {out.shape}"
    # vmap
    batch = jr.normal(key, (8, 16))
    batch_out = jax.vmap(mlp)(batch)
    assert batch_out.shape == (8, 2)
    # grad
    grad = jax.grad(lambda u: mlp(u).sum())(x)
    assert grad.shape == (16,)
    print(f"MLP: forward {x.shape}→{out.shape}, vmap {batch.shape}→{batch_out.shape}, grad {grad.shape}  ✓")
