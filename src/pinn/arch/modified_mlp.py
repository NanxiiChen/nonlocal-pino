"""Modified MLP with learnable gate vectors (u, v) and tanh interpolation."""

from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class ModifiedMLP(eqx.Module):
    """Modified MLP: gate vectors u, v are fixed per input; each hidden layer
    applies tanh then linearly interpolates between u and v.

    Single-sample: (in_dim,) → (out_dim,).
    """
    u_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    hidden_layers: list
    output: eqx.nn.Linear
    activation: Callable = eqx.field(static=True)

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int, activation: Callable = jnp.tanh, *, key: jr.PRNGKey):
        k0, k1, k_out, *k_hidden = jr.split(key, 3 + num_layers)
        self.u_proj = eqx.nn.Linear(in_dim, hidden_dim, key=k0)
        self.v_proj = eqx.nn.Linear(in_dim, hidden_dim, key=k1)
        self.hidden_layers = [eqx.nn.Linear(in_dim, hidden_dim, key=k_hidden[0])]
        for ki in k_hidden[1:]:
            self.hidden_layers.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=ki))
        self.output = eqx.nn.Linear(hidden_dim, out_dim, key=k_out)
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        u = self.activation(self.u_proj(x))
        v = self.activation(self.v_proj(x))
        for layer in self.hidden_layers:
            z = jnp.tanh(layer(x))
            x = z * u + (1.0 - z) * v
        return self.output(x)


if __name__ == "__main__":
    import jax
    key = jr.PRNGKey(0)
    mmlp = ModifiedMLP(in_dim=16, hidden_dim=32, out_dim=2, num_layers=3, key=key)
    x = jr.normal(key, (16,))
    out = mmlp(x)
    assert out.shape == (2,), f"shape: {out.shape}"
    # vmap
    batch = jr.normal(key, (8, 16))
    batch_out = jax.vmap(mmlp)(batch)
    assert batch_out.shape == (8, 2)
    # grad
    grad = jax.grad(lambda u: mmlp(u).sum())(x)
    assert grad.shape == (16,)
    print(f"ModifiedMLP: forward {x.shape}→{out.shape}, vmap {batch.shape}→{batch_out.shape}, grad {grad.shape}  ✓")
