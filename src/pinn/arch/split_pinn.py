"""Split-PINN: DeepONet-style architecture with independent spatial and temporal
branches, fused via dot product.  Efficient: O(N_x + N_t) forward cost per
iteration instead of O(N_x · N_t).

SVD of FEM 1D corrosion data shows rank 5 captures >99.9% energy,
so rank p=10 gives comfortable margin.
"""

import jax.numpy as jnp
import jax
import jax.random as jr
import equinox as eqx

from .embedding import FourierEmbedding
from .mlp import MLP


class _SpatialBranch(eqx.Module):
    """x → FourierEmb → MLP → F_x of shape (2p,)."""
    emb: FourierEmbedding
    trunk: MLP

    def __init__(self, spatial_dim, emb_dim, emb_scale, hidden_dim, rank_p,
                 num_layers, activation, *, key):
        k_emb, k_trunk = jr.split(key)
        self.emb = FourierEmbedding(spatial_dim, emb_dim, emb_scale, key=k_emb)
        self.trunk = MLP(2 * emb_dim, hidden_dim, 2 * rank_p, num_layers,
                         activation=activation, key=k_trunk)

    def __call__(self, x):
        return self.trunk(self.emb(x))


class _TemporalBranch(eqx.Module):
    """t → FourierEmb → MLP → F_t of shape (2p,)."""
    emb: FourierEmbedding
    trunk: MLP

    def __init__(self, emb_dim, emb_scale, hidden_dim, rank_p, num_layers,
                 activation, *, key):
        k_emb, k_trunk = jr.split(key)
        self.emb = FourierEmbedding(1, emb_dim, emb_scale, key=k_emb)
        self.trunk = MLP(2 * emb_dim, hidden_dim, 2 * rank_p, num_layers,
                         activation=activation, key=k_trunk)

    def __call__(self, t):
        return self.trunk(self.emb(jnp.atleast_1d(t)))


class _SplitPINN(eqx.Module):
    """Split-PINN: φ,c = dot_product(F_x[:p], F_t[:p]), dot_product(F_x[p:], F_t[p:])."""
    spatial: _SpatialBranch
    temporal: _TemporalBranch
    rank_p: int = eqx.field(static=True)

    def __call__(self, x, t):
        """Single-sample: (dim,) × () → (2,) for (φ, c)."""
        fx = self.spatial(jnp.atleast_1d(x)).reshape(2, self.rank_p)
        ft = self.temporal(jnp.atleast_1d(t)).reshape(2, self.rank_p)
        return jnp.einsum('fk,fk->f', fx, ft)

    def spatial_batch(self, x_all):
        """Batched spatial forward: (N_x, dim) → (N_x, 2p)."""
        return jax.vmap(self.spatial)(x_all)

    def temporal_batch(self, t_batch):
        """Batched temporal forward: (N_t,) → (N_t, 2p)."""
        return jax.vmap(self.temporal)(t_batch)

    def temporal_derivative(self, t):
        """Time derivative of temporal branch: () → (2p,)."""
        return jnp.squeeze(jax.jacfwd(self.temporal)(jnp.atleast_1d(t)))

    def combine(self, Fx, Ft):
        """Dot-product fusion: (N_x, 2p) × (N_t, 2p) → (N_x, N_t, 2)."""
        Fx = Fx.reshape(-1, 2, self.rank_p)
        Ft = Ft.reshape(-1, 2, self.rank_p)
        return jnp.einsum('ifk,jfk->ijf', Fx, Ft)

    def project(self, Fx, vt):
        """Project spatial features onto a single temporal vector.

        (N_x, 2p) × (2p,) → (N_x, 2) — works for both Ft (field values)
        and dFt/dt (time derivatives).
        """
        Fx = Fx.reshape(-1, 2, self.rank_p)
        vt = vt.reshape(2, self.rank_p)
        return jnp.einsum('ifk,fk->if', Fx, vt)


_ACTIVATIONS = {
    'tanh': jnp.tanh, 'relu': jax.nn.relu, 'gelu': jax.nn.gelu,
    'silu': jax.nn.silu, 'swish': jax.nn.swish,
}


def make_split_pinn(spatial_dim, rank_p=10, hidden_dim=64, num_layers=4, *,
                    spatial_emb_dim=64, spatial_scale=2.0,
                    temporal_emb_dim=64, temporal_scale=2.0,
                    activation='gelu', key):
    """Factory: create a Split-PINN model."""
    act = _ACTIVATIONS.get(activation, jax.nn.gelu)
    if isinstance(act, str):
        raise ValueError(f"Unknown activation: {activation}. Options: {list(_ACTIVATIONS)}")

    k_spatial, k_temporal = jr.split(key)
    spatial = _SpatialBranch(spatial_dim, spatial_emb_dim, spatial_scale,
                             hidden_dim, rank_p, num_layers,
                             activation=act, key=k_spatial)
    temporal = _TemporalBranch(temporal_emb_dim, temporal_scale,
                               hidden_dim, rank_p, num_layers,
                               activation=act, key=k_temporal)
    return _SplitPINN(spatial, temporal, rank_p)


if __name__ == "__main__":
    key = jr.PRNGKey(0)
    rank_p = 10
    Nx, Nt = 101, 20
    net = make_split_pinn(spatial_dim=1, rank_p=rank_p, hidden_dim=32,
                          num_layers=3, key=key)

    # Single-sample
    phi_c = net(jnp.array([0.5]), jnp.array(0.3))
    assert phi_c.shape == (2,)
    print(f"single: {phi_c.shape} ✓  φ={float(phi_c[0]):.4f}, c={float(phi_c[1]):.4f}")

    # Batched spatial
    x_all = jnp.linspace(0, 1, Nx)[:, None]
    Fx = net.spatial_batch(x_all)
    assert Fx.shape == (Nx, 2 * rank_p)
    print(f"spatial_batch: {Fx.shape} ✓")

    # Batched temporal
    t_batch = jnp.linspace(0, 1, Nt)
    Ft = net.temporal_batch(t_batch)
    assert Ft.shape == (Nt, 2 * rank_p)
    print(f"temporal_batch: {Ft.shape} ✓")

    # Combine
    out = net.combine(Fx, Ft)
    assert out.shape == (Nx, Nt, 2)
    print(f"combine: {out.shape} ✓")

    # Project (field and time derivative)
    Ft0 = net.temporal(jnp.array(0.3))
    dFt = net.temporal_derivative(jnp.array(0.3))
    field = net.project(Fx, Ft0)
    d_dt = net.project(Fx, dFt)
    assert field.shape == (Nx, 2) and d_dt.shape == (Nx, 2)
    print(f"project: field {field.shape}, d_dt {d_dt.shape} ✓")

    # Grad through everything
    loss = jnp.mean(out[..., 0]**2 + out[..., 1]**2)
    grad = jax.grad(lambda m: jnp.sum(jax.vmap(m, in_axes=(0, None))(x_all, jnp.array(0.0))[:, 0]))(net)
    print(f"gradient OK ({sum(1 for _ in jax.tree.leaves(grad))} leaves) ✓")

    print("All tests passed.")
