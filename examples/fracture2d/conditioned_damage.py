"""Geometry/known-history conditioned neural nonlocal damage, without correction.

P_ij=-D_ij/D_ii (i!=j), P_ii=0 is a nonnegative row-stochastic graph
average. P^k H are known-coefficient features, not damage-equation iterates.
The same AT2 residual acts on the neural nodal output; no direct targets enter.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pinn.arch import make_pinn
from examples.fracture2d.train import _nonnegative_damage


def make_features(operator, points, *, gc0, length, tip, powers, roundoff_factor):
    D = operator.to_scipy().tocsr()
    powers = tuple(powers)
    if (points.dtype != np.float64 or points.shape != (D.shape[0], 2)
            or not np.isfinite(points).all() or D.dtype != np.float64
            or not np.isfinite(D.data).all() or not np.isfinite([gc0, length]).all()
            or min(gc0, length) <= 0 or not powers or powers[0] != 0
            or any(type(k) is not int or k < 0 for k in powers)
            or tuple(sorted(set(powers))) != powers or len(tip) != 2
            or not np.isfinite(tip).all() or not np.isfinite(roundoff_factor)
            or roundoff_factor < 1 or not jax.config.jax_enable_x64):
        raise ValueError('invalid explicit FP64 conditioned feature geometry/powers')
    diagonal = D.diagonal()
    off = D.copy()
    off.setdiag(0.)
    off.eliminate_zeros()
    off.sort_indices()
    if np.min(diagonal) <= 0 or np.any(off.data > 0):
        raise ValueError('history features require a positive graph diagonal/nonpositive offdiagonal')
    row_error = np.max(abs(np.asarray(off.sum(axis=1)).ravel() + diagonal) / diagonal)
    if row_error > roundoff_factor * np.finfo(float).eps:
        raise ValueError('history graph average rows do not sum to one')
    counts = np.diff(off.indptr)
    width = int(np.max(counts))
    columns = np.zeros((len(points), width), np.int32)
    weights = np.zeros_like(columns, np.float64)
    for i in range(len(points)):
        start, end = off.indptr[i:i + 2]
        columns[i, :end-start] = off.indices[start:end]
        weights[i, :end-start] = -off.data[start:end] / diagonal[i]
    columns, weights = jnp.asarray(columns), jnp.asarray(weights)
    xy = jnp.asarray(points)
    offset = (xy-jnp.asarray(tip)) / length
    geometry = jnp.column_stack((2*xy[:, 0]-1, 2*xy[:, 1]-1,
        jnp.tanh(offset[:, 0]), jnp.tanh(offset[:, 1]),
        jnp.exp(-.5*jnp.sum(offset**2, axis=1)), jnp.exp(-.5*offset[:, 1]**2)))

    @jax.jit
    def features(H):
        H = jax.lax.stop_gradient(H)
        current, values = H, []
        for k in range(powers[-1]+1):
            if k in powers:
                ratio = 2*current/gc0
                values.extend((jnp.log1p(ratio), ratio/(1+ratio)))
            if k < powers[-1]:
                current = jnp.sum(weights * current[columns], axis=1)
        return jnp.concatenate((geometry, jnp.column_stack(values)), axis=1)
    return features, dict(input_dimension=6+2*len(powers), powers=list(powers),
        max_row_sum_defect=float(row_error), graph_row_width=width,
        feature_definition='six geometry features and log1p/bounded P^k(2H/gc0) pairs')


def _build_network(c, input_dimension):
    a = c.arch
    if (a.type != 'mlp' or a.activation not in ('swish', 'tanh')
            or type(a.hidden_dim) is not int or a.hidden_dim < 1
            or type(a.num_layers) is not int or a.num_layers < 1
            or a.use_spatial_emb or a.tensor_split or a.multi_head):
        raise ValueError('conditioned trial requires an explicit ordinary MLP without Fourier embedding')
    net = make_pinn(spatial_dim=input_dimension, out_dim=1, trunk_type=a.type,
        hidden_dim=a.hidden_dim, num_layers=a.num_layers, activation=a.activation,
        use_spatial_emb=a.use_spatial_emb, use_temporal_emb=False, static=True,
        spatial_emb_dim=a.spatial_emb_dim, spatial_scale=a.spatial_scale,
        temporal_emb_dim=1, temporal_scale=1., tensor_split=a.tensor_split,
        multi_head=a.multi_head, film_hidden_dim=a.film_hidden_dim,
        film_num_layers=a.film_num_layers, key=jax.random.PRNGKey(c.train.seed+1))
    net = jax.tree.map(lambda x: x.astype(jnp.float64), net)
    return eqx.tree_at(lambda n: (n.output.weight, n.output.bias), net,
                       (jnp.zeros_like(net.output.weight), jnp.zeros_like(net.output.bias)))


def make_problem(features, matvec, c):
    gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0

    def image(net, H, load):
        raw = c.physical.E/gc0 * load**2 * jax.vmap(net)(features(H))[:, 0]
        return _nonnegative_damage(raw, c.physical.damage_floor)

    def residual(net, H, load):
        d = image(net, H, load)
        return (gc0+2*H)*d + gc1*matvec(d) - 2*H

    def metrics(net, H, load):
        d = image(net, H, load)
        parts = (gc0*d, gc1*matvec(d), 2*H*(d-1))
        return dict(AT2_relative=jnp.max(abs(sum(parts))) / sum(jnp.max(abs(p)) for p in parts),
                    damage_min=jnp.min(d), damage_max=jnp.max(d))

    def mse(net, H, load):
        scale = 2*jnp.sqrt(jnp.mean(H**2))
        return jnp.mean((residual(net, H, load)/scale)**2)

    def l4(net, H, load):
        scale = 2*jnp.sqrt(jnp.mean(H**2))
        return jnp.sqrt(jnp.mean((residual(net, H, load)/scale)**4)
                        + c.conditioned.loss_floor**4)

    return image, metrics, {'residual_mse': mse, 'residual_l4': l4}
