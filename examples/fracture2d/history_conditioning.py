"""Experimental input conditioning; physical H and the AT2 residual stay intact."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.fracture2d.history_network import HistoryOperator
from examples.fracture2d.losses import damage_residual, residual_mean


def make_conditioned_loss(*, coarse, fine, points, columns, values, gc0, gc1, floor):
    """Fixed task shapes; a traced flag switches the existing L1/L2 formulas."""
    def image(net, context):
        return net(context, points, coarse, fine, gc0=gc0)

    def objective(net, batch):
        batch = jax.tree.map(jax.lax.stop_gradient, batch)
        def body(total, task):
            H, ids, weight = task
            def single(_):
                d = image(net, dict(H=H, fine_ids=ids))
                r = damage_residual(d, H, jnp.sum(values*d[columns], axis=1),
                    gc0=gc0, gc1=gc1)
                n = H.shape[0]
                controls = dict(population_size=n, floor=floor)
                probability, weights = jnp.full(n, 1/n, jnp.float64), jnp.ones(n, jnp.float64)
                l1 = residual_mean(r, H, probability, weights, norm='l1', **controls)
                l2 = residual_mean(r, H, probability, weights, norm='l2', **controls)
                return jnp.where(batch['l2'], l2, l1)
            value = jax.lax.cond(weight > 0, single, lambda _: jnp.zeros((), jnp.float64), None)
            return total+weight*value, None
        return jax.lax.scan(body, jnp.zeros((), jnp.float64),
            (batch['H'], batch['fine_ids'], batch['weights']))[0]
    return image, objective


def make_conditioned_step(objective, optimizer, *, old_count, fraction):
    @eqx.filter_jit
    def step(net, state, current, replay, key, enabled, l2):
        batch = replay.batch(current, key, enabled, old_count=old_count, fraction=fraction)
        batch = dict(batch, l2=l2)
        value, gradient = eqx.filter_value_and_grad(objective)(net, batch)
        updates, state = optimizer.update(gradient, state, eqx.filter(net, eqx.is_inexact_array))
        return eqx.apply_updates(net, updates), state, value
    return step


def history_inputs(history, *, gc0, normalized):
    """Positive contexts only. Return encoder H, local channels and amplitude.

    Normalize by a stable nodal RMS and retain log(2 RMS(H)/gc0) separately.
    The returned encoder H is an input feature, never a physical PDE history.
    """
    H = jax.lax.stop_gradient(history)
    maximum = jnp.max(H)
    scale = maximum*jnp.sqrt(jnp.mean((H/maximum)**2))
    ratio = jnp.where(normalized, H/scale, 2*H/gc0)
    local = jnp.stack((jnp.log1p(ratio), ratio/(1+ratio)), axis=-1)
    amplitude = jnp.where(normalized, jnp.log(scale)+jnp.log(2/gc0), 0.)
    encoder_history = jnp.where(normalized, (H/scale)*(gc0/2), H)
    return encoder_history, local, amplitude


class ConditionedHistoryOperator(HistoryOperator):
    """Capacity-matched modes with nontrainable, dynamic Boolean array flags.

    __call__ evaluates the full cloud in H order for the existing loss factory.
    predict_queries supports reordered/subset queries through explicit indices.
    """
    normalized: jax.Array
    local_history: jax.Array

    def __init__(self, *, width, point_depth, query_depth, head_depth, heads,
                 initial_damage, normalized, local_history, key):
        if type(normalized) is not bool or type(local_history) is not bool:
            raise ValueError('conditioning modes must be explicit booleans')
        super().__init__(width=width, point_depth=point_depth, query_depth=query_depth,
            head_depth=head_depth, heads=heads, initial_damage=initial_damage, key=key)
        head = eqx.nn.MLP(width+3, 1, 2*width, head_depth,
            activation=jax.nn.silu, dtype=jnp.float64, key=jax.random.split(key, 5)[4])
        self.head = eqx.tree_at(lambda n:(n.layers[-1].weight,n.layers[-1].bias), head,
            (jnp.zeros_like(head.layers[-1].weight),
             jnp.full_like(head.layers[-1].bias,np.log(initial_damage/(1-initial_damage)))))
        self.normalized = jnp.asarray(normalized)
        self.local_history = jnp.asarray(local_history)

    def predict_queries(self, context, points, query_indices, coarse, fine, *, gc0):
        query_indices = eqx.error_if(query_indices,
            jnp.any((query_indices < 0) | (query_indices >= len(context['H']))),
            'query index out of range for the supplied history')
        return self._predict_queries(context, points, query_indices, coarse, fine, gc0=gc0)

    def _predict_queries(self, context, points, query_indices, coarse, fine, *, gc0):
        H = jax.lax.stop_gradient(context['H'])
        selected = jax.lax.stop_gradient(context['fine_ids'])
        query_indices = jax.lax.stop_gradient(query_indices)
        if (H.ndim != 1 or points.ndim != 2 or points.shape[1] != 2
                or query_indices.shape != (len(points),)
                or query_indices.dtype != jnp.int32):
            raise ValueError('conditioned queries require matching explicit int32 indices')
        encoded, local, amplitude = history_inputs(H, gc0=gc0, normalized=self.normalized)
        coarse_tokens, coarse_valid = self._tokens(encoded, coarse, jnp.arange(len(coarse.valid)), gc0)
        fine_tokens, fine_valid = self._tokens(encoded, fine, selected, gc0)
        tokens = jax.vmap(self.token_norm)(jnp.concatenate((coarse_tokens,fine_tokens)))
        valid = jnp.concatenate((coarse_valid,fine_valid))
        query = jax.vmap(self.query_mlp)(points)
        values = query+self.attention(jax.vmap(self.query_norm)(query), tokens, tokens,
            mask=jnp.broadcast_to(valid,(len(points),len(valid))))
        local = jnp.where(self.local_history, local[query_indices], 0.)
        features = jnp.concatenate((values, local,
            jnp.broadcast_to(amplitude,(len(points),1))),axis=-1)
        return jax.nn.sigmoid(jax.vmap(self.head)(features)[:,0])

    def __call__(self, context, points, coarse, fine, *, gc0):
        if len(points) != len(context['H']):
            raise ValueError('full-cloud call requires queries in H order; use predict_queries with indices')
        return self._predict_queries(context, points, jnp.arange(len(points),dtype=jnp.int32),
            coarse, fine, gc0=gc0)
