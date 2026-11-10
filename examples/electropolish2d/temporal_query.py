"""Explicit time-dependent spatial reads for the Electropolish comparison."""
import equinox as eqx
import jax
import jax.numpy as jnp

from nonlocal_pinn.nn.conditional import ConditionalFieldOperator


class TemporalQueryOperator(eqx.Module):
    backbone: ConditionalFieldOperator
    time_projection: eqx.nn.Linear | None
    kind: str = eqx.field(static=True)

    def __init__(self, backbone, kind, *, key):
        if kind not in ('query_film', 'joint_query') or backbone.cross_attention is None:
            raise ValueError('Temporal query study requires a declared variant and cross-attention')
        self.kind = kind
        self.time_projection = None
        if kind == 'joint_query':
            self.time_projection = eqx.nn.Linear(1 + 2 * len(backbone.time_frequencies),
                                                 backbone.width, use_bias=False, key=key)
            backbone = eqx.tree_at(lambda net: net.time_film, backbone, None)
        self.backbone = backbone

    def prepare(self, condition, x, local):
        return self.backbone.project_queries(x, local), self.backbone.encode(condition)

    def _decode_time(self, prepared, time):
        query, context = prepared
        net = self.backbone
        if self.kind == 'query_film':
            film = net._temporal(time)
            query = (1 + film[0, 0]) * query + film[0, 1]
            hidden = net.cross_attention(query, context)
            return jax.vmap(net._point, in_axes=(0, None))(hidden, film.at[0].set(0))

        # Split projection of the concatenated spatial and temporal features.
        # Only the spatial part has a bias, so the sum is one affine query map.
        query = query + self.time_projection(net.temporal_features(time))
        hidden = net.cross_attention(query, context)
        for layer in net.layers:
            hidden = jax.nn.gelu(jax.vmap(layer)(hidden))
        return jax.vmap(net.output)(hidden)

    def decode(self, prepared, times):
        if times.ndim != 1:
            raise ValueError('Times must have shape (Nt,)')
        return jax.vmap(self._decode_time, in_axes=(None, 0))(prepared, times)

    def __call__(self, condition, x, times, local):
        return self.decode(self.prepare(condition, x, local), times)
