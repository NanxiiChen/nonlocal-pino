"""Opt-in joint space-time queries; both geometry streams remain independent."""
import equinox as eqx
import jax
import jax.numpy as jnp


class JointPrepared(eqx.Module):
    """Spatial projection plus shared context; slicing selects queries only."""
    query: object
    encoded: tuple

    @property
    def ndim(self):
        return self.query.ndim

    @property
    def shape(self):
        return self.query.shape

    def __len__(self):
        return len(self.query)

    def __getitem__(self, selection):
        return JointPrepared(self.query[selection], self.encoded)


class JointQueryOperator(eqx.Module):
    backbone: object
    temporal_projection: object

    def __init__(self, backbone, *, key):
        self.backbone = eqx.tree_at(lambda net: net.time, backbone, None)
        self.temporal_projection = eqx.nn.Linear(
            1+2*backbone.time_features, backbone.width, use_bias=False, key=key)

    @property
    def width(self):
        return self.backbone.width

    @property
    def time_features(self):
        return self.backbone.time_features

    @property
    def time_max_frequency(self):
        return self.backbone.time_max_frequency

    def encode(self, features, weights):
        return self.backbone.encode(features, weights)

    def prepare_encoded(self, encoded, points):
        if points.ndim != 2 or points.shape[1] != len(self.backbone.extent):
            raise ValueError('Joint queries require spatial points(N,2)')
        query = jax.vmap(self.backbone.spatial)(self.backbone.position(points))
        return JointPrepared(query, encoded)

    def decode_temporal(self, prepared, temporal):
        if not isinstance(prepared, JointPrepared):
            raise ValueError('Joint-query decoding requires its spatial/context preparation')
        contexts, weights = prepared.encoded
        def at_time(features):
            hidden = prepared.query + self.temporal_projection(features)
            for block in self.backbone.cross:
                hidden = block(hidden, contexts, weights)
            for layer in self.backbone.layers:
                hidden = jax.nn.gelu(jax.vmap(layer)(hidden))
            return self.backbone.output_scale*jax.vmap(self.backbone.output)(hidden)
        return jax.vmap(at_time)(temporal)


def evolution_queries(prepared, n_graph, n_ic):
    """Exclude soft-IC-only queries without duplicating condition streams."""
    if isinstance(prepared, JointPrepared):
        return JointPrepared(jnp.concatenate((prepared.query[:n_graph],
                                              prepared.query[n_graph+n_ic:])), prepared.encoded)
    return jnp.concatenate((prepared[:n_graph], prepared[n_graph+n_ic:]))
