"""Optional fixed-shape point-patch encoder and coordinate-query damage model.

Local abstraction follows PointNet++ (arXiv:1706.02413); the coordinate
cross-attention decoder follows the PI-CViT pattern. Selection by bounded-H
patch contrast is an experimental input heuristic, not a physical constraint.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class PatchLevel(eqx.Module):
    indices: jax.Array
    mask: jax.Array
    relative: jax.Array
    weights: jax.Array
    metadata: jax.Array
    valid: jax.Array


def build_patches(points, volumes, *, cells):
    """Fixed grid grouping of material points; exact padded maximum, no truncation."""
    if (type(cells) is not int or cells < 1 or points.ndim != 2
            or points.shape[1] != 2 or points.dtype != np.float64
            or volumes.shape != (len(points),) or volumes.dtype != np.float64
            or not np.isfinite(points).all() or not np.isfinite(volumes).all()
            or len(points) == 0 or np.min(volumes) <= 0):
        raise ValueError('patch geometry requires finite FP64 2D points and positive volumes')
    low, span = points.min(0), np.ptp(points, axis=0)
    if np.min(span) <= 0:
        raise ValueError('patch geometry must span both coordinates')
    bins = np.minimum(((points-low)/span*cells).astype(np.int32), cells-1)
    ids = bins[:, 0]+cells*bins[:, 1]
    counts = np.bincount(ids, minlength=cells*cells)
    width = int(counts.max())
    indices = np.zeros((cells*cells, width), np.int32)
    mask = np.zeros_like(indices, bool)
    relative = np.zeros((*indices.shape, 2), np.float64)
    weights = np.zeros_like(indices, np.float64)
    grid = np.column_stack((np.tile(np.arange(cells), cells),
                            np.repeat(np.arange(cells), cells)))
    centers = low+(grid+.5)*span/cells
    order = np.argsort(ids, kind='stable')
    offsets = np.concatenate(([0], np.cumsum(counts)))
    for patch, count in enumerate(counts):
        selected = order[offsets[patch]:offsets[patch+1]]
        indices[patch, :count] = selected
        mask[patch, :count] = True
        relative[patch, :count] = (points[selected]-centers[patch])/(span/cells)
        if count:
            weights[patch, :count] = volumes[selected]/volumes[selected].sum()
    metadata = np.column_stack((2*(centers-low)/span-1,
                                np.full((cells*cells, 2), 1/cells)))
    return PatchLevel(*map(jnp.asarray, (indices, mask, relative, weights,
                                         metadata, counts > 0)))


def select_fine(history, patches, *, gc0, count):
    """Fixed token count; score is the within-patch range of bounded history."""
    H = jax.lax.stop_gradient(history)
    local = 2*H[patches.indices]/(gc0+2*H[patches.indices])
    maximum = jnp.max(jnp.where(patches.mask, local, -jnp.inf), axis=1)
    minimum = jnp.min(jnp.where(patches.mask, local, jnp.inf), axis=1)
    score = jnp.where(patches.valid, maximum-minimum, -jnp.inf)
    return jax.lax.stop_gradient(jax.lax.top_k(score, count)[1])


def make_context(history, patches, *, gc0, count):
    H = np.asarray(history)
    if (H.ndim != 1 or H.dtype != np.float64 or len(H) == 0
            or not np.isfinite(H).all() or np.min(H) < 0 or np.max(H) <= 0
            or not np.isfinite(gc0) or gc0 <= 0 or type(count) is not int
            or not 0 < count <= int(np.asarray(patches.valid).sum())
            or int(np.max(np.asarray(patches.indices))) >= len(H)):
        raise ValueError('history context requires positive forcing and enough nonempty fixed patches')
    H = jax.lax.stop_gradient(jnp.asarray(H))
    return dict(H=H, fine_ids=select_fine(H, patches, gc0=gc0, count=count))


class HistoryOperator(eqx.Module):
    point_mlp: eqx.nn.MLP
    token_mlp: eqx.nn.MLP
    query_mlp: eqx.nn.MLP
    token_norm: eqx.nn.LayerNorm
    query_norm: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    head: eqx.nn.MLP

    def __init__(self, *, width, point_depth, query_depth, head_depth,
                 heads, initial_damage, key):
        if (any(type(n) is not int or n < 1 for n in
                (width, point_depth, query_depth, head_depth, heads))
                or width % heads or not np.isfinite(initial_damage)
                or not 0 < initial_damage < 1 or not jax.config.jax_enable_x64):
            raise ValueError('history operator requires FP64, compatible dimensions and bounded initial damage')
        kp, kt, kq, ka, kh = jax.random.split(key, 5)
        self.point_mlp = eqx.nn.MLP(4, width, width, point_depth,
                                   activation=jax.nn.silu, dtype=jnp.float64, key=kp)
        self.token_mlp = eqx.nn.MLP(2*width+4, width, 2*width, 1,
                                   activation=jax.nn.silu, dtype=jnp.float64, key=kt)
        self.query_mlp = eqx.nn.MLP(2, width, 2*width, query_depth,
                                   activation=jax.nn.silu, dtype=jnp.float64, key=kq)
        self.token_norm = eqx.nn.LayerNorm(width, dtype=jnp.float64)
        self.query_norm = eqx.nn.LayerNorm(width, dtype=jnp.float64)
        self.attention = eqx.nn.MultiheadAttention(heads, width, dropout_p=0., dtype=jnp.float64, key=ka)
        head = eqx.nn.MLP(width, 1, 2*width, head_depth,
                          activation=jax.nn.silu, dtype=jnp.float64, key=kh)
        self.head = eqx.tree_at(lambda n: (n.layers[-1].weight, n.layers[-1].bias),
            head, (jnp.zeros_like(head.layers[-1].weight),
                   jnp.full_like(head.layers[-1].bias, np.log(initial_damage/(1-initial_damage)))))

    def _tokens(self, history, level, selected, gc0):
        mask, indices = level.mask[selected], level.indices[selected]
        ratio = 2*history[indices]/gc0
        features = jnp.concatenate((level.relative[selected],
            jnp.log1p(ratio)[..., None], (ratio/(1+ratio))[..., None]), axis=-1)
        features = jax.vmap(jax.vmap(self.point_mlp))(features)
        mean = jnp.sum(level.weights[selected][..., None]*features, axis=1)
        maximum = jnp.max(jnp.where(mask[..., None], features, -jnp.inf), axis=1)
        valid = level.valid[selected]
        maximum = jnp.where(valid[:, None], maximum, 0.)
        tokens = jax.vmap(self.token_mlp)(jnp.concatenate(
            (mean, maximum, level.metadata[selected]), axis=1))
        return tokens, valid

    def __call__(self, context, points, coarse, fine, *, gc0):
        H = jax.lax.stop_gradient(context['H'])
        selected = jax.lax.stop_gradient(context['fine_ids'])
        if H.ndim != 1 or points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('history operator expects a field vector and 2D queries')
        coarse_tokens, coarse_valid = self._tokens(H, coarse, jnp.arange(len(coarse.valid)), gc0)
        fine_tokens, fine_valid = self._tokens(H, fine, selected, gc0)
        tokens = jax.vmap(self.token_norm)(jnp.concatenate((coarse_tokens, fine_tokens)))
        valid = jnp.concatenate((coarse_valid, fine_valid))
        query = jax.vmap(self.query_mlp)(points)
        normalized = jax.vmap(self.query_norm)(query)
        values = query+self.attention(normalized, tokens, tokens,
            mask=jnp.broadcast_to(valid, (len(points), len(valid))))
        return jax.nn.sigmoid(jax.vmap(self.head)(values)[:, 0])


class HistoryReplay(eqx.Module):
    histories: jax.Array
    fine_ids: jax.Array
    count: jax.Array

    def __init__(self, *, capacity, nodes, fine_count):
        if any(type(n) is not int or n < 1 for n in (capacity, nodes, fine_count)):
            raise ValueError('positive fixed replay dimensions required')
        self.histories = jnp.zeros((capacity, nodes), jnp.float64)
        self.fine_ids = jnp.zeros((capacity, fine_count), jnp.int32)
        self.count = jnp.asarray(0, jnp.int32)

    def append(self, context):
        """Host event only, once for each load start/end; return a new buffer."""
        count = int(self.count)
        if count >= self.histories.shape[0]:
            raise ValueError('history replay capacity exhausted; increase explicit capacity before a new run')
        H, ids = np.asarray(context['H']), np.asarray(context['fine_ids'])
        if (H.shape != self.histories.shape[1:] or ids.shape != self.fine_ids.shape[1:]
                or H.dtype != np.float64 or ids.dtype != np.int32
                or not np.isfinite(H).all() or H.min() < 0 or H.max() <= 0 or ids.min() < 0):
            raise ValueError('invalid replay snapshot shape, dtype or positive history')
        return eqx.tree_at(lambda b: (b.histories, b.fine_ids, b.count), self,
            (self.histories.at[count].set(context['H']), self.fine_ids.at[count].set(context['fine_ids']),
             self.count+jnp.asarray(1, jnp.int32)))

    def batch(self, current, key, enabled, *, old_count, fraction):
        """Fixed shape even with an empty bank; inactive tasks get finite inputs."""
        if type(old_count) is not int or old_count < 1 or not 0 < fraction < 1:
            raise ValueError('explicit fixed replay batch count and fraction required')
        active = jnp.logical_and(enabled, self.count > 0)
        indices = jax.random.randint(key, (old_count,), 0, jnp.maximum(self.count, 1), dtype=jnp.int32)
        old_H = jnp.where(active, self.histories[indices], current['H'][None])
        old_ids = jnp.where(active, self.fine_ids[indices], current['fine_ids'][None])
        old_weight = jnp.where(active, fraction, 0.)
        return jax.tree.map(jax.lax.stop_gradient, dict(
            H=jnp.concatenate((current['H'][None], old_H)),
            fine_ids=jnp.concatenate((current['fine_ids'][None], old_ids)),
            weights=jnp.concatenate((jnp.reshape(1-old_weight, (1,)),
                                     jnp.full((old_count,), old_weight/old_count)))))
