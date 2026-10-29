"""Conditioned field operator with explicit reusable spatial preparation.

Inspired by PI-CViT's patch encoder, coordinate cross-attention and time FiLM:
https://github.com/NanxiiChen/PI-CViT/blob/main/models/cvit.py
This is an independently written separable variant: attention is spatial and
condition dependent, followed by a time-modulated point decoder. It is not an
algebraic refactor of upstream time-dependent cross-attention.
"""
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class AttentionBlock(eqx.Module):
    norm_q: eqx.nn.LayerNorm
    norm_kv: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    norm_out: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP

    def __init__(self, width, heads, mlp_ratio, norm_eps, *, key):
        k_attn, k_mlp = jr.split(key)
        self.norm_q = eqx.nn.LayerNorm(width, eps=norm_eps)
        self.norm_kv = eqx.nn.LayerNorm(width, eps=norm_eps)
        self.attention = eqx.nn.MultiheadAttention(
            heads, width, use_query_bias=True, use_key_bias=True,
            use_value_bias=True, use_output_bias=True, key=k_attn)
        self.norm_out = eqx.nn.LayerNorm(width, eps=norm_eps)
        self.mlp = eqx.nn.MLP(width, width, width_size=width * mlp_ratio,
                              depth=1, activation=jax.nn.gelu, key=k_mlp)

    def __call__(self, query, context):
        q = jax.vmap(self.norm_q)(query)
        kv = jax.vmap(self.norm_kv)(context)
        hidden = query + self.attention(q, kv, kv)
        return hidden + jax.vmap(self.mlp)(jax.vmap(self.norm_out)(hidden))


class ConditionalFieldOperator(eqx.Module):
    """Map a condition image and arbitrary queries to raw field channels.

    ``prepare(condition, x, local)`` runs once per condition/spatial point set.
    ``decode(prepared, times)`` returns (Nt, Nx, out_dim) with no time coupling.
    Local features contain known input-function values, never future labels.
    All model-dependent preparation stays inside the differentiated forward
    call; callers must not reuse it after an optimizer update.
    """
    patch_projection: eqx.nn.Linear
    encoder: tuple
    encoder_norm: eqx.nn.LayerNorm
    spatial_projection: eqx.nn.Linear
    cross_attention: AttentionBlock | None
    condition_film: eqx.nn.MLP | None
    time_film: eqx.nn.MLP
    layers: tuple
    output: eqx.nn.Linear
    spatial_frequencies: jax.Array
    image_shape: tuple = eqx.field(static=True)
    patch_shape: tuple = eqx.field(static=True)
    extent: tuple = eqx.field(static=True)
    width: int = eqx.field(static=True)
    local_dim: int = eqx.field(static=True)

    def __init__(self, *, image_shape, patch_shape, extent, width, encoder_depth,
                 heads, mlp_ratio, decoder_depth, spatial_features,
                 spatial_scale, time_features, time_scale, local_dim, out_dim,
                 conditioning, norm_eps, key):
        if not all(math.isfinite(v) for v in (*extent, spatial_scale, time_scale, norm_eps)):
            raise ValueError('Conditional operator scales must be finite')
        dim = len(image_shape)-1
        if dim not in (2, 3) or len(patch_shape) != dim or len(extent) != dim:
            raise ValueError('Condition must be channels-first 2D/3D with matching patch/extent dimensions')
        channels, *resolution = image_shape
        if (min(image_shape) <= 0 or min(patch_shape) <= 0
                or any(n % p for n, p in zip(resolution, patch_shape))
                or width < 2*dim or width % (2*dim) or heads <= 0
                or width % heads or encoder_depth < 1 or decoder_depth < 1
                or spatial_features < 1 or time_features < 1 or mlp_ratio < 1
                or local_dim < 0 or out_dim < 1 or min(extent) <= 0
                or spatial_scale <= 0 or time_scale <= 0 or norm_eps <= 0):
            raise ValueError('Invalid conditional operator shapes/scales/attention dimensions')
        if conditioning not in ('cross_attention', 'film'):
            raise ValueError(f'Unknown conditioning: {conditioning}')
        keys = iter(jr.split(key, encoder_depth + decoder_depth + 7))
        self.patch_projection = eqx.nn.Linear(channels * math.prod(patch_shape), width, key=next(keys))
        self.encoder = tuple(AttentionBlock(width, heads, mlp_ratio, norm_eps,
                                            key=next(keys)) for _ in range(encoder_depth))
        self.encoder_norm = eqx.nn.LayerNorm(width, eps=norm_eps)
        self.spatial_frequencies = jr.normal(next(keys), (dim, spatial_features)) * spatial_scale
        self.spatial_projection = eqx.nn.Linear(dim + 2 * spatial_features + local_dim,
                                                width, key=next(keys))
        k_condition = next(keys)
        self.cross_attention = (AttentionBlock(width, heads, mlp_ratio, norm_eps,
                                               key=k_condition)
                                if conditioning == 'cross_attention' else None)
        self.condition_film = (eqx.nn.MLP(width, 2 * width, width_size=width,
                                          depth=1, activation=jax.nn.gelu, key=k_condition)
                               if conditioning == 'film' else None)
        self.time_film = eqx.nn.MLP(1 + 2 * time_features, 2 * width * decoder_depth,
                                   width_size=width, depth=1, activation=jax.nn.gelu,
                                   key=next(keys))
        # Frequencies are deterministic coordinate metadata, not parameters.
        self.time_frequencies = tuple(time_scale * (i + 1) for i in range(time_features))
        self.layers = tuple(eqx.nn.Linear(width, width, key=next(keys))
                            for _ in range(decoder_depth))
        self.output = eqx.nn.Linear(width, out_dim, key=next(keys))
        self.image_shape, self.patch_shape = tuple(image_shape), tuple(patch_shape)
        self.extent, self.width, self.local_dim = tuple(extent), width, local_dim

    time_frequencies: tuple = eqx.field(static=True)

    def encode(self, condition):
        if condition.shape != self.image_shape:
            raise ValueError(f'Condition shape {condition.shape}, expected {self.image_shape}')
        if len(self.extent) == 2:
            # Preserve the historical 2D patch order and positional arithmetic.
            channels, height, width = self.image_shape
            ph, pw = self.patch_shape
            ny, nx = height // ph, width // pw
            patches = condition.reshape(channels, ny, ph, nx, pw)
            patches = patches.transpose(1, 3, 0, 2, 4).reshape(ny * nx, -1)
            yy, xx = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing='ij')
            omega = 10000.0 ** (-jnp.arange(self.width // 4) / (self.width // 4))
            px, py = xx.reshape(-1, 1) * omega, yy.reshape(-1, 1) * omega
            pos = jnp.concatenate([jnp.sin(px), jnp.cos(px), jnp.sin(py), jnp.cos(py)], -1)
        else:
            # Volume order C,Z,Y,X; query coordinates retain the order X,Y,Z.
            channels, depth, height, width = self.image_shape
            pd, ph, pw = self.patch_shape
            nz, ny, nx = depth//pd, height//ph, width//pw
            patches = condition.reshape(channels, nz, pd, ny, ph, nx, pw)
            patches = patches.transpose(1, 3, 5, 0, 2, 4, 6).reshape(nz*ny*nx, -1)
            zz, yy, xx = jnp.meshgrid(jnp.arange(nz), jnp.arange(ny), jnp.arange(nx), indexing='ij')
            count = self.width//6
            omega = 10000.0 ** (-jnp.arange(count)/count)
            phases = [axis.reshape(-1, 1)*omega for axis in (xx, yy, zz)]
            pos = jnp.concatenate([value for phase in phases for value in (jnp.sin(phase), jnp.cos(phase))], -1)
        hidden = jax.vmap(self.patch_projection)(patches) + pos.astype(condition.dtype)
        for block in self.encoder:
            hidden = block(hidden, hidden)
        return jax.vmap(self.encoder_norm)(hidden)

    def prepare_encoded(self, encoded, x, local):
        if x.ndim != 2 or x.shape[1] != len(self.extent) or local.shape != (len(x), self.local_dim):
            raise ValueError(f'Queries must be (Nx,{len(self.extent)}) with aligned (Nx,local_dim) input features')
        xn = x / jnp.asarray(self.extent, dtype=x.dtype)
        proj = jnp.pi * (xn @ jax.lax.stop_gradient(self.spatial_frequencies))
        features = jnp.concatenate([xn, jnp.sin(proj), jnp.cos(proj), local], -1)
        hidden = jax.vmap(self.spatial_projection)(features)
        if self.cross_attention is not None:
            return self.cross_attention(hidden, encoded)
        gamma, beta = self.condition_film(encoded.mean(0)).reshape(2, self.width)
        return (1.0 + gamma) * hidden + beta

    def prepare(self, condition, x, local):
        return self.prepare_encoded(self.encode(condition), x, local)

    def _temporal(self, t):
        phase = jnp.pi * t * jnp.asarray(self.time_frequencies, dtype=t.dtype)
        embedding = jnp.concatenate([jnp.atleast_1d(t), jnp.sin(phase), jnp.cos(phase)])
        return self.time_film(embedding).reshape(len(self.layers), 2, self.width)

    def _point(self, hidden, film):
        for layer, modulation in zip(self.layers, film):
            hidden = jax.nn.gelu((1.0 + modulation[0]) * layer(hidden) + modulation[1])
        return self.output(hidden)

    def decode(self, prepared, times):
        if times.ndim != 1:
            raise ValueError('Times must have shape (Nt,)')
        films = jax.vmap(self._temporal)(times)
        return jax.vmap(lambda film: jax.vmap(self._point, in_axes=(0, None))(
            prepared, film))(films)

    def paired(self, condition, x, times, local):
        """Aligned arbitrary (x_i,t_i), for checks or jointly sampled queries."""
        if times.shape != (len(x),):
            raise ValueError('Paired time/query lengths must agree')
        hidden = self.prepare(condition, x, local)
        return jax.vmap(self._point)(hidden, jax.vmap(self._temporal)(times))

    def __call__(self, condition, x, times, local):
        return self.decode(self.prepare(condition, x, local), times)
