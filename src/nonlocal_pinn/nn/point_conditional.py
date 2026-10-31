"""Point-measure conditioning with shared spatial preparation and time decoding."""
import math
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from nonlocal_pinn.nn.conditional import AttentionBlock


def _validate_time_bandwidth(count, maximum):
    if maximum is not None and (isinstance(maximum, bool) or not math.isfinite(maximum)
            or maximum < 1 or count < 2):
        raise ValueError('Explicit time_max_frequency requires >=2 features and finite maximum>=1')


def _time_frequencies(count, maximum, dtype):
    # Missing optional metadata retains legacy arithmetic and checkpoint leaves.
    if maximum is None:
        return jnp.arange(1, count+1, dtype=dtype)
    return jnp.linspace(1., maximum, count, dtype=dtype)


def measure_attention(block, query, context, weights):
    """The existing attention block with a positive context integration measure."""
    q=jax.vmap(block.norm_q)(query)
    kv=jax.vmap(block.norm_kv)(context)
    a=block.attention
    def project(layer,x):
        return jax.vmap(layer)(x).reshape(len(x),a.num_heads,-1)
    q,k,v=project(a.query_proj,q),project(a.key_proj,kv),project(a.value_proj,kv)
    logits=jnp.einsum('mhd,nhd->hmn',q,k)/math.sqrt(q.shape[-1])
    scores=jax.nn.softmax(logits+jnp.log(weights)[None,None,:],axis=-1)
    mixed=jnp.einsum('hmn,nhd->mhd',scores,v).reshape(len(query),-1)
    hidden=query+jax.vmap(a.output_proj)(mixed)
    return hidden+jax.vmap(block.mlp)(jax.vmap(block.norm_out)(hidden))


class PointFieldOperator(eqx.Module):
    lift: object
    latent: object
    key_projection: object
    value_projection: object
    encoder: tuple
    spatial: object
    cross: object
    time: object
    layers: tuple
    output: object
    frequencies: object
    extent: tuple = eqx.field(static=True)
    heads: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    time_features: int = eqx.field(static=True)
    time_max_frequency: float | None = eqx.field(static=True)
    output_scale: float = eqx.field(static=True)

    def __init__(self, *, input_dim, extent, width, heads, latent_tokens,
                 encoder_depth, decoder_depth, spatial_features, spatial_scale,
                 time_features, output_scale, norm_eps, key, time_max_frequency=None):
        integers = (input_dim, width, heads, encoder_depth,
                    decoder_depth, spatial_features, time_features)
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in integers):
            raise ValueError('Point operator dimensions must be positive integers')
        if isinstance(latent_tokens,bool) or not isinstance(latent_tokens,int) or latent_tokens<0:
            raise ValueError('latent_tokens must be nonnegative; zero explicitly retains all input tokens')
        if width % heads or len(extent) != 2:
            raise ValueError('Require two spatial dimensions and width divisible by heads')
        if not all(math.isfinite(v) and v > 0 for v in (*extent, spatial_scale, output_scale, norm_eps)):
            raise ValueError('Point operator scales must be positive finite')
        _validate_time_bandwidth(time_features, time_max_frequency)
        self.time_max_frequency = time_max_frequency
        keys = iter(jr.split(key, 10+encoder_depth+decoder_depth))
        self.lift = eqx.nn.MLP(input_dim, width, width, 1, activation=jax.nn.gelu, key=next(keys))
        kl,kk,kv=next(keys),next(keys),next(keys)
        self.latent = .02*jr.normal(kl, (latent_tokens, width)) if latent_tokens else None
        self.key_projection = eqx.nn.Linear(width, width, key=kk) if latent_tokens else None
        self.value_projection = eqx.nn.Linear(width, width, key=kv) if latent_tokens else None
        self.encoder = tuple(AttentionBlock(width, heads, 2, norm_eps, key=next(keys))
                             for _ in range(encoder_depth))
        self.frequencies = jr.normal(next(keys), (2, spatial_features))*spatial_scale
        self.spatial = eqx.nn.Linear(2+2*spatial_features, width, key=next(keys))
        self.cross = AttentionBlock(width, heads, 2, norm_eps, key=next(keys))
        self.time = eqx.nn.MLP(1+2*time_features, 2*width*decoder_depth, width, 1,
                              activation=jax.nn.gelu, key=next(keys))
        self.layers = tuple(eqx.nn.Linear(width, width, key=next(keys)) for _ in range(decoder_depth))
        self.output = eqx.nn.Linear(width, 1, key=next(keys))
        self.extent, self.heads, self.width = tuple(extent), heads, width
        self.time_features, self.output_scale = time_features, output_scale

    def encode(self, features, weights):
        if features.ndim != 2 or weights.shape != features.shape[:1]:
            raise ValueError('Point features and quadrature weights must align')
        h = jax.vmap(self.lift)(features)
        if self.latent is None:
            for block in self.encoder:
                h=measure_attention(block,h,h,weights)
            return h,weights
        keys = jax.vmap(self.key_projection)(h).reshape(-1, self.heads, self.width//self.heads)
        values = jax.vmap(self.value_projection)(h).reshape(keys.shape)
        query = self.latent.reshape(-1, self.heads, self.width//self.heads)
        scores = jnp.einsum('mhd,nhd->hmn', query, keys)/math.sqrt(self.width//self.heads)
        # Normalized integral attention; weights are positive boundary arc lengths.
        attention = jax.nn.softmax(scores+jnp.log(weights)[None, None, :], axis=-1)
        z = self.latent+jnp.einsum('hmn,nhd->mhd', attention, values).reshape(self.latent.shape)
        for block in self.encoder:
            z = block(z, z)
        return z

    def prepare_encoded(self, encoded, points):
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('Spatial queries must have shape(N,2)')
        x = points/jnp.asarray(self.extent, dtype=points.dtype)
        angles = jnp.pi*x@jax.lax.stop_gradient(self.frequencies)
        h = jax.vmap(self.spatial)(jnp.concatenate((x, jnp.sin(angles), jnp.cos(angles)), axis=-1))
        if self.latent is None:
            context,weights=encoded
            return measure_attention(self.cross,h,context,weights)
        return self.cross(h, encoded)

    def decode(self, prepared, times):
        frequencies = _time_frequencies(self.time_features, self.time_max_frequency, times.dtype)
        angles = jnp.pi*times[:, None]*frequencies
        temporal = jnp.concatenate((times[:, None], jnp.sin(angles), jnp.cos(angles)), axis=-1)
        modulation = jax.vmap(self.time)(temporal).reshape(len(times), len(self.layers), 2, self.width)
        h = jnp.broadcast_to(prepared, (len(times), *prepared.shape))
        for i, layer in enumerate(self.layers):
            h = jax.vmap(jax.vmap(layer))(h)
            h = jax.nn.gelu(h*(1+modulation[:, i, 0, None, :])+modulation[:, i, 1, None, :])
        return self.output_scale*jax.vmap(jax.vmap(self.output))(h)[..., 0]


def normalized_integral_attention(query, keys, values, weights):
    """GNOT eqs3-6, with quadrature instead of an unweighted point sum.

    Inputs have shapes (Nq,H,D), (Nk,H,D), (Nk,H,Dv), (Nk,).
    Each query/key is softmax-normalized over D. Associative contraction avoids
    an Nq-by-Nk array. Zero weights mask padding; each stream must have positive
    total measure. Normalizing weights first changes neither the ratio nor units.
    https://proceedings.mlr.press/v202/hao23c/hao23c.pdf
    """
    if (query.ndim != 3 or keys.ndim != 3 or values.ndim != 3
            or query.shape[1:] != keys.shape[1:] or keys.shape[:2] != values.shape[:2]
            or weights.shape != keys.shape[:1]):
        raise ValueError('Integral attention requires aligned query/key/value/head/measure shapes')
    query=jax.nn.softmax(query,axis=-1)
    keys=jax.nn.softmax(keys,axis=-1)
    measure=weights/jnp.sum(weights)
    moment=jnp.einsum('n,nhd,nhe->hde',measure,keys,values)
    normalization=jnp.einsum('mhd,hd->mh',query,jnp.einsum('n,nhd->hd',measure,keys))
    return jnp.einsum('mhd,hde->mhe',query,moment)/normalization[...,None]


class HeterogeneousAttention(eqx.Module):
    """Separate stream projections and normalization, then an equal stream mean."""
    norm_query: object
    norm_context: tuple
    query: object
    keys: tuple
    values: tuple
    output: object
    norm_output: object
    mlp: object
    heads: int = eqx.field(static=True)

    def __init__(self,width,heads,streams,norm_eps,*,key):
        keys=iter(jr.split(key,3+2*streams))
        self.norm_query=eqx.nn.LayerNorm(width,eps=norm_eps)
        self.norm_context=tuple(eqx.nn.LayerNorm(width,eps=norm_eps) for _ in range(streams))
        self.query=eqx.nn.Linear(width,width,key=next(keys))
        self.keys=tuple(eqx.nn.Linear(width,width,key=next(keys)) for _ in range(streams))
        self.values=tuple(eqx.nn.Linear(width,width,key=next(keys)) for _ in range(streams))
        self.output=eqx.nn.Linear(width,width,key=next(keys))
        self.norm_output=eqx.nn.LayerNorm(width,eps=norm_eps)
        self.mlp=eqx.nn.MLP(width,width,2*width,1,activation=jax.nn.gelu,key=next(keys))
        self.heads=heads

    def __call__(self,query,contexts,measures):
        if len(contexts)!=len(self.keys) or len(measures)!=len(contexts):
            raise ValueError('Every heterogeneous stream requires its own context and measure')
        q=jax.vmap(self.query)(jax.vmap(self.norm_query)(query)).reshape(len(query),self.heads,-1)
        messages=[]
        for context,measure,norm,k,v in zip(contexts,measures,self.norm_context,self.keys,self.values,strict=True):
            context=jax.vmap(norm)(context)
            key=jax.vmap(k)(context).reshape(len(context),self.heads,-1)
            value=jax.vmap(v)(context).reshape(key.shape)
            messages.append(normalized_integral_attention(q,key,value,measure).reshape(query.shape))
        mixed=jnp.mean(jnp.stack(messages),axis=0)
        hidden=query+jax.vmap(self.output)(mixed)
        return hidden+jax.vmap(self.mlp)(jax.vmap(self.norm_output)(hidden))


class HeterogeneousPointOperator(eqx.Module):
    """General boundary/function point streams and independent coordinate queries.

    GNOT-inspired heterogeneous normalized attention; not its full architecture.
    No query self-attention or geometric MoE, allowing exact query chunking.
    Repeated spatial condition reads are shared across time; decode uses the
    existing temporal FiLM. Stream features start with their spatial coordinates.
    No pair distances, fixed neighborhoods or particular hole identities enter.
    """
    lifts: tuple
    spatial: object
    cross: tuple
    time: object
    layers: tuple
    output: object
    frequencies: object
    input_dims: tuple = eqx.field(static=True)
    extent: tuple = eqx.field(static=True)
    width: int = eqx.field(static=True)
    time_features: int = eqx.field(static=True)
    time_max_frequency: float | None = eqx.field(static=True)
    output_scale: float = eqx.field(static=True)

    def __init__(self,*,input_dims,extent,width,heads,condition_depth,decoder_depth,
                 spatial_features,spatial_scale,time_features,output_scale,norm_eps,key,time_max_frequency=None):
        dimensions=(*input_dims,width,heads,condition_depth,decoder_depth,spatial_features,time_features)
        if (not input_dims or any(isinstance(n,bool) or not isinstance(n,int) or n<1 for n in dimensions)
                or not extent or any(n<len(extent) for n in input_dims) or width%heads):
            raise ValueError('Positive point dimensions/depths, leading coordinates and divisible heads required')
        if not all(math.isfinite(v) and v>0 for v in (*extent,spatial_scale,output_scale,norm_eps)):
            raise ValueError('Point coordinate/attention/decoder scales must be positive finite')
        _validate_time_bandwidth(time_features, time_max_frequency)
        self.time_max_frequency = time_max_frequency
        keys=iter(jr.split(key,len(input_dims)+condition_depth+decoder_depth+4))
        self.frequencies=jr.normal(next(keys),(len(extent),spatial_features))*spatial_scale
        self.lifts=tuple(eqx.nn.MLP(n+2*spatial_features,width,width,1,
            activation=jax.nn.gelu,key=next(keys)) for n in input_dims)
        self.spatial=eqx.nn.Linear(len(extent)+2*spatial_features,width,key=next(keys))
        self.cross=tuple(HeterogeneousAttention(width,heads,len(input_dims),norm_eps,key=next(keys))
                         for _ in range(condition_depth))
        self.time=eqx.nn.MLP(1+2*time_features,2*width*decoder_depth,width,1,
                            activation=jax.nn.gelu,key=next(keys))
        self.layers=tuple(eqx.nn.Linear(width,width,key=next(keys)) for _ in range(decoder_depth))
        self.output=eqx.nn.Linear(width,1,key=next(keys))
        self.input_dims,self.extent=tuple(input_dims),tuple(extent)
        self.width,self.time_features,self.output_scale=width,time_features,output_scale

    def position(self,points):
        x=points/jnp.asarray(self.extent,dtype=points.dtype)
        angles=jnp.pi*x@jax.lax.stop_gradient(self.frequencies)
        return jnp.concatenate((x,jnp.sin(angles),jnp.cos(angles)),axis=-1)

    def encode(self,features,weights):
        if len(features)!=len(self.lifts) or len(weights)!=len(features):
            raise ValueError('Explicit point feature/measure tuple required for every stream')
        encoded=[]
        for f,w,lift,n in zip(features,weights,self.lifts,self.input_dims,strict=True):
            if f.ndim!=2 or f.shape[1]!=n or w.shape!=f.shape[:1]:
                raise ValueError('Point stream features and quadrature measures must align')
            w=eqx.error_if(w,jnp.any(~jnp.isfinite(w))|jnp.any(w<0)|(jnp.sum(w)<=0),
                           'Point stream measure must be finite, nonnegative, and have positive total')
            # Mask padded rows BEFORE learned operations, avoiding arbitrary dummy
            # coordinates influencing moments or their parameter gradients.
            f=jnp.where(w[:,None]>0,f,0.)
            position=self.position(f[:,:len(self.extent)])
            encoded.append(jax.vmap(lift)(jnp.concatenate((position,f[:,len(self.extent):]),axis=-1)))
        return tuple(encoded),tuple(weights)

    def prepare_encoded(self,encoded,points):
        if points.ndim!=2 or points.shape[1]!=len(self.extent):
            raise ValueError('Spatial query dimension must match the declared coordinate extent')
        contexts,weights=encoded
        h=jax.vmap(self.spatial)(self.position(points))
        for block in self.cross:
            h=block(h,contexts,weights)
        return h

    def decode(self,prepared,times):
        return PointFieldOperator.decode(self,prepared,times)
