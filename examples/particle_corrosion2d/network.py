"""Two geometry point streams, coordinate-only queries and two physical fields."""
import equinox as eqx
import jax
import jax.numpy as jnp

from nonlocal_pinn.nn.point_conditional import HeterogeneousPointOperator, _time_frequencies


def build_network(c, key):
    """Reuse the generic operator without adding a particle-specific attention."""
    a = c.arch
    if tuple(a.input_dims) != (3, 5):
        raise ValueError('Particle inputs must be domain(x,y,phi0) and boundary(x,y,nx,ny,kind)')
    if tuple(a.extent) != (c.geometry.length, c.geometry.height):
        raise ValueError('Architecture coordinate extent must match the physical rectangle')
    net = HeterogeneousPointOperator(
        input_dims=tuple(a.input_dims), extent=tuple(a.extent), width=a.width,
        heads=a.heads, condition_depth=a.condition_depth, decoder_depth=a.decoder_depth,
        spatial_features=a.spatial_features, spatial_scale=a.spatial_scale,
        time_features=a.time_features, time_max_frequency=a.time_max_frequency,
        output_scale=a.output_scale, norm_eps=a.norm_eps, key=key)
    # The shared class retains its scalar convenience decoder. Replace its head
    # functionally, then use the explicit multichannel decoder below.
    output = eqx.nn.Linear(a.width, 2, key=jax.random.fold_in(key, 741))
    net = eqx.tree_at(lambda model: model.output, net, output)
    if c.train.compute_dtype not in ('float32', 'float64'):
        raise ValueError('Explicit compute_dtype must be float32 or float64')
    return jax.tree.map(lambda value: value.astype(c.train.compute_dtype)
                        if eqx.is_inexact_array(value) else value, net)


def encode(net, data):
    """Encode known functions and geometry only, with their separate measures."""
    return net.encode((data['domain_tokens'], data['boundary_tokens']),
                      (data['domain_weights'], data['boundary_weights']))


def decode(net, prepared, times, c):
    """Return raw (time,point,2) channels; all spatial work is shared in time."""
    if times.ndim != 1 or prepared.ndim != 2 or prepared.shape[1] != net.width:
        raise ValueError('Decoder requires times(Nt) and prepared(Nx,width)')
    tau = times / c.sample.t_max
    frequencies = _time_frequencies(net.time_features, net.time_max_frequency, times.dtype)
    angles = jnp.pi * tau[:, None] * frequencies
    temporal = jnp.concatenate((tau[:, None], jnp.sin(angles), jnp.cos(angles)), axis=-1)
    films = jax.vmap(net.time)(temporal).reshape(len(times), len(net.layers), 2, net.width)
    hidden = jnp.broadcast_to(prepared, (len(times), *prepared.shape))
    for i, layer in enumerate(net.layers):
        hidden = jax.vmap(jax.vmap(layer))(hidden)
        hidden = jax.nn.gelu(hidden * (1+films[:, i, 0, None, :]) + films[:, i, 1, None, :])
    return net.output_scale * jax.vmap(jax.vmap(net.output))(hidden)


def physical_fields(raw, c):
    """Bounded metal phase and unrestricted KKS liquid concentration."""
    if raw.shape[-1] != 2:
        raise ValueError('Physical map requires raw phase and liquid channels')
    phi = .5 * (1+jnp.tanh(raw[..., 0]))
    h = phi**2 * (3-2*phi)
    concentration = c.physical.c_le*raw[..., 1] + (c.physical.c_se-c.physical.c_le)*h
    return jnp.stack((phi, concentration), axis=-1)


def time_jets(net, prepared, times, c):
    """Physical-time JVP, including the complete output-map chain rule."""
    raw, raw_t = jax.jvp(lambda ts: decode(net, prepared, ts, c),
                         (times,), (jnp.ones_like(times),))
    fields, fields_t = jax.jvp(lambda value: physical_fields(value, c), (raw,), (raw_t,))
    return raw, fields, fields_t


def predict_fields(net, data, points, times, c):
    prepared = net.prepare_encoded(encode(net, data), points)
    return physical_fields(decode(net, prepared, times, c), c)
