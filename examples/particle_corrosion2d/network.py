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
    if 'zero_output' in a and a.zero_output:
        output = eqx.tree_at(lambda layer: (layer.weight, layer.bias), output,
                            (jnp.zeros_like(output.weight), jnp.zeros_like(output.bias)))
    net = eqx.tree_at(lambda model: model.output, net, output)
    if c.train.compute_dtype not in ('float32', 'float64'):
        raise ValueError('Explicit compute_dtype must be float32 or float64')
    return jax.tree.map(lambda value: value.astype(c.train.compute_dtype)
                        if eqx.is_inexact_array(value) else value, net)


def encode(net, data):
    """Encode known functions and geometry only, with their separate measures."""
    return net.encode((data['domain_tokens'], data['boundary_tokens']),
                      (data['domain_weights'], data['boundary_weights']))


def mouth_mask(points, c):
    """Prescribed mouth coordinates, with endpoint comparisons in compute dtype."""
    # Cast the actual endpoint coordinates, not a rounded x-minus-center:
    # subtraction can otherwise misclassify a boundary endpoint in FP32.
    left = jnp.asarray(c.geometry.pit_x-c.geometry.mouth_half_width, dtype=points.dtype)
    right = jnp.asarray(c.geometry.pit_x+c.geometry.mouth_half_width, dtype=points.dtype)
    return ((jnp.abs(points[..., 1]) < 1e-12) &
            (points[..., 0] >= left) & (points[..., 0] <= right))


def initial_latent(points, c):
    radius = jnp.linalg.norm(points-jnp.asarray((c.geometry.pit_x, 0.), dtype=points.dtype), axis=-1)
    return c.geometry.interface_k*(radius-c.geometry.pit_radius)


def initial_phase(points, c):
    """JAX version of the prescribed pit, including exact mouth-node values."""
    phi = .5*(1+jnp.tanh(initial_latent(points, c)))
    return jnp.where(mouth_mask(points, c), 0., phi)


def lifting_gate(times, c):
    """Bounded exponential initial lifting; scale is in dimensionless physical time."""
    if c.phase.initial_lifting.kind == 'latent_additive':
        return jnp.log1p(times/c.phase.initial_lifting.time_scale)
    return -jnp.expm1(-times/c.phase.initial_lifting.time_scale)


def decode(net, prepared, times, c, points=None):
    """Return raw (time,point,2) channels; all spatial work is shared in time."""
    if times.ndim != 1 or prepared.ndim != 2 or prepared.shape[1] != net.width:
        raise ValueError('Decoder requires times(Nt) and prepared(Nx,width)')
    tau = times / c.sample.t_max
    if 'time_coordinate' in c.arch:
        scale = c.arch.time_coordinate.time_scale
        tau = jnp.log1p(times/scale)/jnp.log1p(c.sample.t_max/scale)
    frequencies = _time_frequencies(net.time_features, net.time_max_frequency, times.dtype)
    angles = jnp.pi * tau[:, None] * frequencies
    temporal = jnp.concatenate((tau[:, None], jnp.sin(angles), jnp.cos(angles)), axis=-1)
    films = jax.vmap(net.time)(temporal).reshape(len(times), len(net.layers), 2, net.width)
    hidden = jnp.broadcast_to(prepared, (len(times), *prepared.shape))
    for i, layer in enumerate(net.layers):
        hidden = jax.vmap(jax.vmap(layer))(hidden)
        hidden = jax.nn.gelu(hidden * (1+films[:, i, 0, None, :]) + films[:, i, 1, None, :])
    raw = net.output_scale * jax.vmap(jax.vmap(net.output))(hidden)
    if 'phase' in c and 'initial_lifting' in c.phase:
        if points is None or points.shape != (len(prepared), 2):
            raise ValueError('Initial lifting requires all prepared query coordinates')
        phi0 = initial_phase(points, c)
        liquid0 = phi0**2*(3-2*phi0)
        gate = lifting_gate(times, c)[:, None]
        if c.phase.initial_lifting.kind == 'latent_additive':
            raw = raw.at[..., 0].set(initial_latent(points, c)[None, :]+gate*raw[..., 0])
        raw = raw.at[..., 1].set(liquid0[None, :]+gate*raw[..., 1])
    if 'boundary' in c:
        if points is None:
            raise ValueError('Exact mouth data requires query coordinates')
        raw = raw.at[..., 1].set(jnp.where(mouth_mask(points, c)[None, :], 0., raw[..., 1]))
    return raw


def physical_fields(raw, c, points=None, times=None):
    """Bounded metal phase and unrestricted KKS liquid concentration."""
    if raw.shape[-1] != 2:
        raise ValueError('Physical map requires raw phase and liquid channels')
    phi = .5 * (1+jnp.tanh(raw[..., 0]))
    if 'phase' in c and 'initial_lifting' in c.phase:
        if points is None or times is None or raw.shape != (len(times), len(points), 2):
            raise ValueError('Initial lifting requires raw(Nt,Nx,2), points and times')
        if c.phase.initial_lifting.kind == 'bounded_exponential':
            gate = lifting_gate(times, c)[:, None]
            phi = (1-gate)*initial_phase(points, c)[None, :]+gate*phi
    if 'boundary' in c:
        if points is None:
            raise ValueError('Exact mouth data requires query coordinates')
        phi = jnp.where(mouth_mask(points, c)[None, :], 0., phi)
    h = phi**2 * (3-2*phi)
    concentration = c.physical.c_le*raw[..., 1] + (c.physical.c_se-c.physical.c_le)*h
    return jnp.stack((phi, concentration), axis=-1)


def time_jets(net, prepared, times, c, points=None):
    """JVP in supplied dimensionless time, including the output-map chain rule."""
    def evaluate(ts):
        raw = decode(net, prepared, ts, c, points)
        return raw, physical_fields(raw, c, points, ts)
    (raw, fields), (_, fields_t) = jax.jvp(evaluate, (times,), (jnp.ones_like(times),))
    return raw, fields, fields_t


def predict_fields(net, data, points, times, c):
    prepared = net.prepare_encoded(encode(net, data), points)
    return physical_fields(decode(net, prepared, times, c, points), c, points, times)
