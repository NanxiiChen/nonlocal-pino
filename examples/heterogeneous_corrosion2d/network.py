"""Material-image conditioning with shared space and independent time queries."""
import jax.numpy as jnp

from nonlocal_pinn.nn.conditional import ConditionalFieldOperator


def build_network(c, key):
    """No initial field or local material feature is supplied to query tokens."""
    a = c.arch
    return ConditionalFieldOperator(
        image_shape=tuple(c.operator.image_shape), patch_shape=tuple(a.patch_shape),
        extent=tuple(a.extent), width=a.width, encoder_depth=a.encoder_depth,
        heads=a.heads, mlp_ratio=a.mlp_ratio, decoder_depth=a.decoder_depth,
        spatial_features=a.spatial_features, spatial_scale=a.spatial_scale,
        time_features=a.time_features, time_scale=a.time_scale, local_dim=0,
        out_dim=2, conditioning=a.conditioning, norm_eps=a.norm_eps, key=key)


def physical_fields(raw, c):
    """Bounded metal fraction and unrestricted KKS liquid concentration."""
    phi = .5 * (1 + jnp.tanh(raw[..., 0]))
    h = phi**2 * (3 - 2 * phi)
    conc = c.physical.c_le * raw[..., 1] + (c.physical.c_se - c.physical.c_le) * h
    return jnp.stack((phi, conc), axis=-1)


def prepare(net, image, points):
    return net.prepare(image, points, jnp.empty((len(points), 0), dtype=points.dtype))


def predict_fields(net, image, points, times, c):
    """Return physical fields with shape (time, point, phi/c)."""
    prepared = prepare(net, image, points)
    return physical_fields(net.decode(prepared, times / c.sample.t_max), c)
