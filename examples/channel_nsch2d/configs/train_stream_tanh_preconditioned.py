"""Same scaled-tanh output, inverse-Jacobian phase-transport residual."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_stream_tanh import get_config as baseline


def get_config():
    c=baseline()
    c.residual_transform=ConfigDict(dict(kind='phase_inverse_map_jacobian',
        minimum_jacobian=1e-6))
    return c
