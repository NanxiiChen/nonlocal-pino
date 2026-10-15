"""Accepted52 conversion with the certified diagnostic's latent RMS scaling."""
from examples.fracture2d._legacy.configs.plain_sigmoid_parent import get_config as base


def get_config():
    c = base()
    c.continuation.no_reference.parameterization = 'latent_basis_rms'
    c.continuation.no_reference.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/plain_sigmoid_parent52_v2_20261012'
    return c
