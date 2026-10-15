"""Explicit latent-column RMS fork of the failed inherited-scaling trial."""
from examples.fracture2d._legacy.configs.plain_sigmoid_trial import get_config as base


def get_config():
    c = base()
    c.continuation.no_reference.parameterization = 'latent_basis_rms'
    c.continuation.no_reference.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/plain_sigmoid_fixedH_v2_20261012'
    return c
