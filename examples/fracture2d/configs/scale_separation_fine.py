"""Refined scale separation, with six material spacings per AT2 length."""

from examples.fracture2d.configs.scale_separation import get_config as pilot_config


def get_config():
    c = pilot_config()
    c.data.cloud = "/root/autodl-tmp/data/fracture2d/scale_separation_20261001/cloud_h004_api.npz"
    c.data.cloud_sha256 = "2fa8da86dcb227947605b99df6fce2bb6906dbc8f794caea64d13c906bb21ae4"
    c.geometry.h_grip = 0.004
    c.nl.delta_u = 0.008
    c.validation.widths = [0.004, 0.008, 0.016, 0.024, 0.048]
    c.output.dir = "/root/autodl-tmp/tf-logs/fracture2d/scale_sep_h004_du008_20261001"
    return c
