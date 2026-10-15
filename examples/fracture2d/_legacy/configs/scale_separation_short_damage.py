"""User-accepted fine-cloud direct comparison model; retain warning provenance."""

from examples.fracture2d._legacy.configs.scale_separation_b import get_config as b_config


def get_config():
    c = b_config()
    c.nl.delta_d = 0.008
    # Keep the model-config bytes of the verified historical run reproducible.
    c.output.dir = "/root/autodl-tmp/tf-logs/fracture2d/b_short_damage_fresh_20261008"
    return c
