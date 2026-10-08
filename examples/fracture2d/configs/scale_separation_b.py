"""Configuration B: unchanged fine mechanics, shorter nonlocal damage radius."""

from examples.fracture2d.configs.scale_separation_fine import get_config as fine_config


def get_config():
    c = fine_config()
    c.nl.delta_d = 0.024
    c.output.dir = "/root/autodl-tmp/tf-logs/fracture2d/scale_sep_b_validation_20261007"
    return c
