"""Matched pilot with only prescribed IC; no embedded front-motion law."""

from examples.icemelting3d.configs.accuracy_pilot import get_config as base_config


def get_config():
    c = base_config()
    c.experiment.output_dir = "reports/icemelting-weak-prior-20261024"
    c.experiment.methods = ["ic_phase", "ic_balanced", "ic_curriculum"]
    return c
