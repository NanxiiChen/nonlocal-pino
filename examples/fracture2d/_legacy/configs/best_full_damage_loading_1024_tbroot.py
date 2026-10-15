"""Recover the authorized1024 run after the monitor was killed with the server."""
from examples.fracture2d._legacy.configs.best_full_damage_loading_1024 import get_config as base


def get_config():
    c=base()
    c.output_dir='/root/autodl-tmp/tf-logs/fracture2d/best_full_pinn_loading_1024_tbroot_20261014'
    return c
