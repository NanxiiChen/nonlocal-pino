"""Same .0052 parent,644 unrelaxed outers,128 Adam updates per outer."""
from examples.fracture2d._legacy.configs.staggered_damage_644 import get_config as baseline_config


def get_config():
    c = baseline_config()
    c.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/staggered_damage_644_inner128_20261013'
    c.start_sweep = 0
    c.inner_updates = 128
    c.restart_root = None
    c.restart_pins = {}
    c.checkpoints = [1, 8, 32, 64, 128, 129, 136, 192, 256, 384, 512, 644]
    return c
