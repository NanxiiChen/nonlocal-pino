"""Restart the fine prefix with1024 outer sweeps and TensorBoard monitoring."""
from examples.fracture2d._legacy.configs.best_full_damage_loading import get_config as base


def get_config():
    c=base()
    c.output_dir='/root/autodl-tmp/tf-logs/fracture2d/best_full_pinn_loading_1024_20261014'
    c.max_outer_sweeps=1024
    c.checkpoints=[1,8,32,64,128,256,512,644,1024]
    return c
