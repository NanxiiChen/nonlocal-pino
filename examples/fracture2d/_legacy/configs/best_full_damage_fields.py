"""Field monitor using only the user's canonical autodl-tmp log root."""
from examples.fracture2d._legacy.configs.best_full_damage_tensorboard import get_config as base


def get_config():
    c=base()
    c.solution_root='/root/autodl-tmp/tf-logs/fracture2d/best_full_pinn_loading_1024_tbroot_20261014/fine'
    c.event_dir='/root/autodl-tmp/tf-logs/fracture2d/events/best_full_pinn_loading_1024_tbroot_20261014'
    return c
