"""One canonical TensorBoard event stream for the newly computed trajectory."""
from examples.fracture2d.configs.best_full_damage_tensorboard import get_config as base


def get_config():
    c = base()
    c.run_root = '/root/autodl-tmp/tf-logs/fracture2d/cold_full_pinn_loading_0060_20261014'
    c.prefix_solution_root = c.run_root + '/prefix/coarse'
    c.solution_root = c.run_root + '/fine'
    c.event_dir = '/root/autodl-tmp/tf-logs/fracture2d/events/cold_full_pinn_loading_0060_20261014'
    return c
