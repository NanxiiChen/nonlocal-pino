"""Cold shear FEM-energy trial with five globally sampled past histories."""
from examples.fracture2d_shear.configs.train_base import get_config as baseline_config


def get_config():
    c = baseline_config()
    c.history.replay_enabled = True
    c.history.replay_sampling = 'completed_without_replacement'
    c.history.replay_batch = 5
    c.history.replay_fraction = .1
    c.history.replay_every = 4
    c.history.current_only_steps = 32
    c.output.root = '/root/autodl-tmp/tf-logs/fracture2d_shear/pinn_history_conditioned_replay_random5'
    c.output.event_dir = c.output.root + '/events'
    c.output.resume = False
    return c
