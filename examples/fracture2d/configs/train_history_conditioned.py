"""Short true-zero conditioned-history pilot with one-time L1/L2 bootstrap."""
from examples.fracture2d.configs.train_history import get_config as history_config


def get_config():
    c = history_config()
    c.arch.type = 'conditioned_history_operator'
    c.history.normalized = True
    c.history.local_history = True
    c.history.bootstrap_l1_steps = 2048
    c.history.bootstrap_l2_steps = 2048
    c.history.replay_enabled = False
    c.history.replay_every = 8
    c.history.current_only_steps = 32
    c.loss.norm = 'l2'
    c.train.learning_rate = 1e-4
    c.train.minimum_rate = 1e-4
    c.loading.u_final = .0010
    c.loading.advance_on_budget = False
    c.output.root = '/root/autodl-tmp/tf-logs/fracture2d/runs/pinn_history_conditioned_pilot'
    c.output.event_dir = c.output.root + '/events'
    return c
