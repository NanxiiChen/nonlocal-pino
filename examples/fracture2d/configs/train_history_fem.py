"""Optional single-geometry P1 energy-trained conditioned-history pilot."""
from examples.fracture2d.configs.train_history_conditioned import get_config as conditioned_config


def get_config():
    c = conditioned_config()
    c.damage.discretization = 'fem'
    c.loss.norm = 'energy'
    c.history.bootstrap_energy_steps = 1024
    # Energy has its own one-time bootstrap, with no residual-loss phases.
    del c.history.bootstrap_l1_steps
    del c.history.bootstrap_l2_steps
    c.history.replay_enabled = False
    c.output.root = '/root/autodl-tmp/tf-logs/fracture2d/runs/pinn_history_fem_pilot'
    c.output.event_dir = c.output.root + '/events'
    c.output.reference_root = ''  # Set only to a matched P1 direct run.
    return c
