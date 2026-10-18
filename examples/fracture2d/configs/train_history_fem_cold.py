"""Fresh FE-energy cold path with128outer blocks and budget-triggered rates."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.train_history_fem_full import get_config as full_config


def get_config():
    c = full_config()
    c.loading.prefix_outer_steps = 128
    c.loading.outer_steps = 128
    c.train.rate_control = ConfigDict(dict(mode='budget_finishing',
        rates=[1e-4, 3e-5, 1e-5], outers_per_rate=128))
    c.output.root = '/root/autodl-tmp/tf-logs/fracture2d/runs/pinn_history_fem_cold_budget128'
    c.output.event_dir = c.output.root + '/events'
    c.output.resume = False
    return c
