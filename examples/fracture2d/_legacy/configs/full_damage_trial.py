"""Same .0052 parent -> fixed .0053, all71307 nodes, empirical optimizer choice."""
from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.staggered_damage_trial import get_config as parent_config


def get_config():
    c = parent_config()
    c.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/full_damage_trial_20261013'
    c.cases = [dict(name='pinn_full', solver='pinn', omega=1., max_sweeps=644)]
    c.full_training = ConfigDict(dict(norm='l1', inner_updates=128,
        pilot_updates=128, selection='minimum_full_loss', sampling='all_nodes',
        stop_early=False, adam_memory='parent_persistent',
        lbfgs=dict(maxiter=128, maxfun=8192, memory=20, max_linesearch=50,
                   gtol=1e-12, ftol=0.)))
    c.checkpoints = [1, 8, 32, 64, 128, 192, 256, 384, 512, 644]
    return c
