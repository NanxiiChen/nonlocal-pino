"""Stage1: known-own-H/local-geometry conditioned neural AT2 damage."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.damage_optimizer_trial import get_config as base


def get_config():
    c = base()
    c.arch.hidden_dim = 96
    c.arch.num_layers = 3
    c.arch.use_spatial_emb = False
    c.arch.tensor_split = False
    c.arch.multi_head = False
    c.conditioned = ConfigDict()
    c.conditioned.powers = [0, 1, 2, 4, 8, 16]
    c.conditioned.tip = [.5, .5]
    c.conditioned.loss_floor = 1e-15
    c.conditioned.methods = ['residual_mse', 'residual_l4']
    c.conditioned.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/conditioned_damage_fixedH_20261011'
    c.optimization.lbfgs.maxiter = 2048
    c.optimization.lbfgs.check_every = 32
    c.optimization.lbfgs.residual_limit = 5e-4
    return c
