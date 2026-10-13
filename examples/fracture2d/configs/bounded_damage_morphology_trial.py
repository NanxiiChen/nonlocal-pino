"""Fresh bounded heads; fixed own-H diagnostics and certified52 physical restart."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.restart_damage_loading_128 import get_config as base


def get_config():
    c = base()
    c.head = ConfigDict()
    a = c.head
    a.mode = 'history_comparison'
    a.warm_root = ''
    a.warm_pins = []
    a.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/bounded_damage_morphology_20261013'
    a.heads = ['sigmoid', 'exp_square', 'identity']
    a.coupled_heads = ['sigmoid', 'exp_square']
    a.initial_damage = .03
    a.fixed_indices = [52, 60]
    a.source52 = c.core.restart.source_fields
    a.source52_sha256 = c.core.restart.source_fields_sha256
    a.source60 = '/root/autodl-tmp/tf-logs/fracture2d/restart_damage_loading_128_20261012/state_0060/fields.npz'
    a.source60_sha256 = '7ea0cbbf1cae9aaea64719193f0022869dbc4978db5e4873a1e562cbb94167ca'
    a.indices = list(range(53, 61))
    a.fixed_iterations = 4096
    a.inner_iterations = 512
    a.outer_iterations = 128
    a.checkpoints = [8, 32, 64, 128]
    a.background_band_lengths = 3.
    a.background_threshold = .1
    a.user_provisional = True
    a.lbfgs = ConfigDict(dict(maxiter=4096, check_every=128, memory=20,
        max_linesearch=30, gtol=1e-12, ftol=0., residual_limit=.001,
        range_tolerance=1e-10))
    return c
