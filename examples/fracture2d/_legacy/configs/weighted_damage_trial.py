"""Explicit fixed-checkpoint pilot; source weights and physics are preserved."""
from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.linear_damage_pending_continue import get_config as base


def get_config():
    c = base()
    c.weighting = ConfigDict()
    c.weighting.snapshot_dir = '/root/autodl-tmp/tf-logs/fracture2d/damage_error_maps_20261012_outer0153'
    c.weighting.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/weighted_damage_trial_20261012'
    c.weighting.fields_sha256 = '2b0600697bdbd6a5070a9439b606ffe8c10ff40fc1a32a0f43723e2fd7991c85'
    c.weighting.network_sha256 = '426fe9a599e290ec254f0482a85d9bcd5ca2448d0a8aeb0f8861527aeefa63e0'
    c.weighting.modes = ['uniform', 'static', 'damage']
    c.weighting.boost = 9.
    c.weighting.damage_power = 2.
    c.weighting.segment_start = [.5, .5]
    c.weighting.segment_end = [1., .5]
    c.weighting.band_width = .024
    c.weighting.power = 16
    c.weighting.maxiter = 2048
    c.weighting.penalty = 1000.
    c.weighting.load = .0053
    c.weighting.core_count = 1024
    c.weighting.sample_seed = 42
    return c
