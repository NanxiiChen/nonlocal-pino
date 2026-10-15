"""Same checkpoint/H comparison with exact sampled NN supports."""
from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.linear_damage_pending_continue import get_config as base


def get_config():
    c = base()
    c.core = ConfigDict()
    c.core.source_dir = '/root/autodl-tmp/tf-logs/fracture2d/core_halo_source_20261012_outer0192'
    c.core.source_audit = '/root/autodl-tmp/tf-logs/fracture2d/physical_fields_status_20261012_outer0192/error_report.json'
    c.core.fields_sha256 = '4bec2f9bb6ae269361197481107ea44cc2a303a7807698febd61fb149ae6bf0f'
    c.core.network_sha256 = '48b0801a029573be8f285bb88d702d103ecb5f672f19e3e767e43b1d8cb7863f'
    c.core.source_outer = 192
    c.core.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/core_halo_damage_trial_20261012'
    c.core.cases = ['full_original', 'full_cached', 'sample_uniform', 'sample_crack']
    c.core.load = .0053
    c.core.count = 2048
    c.core.block_iterations = {name:256 for name in c.core.cases}
    c.core.sampling_profile = 'damage_gaussian'
    c.core.halo_capacity = 32768
    c.core.density_power = 2.
    c.core.total_iterations = 2048
    c.core.seed = 42
    c.core.penalty = 1000.
    c.core.power = 16
    c.core.benchmark_evaluations = 20
    c.core.boost = 9.
    c.core.damage_power = 2.
    c.core.segment_start = [.5, .5]
    c.core.segment_end = [1., .5]
    c.core.band_width = .024
    c.core.probability_mixture = .8
    return c
