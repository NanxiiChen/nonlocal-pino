"""Accepted own-history optimizer fork; peak objective first, physical gates fixed."""
from pathlib import Path
from ml_collections import ConfigDict
from examples.fracture2d.configs.compact_damage_loading_full import get_config as base
from examples.fracture2d.local_damage_trial import sha


def get_config():
    c = base()
    c.loading.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_loading_peak_first_20261011'
    c.loading.methods = ['residual_l16', 'residual_mse']
    c.loading.fork = ConfigDict()
    c.loading.fork.parent_root = '/root/autodl-tmp/tf-logs/fracture2d/compact_damage_loading_original_gates_20261011'
    c.loading.fork.parent_identity_sha256 = 'b0dfc56e4eb3f75e9b02b6c891d39250ea28225f75d6174250e60222ac276e24'
    c.loading.fork.prefix_length = 17
    c.loading.fork.runner_sha256 = sha(Path(__file__).resolve().parents[1]/'compact_damage_fork.py')
    return c
