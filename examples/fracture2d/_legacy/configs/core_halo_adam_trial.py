"""Persistent Adam rate controls at exact physical192/currentH/.0053."""
from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.core_halo_damage_resample import get_config as base


def get_config():
    c=base()
    c.core.output_dir='/root/autodl-tmp/tf-logs/fracture2d/core_halo_adam_trial_20261012'
    c.core.adam=ConfigDict()
    c.core.adam.rates=[1e-3,1e-4,1e-5]
    c.core.adam.total_iterations=2048
    c.core.adam.resample_every=5
    c.core.adam.check_every=128
    c.core.adam.decay_steps=2048
    c.core.adam.decay_rate=.1
    c.core.adam.minimum_fraction=.1
    c.core.adam.b1=.9
    c.core.adam.b2=.999
    c.core.adam.eps=1e-8
    c.core.adam.eps_root=0.
    c.core.adam.grad_clip=1000.
    c.core.adam.polish_iterations=2048
    c.core.adam.full_lbfgs_baseline='/root/autodl-tmp/tf-logs/fracture2d/core_halo_damage_trial_20261012/full_cached'
    c.core.adam.full_lbfgs_audit='/root/autodl-tmp/tf-logs/fracture2d/core_halo_damage_trial_20261012/comparison_audit.json'
    c.core.adam.full_lbfgs_network_sha256='1348a060699763e0921113c014bbcf7219613f3d8365e663219f1bca35e27926'
    return c
