"""Certified52 restart,128 alternating updates/load, unchanged512 inner budget."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.core_halo_adam_trial import get_config as base


def get_config():
    c=base();c.core.restart=ConfigDict();a=c.core.restart
    a.user_authorized=True
    a.output_dir='/root/autodl-tmp/tf-logs/fracture2d/restart_damage_loading_128_20261012'
    a.source_fields='/root/autodl-tmp/tf-logs/fracture2d/linear_damage_loading_20261012/state_0052/fields.npz'
    a.source_fields_sha256='94ea359d4d84d074efd6b68fe5675b06fbf52831ab852f5dfce5ffa9e7654ffb'
    a.source_network='/root/autodl-tmp/tf-logs/fracture2d/linear_damage_loading_20261012/state_0052/network.eqx'
    a.source_network_sha256='d174aff62164b58bccdb8605088fd0c3374982be6fe84dca7e7094b5226bd065'
    a.source_report='/root/autodl-tmp/tf-logs/fracture2d/linear_damage_loading_20261012/state_0052/result.json'
    a.source_report_sha256='0ce5f8a20309a2cfd040d3ee6a5dc5623946b43bacca9d87e822395bfe9bfe81'
    a.source_parameterization='/root/autodl-tmp/tf-logs/fracture2d/linear_damage_loading_20261012/parameterization.npz'
    a.source_parameterization_sha256='777fe3ea578a8da719a7efd22e74685ac182bf0c82ca5bb74e012ce6763fe206'
    a.source_audit='/root/autodl-tmp/tf-logs/fracture2d/linear_damage_loading_20261012/audit.json'
    a.initial_index=52
    a.indices=list(range(53,76))
    a.outer_updates=128
    a.inner_updates=512
    a.checkpoint_sweeps=[8,32,64,128]
    a.range_penalties=[1e3,1e4,1e5,1e6]
    a.crack_threshold=.9
    a.tip=[.5,.5]
    a.tip_radius=.024
    a.right_edge=1.
    a.coordinate_tolerance=1e-12
    a.drift_envelope=1.08
    return c
