"""User-authorized provisional load advancement; original diagnostics retained."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.core_halo_adam_trial import get_config as base


def get_config():
    c=base()
    c.core.provisional=ConfigDict()
    a=c.core.provisional
    a.user_authorized=True
    a.output_dir='/root/autodl-tmp/tf-logs/fracture2d/provisional_damage_loading_20261012'
    a.source_fields='/root/autodl-tmp/tf-logs/fracture2d/adam_polished_physical_diagnostic_20261012/relaxed_candidate_fields.npz'
    a.source_fields_sha256='89aecae1917eff89e878e340ffe19f9819f411f214b5484b3a92b5314843d62f'
    a.source_report='/root/autodl-tmp/tf-logs/fracture2d/adam_polished_physical_diagnostic_20261012/physical_report.json'
    a.source_network='/root/autodl-tmp/tf-logs/fracture2d/core_halo_adam_trial_20261012/sampled_polish/network.eqx'
    a.source_network_sha256='94f6d935d789739390102596bc988f2453f32b520e76d335aefbe3103b2e673e'
    a.source_network_fields='/root/autodl-tmp/tf-logs/fracture2d/core_halo_adam_trial_20261012/sampled_polish/fields.npz'
    a.source_network_fields_sha256='87c1d4c2754535da9f79d99be30bdb3efd1d75ef95373551c3c8652a7483db54'
    a.source_network_audit='/root/autodl-tmp/tf-logs/fracture2d/core_halo_adam_trial_20261012/comparison_audit.json'
    a.initial_index=53
    a.indices=list(range(54,76))
    a.outer_updates=8
    a.inner_updates=512
    a.range_penalties=[1e3,1e4,1e5,1e6]
    a.crack_threshold=.9
    a.tip=[.5,.5]
    a.tip_radius=.024
    a.right_edge=1.
    a.coordinate_tolerance=1e-12
    a.drift_envelope=1.08
    return c
