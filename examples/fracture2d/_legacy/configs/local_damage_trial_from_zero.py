"""Own-history 0-to-.0075 verification of the larger-budget direct trial."""
from examples.fracture2d._legacy.configs.local_damage_trial_2048 import get_config as base


def get_config():
    c = base()
    c.data.reference_indices = list(range(1, 76))
    c.data.end_state_index = 75
    # Small-load volume-weighted damage RHS requires a smaller absolute floor.
    # The relative tolerance and every outer/physical gate remain unchanged.
    c.validation.cg_atol = 1e-13
    c.trial.output_base = '/root/autodl-tmp/tf-logs/fracture2d/local_damage_from_zero_20261011'
    return c
