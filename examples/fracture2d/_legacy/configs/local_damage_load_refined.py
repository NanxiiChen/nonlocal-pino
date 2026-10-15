"""Own-history loading sensitivity; quarter increments around critical fracture."""
from examples.fracture2d._legacy.configs.local_damage_trial_from_zero import get_config as base


def get_config():
    c = base()
    c.data.load_step = 2.5e-5
    # .0001-.0052, then .005225-.0056, then .0057-.0075.
    # These indices define this trial's schedule, not imported reference fields.
    c.data.reference_indices = [4*i for i in range(1, 53)]+list(range(209, 225))+[4*i for i in range(57, 76)]
    c.data.end_state_index = len(c.data.reference_indices)
    c.trial.output_base = '/root/autodl-tmp/tf-logs/fracture2d/local_damage_load_refined_20261011'
    return c
