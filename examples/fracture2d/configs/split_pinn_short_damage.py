"""Fresh FP64 PINN on the user-accepted direct engineering comparison model."""

from examples.fracture2d.configs.split_pinn_fine import get_config as fine_config


def get_config():
    c = fine_config()
    c.nl.delta_d = 0.008
    c.train.operator_mode = "analytic_padded_damage"
    c.data.reference_root = "/root/autodl-tmp/data/fracture2d/scale_separation_20261001/warning_short_damage_from52_20261008"
    c.data.reference_policy = "user_accepted_warning"
    c.data.reference_policy_sha256 = "59d16a53f4dc8390614f2ac54b856b14ab4e85e402f66a5644239deba947f1f9"
    # First verify fresh loading and checkpoint recovery before endpoint extension.
    c.data.end_state_index = 2
    c.output.dir = "/root/autodl-tmp/tf-logs/fracture2d/pinn_short_damage_first2_20261009"
    c.output.event_dir = "/root/autodl-tmp/tf-logs/fracture2d/events/pinn_short_damage_first2_20261009"
    c.output.fork_from = ""
    c.output.fork_after_index = 0
    return c
