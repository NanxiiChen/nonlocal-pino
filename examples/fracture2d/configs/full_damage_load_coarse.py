"""Cold all-node L1/Adam loading; paired control varies only early load step."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.l1_damage_loading_from_zero import get_config as base


def get_config():
    c = base()
    c.stagger.omega = 1.
    c.full_loading = ConfigDict(dict(
        suite_root='/root/autodl-tmp/tf-logs/fracture2d/full_damage_load_steps_20261014',
        case='coarse', pre_step=5e-4, transition_load=.0050,
        post_step=1e-4, end_load=.0053,
        inner_updates=128, prefix_outer_cap=128, critical_outer_cap=644,
        critical_load=.0053, stop_on_field_stability=True,
        damage_change_limit=1e-5, displacement_change_limit=1e-4,
        minimum_sweeps=3, consecutive_passes=3,
        checkpoint_sweeps=[1, 8, 32, 64, 128, 256, 384, 512, 644],
        sampling='all_nodes', symmetry_constraint=False,
        advance_on_budget_exit=True, budget_exit_is_certified=False,
        reference_root='/root/autodl-tmp/tf-logs/fracture2d/local_damage_load_refined_20261011_direct',
        oracle_loads=[.0050, .0052, .0053], cpu_threads=4))
    # Retain only settings used by the common cold-network/Adam implementation.
    keys = ('basis_path', 'basis_sha256', 'output_head', 'history_feature_scaling',
            'parameterization', 'trunk_gain', 'loss_norm', 'optimizer',
            'initial_physical_damage', 'initial_physical_history', 'learning_rate',
            'decay_steps', 'decay_rate', 'minimum_rate', 'grad_clip', 'b1', 'b2',
            'eps', 'eps_root', 'shampoo_beta', 'correct_bias',
            'precondition_frequency', 'max_precond_dim', 'precondition_1d', 'weight_decay')
    c.l1_loading = ConfigDict({k: c.l1_loading[k] for k in keys})
    return c
