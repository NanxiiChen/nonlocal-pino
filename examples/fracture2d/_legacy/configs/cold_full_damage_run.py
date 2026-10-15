"""Fresh physical trajectory: fine prefix, audited critical-load continuation."""
from ml_collections import ConfigDict


def get_config():
    return ConfigDict(dict(
        suite_root='/root/autodl-tmp/tf-logs/fracture2d/cold_full_pinn_loading_0060_20261014',
        prefix_config='examples/fracture2d/_legacy/configs/cold_full_damage_prefix.py',
        continuation_config='examples/fracture2d/_legacy/configs/cold_full_damage_tail.py',
        monitor_config='examples/fracture2d/_legacy/configs/cold_full_damage_fields.py',
        prefix_case='coarse', prefix_index=52, prefix_load=.0052,
        load_step=1e-4, transition_load=.0050, end_load=.0060,
        inner_updates=128, max_outer_sweeps=1024,
        damage_change_limit=1e-5, displacement_change_limit=1e-4,
        minimum_sweeps=3, consecutive_passes=3, learning_rate=1e-5,
        checkpoints=[1,8,32,64,128,256,512,644,768,808,896,1024],
        oracle_loads=[.0053,.0055,.0058,.0060],
        sampling='all_nodes', loss_norm='l1', optimizer='adam', omega=1.,
        symmetry_constraint=False, advance_on_budget_exit=True,
        residual_is_stopping_gate=False, cpu_threads=4,
        event_dir='/root/autodl-tmp/tf-logs/fracture2d/events/cold_full_pinn_loading_0060_20261014'),
        allow_dotted_keys=True)
