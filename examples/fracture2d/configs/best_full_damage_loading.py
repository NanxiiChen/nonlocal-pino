"""Audited fine prefix, full-node L1/Adam and longer same-load coupling to .0065."""
from ml_collections import ConfigDict


def get_config():
    return ConfigDict(dict(
        output_dir='/root/autodl-tmp/tf-logs/fracture2d/best_full_pinn_loading_20261014',
        source_root='/root/autodl-tmp/tf-logs/fracture2d/full_damage_load_steps_20261014/fine',
        source_index=52, source_load=.0052, source_optimizer_count=384000,
        source_pins={
            'identity.json': '54a26eb7ec6894f0a29bda9871e43cc3210fd84a4bac0f03270967f23c7eef2d',
            'parameterization.npz': '12e5a9c11853fedc394a1ceef6bfb3b8e93fe04bdb2877307664e344282f89cd',
            'independent_review/audit.json': '222d783a57c78c594c491aa6405d88912cb9e2c4de4388b27326c81e010438b5',
            'state_0052/result.json': '2b387ac60aeae9ba664ef662322d3bf73f7e3d3a4aa73d81c80132b34c76aea4',
            'state_0052/fields.npz': '0667a95e17abdcec0a77ac8f37aa6c91a863552b10d339bce2370eac592421a7',
            'state_0052/network.eqx': '948a30dcfb976d7094f0f49f00b5f0cb1165bfa7b58214e0e8e618732e79c936',
            'state_0052/optimizer.eqx': '77d04461dee9147f9b5b9d4baea3a08e5862e764b6f226175c8416dc15dd844c'},
        load_step=1e-4, transition_load=.0050, end_load=.0065,
        inner_updates=128, max_outer_sweeps=4096,
        damage_change_limit=1e-5, displacement_change_limit=1e-4,
        minimum_sweeps=3, consecutive_passes=3, learning_rate=1e-5,
        checkpoints=[1, 8, 32, 64, 128, 256, 512, 644, 1024, 1536, 2048, 3072, 4096],
        oracle_loads=[.0053, .0055, .0058, .0060, .0065],
        sampling='all_nodes', loss_norm='l1', optimizer='adam', omega=1.,
        symmetry_constraint=False, advance_on_budget_exit=True,
        residual_is_stopping_gate=False, cpu_threads=4), allow_dotted_keys=True)
