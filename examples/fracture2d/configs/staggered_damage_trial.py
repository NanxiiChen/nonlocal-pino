"""Common-parent fixed .0053 control; no unready load advancement."""
from ml_collections import ConfigDict


def get_config():
    source = '/root/autodl-tmp/tf-logs/fracture2d/l1_damage_loading_from_zero_20261013'
    return ConfigDict(dict(
        output_dir='/root/autodl-tmp/tf-logs/fracture2d/staggered_damage_trial_20261013',
        source_root=source, parent_index=12, parent_load=.0052, load=.0053,
        parent_count=786432,
        pins={
            'identity.json': '3227112c615b9c78a2a00578609077920bbbf97b1ed966c4847b45c02c5be2cd',
            'parameterization.npz': '12e5a9c11853fedc394a1ceef6bfb3b8e93fe04bdb2877307664e344282f89cd',
            'sampling_probabilities.npy': 'cdd3c67f4199ce1f1fb64ef2863026ed50c085a19157deb2c790d0af0d3d9014',
            'independent_review/audit.json': '78a0defa591c278cee518d2b0d2451af3a8d2d7ee6688efc0bfcbb9c86a9b8a9',
            'state_0012/result.json': '5913400b50c02e179adcf5e311c60aef037c0dcb214f2f9cc25f7bfea0042220',
            'state_0012/fields.npz': 'bc865306a5bc1b8bffae1e45f8d097f65a6d99c396c425a70ec938f846b9cd36',
            'state_0012/network.eqx': 'aa674be3cb96c6e5d3de06080f432af3b929c965f5f77ab200d895dc1a498417',
            'state_0012/optimizer.eqx': 'e5d1acda8fda04458992115991245611c4ba0aa87fe3eb539c3a6fc36ad3cbbe'},
        cases=[dict(name='direct_half', solver='direct', omega=.5, max_sweeps=2048),
               dict(name='direct_full', solver='direct', omega=1., max_sweeps=2048),
               dict(name='pinn_full', solver='pinn', omega=1., max_sweeps=128),
               dict(name='pinn_half', solver='pinn', omega=.5, max_sweeps=128)],
        displacement_change_limit=1e-4, damage_change_limit=1e-5,
        minimum_sweeps=3, consecutive_passes=3,
        loose_displacement_limit=1e-3, loose_damage_limit=1e-3,
        checkpoints=[1, 8, 32, 64, 128, 256, 512, 1024, 2048],
        final_mechanics_synchronization=True, advance_on_budget_exit=False,
        cpu_threads=4), allow_dotted_keys=True)
