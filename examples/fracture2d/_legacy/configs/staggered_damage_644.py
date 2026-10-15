"""Extend the audited unrelaxed512-inner branch to644 total outers."""
from examples.fracture2d._legacy.configs.staggered_damage_trial import get_config as parent_config
from ml_collections import ConfigDict


def get_config():
    c = parent_config()
    c.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/staggered_damage_644_inner512_20261013'
    c.cases = [dict(name='pinn_full', solver='pinn', omega=1., max_sweeps=644)]
    c.start_sweep = 128
    c.inner_updates = 512
    c.stop_early = False
    c.restart_root = '/root/autodl-tmp/tf-logs/fracture2d/staggered_damage_trial_20261013'
    c.restart_pins = ConfigDict({
        'identity.json': '950f3afb2883008131d2ec963fdfdf4fba68de4307a6d6ca1525e31835e76805',
        'independent_review/audit.json': 'b433b53aa798514e7cdd0b5ab9670c9a96aa40e56f11636ccb3fca450280f0b2',
        'pinn_full/sampling_blocks.jsonl': '582e488b2e67293c5ca2e3af38b97a0fa30073d8becf6c4361e134605d9d1214',
        'pinn_full/outer_0128/fields.npz': 'bcbe0b77e77b8ce7e6fae56d04402e6ca0fc13396f2617344cbded81a6486d55',
        'pinn_full/outer_0128/report.json': 'c0d0d83ad58fdd89ee326a01ae389ec324ecd4900be7afbe6fde447e8d987262',
        'pinn_full/outer_0128/network.eqx': '6b25ac0e4b8d435641be5422d1feb37583b489cd41f88ddfe7544213629ac0d3',
        'pinn_full/outer_0128/optimizer.eqx': 'f427bee414d0abbebf2512f1f05d3ad9ec65e8cca4fda7d72e163189ecf328b3'}, allow_dotted_keys=True)
    c.checkpoints = [129, 136, 192, 256, 384, 512, 644]
    return c
