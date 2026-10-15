"""Uniform/proposed/capped weighted L1 from the same audited all-node644 state."""
from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.full_damage_trial import get_config as parent_config


def get_config():
    c = parent_config()
    c.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/weighted_damage_fixedH_20261013'
    c.weighting = ConfigDict(dict(
        source_root='/root/autodl-tmp/tf-logs/fracture2d/full_damage_trial_20261013',
        source_count=868864, load=.0053, updates=4096, checkpoints=[128, 512, 2048, 4096],
        weight_update='frozen_source_physical_damage', physical_state_promoted=False,
        oracle_evaluation_only=True, region_threshold=.1,
        cases=[dict(name='uniform', kind='uniform', epsilon=.001, cap=None),
               dict(name='reciprocal', kind='reciprocal', epsilon=.001, cap=None),
               dict(name='capped50', kind='reciprocal', epsilon=.001, cap=50.)],
        pins=ConfigDict({
            'identity.json':'3212d63d819a5fce3e967a4498c89803e8ebcc3836e627abd92e8069000017a3',
            'selection.json':'27cf3cfd459beadfa304e7311fddf82f3b994c98084ec61decd336b1f87d3dbb',
            'independent_review/audit.json':'dd275f7c5adad71bbe026b459712ba89df01e674d1bfb4217e966f44cb28157e',
            'pinn_full/final/fields.npz':'a99a6110d4ee29ee518062ea0063dc408910ce420a1ea94517d92df033069f61',
            'pinn_full/final/network.eqx':'5818d0b3bc3cea3b1ceb5a1db91d6393eba48796b8818745ce906016730b4b3d',
            'pinn_full/final/optimizer.eqx':'f38c967129eabc0924cd147b9ecb879ad5d1a009a3d8e271f64c6fd77ad96b5c',
            'pinn_full/final/report.json':'4bc9392c00a8333b4772fcddfc57c678bf426992d94ee8c325b0053fe041e88b'
        },allow_dotted_keys=True)),allow_dotted_keys=True)
    return c
