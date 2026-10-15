"""Common own-L1 .0052 parent and first .0053 frozen-H optimizer control."""
from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.full_damage_trial import get_config as parent_config


def get_config():
    c = parent_config()
    c.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/norm_optimizer_damage_20261014'
    c.norm_trial = ConfigDict(dict(power=16, floor=1e-15,
        adam_initialization='fresh_identical_zero_moments', updates=4096,
        checkpoints=[128,512,2048,4096], sampling='all_nodes',
        screen_source='/root/autodl-tmp/tf-logs/fracture2d/full_damage_trial_20261013',
        screen_pins=ConfigDict({
            'identity.json':'3212d63d819a5fce3e967a4498c89803e8ebcc3836e627abd92e8069000017a3',
            'screen/adam/fields.npz':'7bb069e805d75a888c495041ee7db8ced9daae508a556061fee221ad371cc9a8',
            'screen/adam/report.json':'08c5aaa2b56f39a5980d621f9eaeb59d4ea9f1a443e0773f532e79a7752fd9c3',
            'independent_review/audit.json':'dd275f7c5adad71bbe026b459712ba89df01e674d1bfb4217e966f44cb28157e'
        },allow_dotted_keys=True),
        cases=[dict(name='l1_adam',norm='l1',optimizer='adam'),
               dict(name='l16_adam',norm='l16',optimizer='adam'),
               dict(name='l16_lbfgs',norm='l16',optimizer='lbfgs')],
        lbfgs=dict(maxiter=4096,maxfun=65536,memory=20,max_linesearch=50,gtol=1e-12,ftol=0.),
        coupled_inner=128,coupled_outers=644,
        selection='L16_field_L2_and_max_residual_no_worse_than_L1_then_min_field_L2',
        oracle_evaluation_only=True,physical_state_promoted=False),allow_dotted_keys=True)
    return c
