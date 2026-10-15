"""Same-own-H linear-head extra training and inequality AL control."""
from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.plain_damage_head_trial import get_config as base


def get_config():
    c = base()
    c.range = ConfigDict()
    c.range.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/range_damage_trial_20261012'
    c.range.source_dir = c.head.output_dir
    c.range.source = ConfigDict({
        'identity.json':'159008293c1d2d4de8001e29a020494ec7e3d92a626c16e4bd677d18a7460380',
        'audit.json':'a2d595882a1d6cc738e03968292c0e87327511e33d72e071c5dd6faf4c6bb12b',
        'own_fields.npz':'68fbf1fd5c304db4ebcd663677aedaed0c4aa6c2c5a3a48540342554d48258ac',
        'identity/network.eqx':'8cf9ad7ce88a89aed2ec710d00479d47958619895b583e1e1b5f8a180d9a8b1e',
        'identity/fields.npz':'cea05e307bd3fb3afc5ad9469cec58ab1ecc18c2f5591712b3e4152fd0d860c5',
        'identity/parameterization.npz':'f55ece6b61fa5bfd57995e4e7fed6a970c49a4359468921f1c02887c3b727ad3',
    },allow_dotted_keys=True)
    c.range.penalties = [1.,10.,100.,1000.]
    c.range.method = 'residual_l16'
    c.range.activity_thresholds = [1e-7,1e-6,.1,.5,.9]
    c.range.lbfgs = ConfigDict(c.head.lbfgs.to_dict())
    return c
