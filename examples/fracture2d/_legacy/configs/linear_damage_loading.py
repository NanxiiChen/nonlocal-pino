"""Own-zero mechanics/identity-head damage loading from certified own-H NN."""
from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.range_damage_extend import get_config as base


def get_config():
    c = base()
    c.loading = ConfigDict()
    c.loading.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/linear_damage_loading_20261012'
    c.loading.source_dir = c.extension.output_dir
    c.loading.indices = list(range(1,76))
    c.loading.penalties = [1e3,1e4,1e5,1e6]
    c.loading.source = ConfigDict({
        'identity.json':'7b480865b20d805e6a64a98324a0214ac5f49cb34c930bc8dfcc2b365229296a',
        'audit.json':'969c4c5582460b1aaf096740ac6458337f2d021467c022d2cb3ae92dc938beec',
        'own_fields.npz':'85b5275fb2da746f7cb4e0222cc4fce23eceb6d0b2da28863dbacd8a6dd4bd58',
        'parameterization.npz':'777fe3ea578a8da719a7efd22e74685ac182bf0c82ca5bb74e012ce6763fe206',
        'AL_0002/network.eqx':'c4bf92ca319482fa6ce2d6ddf7b3b3d4670cc26aa2395aa8ea94643f4682eae9',
        'AL_0002/fields.npz':'fd4d20676357a307aaeb68fd8f29a46f384d134b8ed3e6883bf38396118b8ce5',
        'result.json':'c88ca8949d8b9ada54ecef999320e5eb3a206738e7400fe9e15110de3c31338f',
    },allow_dotted_keys=True)
    return c
