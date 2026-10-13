"""Same-own-H continuation of the range optimizer after equation readiness."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.range_damage_trial import get_config as base


def get_config():
    c = base()
    c.extension = ConfigDict()
    c.extension.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/range_damage_extend_20261012'
    c.extension.source_dir = c.range.output_dir
    c.extension.penalties = [1e4,1e5,1e6]
    c.extension.source = ConfigDict({
        'identity.json':'03ef65846f38f031010fd1990b409f645fe40edf5a62e918ac2786b8e30bc481',
        'audit.json':'63f0ab524717c38a0f2a17c29bd78b2d3675ba0ff8e72e8cc8499a0a6edf6790',
        'own_fields.npz':'85b5275fb2da746f7cb4e0222cc4fce23eceb6d0b2da28863dbacd8a6dd4bd58',
        'parameterization.npz':'777fe3ea578a8da719a7efd22e74685ac182bf0c82ca5bb74e012ce6763fe206',
        'AL_0004/network.eqx':'644723ece5c57a20b1fb40cca6ddf4b8b6be9a1f036c53ebcdaf337965de955b',
        'AL_0004/fields.npz':'dcca4e95bd316f01c00bbbbe381a562bfd410a60c1029db3605b59cda8167e01',
        'result.json':'b945c1a755ad01dd6ff706b5e1bc98b8d992ab25530cd08917a0e671dfbb12e0',
    },allow_dotted_keys=True)
    return c
