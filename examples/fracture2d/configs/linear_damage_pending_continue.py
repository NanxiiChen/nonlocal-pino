"""Same-physics recovery from certified interrupted physical126."""
from ml_collections import ConfigDict
from examples.fracture2d.configs.linear_damage_relaxed_continue import get_config as base


def get_config():
    c = base()
    c.loading.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/linear_damage_pending_continue_20261012'
    c.continuation = ConfigDict()
    c.continuation.source_dir = '/root/autodl-tmp/tf-logs/fracture2d/linear_damage_relaxed_continue_20261012'
    c.continuation.accepted_parent_index = 52
    c.continuation.completed_outer = 126
    c.continuation.negative_atol = .001
    c.continuation.positive_atol = 2e-6
    c.continuation.source = ConfigDict({
        'identity.json': '1ac6d4735da6584ae0494a41024bf74d552a400b54e2e7461ebeb6007c508133',
        'initial_restore_audit.json': '028ab9df67b7fe0a9d9e5ee9f681522f3a3cda7b91c718bfa01eccac1b40163c',
        'recovery_audit.json': '3cc1396ee7f20d27c1f5154071d9c5895537d504ee6982e8cdc0d182d7679a2e',
        'pending_0126_audit.json': '18709a48cf639cddf97b4b2a912d9c81ca85b6cf7beb388a1d46ee8916754cdf',
        'parameterization.npz': '777fe3ea578a8da719a7efd22e74685ac182bf0c82ca5bb74e012ce6763fe206',
        'accepted_parent_result.json': '0ce5f8a20309a2cfd040d3ee6a5dc5623946b43bacca9d87e822395bfe9bfe81',
        'accepted_parent_fields.npz': '94ea359d4d84d074efd6b68fe5675b06fbf52831ab852f5dfce5ffa9e7654ffb',
        'state_0053/pending.json': '2565395d937a1def4368d8ef5069b773878c8df068742dcd4e9ef6062232bad5',
        'state_0053/pending_0126_fields.npz': 'da35ea65812c3913c61f3d00496e81bed73ac75ec41324cfd524cef27439aada',
        'state_0053/pending_0126_network.eqx': '86181055ae5bea4a51d2ab249638925a9737d92bc045f000b8de5febe00dddd8',
        'state_0053/sweep_0126.json': '7ff84bee0616fe23e970ad40bf22c224098d8abec0e5ef11f74bbc9ef952665d',
    },allow_dotted_keys=True)
    return c
