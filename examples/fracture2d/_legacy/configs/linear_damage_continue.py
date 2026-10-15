"""Same-load linear neural recovery with user-accepted microunit range error."""
from ml_collections import ConfigDict
from examples.fracture2d._legacy.configs.linear_damage_loading import get_config as base


def get_config():
    c = base()
    c.range.lbfgs.range_tolerance = 2e-6
    c.reporting.range_atol = 2e-6
    c.loading.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/linear_damage_continue_20261012'
    c.loading.indices = list(range(53,76))
    c.continuation = ConfigDict()
    c.continuation.source_dir = '/root/autodl-tmp/tf-logs/fracture2d/linear_damage_loading_20261012'
    c.continuation.accepted_parent_index = 52
    c.continuation.completed_outer = 122
    c.continuation.range_atol = 2e-6
    c.continuation.source = ConfigDict({'identity.json': 'a3f88ff3cb49928cee276e783ad00e42d8d57c18a7968ee3a30d0fb559ae2cb8', 'audit.json': '93f14bd4c470484c82f0f421b7df65cfb6a3af71292ec411b64cbf340d8fd00f', 'result.json': 'f13fd99d4436ce54490362ef5e528f8a21055c60560451a20bedaefd8117bb70', 'parameterization.npz': '777fe3ea578a8da719a7efd22e74685ac182bf0c82ca5bb74e012ce6763fe206', 'state_0052/result.json': '0ce5f8a20309a2cfd040d3ee6a5dc5623946b43bacca9d87e822395bfe9bfe81', 'state_0052/fields.npz': '94ea359d4d84d074efd6b68fe5675b06fbf52831ab852f5dfce5ffa9e7654ffb', 'state_0052/network.eqx': 'd174aff62164b58bccdb8605088fd0c3374982be6fe84dca7e7094b5226bd065', 'state_0053/result.json': '05bd3fc2421d96880bf67938956f334edcbc3336d0b49163d97059a34e12d749', 'state_0053/fields.npz': 'a2fbfe161b2b38bf3d28d7ec297669b5a60039b559968bea1e57dfa5c438965e', 'state_0053/network.eqx': 'b040b03d2f6d57c5f67f7770969649c6b99dd5100dd0af0c9c0c2ec12dcbcbf1', 'state_0053/pending.json': '66d5fde655d01499fcdd061deb280b33a13b28feebb9de6393536a64f4c08c61', 'state_0053/pending_0122_fields.npz': '7d6729dd8d345877548ae91347043171054e8ddc86267cb41690e853a3f47cef', 'state_0053/pending_0122_network.eqx': '3a8433d98928711e18f4e13593885d0e490e36be0dee13ffb83d6212fa904e9e'},allow_dotted_keys=True)
    return c
