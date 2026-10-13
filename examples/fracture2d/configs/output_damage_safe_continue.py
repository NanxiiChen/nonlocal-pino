"""Original53..75 loading from accepted52 with a bounded frozen initializer."""
from examples.fracture2d.configs.output_damage_continue import get_config as base


def get_config():
    c = base()
    c.continuation.parent = '/root/autodl-tmp/tf-logs/fracture2d/output_damage_sigmoid_full_20261011/state_0052'
    c.continuation.parent_field_sha256 = '00c8fb8cd09a93e9115084bca5261d01f5247f0ee08a88068908f4da02302816'
    c.continuation.parent_network_sha256 = '895ca01dc095cc77f637d56496fd8b34e19e6d7d6413c5a42acb44db128ca8ef'
    c.continuation.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/output_damage_sigmoid_safe_full_20261012'
    c.continuation.first_index = 53
    c.continuation.last_index = 75
    c.continuation.initializer = 'cap_frozen_predictor'
    return c
