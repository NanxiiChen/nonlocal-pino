"""Verify continuation provenance, then reuse independent native field replay."""
import argparse
import json
from pathlib import Path
import time

import jax
from ml_collections import ConfigDict
import numpy as np

from examples.fracture2d.local_damage_trial import sha
from examples.fracture2d.staggered_damage_continue import restart_plan
from examples.fracture2d.staggered_damage_audit import main as audit_fields, read_fields


def main(root, *, watch):
    while not (root/'preflight.json').exists():
        if (root/'exit.json').exists(): raise RuntimeError('continuation failed before restore')
        if not watch: raise RuntimeError('continuation preflight unavailable')
        time.sleep(5)
    identity = json.loads((root/'identity.json').read_text())
    a = ConfigDict(identity['config'], allow_dotted_keys=True)
    source = Path(a.source_root)
    original = json.loads((source/'identity.json').read_text())['config']['l1_loading']
    plan = restart_plan(a, original['inner_updates'], original['pool_size'])
    assert identity['initial_count'] == plan['initial_count']
    assert identity['source_config']['l1_loading']['inner_updates'] == a.inner_updates
    assert identity['preceding_load_history_unchanged'] and not a.stop_early
    if a.start_sweep:
        previous = Path(a.restart_root)
        for name, digest in a.restart_pins.items(): assert sha(previous/name) == digest, name
        blocks = [json.loads(line) for line in (previous/'pinn_full/sampling_blocks.jsonl').read_text().splitlines()]
        assert identity['rng_state'] == blocks[-1]['rng_after']
        stage = read_fields(previous/'pinn_full'/f'outer_{a.start_sweep:04d}'/'fields.npz')
        parent = read_fields(source/f'state_{a.parent_index:04d}'/'fields.npz')
        np.testing.assert_array_equal(stage['H_parent'], parent['H'])
        np.testing.assert_array_equal(stage['d_parent'], parent['d'])
    else:
        blocks = [json.loads(line) for line in (source/'sampling_blocks.jsonl').read_text().splitlines()]
        assert identity['rng_state'] == blocks[a.parent_count//original['pool_size']-1]['rng_after']
    audit_fields(root, watch=watch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    parser.add_argument('--watch', action='store_true'); args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    main(Path(args.root), watch=args.watch)
