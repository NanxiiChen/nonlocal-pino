"""Declared accepted-parent fork bounding only the frozen initializer."""
import argparse
import json
from pathlib import Path
import runpy

import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from examples.fracture2d._legacy import output_damage_continue as continuation
from examples.fracture2d._legacy.local_damage_trial import sha
from examples.fracture2d._legacy.output_damage_trial import setup


def bounded_initializer(frozen_image, *, epsilon):
    """Keep the logit seed finite without clipping unknown physical damage."""
    if not np.isfinite(epsilon) or not 0 < epsilon < .5:
        raise ValueError('initializer epsilon must lie strictly between0 and.5')

    @jax.jit
    def image(H, load):
        return jnp.minimum(frozen_image(H, load), 1-epsilon)

    return image


def safe_setup(c):
    if c.continuation.initializer != 'cap_frozen_predictor':
        raise ValueError('safe continuation requires explicit frozen initializer cap')
    s = setup(c)
    return {**s, 'frozen_image': bounded_initializer(s['frozen_image'],
        epsilon=c.head_trial.initialization_epsilon)}


def verify_parent(c):
    """Verify certified old ancestry and archived original continuation code."""
    parent = Path(c.continuation.parent)
    root = parent.parent
    identity_path = root/'identity.json'
    identity = json.loads(identity_path.read_text())
    audit_path = root/'audit.json'
    audit = json.loads(audit_path.read_text())
    original = ConfigDict(identity['config'])
    old_base, new_base = original.to_dict(), c.to_dict()
    del old_base['continuation']
    del new_base['continuation']
    if (identity['schema'] != 'certified-sigmoid-own-history-continuation-v1'
            or old_base != new_base or not identity['own_history']
            or identity['direct_targets'] or identity['damage_correction']
            or original.continuation.mode != c.continuation.mode
            or original.continuation.methods != c.continuation.methods
            or original.continuation.lbfgs != c.continuation.lbfgs
            or not audit['pass_replay'] or not audit['independent_numpy_network_replay']
            or not audit['own_history'] or not audit['parent_unchanged']
            or audit['total_ready_loads'] != c.continuation.first_index-1
            or not json.loads((root/'compilation_parity.json').read_text())['pass_parity']):
        raise ValueError('initializer fork parent model/audit/controls differ')
    previous, report, protected = continuation.verify_parent(original)
    for path, digest in identity['source'].items():
        if sha(path) != digest:
            raise ValueError(f'initializer fork protected source changed: {path}')
        protected[path] = digest
    for name, digest in identity['code'].items():
        archived = root/'code_snapshot'/name
        if sha(archived) != digest:
            raise ValueError(f'initializer fork archived code changed: {name}')
        # Orchestration is deliberately updated; certified numerical modules
        # must still match their original runtime source as well as the archive.
        if name != 'output_damage_continue.py' and sha(Path(__file__).with_name(name)) != digest:
            raise ValueError(f'initializer fork certified numerical code changed: {name}')
        protected[str(archived)] = digest
    for path in (identity_path, audit_path, root/'audit_script.py', root/'compilation_parity.json'):
        protected[str(path)] = sha(path)
    for index in range(original.continuation.first_index, c.continuation.first_index):
        folder = root/f'state_{index:04d}'
        fields, current = continuation.accepted_state(folder, report['state_sha256'])
        with np.load(folder/'fields.npz', allow_pickle=False) as f:
            np.testing.assert_array_equal(f['d_parent'], previous['d'])
            np.testing.assert_array_equal(f['H_parent'], previous['H'])
        if (current['index'] != index or current['load'] != index*c.data.load_step
                or np.any(fields['H'] < previous['H'])):
            raise ValueError('initializer fork physical ancestry/load differs')
        audited = [row for row in audit['reports'] if row['index'] == index]
        if len(audited) != 1 or audited[0]['status'] != 'ready':
            raise ValueError('initializer fork accepted state missing from independent audit')
        for name in ('fields.npz', 'network.eqx', 'result.json'):
            path = folder/name
            protected[str(path)] = sha(path)
        previous, report = fields, current
    if (parent.name != f"state_{report['index']:04d}"
            or report['state_sha256'] != c.continuation.parent_field_sha256
            or report['network_sha256'] != c.continuation.parent_network_sha256):
        raise ValueError('initializer fork parent pins differ')
    return previous, report, protected


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    c = runpy.run_path(args.config)['get_config']()
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    continuation.run(c, resume=args.resume, parent_loader=verify_parent,
        setup_fn=safe_setup, extra_code_names=['output_damage_safe_continue.py'])
