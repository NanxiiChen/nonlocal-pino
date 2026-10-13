"""Audited own-neural accepted-prefix fork for optimizer-policy experiments."""
import argparse
import copy
import json
from pathlib import Path
import runpy
import shutil
import jax
import numpy as np
from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.local_damage_trial import sha
from examples.fracture2d.compact_damage_loading import run


def comparable(config):
    result = copy.deepcopy(config)
    result['data']['end_state_index'] = 0
    result['output']['resume'] = False
    for key in ('output_dir', 'methods', 'lbfgs'):
        del result['loading'][key]
    # Explicit provenance is absent in legacy v1 roots and present in forks;
    # it does not change any numerical input or physical criterion.
    if 'fork' in result['loading']:
        del result['loading']['fork']
    return result


def prepare(c, *, resume):
    controls = c.loading.fork
    if type(controls.prefix_length) is not int or not 1 <= controls.prefix_length < c.data.end_state_index:
        raise ValueError('fork needs an explicit accepted prefix before the requested end')
    parent, root = Path(controls.parent_root), Path(c.loading.output_dir)
    if root.resolve() == parent.resolve() or sha(parent/'identity.json') != controls.parent_identity_sha256:
        raise ValueError('fork parent identity/path differs')
    identity = json.loads((parent/'identity.json').read_text())
    if (identity['schema'] != 'own-zero-compact-neural-loading-v1'
            or not identity['own_zero_initial_state'] or identity['direct_targets']
            or identity['damage_correction'] or comparable(identity['config']) != comparable(c.to_dict())):
        raise ValueError('fork must retain the same own-zero model/geometry/physical gates')
    if sha(__file__) != controls.runner_sha256:
        raise ValueError('fork runner differs')
    manifest, parent_sha, d_previous, H_previous = [], None, None, None
    for index in range(1, controls.prefix_length+1):
        folder = parent/f'state_{index:04d}'
        r = json.loads((folder/'result.json').read_text())
        if (r['index'] != index or r['load'] != c.data.reference_indices[index-1]*c.data.load_step
                or r['status'] != 'ready' or not r['physical_state_promoted']
                or not r['outer_convergence']['converged']
                or not all(v['pass'] for v in r['outer_convergence']['checks'].values())
                or r['parent_sha256'] != parent_sha
                or sha(folder/'fields.npz') != r['state_sha256']
                or sha(folder/'network.eqx') != r['network_sha256']):
            raise ValueError(f'fork state {index} is rejected, changed or has invalid ancestry')
        with np.load(folder/'fields.npz', allow_pickle=False) as f:
            if index == 1:
                d_previous, H_previous = np.zeros_like(f['d']), np.zeros_like(f['H'])
            np.testing.assert_array_equal(f['d_parent'], d_previous)
            np.testing.assert_array_equal(f['H_parent'], H_previous)
            if (not np.isfinite(f['q']).all() or not np.isfinite(f['d']).all()
                    or not np.isfinite(f['H']).all() or np.min(f['H']-H_previous) < 0):
                raise ValueError(f'fork state {index} has invalid fields/history')
            d_previous, H_previous = f['d'].copy(), f['H'].copy()
        manifest.append(dict(index=index, fields_sha256=r['state_sha256'],
            network_sha256=r['network_sha256'], report_sha256=sha(folder/'result.json')))
        parent_sha = r['state_sha256']
    record = dict(parent_root=str(parent), parent_identity_sha256=controls.parent_identity_sha256,
                  prefix=manifest, runner_sha256=controls.runner_sha256,
                  parent_numerical_runner_sha256=identity['runner_sha256'],
                  numerical_runner_sha256=sha(run.__code__.co_filename))
    canonical = c.to_dict()
    canonical['data']['end_state_index'] = 0
    canonical['output']['resume'] = False
    # Parent code remains pinned in provenance; the independently checked
    # optimizer-objective extension is pinned as the new numerical runner.
    new_identity = dict(identity, config=canonical, runner_sha256=record['numerical_runner_sha256'])
    if resume:
        if json.loads((root/'fork.json').read_text()) != record:
            raise ValueError('fork provenance changed')
    else:
        root.mkdir()
        atomic_json(root/'identity.json', new_identity)
        atomic_json(root/'fork.json', record)
        for item in manifest:
            name = f"state_{item['index']:04d}"
            (root/name).mkdir()
            for filename in ('fields.npz', 'network.eqx', 'result.json'):
                shutil.copy2(parent/name/filename, root/name/filename)
        print(json.dumps(dict(event='accepted_neural_prefix_fork', prefix=controls.prefix_length,
                              parent=str(parent), output=str(root))), flush=True)
    c.output.resume = True


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    c = runpy.run_path(args.config)['get_config']()
    prepare(c, resume=args.resume)
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    run(c)
