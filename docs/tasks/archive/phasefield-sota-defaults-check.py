"""Audit promoted configs and unchanged samplers against direct saved paths."""
import ast
import importlib
import json
from pathlib import Path
import subprocess

import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from examples.corrosion2d import training_data
from examples.corrosion2d.residual_experiment import build_model, load_reference

ROOT = Path(__file__).resolve().parents[3]
WINNERS = {
    'corrosion2d': Path('/root/autodl-tmp/tf-logs/corrosion2d/20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-bounded-seed42'),
    'electropolish2d': Path('/root/autodl-tmp/tf-logs/electropolish2d/20261025-data-physical-icrandom_interior-bcrandom-phase-stagger0-seed7'),
    'icemelting3d': Path('/root/autodl-tmp/tf-logs/icemelting3d/20261024-5s-soap-lr1e3-seed42'),
}


def flatten(d, prefix=''):
    result = {}
    for key, value in d.items():
        key = prefix+'.'+key if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, key))
        else:
            result[key] = value
    return result


def main():
    if jax.default_backend() != 'cpu':
        raise ValueError('Run with JAX_PLATFORMS=cpu')
    output = {'scope': 'Configuration/initialization/data parity, not a new full training replay', 'cases': {}}
    for case, run in WINNERS.items():
        config = importlib.import_module(f'examples.{case}.configs.train_base').get_config()
        saved_config = ConfigDict(json.loads((run/'config.json').read_text()))
        old, new = flatten(json.loads(saved_config.to_json())), flatten(json.loads(config.to_json()))
        extra = set(new)-set(old)
        assert not set(old)-set(new)
        if case == 'corrosion2d':
            assert extra == {'experiment.lr_tail_start', 'experiment.lr_tail_end', 'experiment.ch_rms_after'}
            assert config.experiment.lr_tail_start == config.experiment.ch_rms_after == 0
        else:
            assert not extra
        diff = {key: {'saved': old[key], 'default': new[key]}
                for key in old if old[key] != new[key]}
        allowed = {'log.run_name'}
        if case == 'icemelting3d':
            allowed |= {'train.iterations', 'eval.graph_reference_path'}
        assert set(diff) <= allowed, diff
        assert config.train.iterations == 5000 and config.train.optimizer == 'soap'
        assert config.train.lr_init == .001
        assert (run/'ckpt_005000.eqx').is_file()
        key = jax.random.split(jax.random.PRNGKey(config.train.seed))[0]
        if case == 'icemelting3d':
            from examples.icemelting3d.train import _build_phase_network
            template = _build_phase_network(config, key)
            saved_template = _build_phase_network(saved_config, key)
            reference_path = ROOT/'reports/icemelting-weak-prior-20261024/reference.npz'
            assert reference_path.is_file()
        else:
            reference = load_reference(config)
            assert reference['phi'].shape == (21, 51, 101)
            reference_path = Path(config.experiment.reference_path)
            template = build_model(config, key)
            saved_template = build_model(saved_config, key)
        current_leaves, saved_leaves = (jax.tree.leaves(eqx.filter(net, eqx.is_array))
                                       for net in (template, saved_template))
        for x, y in zip(current_leaves, saved_leaves):
            np.testing.assert_array_equal(x, y)
        restored = eqx.tree_deserialise_leaves(run/'model_initial.eqx', template)
        a, b = (jax.tree.leaves(eqx.filter(net, eqx.is_array)) for net in (template, restored))
        assert len(a) == len(b)
        max_difference = max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x, y in zip(a, b))
        for x, y in zip(a, b):
            # GPU normal initialization and CPU normal initialization can round
            # differently; same-CPU old/new construction above must be exact.
            np.testing.assert_allclose(x, y, atol=2e-6, rtol=0)
        output['cases'][case] = {'run': str(run), 'checkpoint': str(run/'ckpt_005000.eqx'),
            'reference': str(reference_path), 'allowed_config_differences': diff,
            'explicit_new_fields': {key: new[key] for key in sorted(extra)},
            'same_cpu_saved_vs_default_initialization': 'exact array equality',
            'cpu_vs_saved_gpu_initial_parameter_max_abs_difference': max_difference}
    # Check extracted functions against the unmodified original CLI sources.
    parity = {}
    for case, prefix in (('corrosion2d', 'corrosion'), ('electropolish2d', 'electropolish')):
        source = subprocess.check_output(['git', 'show', f'HEAD:examples/{case}/train.py'], cwd=ROOT, text=True)
        tree = ast.parse(source)
        namespace = {'np': np, 'jnp': jnp}
        config = importlib.import_module(f'examples.{case}.configs.train_base').get_config()
        for old_name, new_name in (('_bc_info', prefix+'_boundary_points'),
                                   ('_sample_ic_points', prefix+'_initial_points')):
            node = next(v for v in tree.body if isinstance(v, ast.FunctionDef) and v.name == old_name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<original-sampler>', 'exec'), namespace)
            old_fn, new_fn = namespace[old_name], getattr(training_data, new_name)
            if case == 'electropolish2d':
                args = (config,)
            elif old_name == '_bc_info':
                args = (config.pits.entries, config.sample.N_bc_dir, config.sample.N_bc_solid)
            else:
                args = (config.pits.entries, config.sample.N_ic_global, config.sample.N_ic_local)
            expected, actual = old_fn(*args), new_fn(*args)
            if isinstance(expected, tuple):
                for x, y in zip(expected, actual):
                    np.testing.assert_array_equal(x, y)
            else:
                np.testing.assert_array_equal(expected, actual)
            parity[new_name] = 'exact array equality against original CLI function'
    output['sampler_parity'] = parity
    Path(__file__).with_suffix('.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
