"""Validate explicit map defaults and saved-config compatibility using paths."""
import importlib
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())
sys.path.insert(0, str(ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from examples.twophase2d.network import PhaseNetwork
from examples.twophase2d.problem import validate_config
from examples.twophase2d.fields import predict_fields
from examples.twophase2d.configs.train_vortex_conserved import get_config


def main():
    if jax.default_backend() != 'cpu':
        raise ValueError('Run with JAX_PLATFORMS=cpu')
    configs = {}
    for p in sorted((ROOT / 'examples/twophase2d/configs').glob('train_*.py')):
        c = importlib.import_module('examples.twophase2d.configs.' + p.stem).get_config()
        validate_config(c)
        configs[p.stem] = c.phase.map
    assert configs['train_base'] == configs['train_vortex_conserved'] == 'additive'
    assert configs['train_vortex_conserved_local'] == 'additive'
    for name in ('train_vortex_accuracy', 'train_vortex_causal_conserved', 'train_vortex_initial_time'):
        assert configs[name] == 'inverse_cubic'

    run_root = Path('/root/autodl-tmp/tf-logs/twophase2d')
    additive_run = run_root / '20261027-vortex-map-ablation-additive'
    saved = json.loads((additive_run / 'config.json').read_text())
    current = json.loads(get_config().to_json())
    assert current.pop('phase') == {'map': 'additive'}
    del saved['train']['seed']
    del saved['log']['run_name']
    del current['log']['run_name']
    assert current == saved, 'Default differs from the tested additive strategy'

    # Invalid strings fail early; no silent map fallback.
    invalid = get_config()
    invalid.phase.map = 'misspelled_map'
    try:
        validate_config(invalid)
    except ValueError as error:
        assert 'phase.map' in str(error)
    else:
        raise AssertionError('Invalid map was accepted')

    replay = {}
    # Replay full grids at t=0 and t=T for both saved schemas.
    for name in ('additive', 'cubic'):
        run = run_root / ('20261027-vortex-map-ablation-' + name)
        c = ConfigDict(json.loads((run / 'config.json').read_text()))
        validate_config(c)
        if name == 'additive':
            assert 'phase' not in c
        else:
            assert c.phase.map == 'inverse_cubic'
        jax.config.update('jax_enable_x64', True)
        with jax.enable_x64(False):
            key = jax.random.split(jax.random.PRNGKey(c.train.seed))[0]
            template = PhaseNetwork(c, key)
        template = jax.tree.map(lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, template)
        net = eqx.tree_deserialise_leaves(run / 'model_final.eqx', template)
        with np.load(c.reference.path, allow_pickle=False) as reference:
            points = jnp.asarray(reference['points'])
        with np.load(run / 'prediction.npz', allow_pickle=False) as fields:
            expected = np.stack((fields['phi'][[0, -1]], fields['q'][[0, -1]]), axis=-1)
        times = jnp.asarray([0., c.physical.terminal_time])
        evaluate = eqx.filter_jit(lambda model: jax.vmap(lambda t: predict_fields(model, points, t, c))(times))
        actual = np.asarray(evaluate(net)).reshape(expected.shape)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
        replay[name] = {'saved_map': net.phase_map, 'max_absolute': float(np.max(np.abs(actual - expected)))}

    result = {'status': 'passed', 'preset_count': len(configs), 'maps': configs,
              'default_matches_fresh_additive_strategy': True,
              'invalid_map_rejected': True, 'saved_checkpoint_replays': replay}
    output = Path(__file__).with_name('twophase-optional-map-validation.json')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
