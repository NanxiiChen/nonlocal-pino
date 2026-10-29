"""Fresh GPU replay of two predeclared final SOAP checkpoint fields."""
import json
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
from ml_collections import ConfigDict

from examples.icemelting3d.operator_data import build_network, load_dataset, validate_config
from examples.icemelting3d.operator_training import make_predictor
from nonlocal_pinn.training.runtime import configure_gpu_policy

run = Path('/root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-soap-01')
c = ConfigDict(json.loads((run / 'config.json').read_text()))
validate_config(c)
configure_gpu_policy(c.train.gpu_policy)
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
assert jax.default_backend() == 'gpu'
manifest, refs = load_dataset(c)
net = eqx.tree_deserialise_leaves(run / 'model_final.eqx', build_network(c, jax.random.key(0)))
assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(net) if eqx.is_array(a))
predict = make_predictor(c)
records = []
for index in (manifest['monitor_indices'][0], manifest['test_indices'][0]):
    ref = refs[index]
    replay = predict(net, ref['parameters'], ref['points'], ref['time'])
    with np.load(run / f'prediction_{index:03d}.npz') as stored:
        gap = float(abs(replay - stored['prediction']).max())
    assert gap < 2e-5, (index, gap)
    records.append(dict(index=index, maximum_absolute_replay_gap=gap))
result = dict(all_checks_pass=True, checkpoint=str(run / 'model_final.eqx'),
              backend=jax.default_backend(), tolerance=2e-5, records=records)
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
