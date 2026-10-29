"""Reload both saved final checkpoints for local-FEM evaluation after MPI ends."""
import json
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
from ml_collections import ConfigDict

from examples.icemelting3d.operator_data import build_network
from examples.icemelting3d.operator_training import make_predictor
from nonlocal_pinn.training.runtime import configure_gpu_policy

OUT = Path(__file__).resolve().parent
validation = json.loads((OUT/'melting-local-fem-validation.json').read_text())
assert validation['status'] in ('passed', 'accepted_by_user')
runs = dict(Adam=Path('/root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-01'),
            SOAP=Path('/root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-soap-01'))
configs = {name: ConfigDict(json.loads((run/'config.json').read_text())) for name, run in runs.items()}
c = configs['Adam']
configure_gpu_policy(c.train.gpu_policy)
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
assert jax.default_backend() == 'gpu'
predict = make_predictor(c)
root = Path(validation['conditions'][0]['path']).parent / 'checkpoint_replay'
root.mkdir(exist_ok=False)
records = []
for name, run in runs.items():
    for key in ('arch', 'initial', 'physical', 'nondim'):
        assert configs[name][key] == c[key]
    checkpoint = run/'model_final.eqx'
    original_bytes = checkpoint.read_bytes()
    net = eqx.tree_deserialise_leaves(checkpoint, build_network(configs[name], jax.random.key(0)))
    assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(net) if eqx.is_array(a))
    for condition in validation['conditions']:
        index = condition['index']
        with np.load(condition['path']) as fem:
            points, slice_points, parameters, times = (fem[key] for key in ('points', 'slice_points', 'parameters', 'time'))
        p = predict(net, parameters, points, times)
        sp = predict(net, parameters, slice_points, times)
        with np.load(run/f'prediction_{index:03d}.npz') as original:
            arrays = {key: original[key] for key in original.files}
        gap = float(abs(p-arrays['prediction']).max())
        slice_gap = float(abs(sp-arrays['slice_prediction']).max())
        assert max(gap, slice_gap) < 2e-5, (name, index, gap, slice_gap)
        arrays['prediction'], arrays['slice_prediction'] = p, sp
        target = root/f'{name.lower()}-{index:03d}.npz'
        np.savez_compressed(target, **arrays)
        records.append(dict(optimizer=name, index=index, path=str(target),
                            checkpoint=str(checkpoint), maximum_replay_gap=gap,
                            maximum_slice_replay_gap=slice_gap))
    assert checkpoint.read_bytes() == original_bytes
report = dict(all_checks_pass=True, records=records, backend=jax.default_backend(),
              scope='Fresh final-checkpoint loading/inference, no gradient/update; common local FEM queries; direct paths, no file hashes.')
(OUT/'melting-local-fem-checkpoint-replay.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
