"""Fresh same-GPU final checkpoint replay at two predeclared held-out inputs."""
import json
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
from ml_collections import ConfigDict

from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.icemelting3d.operator_data import build_network
from examples.icemelting3d.operator_training import make_predictor

ROOT = Path('/root/autodl-tmp/tf-logs/icemelting3d')
PREFIX = '20261031-melting-spatial-01'


def main():
    configure_gpu_policy('ordinary')
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', 'highest')
    if jax.default_backend() != 'gpu':
        raise ValueError('This receipt declares same-GPU replay, not CPU portability')
    records = []
    for mode in ('fixed_pool', 'rolling_pool'):
        run = ROOT/(PREFIX+'-'+mode)
        c = ConfigDict(json.loads((run/'config.json').read_text()))
        template = build_network(c, jax.random.split(jax.random.key(c.train.seed), 3)[0])
        net = eqx.tree_deserialise_leaves(run/'model_final.eqx', template)
        predict = make_predictor(c)
        for index in (2, 5):
            with np.load(run/f'prediction_{index:03d}.npz') as f:
                actual = predict(net, f['parameters'], f['points'], f['time'])
                gap = float(np.max(np.abs(actual-f['prediction'])))
                if gap > 2e-5:
                    raise ValueError(f'{mode}/input{index} checkpoint replay gap{gap}>2e-5')
                records.append(dict(mode=mode, index=index, max_absolute_phi_gap=gap))
    result = dict(status='passed', backend=jax.default_backend(), threshold=2e-5,
        records=records, scope='Fresh same-GPU replay; independent CPU portability not evaluated')
    target = Path('/root/nonlocal-pinn/docs/tasks/melting-spatial-replay.json')
    target.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
