"""Read-only historical SOTA cross-grid evaluation; not a matched control."""
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
from examples.corrosion2d.configs.residual_pilot import get_config
from examples.corrosion2d.residual_experiment import build_model, load_reference


def main():
    if jax.default_backend() != 'cpu':
        raise ValueError('CPU only')
    run = Path('/root/autodl-tmp/tf-logs/electropolish2d/SOTA')
    cfg = ConfigDict(json.loads((run/'config.json').read_text()))
    target = get_config('electropolish2d', 'phase', 42)
    assert cfg.physical.to_dict() == target.physical.to_dict()
    assert cfg.surface.to_dict() == target.surface.to_dict()
    assert cfg.nondim.to_dict() == target.nondim.to_dict()
    ref = load_reference(target)
    points = jnp.asarray(np.stack([ref['X'].ravel(), ref['Y'].ravel()], -1), dtype=jnp.float32)
    times = jnp.asarray(ref['time'], dtype=jnp.float32)
    template = build_model(cfg, jax.random.split(jax.random.PRNGKey(42))[0])

    @eqx.filter_jit
    def predict(net):
        return jax.vmap(lambda t: jax.vmap(net, in_axes=(0, None))(points, t))(times)

    output = {'scope': 'Historical301x151-trained SOTA evaluated on101x51 reference; lr0.005/stagger1000/full spatial batches; NOT matched-budget/discretization optimization evidence',
              'reference_path': target.experiment.reference_path, 'runs': {}}
    for name in ('ckpt_005000.eqx', 'ckpt_005800.eqx'):
        path = run/name
        model = eqx.tree_deserialise_leaves(path, template)
        f = np.asarray(predict(model)).reshape(21, *ref['X'].shape, 2).astype(float)
        result = {}
        for channel, field in enumerate(('phi', 'c')):
            a, b = f[..., channel], ref[field]
            result['terminal_'+field+'_relative_l2'] = float(np.linalg.norm(a[-1]-b[-1])/np.linalg.norm(b[-1]))
            result['pooled_'+field+'_relative_l2'] = float(np.linalg.norm(a-b)/np.linalg.norm(b))
        p = f[-1, ..., 0]
        crossings = ((p[:-1]-.5)*(p[1:]-.5) < 0).sum(axis=0)
        result['terminal_vertical_crossing_histogram'] = {str(int(k)): int(v) for k,v in zip(*np.unique(crossings, return_counts=True))}
        output['runs'][str(path)] = result
        print(name, result, flush=True)
    Path(__file__).with_suffix('.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
