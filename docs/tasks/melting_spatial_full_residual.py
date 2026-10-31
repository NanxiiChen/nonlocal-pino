"""Common full128^3PDE checks without parameter gradients or training updates."""
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.icemelting3d.ellipsoids import image_points, signed_distance
from examples.icemelting3d.operator_data import build_network, physics
from examples.icemelting3d.operator_losses import latent_fields, make_loss_fn
from examples.icemelting3d.operator_training import build_graph, make_predictor
from examples.icemelting3d.spatial_sampling import SpatialSampler

ROOT = Path('/root/autodl-tmp/tf-logs/icemelting3d')
PREFIX = '20261031-melting-spatial-01'


def main():
    configure_gpu_policy('ordinary')
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', 'highest')
    if jax.default_backend() != 'gpu':
        raise ValueError('Require the measured GPU execution path')
    config = json.loads((ROOT/(PREFIX+'-fixed_pool')/'config.json').read_text())
    c = ConfigDict(config); c.sample.spatial.mode = 'full'
    op, points, graph = build_graph(c)
    chunks = SpatialSampler(c, op, np.asarray(points))
    bank = Path(c.operator.dataset_path)
    manifest = json.loads((bank/'manifest.json').read_text())
    indices = (2, 5)
    parameters = jnp.asarray([manifest['parameters'][i] for i in indices], dtype=jnp.float32)
    width = physics(c)[3]
    input_points = jnp.asarray(image_points(tuple(c.arch.image_shape)), dtype=jnp.float32)
    distance = eqx.filter_jit(lambda x: jax.vmap(
        lambda p: signed_distance(x, p, c.initial.sdf_iterations))(parameters)/width)
    images = jnp.tanh(distance(input_points)).reshape((len(indices),)+tuple(c.arch.image_shape))
    times = jnp.asarray([.25, .5, .75, 1.], dtype=jnp.float32)
    objective = eqx.filter_jit(make_loss_fn(c))
    phi_fn = eqx.filter_jit(lambda net, x, u0: jnp.tanh(
        latent_fields(net, net.encode(images[0]), x, u0, times)[0]))
    records = []
    for mode in ('fixed_pool', 'rolling_pool'):
        run = ROOT/(PREFIX+'-'+mode)
        template = build_network(c, jax.random.split(jax.random.key(c.train.seed), 3)[0])
        net = eqx.tree_deserialise_leaves(run/'model_final.eqx', template)
        total = 0.; begin = time.perf_counter(); gap = None
        for k, ((x, sub, mask), weight) in enumerate(chunks.full_chunks()):
            initial = distance(x)
            value = float(objective(net, times, images, initial, x, sub, mask))
            if not np.isfinite(value):
                raise FloatingPointError('Nonfinite common full-grid latent residual')
            total += weight*value
            if k == 32:
                # Ensure large residual-evaluation chunks represent the same
                # physical field as the saved standard1024query predictor.
                full_phi = np.asarray(phi_fn(net, x, initial[0]))
                selected = np.random.default_rng(12).choice(len(x), 512, replace=False)
                separate = make_predictor(c)(net, np.asarray(parameters[0]), np.asarray(x)[selected], np.asarray(times))
                gap = float(np.max(np.abs(full_phi[:, selected]-separate)))
                if gap > 2e-5:
                    raise ValueError(f'GPU full-chunk/query prediction invariance fails{gap}>2e-5')
        records.append(dict(mode=mode, mean_absolute_latent_residual=total,
            seconds_including_JIT=time.perf_counter()-begin,
            max_large_vs_standard_chunk_phi_gap=gap))
    receipt = dict(status='passed', inputs=list(indices), positive_times=np.asarray(times).tolist(),
        graph_nodes=graph['grid_nodes'], chunks=chunks.summary(), records=records,
        scope='Common full spatial PDE check,2reused inputs/4positive times; not a full5000training or full-grid physical L2',
        training_updates=0)
    Path('/root/nonlocal-pinn/docs/tasks/melting-spatial-full-residual.json').write_text(
        json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
