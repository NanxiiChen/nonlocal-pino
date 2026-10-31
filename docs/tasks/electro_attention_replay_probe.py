"""Reproduce the recorded composed-attention CPU/GPU checkpoint difference.

Run separately with JAX_PLATFORMS=cpu or cuda; output arrays stay external.
This diagnoses a saved model and never changes training/math/defaults.
"""
import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from examples.electropolish2d.operator_training import _build_network, physical_fields, _query_initial
from examples.electropolish2d.operator_data import load_dataset


def main(run, output):
    if output.exists():
        raise FileExistsError(output)
    c = ConfigDict(json.loads((run/'config.json').read_text()))
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', 'highest')
    template = _build_network(c.arch, c.operator.image_shape, jax.random.key(0))
    net = eqx.tree_deserialise_leaves(run/'model_final.eqx', template)
    data, metadata, _ = load_dataset(c)
    index = metadata['eval_indices'][0]
    with np.load(run/'prediction.npz') as saved:
        target = saved['prediction'][0]
        times = saved['time'].astype(np.float32)

    @eqx.filter_jit
    def composed(n, image, points, times, known):
        encoded = n.encode(image)
        xn = points/jnp.asarray(n.extent, dtype=points.dtype)
        projection = jnp.pi*(xn@n.spatial_frequencies)
        features = jnp.concatenate([xn, jnp.sin(projection), jnp.cos(projection), _query_initial(known, c)], -1)
        hidden = jax.vmap(n.spatial_projection)(features)
        block = n.cross_attention
        q, kv = jax.vmap(block.norm_q)(hidden), jax.vmap(block.norm_kv)(encoded)
        attention = block.attention(q, kv, kv)
        h = hidden+attention
        prepared = h+jax.vmap(block.mlp)(jax.vmap(block.norm_out)(h))
        fields = physical_fields(n.decode(prepared, times), c)
        return encoded, features, hidden, q, kv, attention, prepared, fields

    points = data['points'].astype(np.float32)
    values = composed(net, jnp.asarray(data['images'][index]), jnp.asarray(points),
        jnp.asarray(times), jnp.asarray(data['initial'][index]))
    arrays = {k:np.asarray(v) for k, v in zip(
        ('encoded', 'features', 'hidden', 'q', 'kv', 'attention', 'prepared', 'fields'), values, strict=True)}
    arrays.update(points=points, image=data['images'][index], times=times,
        parameters=np.concatenate([np.asarray(v).ravel() for v in jax.tree.leaves(eqx.filter(net, eqx.is_array))]))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(json.dumps(dict(backend=jax.default_backend(), output=str(output),
        saved_GPU_field_gap=float(np.max(abs(arrays['fields'].reshape(target.shape)-target))))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.output)
