"""Measure full-grid gradients and complete device-bank memory before training."""
import argparse
import json
from pathlib import Path
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nonlocal_pinn.training.optimizers import make_optimizer
from nonlocal_pinn.training.runtime import configure_gpu_policy
from .configs.train_full_grid import get_config
from .full_grid import prepare_initial_full_bank
from .full_grid_validation import validate_full_control
from .losses import LOSS_NAMES, make_loss_vector, make_step
from .network import build_network
from .train import load_bank, write_json


def probe(inputs, reference, control, output, updates):
    c = get_config()
    if updates <= c.train.gradnorm_update_every or updates > c.train.iterations:
        raise ValueError('Probe must include a warm weight-refresh step within the training budget')
    runtime = validate_full_control(c, control, inputs, reference)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    started = time.perf_counter()
    arrays, manifest = load_bank(inputs, c)
    arrays = prepare_initial_full_bank(arrays, c, manifest, c.operator.online.workers)
    bank = {name: jnp.asarray(value) for name, value in arrays.items()}
    with np.load(Path(control)/'sampling.npz', allow_pickle=False) as trace:
        conditions, times = trace['condition_ids'][:updates], trace['times'][:updates]
    net = build_network(c, jax.random.PRNGKey(runtime['seed']))
    optimizer = make_optimizer(c)
    state = optimizer.init(eqx.filter(net, eqx.is_array))
    weights = jnp.ones(len(LOSS_NAMES), dtype=jnp.float32)
    step = make_step(make_loss_vector(c), optimizer, c, LOSS_NAMES)

    @jax.jit
    def gather(data, ids):
        values = {name: data[name][ids] for name in manifest['fixed_keys']}
        values.update({name: data[name][ids, 0] for name in manifest['graph_keys']})
        return values

    jax.block_until_ready(bank)
    report = dict(config=json.loads(c.to_json()), control=str(Path(control).resolve()),
        device=jax.devices()[0].device_kind, seed=runtime['seed'], updates=[],
        full_bank_device_bytes=sum(value.nbytes for value in bank.values()),
        pde_nodes_min=int(np.sum(arrays['pde_weights'][:, 0] > 0, axis=-1).min()),
        pde_nodes_max=int(np.sum(arrays['pde_weights'][:, 0] > 0, axis=-1).max()),
        setup_seconds=time.perf_counter()-started,
        scope='Finite resource probe on the complete initial bank, without reference evaluation or rolling production.')
    write_json(output/'started.json', report)
    source = Path(__file__).resolve()
    shutil.copyfile(source, output/source.name)
    print(json.dumps(dict(event='ready', **{k:v for k,v in report.items() if k not in ('config','updates')})), flush=True)
    for index in range(updates):
        payload = gather(bank, jnp.asarray(conditions[index]))
        jax.block_until_ready(payload)
        refresh = index % c.train.gradnorm_update_every == 0
        begin = time.perf_counter()
        net, state, weights, values, finite = step(net, state, weights,
            jnp.asarray(times[index]), payload, jnp.asarray(refresh))
        jax.block_until_ready((net, state, weights, values, finite))
        if not bool(finite):
            raise FloatingPointError(f'Nonfinite full-grid probe update {index+1}')
        row = dict(step=index+1, refresh=refresh, seconds=time.perf_counter()-begin,
            loss=np.asarray(values).tolist(), weights=np.asarray(weights).tolist(),
            memory={key:int(value) for key,value in jax.devices()[0].memory_stats().items()})
        report['updates'].append(row)
        print(json.dumps(row), flush=True)
    report.update(first_compile_seconds=report['updates'][0]['seconds'],
        held_median_seconds=float(np.median([row['seconds'] for row in report['updates'][1:] if not row['refresh']])),
        refresh_median_seconds=float(np.median([row['seconds'] for row in report['updates'][1:] if row['refresh']])),
        peak_active_bytes=max(row['memory']['peak_bytes_in_use'] for row in report['updates']),
        total_seconds=time.perf_counter()-started)
    write_json(output/'report.json', report)
    print('COMPLETE '+str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'reference', 'control', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--updates', required=True, type=int)
    args = parser.parse_args()
    probe(args.inputs, args.reference, args.control, args.output, args.updates)


if __name__ == '__main__':
    main()
