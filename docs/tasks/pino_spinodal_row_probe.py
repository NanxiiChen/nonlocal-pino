"""Measure exact dense-metric row sampling without claiming a compact halo."""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def prepare(output):
    from examples.confined_ch2d.configs.train_operator import get_config
    from examples.confined_ch2d.operator_metrics import linear_stable_resolvent
    from examples.confined_ch2d.operator_online import pad_input
    from examples.confined_ch2d.operator_problem import fixed_initial
    c = get_config()
    root = Path(c.operator.dataset_path)
    manifest = json.loads((root/'manifest.json').read_text())
    cases = manifest['train_indices'][:c.operator.condition_batch]
    rows = []
    records = []
    for index in cases:
        path = root/manifest['conditions'][index]['file']
        with np.load(path, allow_pickle=False) as saved:
            data = {key: saved[key] for key in ('points', 'volumes', 'neighbors', 'coefficients',
                                               'condition', 'condition_weights')}
        data['initial'] = fixed_initial(data['points'], c, xp=np)
        p = c.physical
        data['metric'], _ = linear_stable_resolvent(data['volumes'], data['neighbors'], data['coefficients'],
            data['initial'], p.beta, p.epsilon, p.mobility, p.terminal_time)
        nonzero = np.count_nonzero(data['metric'].astype(np.float32), axis=1)
        records.append(dict(index=index, source=str(path), nodes=len(nonzero),
                            minimum_metric_row_support=int(nonzero.min())))
        rows.append(pad_input(data, c))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'inputs.npz', **{key: np.stack([row[key] for row in rows]) for key in rows[0]})
    (output/'input-audit.json').write_text(json.dumps(dict(status='passed', cases=records,
        geometry_count=len(rows), training_labels_loaded=False, matrix_scope='Canonical stable resolvent',
        global_dependencies='Exact mass projection and dense metric retain every physical node'), indent=2)+'\n')


def probe(inputs, rows, output):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from examples.confined_ch2d.configs.train_operator import get_config
    from examples.confined_ch2d.operator_precision import make_compute_optimizer
    from examples.confined_ch2d.operator_training import build_network, make_loss, operator_temporal_samples
    from examples.confined_ch2d.operator_spatial import residual_rows
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    c = get_config()
    c.train.seed = 42
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    with np.load(inputs/'inputs.npz', allow_pickle=False) as saved:
        host = {key: saved[key] for key in saved.files}
    bank = jax.tree.map(jnp.asarray, host)
    items = [{key: value[i] for key, value in host.items()} for i in range(c.operator.condition_batch)]
    key_model, _, key_time = jax.random.split(jax.random.key(c.train.seed), 3)
    model = build_network(c, key_model)
    optimizer = make_compute_optimizer(c)
    state = optimizer.init(eqx.filter(model, eqx.is_array))
    loss = make_loss(c)
    @eqx.filter_jit
    def step(model, state, batch, times):
        value, gradient = eqx.filter_value_and_grad(loss)(model, batch, times)
        updates, state = optimizer.update(gradient, state, model)
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves((gradient, state))
                                   if eqx.is_inexact_array(a)]))
        return eqx.apply_updates(model, updates), state, value, finite
    rng = np.random.default_rng(20261112)
    timings, preparation, values, draws = [], [], [], []
    for i in range(1, 61):
        times, _ = operator_temporal_samples(c, i, jax.random.fold_in(key_time, i))
        tick = time.perf_counter()
        batch = bank
        if rows:
            sampled = [residual_rows(item, rows, rng) for item in items]
            ids = np.stack([d['residual_core_indices'] for d in sampled])
            batch = dict(bank, residual_core_indices=jnp.asarray(ids),
                residual_core_weights=jnp.asarray(np.stack([d['residual_core_weights'] for d in sampled])))
            jax.block_until_ready(batch)
            draws.append(ids)
        preparation.append(time.perf_counter()-tick)
        tick = time.perf_counter()
        model, state, value, finite = step(model, state, batch, times)
        value.block_until_ready()
        timings.append(time.perf_counter()-tick)
        if not bool(finite) or not np.isfinite(value):
            raise FloatingPointError(f'Nonfinite diagnostic update {i}')
        values.append(float(value))
    result = dict(status='passed', sampled_rows=rows, updates=60, physical_query_capacity=host['points'].shape[1],
        B=c.operator.condition_batch, Nt=c.sample.Nt, width=c.arch.width, optimizer=c.train.optimizer,
        warm_median_seconds=float(np.median(np.array(timings[10:])+np.array(preparation[10:]))),
        first_update_seconds=timings[0], peak_active_GiB=jax.devices()[0].memory_stats()['peak_bytes_in_use']/1024**3,
        scope='Fixed16known-input timing batch, fresh SOAP; no accuracy/generalization result or full rolling bank memory',
        input_directory=str(inputs), global_mass_and_dense_metric_unchanged=True,
        update_seconds=timings, preparation_seconds=preparation, loss=values)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if draws:
        np.savez_compressed(output.with_suffix('.npz'), core_ids=np.stack(draws))
    print(json.dumps({k: v for k, v in result.items() if k not in ('update_seconds', 'preparation_seconds', 'loss')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--rows', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.inputs)
    elif args.rows is not None and args.rows >= 0 and args.output is not None:
        probe(args.inputs, args.rows, args.output)
    else:
        parser.error('Timing requires nonnegative --rows and --output')
