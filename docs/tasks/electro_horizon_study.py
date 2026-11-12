"""Finite fixed-thickness horizon/grid study; direct paths, final weights, common FD64.

Run prepare on CPU, probe largest on GPU, then launch the finite run queue.
Training always enters examples/electropolish2d/train.py. No reference labels
enter updates. All subprocess commands, timings and failures are journaled.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO/'.venv/bin/python'
BASE = Path('/root/autodl-tmp/data/electropolish2d/operator_test64_20261028')
LOCAL = Path('/root/autodl-tmp/data/electropolish2d/operator_local_test64_20261029')
ARMS = [('n101-d020', 101, .02), ('n101-d015', 101, .015),
        ('n101-d030', 101, .03), ('n101-d040', 101, .04),
        ('n151-d020', 151, .02), ('n129-d020', 129, .02),
        ('n051-d040', 51, .04), ('n151-ratio2', 151, 2/150), ('n201-d010', 201, .01)]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def prepare(root):
    import jax
    import jax.numpy as jnp
    from ml_collections import ConfigDict
    from examples.electropolish2d.configs.train_operator import get_config
    from examples.corrosion2d.residual_experiment import grid
    from examples.electropolish2d.horizon_study import load_monitor, operator_audit
    from nonlocal_pinn.operators import build_D
    if jax.default_backend() != 'cpu' or jax.config.x64_enabled:
        raise ValueError('Preparation requires CPU with JAX_ENABLE_X64=false, matching training graph dtype')
    root.mkdir(parents=True, exist_ok=False)
    (root/'configs').mkdir(); (root/'logs').mkdir()
    baseline = get_config()
    records = []
    for name, nx, delta in ARMS:
        c = baseline.copy_and_resolve_references()
        c.sample.Nx, c.sample.Ny, c.nl.delta = nx, (nx+1)//2, delta
        c.operator.horizon_study = ConfigDict(dict(monitor_Nx=101, monitor_Ny=51, monitor_delta=.02))
        c.train.seed = 42
        c.log.log_dir, c.log.run_name = str(root/'runs'), name
        c.log.eval_every, c.log.log_every, c.log.checkpoint_step = 1000, 100, 0
        _, _, points = grid(c)
        op = build_D(jnp.asarray(points, dtype=jnp.float32), delta, method='sparse', kernel=c.nl.kernel)
        audit = operator_audit(op, points.astype(np.float32), delta, c.sample.K_core)
        c.sample.spatial.max_nodes = audit['max_nodes']
        c.sample.spatial.max_edges = audit['max_edges']
        if name == ARMS[0][0]:
            load_monitor(c)
        config_path = root/'configs'/f'{name}.json'
        write(config_path, json.loads(c.to_json()))
        records.append(dict(name=name, config=str(config_path), run=str(root/'runs'/name), audit=audit))
        print(name, json.dumps(audit), flush=True)
    probe = read(root/'configs/n151-d020.json')
    probe['train']['iterations'] = 50
    probe['log'].update(run_name='probe-n151-d020', eval_every=50, log_every=10)
    write(root/'configs/probe.json', probe)
    device = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'], text=True)
    write(root/'manifest.json', dict(created=datetime.now(timezone.utc).isoformat(), arms=records,
        device=device, steps=10000, seed=42, probe_config=str(root/'configs/probe.json'),
        allocator_fraction=.95, preallocate=False, pdf_audit_packages='/tmp/pino-pdf-audit',
        phase_thickness='unchanged physical and nondimensional coefficients',
        core_count=256, reference=str(LOCAL), inputs=str(BASE), count=64,
        evaluation_times=np.linspace(0, 1, 21).tolist(),
        metric='per-channel unweighted space-time relative L2, then channel mean, then64sample mean/std(ddof=1)',
        checkpoint='last step, no selection',
        groups={'fixed_h': [r[0] for r in ARMS[:4]],
                'fixed_delta': ['n101-d020', 'n129-d020', 'n151-d020'],
                'fixed_ratio': ['n051-d040', 'n101-d020', 'n151-ratio2', 'n201-d010']}))


def command(root, label, argv):
    tick = time.perf_counter()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_PYTHON_CLIENT_MEM_FRACTION='.95', JAX_ENABLE_X64='false')
    env.pop('JAX_PLATFORMS', None)
    with (root/'logs'/f'{label}.log').open('x') as log:
        process = subprocess.Popen(argv, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        with (root/'journal.jsonl').open('a') as journal:
            journal.write(json.dumps(dict(event='start', label=label, pid=process.pid, argv=argv,
                                          at=datetime.now(timezone.utc).isoformat()))+'\n')
        status = process.wait()
    with (root/'journal.jsonl').open('a') as journal:
        journal.write(json.dumps(dict(event='end', label=label, exit_code=status, seconds=time.perf_counter()-tick,
                                      at=datetime.now(timezone.utc).isoformat()))+'\n')
    if status:
        raise RuntimeError(f'{label} exited{status}; see{root}/logs/{label}.log; queue stopped')


def train_command(config):
    return [str(PYTHON), 'examples/electropolish2d/train.py',
            f'--config=examples/electropolish2d/configs/train_operator_horizon.py:{config}', '--seed=42']


def queue(root, probe):
    probe_config = Path(read(root/'manifest.json')['probe_config'])
    if probe:
        command(root, 'probe', train_command(probe_config))
        return
    probe_run = read(probe_config)['log']['run_name']
    if not (root/'runs'/probe_run/'summary.json').is_file():
        raise ValueError('Complete and inspect the largest-graph probe before the full queue')
    for item in read(root/'manifest.json')['arms']:
        command(root, item['name']+'-train', train_command(item['config']))
        command(root, item['name']+'-eval', [str(PYTHON), str(Path(__file__).resolve()),
                'evaluate', '--root', str(root), '--run', item['run']])
    command(root, 'audit-and-plot', [str(PYTHON), 'scripts/render_electro_horizon_study.py', '--root', str(root)])
    write(root/'complete.json', dict(status='complete', at=datetime.now(timezone.utc).isoformat()))


def evaluate(root, run):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from ml_collections import ConfigDict
    from examples.electropolish2d.operator_training import _build_network, predict_fields
    from examples.electropolish2d.horizon_study import channel_relative_errors
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    spec, config, summary = read(root/'manifest.json'), read(run/'config.json'), read(run/'summary.json')
    if summary['iterations'] != 10000 or summary['training_uses_positive_time_labels']:
        raise ValueError('Only complete cold10000 physics-only runs may be evaluated')
    if (run/'evaluation64.json').exists():
        raise FileExistsError(run/'evaluation64.json')
    manifest = read(BASE/'manifest.json')
    source = read(manifest['source_config'])
    for section in ('physical', 'nondim', 'surface'):
        if config[section] != source[section]:
            raise ValueError('Reference physics/input mismatch: '+section)
    for name in ('src/nonlocal_pinn/nn/conditional.py', 'examples/electropolish2d/operator_training.py',
                 'examples/electropolish2d/losses.py', 'examples/electropolish2d/horizon_study.py'):
        if (run/'source'/name).read_bytes() != (REPO/name).read_bytes():
            raise ValueError('Inference source mismatch: '+name)
    c = ConfigDict(config)
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    with np.load(BASE/'prediction.npz') as data:
        images, initial, xx, yy = (data[k] for k in ('images', 'initial', 'X', 'Y'))
    if images.shape != (64, *c.operator.image_shape) or initial.shape != (64, 5151, 2) or xx.shape != (51, 101):
        raise ValueError('Common64query/encoder arrays have unexpected shapes')
    times = np.asarray(read(LOCAL/'manifest.json')['times'])
    indices = np.searchsorted(times, spec['evaluation_times'])
    np.testing.assert_array_equal(times[indices], spec['evaluation_times'])
    points = jnp.asarray(np.column_stack([xx.ravel(), yy.ravel()]), dtype=jnp.float32)
    ts = jnp.asarray(times[indices], dtype=jnp.float32)
    template = _build_network(c.arch, c.operator.image_shape, jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(run/'model_final.eqx', template)
    predictor = eqx.filter_jit(lambda net, image, known: predict_fields(net, image, points, ts, known, c))
    records, predictions = [], []
    tick = time.perf_counter()
    for i, identity in enumerate(manifest['surface_ids']):
        reference = LOCAL/f'condition_{i:03d}/local101.npz'
        with np.load(reference) as ref:
            np.testing.assert_array_equal(ref['time'], times)
            np.testing.assert_array_equal(ref['X'], xx)
            np.testing.assert_array_equal(ref['Y'], yy)
            target = ref['fields'][indices]
        np.testing.assert_allclose(target[0].reshape(-1, 2), initial[i], rtol=0, atol=3e-7)
        p = np.asarray(predictor(model, jnp.asarray(images[i]), jnp.asarray(initial[i]))).reshape(len(ts), *xx.shape, 2)
        scores = channel_relative_errors(p, target)
        row = dict(index=i, surface_id=identity, reference=str(reference), **scores)
        # Refined local references diagnose target error separately, on the same points/times.
        if i in read(LOCAL/'manifest.json')['refinement_indices']:
            row['refinement'] = {}
            for nx in (201, 401):
                with np.load(LOCAL/f'condition_{i:03d}/local{nx}.npz') as ref:
                    stride = (nx-1)//100
                    refined = ref['fields'][indices, ::stride, ::stride]
                row['refinement'][str(nx)] = channel_relative_errors(p, refined)
        records.append(row); predictions.append(p)
        print('EVAL', run.name, i, scores['channel_mean'], flush=True)
    if len(records) != 64:
        raise ValueError('Expected64reference samples')
    statistics = {k: dict(mean=float(np.mean([r[k] for r in records])),
                         std=float(np.std([r[k] for r in records], ddof=1)))
                  for k in ('channel_mean', 'phi', 'c')}
    np.savez_compressed(run/'evaluation64.npz', prediction=np.stack(predictions), time=times[indices],
                        X=xx, Y=yy, surface_ids=manifest['surface_ids'])
    write(run/'evaluation64.json', dict(run=str(run), checkpoint=str(run/'model_final.eqx'),
        steps=10000, count=64, ddof=1, metric=spec['metric'], time_indices=indices.tolist(),
        statistics=statistics, records=records, seconds=time.perf_counter()-tick))
    print('EVALUATION_COMPLETE', json.dumps(statistics), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'probe', 'run', 'evaluate'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root)
    elif args.action in ('probe', 'run'):
        queue(args.root, args.action == 'probe')
    else:
        if args.run is None:
            parser.error('evaluate requires --run')
        evaluate(args.root, args.run)
