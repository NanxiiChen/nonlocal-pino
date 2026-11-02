"""CPU evaluation of one saved snapshot on the entire reserved geometry cohort."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(run, owner, step, manifest, output, timeout_seconds):
    run, owner, manifest, output = map(Path, (run, owner, manifest, output))
    if not all(p.is_absolute() for p in (run, owner, manifest, output)):
        raise ValueError('All artifact paths must be absolute')
    config = json.loads((run/'config.json').read_text())
    if (step <= 0 or step > config['train']['iterations']
            or step % config['log']['checkpoint_every'] or timeout_seconds <= 0):
        raise ValueError('Require a scheduled snapshot and positive finite wait budget')
    if manifest != Path(config['operator']['validation_manifest']):
        raise ValueError('Use the full manifest frozen before training')
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = run/f'ckpt_{step:06d}.eqx'
    record = dict(status='waiting_for_snapshot_replay', pid=os.getpid(), run=str(run),
        owner=str(owner), checkpoint=str(checkpoint), step=step, manifest=str(manifest),
        output=str(output), started_utc=datetime.now(timezone.utc).isoformat(),
        timeout_seconds=timeout_seconds, stages=[])

    def write():
        temporary = output/'status.tmp'
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temporary.replace(output/'status.json')

    write()
    started = time.monotonic()
    try:
        while True:
            state = json.loads(owner.read_text())
            if Path(state['run']) != run:
                raise ValueError('Owner belongs to another training run')
            complete = False
            if (run/'summary.json').is_file():
                summary = json.loads((run/'summary.json').read_text())
                complete = summary['status'] == 'completed' and summary['iterations'] >= step
            replay = run/f'partial_{step:06d}.json'
            if replay.is_file():
                proof = json.loads(replay.read_text())
                if (proof['status'] != 'partial_checkpoint_metric_replay_passed'
                        or proof['step'] != step or Path(proof['checkpoint']) != checkpoint):
                    raise ValueError('Partial snapshot replay proof mismatch')
                complete = True
            if checkpoint.is_file() and complete:
                break
            if state['status'] != 'training':
                raise RuntimeError(f"Training ended before a validated snapshot: {state['status']}")
            if time.monotonic()-started >= timeout_seconds:
                raise TimeoutError('Snapshot replay did not arrive within the finite budget')
            time.sleep(30)
        checked = 0
        for source in (run/'source').rglob('*.py'):
            if source.read_bytes() != (ROOT/source.relative_to(run/'source')).read_bytes():
                raise ValueError(f'Frozen source differs: {source}')
            checked += 1
        if checked == 0:
            raise ValueError('Missing frozen source tree')
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < 8:
            raise RuntimeError('This postprocessor requires eight available CPU IDs')
        os.sched_setaffinity(0, cpus[-8:-4])
        os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg')
        record.update(status='cpu_evaluation', source_byte_checks=checked)
        write()
        with (output/'evaluation.log').open('x') as log:
            command = [sys.executable, __file__, '--evaluate', '--run', str(run),
                '--owner', str(owner), '--step', str(step), '--manifest', str(manifest),
                '--output', str(output), '--timeout-seconds', str(timeout_seconds)]
            child = subprocess.Popen(command, cwd=ROOT, env=os.environ, stdout=log,
                stderr=subprocess.STDOUT)
            record['stages'].append(dict(stage='reserved_cpu_evaluation', pid=child.pid))
            write()
            code = child.wait()
            if code:
                raise RuntimeError(f'CPU cohort evaluation exited {code}; see evaluation.log')
        record.update(status='snapshot_cohort_completed',
            completed_utc=datetime.now(timezone.utc).isoformat())
        write()
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        write()
        raise


def evaluate(run, step, manifest, output):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.cohort import load
    from examples.channel_nsch2d.cohort_evaluate import field_metrics
    from examples.channel_nsch2d.data import geometry_data
    from examples.channel_nsch2d.geometry import with_nominal
    from examples.channel_nsch2d.network import build_network, encode, fields
    from examples.channel_nsch2d.problem import identity, walls, initial_field, flow_rate
    from examples.channel_nsch2d.response_queries import validate_arrays, protocol
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    from docs.tasks.channel_droplet_metrics import evaluate as droplet_metrics
    from examples.channel_nsch2d.evaluate import plot_fields

    run, manifest, output = map(Path, (run, manifest, output))
    if os.environ['JAX_PLATFORMS'] != 'cpu' or jax.default_backend() != 'cpu':
        raise RuntimeError('Snapshot evaluation must leave the training GPU untouched')
    c = ConfigDict(json.loads((run/'config.json').read_text()))
    cohort = load(manifest, c)
    group_gate = json.loads((manifest.parent/'cohort-reference-validation.json').read_text())
    if (group_gate['status'] != 'passed' or group_gate['cases'] != cohort['count']
            or group_gate['common_query_protocol'] != protocol(c, cohort['query_grid'])):
        raise ValueError('Every frozen cohort reference must pass its original qualification')
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64', c.train.compute_dtype == 'float64')
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    dtype = np.dtype(c.train.compute_dtype)
    key, _ = jax.random.split(jax.random.PRNGKey(c.train.seed))
    checkpoint = run/f'ckpt_{step:06d}.eqx'
    net = eqx.tree_deserialise_leaves(checkpoint, build_network(c, key))
    predict = eqx.filter_jit(lambda model, data, points, times:
        fields(model, encode(model, data), points, times, data['amplitudes'], c))
    results = []
    for case in cohort['cases']:
        parameters = np.asarray(case['amplitudes'])
        gate = json.loads((Path(case['root'])/'reference-validation.json').read_text())
        cc = with_nominal(c, parameters)
        if gate['status'] != 'passed' or gate['identity'] != json.loads(json.dumps(identity(cc))):
            raise ValueError(f"Reference identity/qualification mismatch: {case['index']}")
        with np.load(gate['reference'], allow_pickle=False) as f:
            reference = {k: f[k] for k in f.files}
        if (json.loads(str(reference['identity_json'])) != gate['identity']
                or validate_arrays(reference, c) != group_gate['common_query_protocol']
                or gate['common_queries']['status'] != 'passed'):
            raise ValueError('Reference payload/common-query qualification mismatch')
        np.testing.assert_array_equal(reference['amplitudes'], parameters)
        np.testing.assert_array_equal(reference['time'],
            np.linspace(0, c.physical.terminal_time, c.reference.snapshots))
        expected = (c.reference.snapshots, int(np.prod(cohort['query_grid'])), 5)
        if reference['fields'].shape != expected or not np.isfinite(reference['fields']).all():
            raise ValueError('Finite reference fields on the declared query grid required')
        raw, _ = geometry_data(parameters, c)
        data = {k: jnp.asarray(v, dtype=v.dtype if np.issubdtype(v.dtype, np.integer) else dtype)
                for k, v in raw.items()}
        roots, weights = np.polynomial.legendre.leggauss(64)
        sx, eta = np.meshgrid(np.linspace(0, c.geometry.length, 5), (roots+1)/2, indexing='ij')
        sb, st = walls(sx, parameters, c, xp=np)
        sections = np.stack((sx, sb+eta*(st-sb)), axis=-1).reshape(-1, 2)
        points = np.concatenate((reference['points'], raw['boundary'][:, :2], sections))
        values = np.asarray(predict(net, data, jnp.asarray(points, dtype=dtype),
            jnp.asarray(reference['time'], dtype=dtype)), dtype=np.float64)
        if values.shape != (c.reference.snapshots, len(points), 5) or not np.isfinite(values).all():
            raise FloatingPointError('Finite aligned five-field CPU predictions required')
        n, nb = len(reference['points']), len(raw['boundary'])
        prediction, boundary = values[:, :n], values[:, n:n+nb]
        flux = np.sum(values[:, n+nb:, 2].reshape(len(values), 5, 64)
            * (st-sb)[None]*weights[None, None]/2, axis=-1)
        gaps = dict(initial_phi=float(np.max(abs(prediction[0, :, 0]
                - initial_field(reference['points'], c, xp=np)))),
            initial_velocity=float(np.max(abs(prediction[0, :, 2:4]))),
            wall_velocity=float(np.max(abs(boundary[:, raw['boundary'][:, 4] == 0, 2:4]))),
            section_flux=float(np.max(abs(flux-flow_rate(reference['time'], c, xp=np)[:, None]))))
        if max(gaps.values()) > 2e-5:
            raise ValueError(f"Snapshot hard-constraint check failed: {case['index']}: {gaps}")
        bottom, top = walls(reference['points'][:, 0], parameters, c, xp=np)
        metrics = field_metrics(prediction, reference['fields'], top-bottom)
        folder = output/f"case_{case['index']:03d}"
        folder.mkdir()
        (folder/'config.json').write_text(cc.to_json(indent=2)+'\n')
        payload = folder/'prediction.npz'
        np.savez_compressed(payload, fields=prediction, reference=reference['fields'],
            points=reference['points'], time=reference['time'], amplitudes=parameters)
        shape = droplet_metrics(payload, folder/'droplet-shape')
        plot_fields(payload, folder/'phi-fields', selected_fields=['phi'],
            selected_times=[0., 1.5, 3., 4.5, 6.])
        row = dict(index=case['index'], parameters=parameters.tolist(), metrics=metrics,
            hard_constraint_max_gaps=gaps, terminal_shape=shape['rows'][-1],
            reference=gate['reference'], prediction=str(payload))
        results.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    aggregate = {k: dict(mean=float(np.mean([r['metrics'][k] for r in results])),
        worst=float(np.max([r['metrics'][k] for r in results]))) for k in results[0]['metrics']}
    report = dict(status='saved_snapshot_reserved_evaluation_completed', run=str(run),
        checkpoint=str(checkpoint), step=step, geometries=len(results),
        query_grid=cohort['query_grid'], snapshots=c.reference.snapshots,
        backend=jax.default_backend(), aggregate=aggregate, per_geometry=results,
        scope='Entire reused reserved cohort; partial model evaluation, final online-stream audit pending')
    if step == c.train.iterations and (run/'unseen3/evaluation.json').is_file():
        original = json.loads((run/'unseen3/evaluation.json').read_text())
        gaps = [abs(row['metrics'][k]-old['metrics'][k])
            for row, old in zip(results, original['per_geometry'], strict=True)
            for k in row['metrics']]
        for row, old in zip(results, original['per_geometry'], strict=True):
            if row['index'] != old['index']:
                raise ValueError('Saved cohort ordering differs')
        if max(gaps) > 2e-6:
            raise ValueError(f'CPU metrics disagree with actual saved cohort: {max(gaps)}')
        report['saved_cohort_maximum_metric_gap'] = max(gaps)
    (output/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], aggregate=aggregate), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout-seconds', type=int, required=True)
    parser.add_argument('--evaluate', action='store_true', help='Internal CPU child of the finite owner')
    args = parser.parse_args()
    if args.evaluate:
        evaluate(args.run, args.step, args.manifest, args.output)
    else:
        main(args.run, args.owner, args.step, args.manifest, args.output, args.timeout_seconds)
