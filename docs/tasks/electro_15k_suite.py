"""Finite cold15k Electropolish library/time-fusion suite; no automatic retries."""
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from docs.tasks.electro_library_extension import environment, gpu, read, write

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('fixed64', 'fixed128', 'fixed256', 'fixed512', 'joint_query')
STEPS = list(range(0, 15001, 250))


def prepare(dependency, bundle):
    old = read(dependency / 'journal.json')
    if old['execution'] != 'runtime_serial' or old['iterations'] != 15000:
        raise ValueError('Require the original capture-disabled15k recovery pair')
    if gpu() != old['gpu']:
        raise ValueError('The temporal baseline GPU/driver must be unchanged')
    bundle.mkdir(parents=True, exist_ok=False)
    source = bundle / 'source'
    shutil.copytree(old['source'], source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('docs/tasks/electro_15k_suite.py', 'docs/tasks/pino_time_fusion_support.py'):
        shutil.copyfile(ROOT / name, source / name)
    shutil.copytree(ROOT / 'scripts', source / 'scripts',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    runs, configs = {}, {}
    for arm in ARMS:
        name = bundle.name + '-' + arm
        run = Path(old['runs']['online']).parent / name
        if run.exists():
            raise FileExistsError(run)
        module = 'suite_' + arm
        path = source / 'study_configs' / (module + '.py')
        content = ('from ml_collections import ConfigDict\n'
                   'from examples.electropolish2d.configs.train_operator_threeway import get_config as baseline\n'
                   'def get_config():\n    c = baseline()\n')
        if arm.startswith('fixed'):
            content += ('    c.operator.condition_library = ConfigDict(dict('
                        f'path={old["library"]!r},count={int(arm[5:])},'
                        "sampling='uniform_with_replacement'))\n")
        else:
            content += "    c.arch.temporal_query = 'joint_query'\n"
        content += (f'    c.train.iterations = 15000\n    c.log.checkpoint_step = 10000\n'
                    f'    c.log.run_name = {name!r}\n    return c\n')
        path.write_text(content)
        runs[arm], configs[arm] = str(run), str(path)
    state = dict(status='prepared', dependency=str(dependency), bundle=str(bundle), source=str(source),
                 baseline=old['runs']['online'], fixed32=old['runs']['fixed32'],
                 library=old['library'], gpu=old['gpu'], runs=runs, configs=configs,
                 training_order=list(ARMS), iterations=15000, checkpoint_step=10000,
                 seed=42, stages=[], automatic_retries=False, concurrent_training=False,
                 xla_flags='--xla_gpu_enable_command_buffer=', memory_fraction=.40,
                 postprocessing='Independent audits, then figures; visual review before publication')
    write(bundle / 'journal.json', state)
    print('PREPARED', bundle, flush=True)


def preflight(bundle):
    from examples.electropolish2d.condition_library import load_library
    from nonlocal_pinn.training.runtime import with_run_seed
    state = read(bundle / 'journal.json')
    baseline = read(Path(state['baseline']) / 'config.json')
    source = Path(state['source'])
    previous_source = Path(read(Path(state['dependency']) / 'journal.json')['source'])
    files = [p for top in ('examples', 'src') for p in (source / top).rglob('*.py')]
    for path in files:
        if path.read_bytes() != (previous_source / path.relative_to(source)).read_bytes():
            raise ValueError('Numerical source changed: ' + str(path))
    for arm in ARMS:
        c = with_run_seed(importlib.import_module('study_configs.suite_' + arm).get_config(), 42)
        actual = json.loads(c.to_json())
        expected = json.loads(json.dumps(baseline))
        expected['log']['run_name'] = Path(state['runs'][arm]).name
        if arm.startswith('fixed'):
            expected['operator']['condition_library'] = dict(
                path=state['library'], count=int(arm[5:]), sampling='uniform_with_replacement')
            coefficients, _ = load_library(c)
            with np.load(Path(state['library']) / 'coefficients.npz') as data:
                np.testing.assert_array_equal(coefficients, data['coefficients'][:int(arm[5:])])
        else:
            expected['arch']['temporal_query'] = 'joint_query'
        if actual != expected:
            raise ValueError('Undeclared control difference: ' + arm)
    write(bundle / 'preflight.json', dict(status='passed', source_files_identical=len(files),
          arms=list(ARMS), iterations=15000, all64_monitor=True, nested_original_prefixes=True))


def field_audit(run, step):
    from scripts.pino_comparison_data import norm
    suffix = '' if step == 15000 else f'_step_{step:06d}'
    evaluated = read(run / f'spatial_evaluation64{suffix}.json')
    if evaluated['status'] != 'passed' or evaluated['count'] != 64 or evaluated['iterations'] != step:
        raise ValueError('Require independently loaded64case checkpoint evaluation')
    targets = Path(evaluated['local_reference'])
    take = evaluated['regular_time_indices']
    values, norms = [], []
    with np.load(run / f'spatial_evaluation64{suffix}.npz') as data:
        np.testing.assert_array_equal(data['time'][take], np.linspace(0, 1, 21))
        if data['prediction'].shape[0] != 64:
            raise ValueError('Require all64physical predictions')
        for index, prediction in enumerate(data['prediction']):
            target_path = targets / f'condition_{index:03d}/local101.npz'
            with np.load(target_path) as target:
                for key in ('time', 'X', 'Y'):
                    np.testing.assert_array_equal(data[key], target[key])
                channel = [norm(prediction[take, ..., k], target['fields'][take, ..., k]) for k in range(2)]
            values.append([x['error'] for x in channel])
            norms.append(dict(index=index, reference=str(target_path), channels=channel))
    scores = np.asarray(values)
    return dict(step=step, sample_count=64, channel_errors=values,
                sample_errors=scores.mean(1).tolist(), channel_means=scores.mean(0).tolist(),
                mean=float(scores.mean()), sample_std=float(scores.mean(1).std(ddof=1)),
                reference=str(targets), norms=norms)


def audit(bundle, arm):
    from scripts.pino_comparison_data import history
    from docs.tasks.pino_time_fusion_support import initialized_backbone
    state = read(bundle / 'journal.json')
    run, baseline = Path(state['runs'][arm]), Path(state['baseline'])
    config, original = read(run / 'config.json'), read(baseline / 'config.json')
    expected = json.loads(json.dumps(original))
    expected['log']['run_name'] = config['log']['run_name']
    if arm == 'joint_query':
        expected['arch']['temporal_query'] = 'joint_query'
    else:
        expected['operator']['condition_library'] = dict(
            path=state['library'], count=int(arm[5:]), sampling='uniform_with_replacement')
    if config != expected or read(run / 'summary.json')['iterations'] != 15000:
        raise ValueError('Incomplete or changed fixed-budget control')
    if read(run / 'local_monitor.json') != read(baseline / 'local_monitor.json'):
        raise ValueError('Monitoring inputs/reference/cohort changed')
    pairing = dict(all_actual_sampling_arrays_equal=True, source_bytes_verified=True)
    if arm == 'joint_query':
        pairing.update(initialized_backbone('electropolish', [original, config], [baseline, run]))
    else:
        for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
            if (run / name).read_bytes() != (baseline / name).read_bytes():
                raise ValueError('Initial state changed: ' + name)
        pairing['complete_initial_states_equal'] = True
    with np.load(run / 'sampling.npz') as a, np.load(baseline / 'sampling.npz') as b:
        if a.files != b.files:
            raise ValueError('Sampling schema changed')
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        if len(a['times']) != 15000:
            raise ValueError('Incomplete actual sampling')
    for candidate in (run, baseline):
        for name in read(candidate / 'source/files.json'):
            if (candidate / 'source' / name).read_bytes() != (Path(state['source']) / name).read_bytes():
                raise ValueError('Frozen numerical source changed: ' + name)
    coefficients = np.load(run / 'online_coefficients.npy')
    if coefficients.shape != (15000, 8, 6, 3) or not np.isfinite(coefficients).all():
        raise ValueError('Incomplete actual initial-condition draws')
    if np.any(np.load(run / 'online_accepted.npy') < 8):
        raise ValueError('Insufficient accepted profiles')
    unique = np.unique(coefficients.reshape(-1, 18), axis=0)
    if arm == 'joint_query':
        for name in ('online_coefficients.npy', 'online_accepted.npy'):
            np.testing.assert_array_equal(np.load(run / name), np.load(baseline / name))
        with np.load(run / 'paired_inputs.npz') as a, np.load(baseline / 'paired_inputs.npz') as b:
            if a.files != b.files:
                raise ValueError('Prescribed-input witness schema changed')
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        if len(unique) != 120000:
            raise ValueError('Online draws should be120000distinct profiles')
    else:
        with np.load(Path(state['library']) / 'coefficients.npz') as data:
            allowed = data['coefficients'][:int(arm[5:])].reshape(-1, 18)
        np.testing.assert_array_equal(unique, np.unique(allowed, axis=0))
    curves = np.asarray(history(run / 'local_metrics.jsonl', STEPS, list(range(64)), ('phi', 'c'), ('phi', 'c')))
    final, intermediate = field_audit(run, 15000), field_audit(run, 10000)
    for data, index in ((final, -1), (intermediate, STEPS.index(10000))):
        np.testing.assert_allclose(data['channel_errors'], curves[index], rtol=3e-6, atol=2e-8)
    spatial = read(run / 'spatial_sampling.json')
    if spatial['mode'] != 'rolling_pool' or spatial['entries_used'] != 15000:
        raise ValueError('Spatial sampling did not retain rolling behavior')
    result = dict(status='passed', arm=arm, run=str(run), baseline=str(baseline),
                  configuration=str(run / 'config.json'), checkpoint=str(run / 'model_final.eqx'),
                  pairing=pairing, unique_training_profiles=len(unique), steps=15000,
                  history=curves.tolist(), final=final, intermediate=intermediate,
                  metric='Equal-point full-space-time L2 per physical channel, then channel/sample mean')
    write(run / 'suite15k_audit.json', result)
    print('AUDITED', arm, final['mean'], flush=True)


def run(bundle):
    state = read(bundle / 'journal.json')
    if state['status'] != 'prepared' or read(bundle / 'preflight.json')['status'] != 'passed':
        raise ValueError('Exactly one prepared and qualified owner is required')
    state.update(status='waiting_dependency', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat())
    write(bundle / 'journal.json', state)
    dependency = Path(state['dependency'])
    while True:
        previous = read(dependency / 'journal.json')
        if previous['status'] == 'complete':
            for path in previous['runs'].values():
                if read(Path(path) / 'library_extension_audit.json')['status'] != 'passed':
                    raise ValueError('Dependency final audits did not pass')
            break
        if previous['status'] == 'failed':
            state.update(status='blocked_dependency', error='Dependency failed; no experiments started')
            write(bundle / 'journal.json', state)
            raise RuntimeError(state['error'])
        if previous['status'] != 'running' or not Path(f'/proc/{previous["pid"]}').exists():
            raise RuntimeError('Dependency is not an active expected owner')
        time.sleep(60)
    if gpu() != state['gpu']:
        raise RuntimeError('Current GPU/driver differs from the FiLM baseline')
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if busy:
        raise RuntimeError('GPU is occupied after dependency completion: ' + busy)
    source = Path(state['source'])
    child = None

    def stage(name, args, platform):
        nonlocal child
        begin = time.monotonic_ns()
        with (bundle / (name + '.log')).open('x') as stream:
            child = subprocess.Popen([sys.executable, *args], cwd=source, stdout=stream,
                                     stderr=subprocess.STDOUT,
                                     env=environment(source, platform, state['xla_flags']))
            row = dict(name=name, args=args, status='running', pid=child.pid,
                       started_utc=datetime.now(timezone.utc).isoformat(), xla_flags=state['xla_flags'])
            state['stages'].append(row)
            write(bundle / 'journal.json', state)
            code = child.wait()
        row.update(status='passed' if code == 0 else 'failed', exit_code=code,
                   seconds=(time.monotonic_ns()-begin)/1e9)
        write(bundle / 'journal.json', state)
        if code:
            raise RuntimeError(name + ' failed; no automatic retry')

    try:
        state['status'] = 'running'
        write(bundle / 'journal.json', state)
        for arm in state['training_order']:
            stage('train-' + arm, ['-m', 'examples.electropolish2d.train',
                  '--config=' + state['configs'][arm], '--seed=42'], 'cuda')
            for step in (10000, 15000):
                suffix = ['--step', str(step)] if step == 10000 else []
                stage(f'evaluate-{arm}-{step}', ['-m', 'docs.tasks.electro_spatial_evaluate',
                      '--run', state['runs'][arm], *suffix], 'cuda')
            stage('audit-' + arm, ['-m', 'docs.tasks.electro_15k_suite', '--action', 'audit',
                  '--bundle', str(bundle), '--arm', arm], 'cpu')
        state.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
        write(bundle / 'journal.json', state)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(bundle / 'journal.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('prepare', 'preflight', 'run', 'audit'), required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--dependency', type=Path)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.dependency is None:
            parser.error('prepare requires --dependency')
        prepare(args.dependency.resolve(strict=True), args.bundle.resolve())
    elif args.action == 'audit':
        if args.arm is None:
            parser.error('audit requires --arm')
        audit(args.bundle.resolve(strict=True), args.arm)
    else:
        {'preflight': preflight, 'run': run}[args.action](args.bundle.resolve(strict=True))
