"""Two cold Electropolish controls using the completed library study's source."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def gpu():
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version',
        '--format=csv,noheader'], text=True).strip()


def environment(source, platform):
    return dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                MKL_NUM_THREADS='1', JAX_ENABLE_X64='false',
                JAX_ENABLE_COMPILATION_CACHE='false', XLA_FLAGS='',
                XLA_PYTHON_CLIENT_PREALLOCATE='true', XLA_PYTHON_CLIENT_MEM_FRACTION='.40',
                JAX_PLATFORMS=platform, PYTHONUNBUFFERED='1',
                PYTHONPATH=f'{source}:{source / "src"}:/tmp/pino-pdf-audit')


def prepare(previous, bundle, prefix, iterations, checkpoint_step):
    old = read(previous / 'journal.json')
    if old['status'] != 'complete':
        raise ValueError('The preceding experiment must be complete')
    bundle.mkdir(parents=True, exist_ok=False)
    source = bundle / 'source'
    shutil.copytree(previous / 'source', source,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(__file__, source / 'docs/tasks/electro_library_extension.py')
    first = Path(old['runs']['electropolish'][0])
    library = Path(read(first / 'config.json')['operator']['condition_library']['path'])
    runs = {arm: str(first.parent / f'{prefix}-{arm}') for arm in ('fixed32', 'online')}
    configs = {}
    for arm in runs:
        if Path(runs[arm]).exists():
            raise FileExistsError(runs[arm])
        path = source / 'study_configs' / f'extension_{arm}.py'
        text = ('from ml_collections import ConfigDict\n'
                'from examples.electropolish2d.configs.train_operator_threeway import get_config as baseline\n'
                'def get_config():\n    c=baseline()\n')
        if arm == 'fixed32':
            text += (f'    c.operator.condition_library=ConfigDict(dict(path={str(library)!r},'
                     "count=32,sampling='uniform_with_replacement'))\n")
        text += f'    c.train.iterations={iterations}\n    c.log.checkpoint_step={checkpoint_step}\n'
        text += f'    c.log.run_name={Path(runs[arm]).name!r}\n    return c\n'
        path.write_text(text)
        configs[arm] = str(path)
    state = dict(status='prepared', previous=str(previous), bundle=str(bundle),
                 source=str(source), runs=runs, configs=configs, first=str(first),
                 old_online='/root/autodl-tmp/tf-logs/electropolish2d/electro-threeway-20261114-03-nonlocal',
                 library=str(library), gpu=gpu(), previous_gpu=old['gpu'],
                 iterations=iterations, checkpoint_step=checkpoint_step, seed=42, concurrent_training=True,
                 memory_fraction_per_process=.40, stages=[], automatic_retries=False)
    write(bundle / 'journal.json', state)
    print(json.dumps(dict(status='prepared', runs=runs, gpu=state['gpu'])), flush=True)


def preflight(bundle):
    import importlib
    from examples.electropolish2d.condition_library import load_library
    from nonlocal_pinn.training.runtime import with_run_seed
    state = read(bundle / 'journal.json')
    previous = read(Path(state['first']) / 'config.json')
    for arm in ('fixed32', 'online'):
        c = with_run_seed(importlib.import_module(f'study_configs.extension_{arm}').get_config(), state['seed'])
        actual = json.loads(c.to_json())
        expected = json.loads(json.dumps(previous))
        expected['log']['run_name'] = Path(state['runs'][arm]).name
        expected['train']['iterations'] = state['iterations']
        expected['log']['checkpoint_step'] = state['checkpoint_step']
        if arm == 'fixed32':
            expected['operator']['condition_library']['count'] = 32
            coefficients, _ = load_library(c)
            with np.load(Path(state['library']) / 'coefficients.npz') as data:
                np.testing.assert_array_equal(coefficients, data['coefficients'][:32])
        else:
            del expected['operator']['condition_library']
        if actual != expected or actual['train']['iterations'] != state['iterations']:
            raise ValueError('Unexpected configuration difference: ' + arm)
        if actual['operator']['comparison_monitor']['indices'] != list(range(64)):
            raise ValueError('All64test conditions must be monitored')
    write(bundle / 'preflight.json', dict(status='passed', fixed32_is_original_prefix=True,
          common_configuration=True, test_count=64, iterations=state['iterations'], source=state['source']))


def audit_arm(bundle, arm):
    from docs.tasks.fixed_library_audit import norms
    state = read(bundle / 'journal.json')
    run = Path(state['runs'][arm])
    first = Path(state['runs']['fixed32'])
    c = read(run / 'config.json')
    other = read(first / 'config.json')
    if arm == 'online':
        del other['operator']['condition_library']
    other['log']['run_name'] = c['log']['run_name']
    budget = state['iterations']
    if c != other or read(run / 'summary.json')['iterations'] != budget:
        raise ValueError('Unexpected online comparison configuration or budget')
    for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
        if ((run / name).read_bytes() != (first / name).read_bytes()
                or (run / name).read_bytes() != (Path(state['first']) / name).read_bytes()):
            raise ValueError('Unpaired initial state: ' + name)
    with np.load(first / 'sampling.npz') as a, np.load(run / 'sampling.npz') as b:
        if a.files != b.files:
            raise ValueError('Different recorded spatial/time fields')
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    with np.load(Path(state['first']) / 'sampling.npz') as a, np.load(run / 'sampling.npz') as b:
        dynamic = ('times', 'draw_ids', 'entry_draw_ids', 'core_ids', 'core_mask', 'nonlocal_halo_counts')
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key][:10000] if key in dynamic else b[key])
    coefficients = np.load(run / 'online_coefficients.npy')
    if coefficients.shape != (budget, 8, 6, 3) or not np.isfinite(coefficients).all():
        raise ValueError('Incomplete online draws')
    unique = np.unique(coefficients.reshape(-1, 18), axis=0)
    if np.any(np.load(run / 'online_accepted.npy') < 8):
        raise ValueError('Insufficient accepted profiles')
    if arm == 'online':
        if len(unique) != budget * 8:
            raise ValueError('Online profiles are not distinct')
        old = np.load(Path(state['old_online']) / 'online_coefficients.npy')
        inputs = dict(historical_online_coefficients_exact=bool(np.array_equal(coefficients[:10000], old)),
                      historical_online_coefficient_max_gap=float(np.max(abs(coefficients[:10000]-old))))
    else:
        from ml_collections import ConfigDict
        from examples.electropolish2d.condition_library import load_library
        library, _ = load_library(ConfigDict(c))
        np.testing.assert_array_equal(unique, np.unique(library.reshape(32, 18), axis=0))
        inputs = dict(actual_draws_restricted_to_original32=True)
    evaluated = read(run / 'spatial_evaluation64.json')
    if evaluated['status'] != 'passed' or evaluated['count'] != 64:
        raise ValueError('Require complete independent final inference')
    target = Path(evaluated['local_reference'])
    errors = []
    with np.load(run / 'spatial_evaluation64.npz') as data:
        take = evaluated['regular_time_indices']
        for i, prediction in enumerate(data['prediction']):
            with np.load(target / f'condition_{i:03d}/local101.npz') as ref:
                np.testing.assert_array_equal(data['time'], ref['time'])
                errors.append(norms(prediction[take], ref['fields'][take]))
    history = [json.loads(line) for line in (run / 'local_metrics.jsonl').read_text().splitlines()]
    if [row['step'] for row in history] != list(range(0, budget + 1, 250)):
        raise ValueError('Incomplete all64monitor schedule')
    logged = [[row['phi'], row['c']] for row in history[-1]['conditions']]
    np.testing.assert_allclose(errors, logged, rtol=3e-6, atol=2e-8)
    errors = np.asarray(errors)
    if errors.shape != (64, 2):
        raise ValueError('Require64two-channel norms')
    sample = errors.mean(1)
    spatial = read(run / 'spatial_sampling.json')
    if spatial['mode'] != 'rolling_pool' or spatial['entries_used'] != budget:
        raise ValueError('Spatial sampling did not roll')
    result = dict(status='passed', run=str(run), checkpoint=str(run / 'model_final.eqx'),
                  steps=budget, sample_count=64, channel_errors=errors.tolist(),
                  sample_errors=sample.tolist(), channel_means=errors.mean(0).tolist(),
                  mean=float(sample.mean()), sample_std=float(sample.std(ddof=1)),
                  reference=str(target), unique_training_profiles=len(unique),
                  initial_state_and_spatial_time_sampling_match_fixed32=True,
                  input_audit=inputs,
                  hardware_qualification='Both new arms share currentGPU; historicalGPU differs',
                  metric='Equal-point full-space-time channel norms, then channel/sample mean')
    midpoint = state['checkpoint_step']
    intermediate = read(run / f'spatial_evaluation64_step_{midpoint:06d}.json')
    if intermediate['status'] != 'passed' or intermediate['iterations'] != midpoint:
        raise ValueError('Missing declared intermediate checkpoint evaluation')
    intermediate_errors = []
    with np.load(run / f'spatial_evaluation64_step_{midpoint:06d}.npz') as data:
        for i, prediction in enumerate(data['prediction']):
            with np.load(target / f'condition_{i:03d}/local101.npz') as ref:
                np.testing.assert_array_equal(data['time'], ref['time'])
                intermediate_errors.append(norms(prediction[intermediate['regular_time_indices']],
                                                 ref['fields'][intermediate['regular_time_indices']]))
    logged = next(row for row in history if row['step'] == midpoint)
    np.testing.assert_allclose(intermediate_errors, [[x['phi'], x['c']] for x in logged['conditions']],
                               rtol=3e-6, atol=2e-8)
    e = np.asarray(intermediate_errors)
    result['intermediate'] = dict(step=midpoint, channel_errors=e.tolist(),
                                  channel_means=e.mean(0).tolist(), mean=float(e.mean()),
                                  sample_std=float(e.mean(1).std(ddof=1)))
    write(run / 'library_extension_audit.json', result)


def run(bundle):
    state = read(bundle / 'journal.json')
    if state['status'] != 'prepared' or read(bundle / 'preflight.json')['status'] != 'passed':
        raise ValueError('Exactly one prepared and preflighted owner is required')
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip()
    if gpu() != state['gpu'] or busy:
        raise RuntimeError('GPU changed or occupied before launch: ' + busy)
    source = Path(state['source'])
    state.update(status='running', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat())
    children = []

    def save():
        write(bundle / 'journal.json', state)

    def start(name, args, platform):
        stream = (bundle / (name + '.log')).open('x')
        begin = time.monotonic_ns()
        child = subprocess.Popen([sys.executable, *args], cwd=source, stdout=stream,
                                 stderr=subprocess.STDOUT, env=environment(source, platform))
        row = dict(name=name, args=args, status='running', pid=child.pid,
                   started_utc=datetime.now(timezone.utc).isoformat())
        state['stages'].append(row)
        item = child, stream, begin, row
        children.append(item)
        save()
        return item

    def finish(item, code):
        child, stream, begin, row = item
        stream.close()
        row.update(exit_code=code, seconds=(time.monotonic_ns()-begin)/1e9,
                   status='passed' if code == 0 else 'failed')
        save()
        if code:
            raise RuntimeError(row['name'] + ' failed; no automatic retry')

    save()
    try:
        active = [start('train-' + arm, ['-m', 'examples.electropolish2d.train',
                    '--config=' + state['configs'][arm], '--seed=42'], 'cuda')
                  for arm in ('fixed32', 'online')]
        while active:
            for item in active[:]:
                code = item[0].poll()
                if code is not None:
                    finish(item, code)
                    active.remove(item)
            if active:
                time.sleep(.5)
        for arm in ('fixed32', 'online'):
            item = start('evaluate-' + arm + '-checkpoint', ['-m', 'docs.tasks.electro_spatial_evaluate',
                         '--run', state['runs'][arm], '--step', str(state['checkpoint_step'])], 'cuda')
            finish(item, item[0].wait())
            item = start('evaluate-' + arm, ['-m', 'docs.tasks.electro_spatial_evaluate',
                         '--run', state['runs'][arm]], 'cuda')
            finish(item, item[0].wait())
        for arm in ('fixed32', 'online'):
            item = start('audit-' + arm, ['-m', 'docs.tasks.electro_library_extension',
                         '--action', 'audit', '--arm', arm, '--bundle', str(bundle)], 'cpu')
            finish(item, item[0].wait())
        state.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
        save()
    except BaseException as error:
        for item in children:
            if item[0].poll() is None:
                item[0].terminate()
                try:
                    code = item[0].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    item[0].kill()
                    code = item[0].wait()
                item[1].close()
                item[3].update(status='interrupted', exit_code=code)
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('prepare', 'preflight', 'run', 'audit'), required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--prefix')
    parser.add_argument('--iterations', type=int, choices=(10000, 15000))
    parser.add_argument('--checkpoint-step', type=int)
    parser.add_argument('--arm', choices=('fixed32', 'online'))
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.previous is None or args.prefix is None or args.iterations is None or args.checkpoint_step is None:
            parser.error('prepare requires --previous, --prefix, --iterations and --checkpoint-step')
        if not 0 < args.checkpoint_step < args.iterations or args.checkpoint_step % 250:
            parser.error('checkpoint-step must be an intermediate evaluation step')
        prepare(args.previous.resolve(strict=True), args.bundle.resolve(), args.prefix, args.iterations, args.checkpoint_step)
    elif args.action == 'audit':
        if args.arm is None:
            parser.error('audit requires --arm')
        audit_arm(args.bundle.resolve(strict=True), args.arm)
    else:
        {'preflight': preflight, 'run': run}[args.action](args.bundle.resolve(strict=True))
