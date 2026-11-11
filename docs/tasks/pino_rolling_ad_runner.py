"""Finite, serial current-recipe derivative controls and production resource probes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
LOGS = Path('/root/autodl-tmp/tf-logs')
INPUTS = Path('/root/autodl-tmp/data/particle_corrosion2d/20261107-spaced-six-01/inputs64')
REFERENCE = INPUTS.parent/'local_dev4'
CASES = {
    'particle_corrosion2d': dict(steps=20000, probe_steps=251, presets=(
        'train_physical_spaced_w256_method_nonlocal',
        'train_physical_spaced_w256_method_local_ad')),
    'electropolish2d': dict(steps=10000, probe_steps=65, presets=(
        'train_operator_rolling_nonlocal', 'train_operator_rolling_local')),
}
SEEDS = (42, 43)
ALLOCATOR = dict(XLA_PYTHON_CLIENT_PREALLOCATE='true', XLA_PYTHON_CLIENT_MEM_FRACTION='0.90')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def device():
    command = ['nvidia-smi', '--query-gpu=name,uuid,memory.total', '--format=csv,noheader,nounits']
    lines = subprocess.check_output(command, text=True).strip().splitlines()
    if len(lines) != 1:
        raise ValueError('This serial owner requires exactly one visible physical GPU')
    name, uuid, mib = [field.strip() for field in lines[0].split(',')]
    return dict(name=name, uuid=uuid, memory_bytes=int(mib)*1024**2)


def sources():
    paths = set()
    for case in CASES:
        paths.update((REPO/'examples'/case).rglob('*.py'))
    paths.update((REPO/'src/nonlocal_pinn').rglob('*.py'))
    paths.update(REPO/relative for relative in (
        'docs/tasks/pino_rolling_ad_runner.py', 'docs/tasks/pino_rolling_ad_report.py',
        'docs/tasks/electro_spatial_evaluate.py'))
    return sorted(paths)


def snapshot(output):
    for source in sources():
        target = output/'source'/source.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def verify_sources(output):
    for source in sources():
        saved = output/'source'/source.relative_to(REPO)
        if not saved.is_file() or saved.read_bytes() != source.read_bytes():
            raise ValueError(f'Frozen study source changed: {source}')


def resources(case, run, steps):
    import numpy as np
    summary, timing = read(run/'summary.json'), read(run/'timing.json')
    c = read(run/'config.json')
    budget_key = 'steps' if case == 'particle_corrosion2d' else 'iterations'
    if summary[budget_key] != steps or c['train']['iterations'] != steps:
        raise ValueError('Trainer did not complete its declared update budget')
    updates = np.asarray(timing['update_seconds'], dtype=float)
    if updates.shape != (steps,) or not np.isfinite(updates).all() or np.any(updates <= 0):
        raise ValueError('Incomplete/nonfinite synchronized update timings')
    key = 'allocator' if case == 'particle_corrosion2d' else 'allocator_stats'
    memories = [row[key] for row in read(run/'memory.json')]
    if not memories or any(item is None for item in memories):
        raise ValueError('Resource gate needs actual GPU allocator measurements')
    peak = max(item['peak_bytes_in_use'] for item in memories)
    index = np.arange(steps)
    warm = index >= 25
    refresh = index % c['train']['gradnorm_update_every'] == 0
    result = dict(case=case, run=str(run), steps=steps, peak_bytes=peak,
                  peak_gib=peak/2**30, first_update_seconds=float(updates[0]))
    for name, mask in (('all', warm), ('held', warm & ~refresh), ('refresh', warm & refresh)):
        if not np.any(mask):
            raise ValueError(f'Probe does not exercise warm {name} updates')
        result[f'warm_{name}_mean_seconds'] = float(np.mean(updates[mask]))
        result[f'warm_{name}_median_seconds'] = float(np.median(updates[mask]))
    return result


def probe_config(output, case, preset):
    target = output/f'{case}-{preset}-probe.py'
    steps = CASES[case]['probe_steps']
    lines = [f'from examples.{case}.configs.{preset} import get_config as base',
             '', 'def get_config():', '    c = base()', f'    c.train.iterations = {steps}',
             f'    c.log.eval_every = {steps}']
    if case == 'particle_corrosion2d':
        lines.append(f'    c.log.checkpoint_every = {steps}')
    else:
        lines.append('    c.log.checkpoint_step = 0')
    lines += ['    return c', '']
    target.write_text('\n'.join(lines))
    return target


def run_owner(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    hardware = device()
    if shutil.disk_usage(LOGS).free < 8*2**30:
        raise ValueError('Less than8GiB free; keep artifacts intact and report capacity before launch')
    if args.mode == 'queue':
        probe = read(args.probe/'resource-results.json')
        if probe['status'] != 'passed' or probe['device'] != hardware:
            raise ValueError('Need passed production probes on the same physical GPU')
        if probe['allocator_environment'] != ALLOCATOR:
            raise ValueError('Probe and queue must use the same explicit allocator policy')
        expected = {(case, method) for case in CASES for method in ('nonlocal', 'local_ad')}
        if {(row['case'], row['method']) for row in probe['results']} != expected:
            raise ValueError('Resource qualification is incomplete')
        verify_sources(args.probe)
    state = dict(status='running', mode=args.mode, pid=os.getpid(), device=hardware,
                 started_utc=datetime.now(timezone.utc).isoformat(), stages=[], runs=[],
                 runtime_seeds=list(SEEDS), gpu_execution='serial', reference_solves=0,
                 allocator_environment=ALLOCATOR,
                 evaluation_scope='Reused local-reference development cohorts; no blind population claim',
                 preprocessing_scope='Shared graph-input harness also used by AD; report its overhead explicitly')
    snapshot(output)
    journal = output/'journal.json'
    write(journal, state)
    common = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', MPLBACKEND='Agg', XLA_FLAGS='',
                  JAX_ENABLE_COMPILATION_CACHE='false', **ALLOCATOR)

    def stage(label, command, *, gpu, run):
        verify_sources(output)
        if shutil.disk_usage(LOGS).free < 4*2**30:
            raise ValueError('Artifact disk headroom below4GiB; preserve work and stop before next stage')
        log = output/f'{label}.log'
        record = dict(label=label, command=command, log=str(log), run=str(run),
                      gpu=gpu, status='running')
        state['stages'].append(record)
        environment = dict(common, JAX_PLATFORMS='cuda' if gpu else 'cpu')
        started = time.perf_counter()
        with log.open('x') as stream:
            child = subprocess.Popen([sys.executable, *command], cwd=REPO, env=environment,
                                     stdout=stream, stderr=subprocess.STDOUT)
            record['pid'] = child.pid
            write(journal, state)
            code = child.wait()
        record.update(pid=None, exit_code=code, seconds=time.perf_counter()-started,
                      status='passed' if code == 0 else 'failed')
        write(journal, state)
        if code:
            raise RuntimeError(f'{label} failed with exit{code}; inspect {log}')
        return record

    measurements = []
    try:
        for case, spec in CASES.items():
            for seed in ((42,) if args.mode == 'probe' else SEEDS):
                pair = {}
                for method, preset in zip(('nonlocal', 'local_ad'), spec['presets'], strict=True):
                    label = f'{case}-s{seed}-{method}'
                    name = f'{output.name}-{label}'
                    run = LOGS/case/name
                    if run.exists():
                        raise FileExistsError(run)
                    config = (probe_config(output, case, preset) if args.mode == 'probe'
                              else REPO/'examples'/case/'configs'/f'{preset}.py')
                    command = ['-m', f'examples.{case}.train']
                    if case == 'particle_corrosion2d':
                        command += ['--config', str(config), '--inputs', str(INPUTS),
                                    '--reference', str(REFERENCE), '--output', str(run)]
                        if method == 'local_ad':
                            command += ['--rolling-method-control', str(pair['nonlocal'])]
                        else:
                            command += ['--seed', str(seed)]
                    else:
                        command += [f'--config={config}', f'--seed={seed}',
                                    f'--config.log.run_name={name}']
                    record = stage(label, command, gpu=True, run=run)
                    steps = spec['probe_steps'] if args.mode == 'probe' else spec['steps']
                    resource = resources(case, run, steps)
                    resource.update(method=method, seed=seed, process_seconds=record['seconds'])
                    measurements.append(resource)
                    state['runs'].append(resource)
                    write(journal, state)
                    if args.mode == 'probe' and resource['peak_bytes'] > .85*hardware['memory_bytes']:
                        raise MemoryError(f'{label} leaves less than15% active-memory headroom')
                    pair[method] = run
                    if case == 'particle_corrosion2d':
                        stage(label+'-online-audit', ['-m', 'examples.particle_corrosion2d.online_validation',
                              '--run', str(run), '--workers', '4'], gpu=False, run=run)
                    elif args.mode == 'queue':
                        stage(label+'-evaluate64-5000', ['-m', 'docs.tasks.electro_spatial_evaluate',
                              '--run', str(run), '--step', '5000'], gpu=True, run=run)
                        stage(label+'-evaluate64', ['-m', 'docs.tasks.electro_spatial_evaluate',
                              '--run', str(run)], gpu=True, run=run)
                if case == 'particle_corrosion2d':
                    stage(f'{case}-s{seed}-pair', ['-m', 'examples.particle_corrosion2d.compare',
                          '--runs', str(pair['nonlocal']), str(pair['local_ad']),
                          '--output', str(output/f'{case}-s{seed}-comparison')], gpu=False, run=output)
                else:
                    stage(f'{case}-s{seed}-pair', ['-m', 'examples.electropolish2d.operator_pairing',
                          '--nonlocal-run', str(pair['nonlocal']), '--local-run', str(pair['local_ad'])],
                          gpu=False, run=output)
        if args.mode == 'probe':
            write(output/'resource-results.json', dict(status='passed', device=hardware,
                  allocator_environment=ALLOCATOR, results=measurements,
                  scope='Cold production-path probes, discarded; no accuracy claim'))
        else:
            stage('final-report', ['-m', 'docs.tasks.pino_rolling_ad_report', '--root', str(output)],
                  gpu=False, run=output)
    except BaseException as error:
        state.update(status='failed', failure=repr(error))
        raise
    else:
        state['status'] = 'complete'
    finally:
        state.update(finished_utc=datetime.now(timezone.utc).isoformat(), pid=None)
        write(journal, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('probe', 'queue'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', type=Path, help='Completed same-device probe directory; required for queue')
    args = parser.parse_args()
    if args.mode == 'queue' and args.probe is None:
        parser.error('queue requires --probe')
    run_owner(args)


if __name__ == '__main__':
    main()
