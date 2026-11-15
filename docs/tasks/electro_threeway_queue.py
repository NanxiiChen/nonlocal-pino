"""Finite same-device three-method cold10k controls after a verified GPU owner."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOGS = Path('/root/autodl-tmp/tf-logs/electropolish2d')
METHODS = ('nonlocal', 'local_ad', 'fem_weak')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def process_ticks(pid):
    # /proc/stat command names may contain spaces; parse after the final ')'.
    text = (Path('/proc')/str(pid)/'stat').read_text()
    return int(text[text.rindex(')')+2:].split()[19])


def audit(bundle, prefix):
    runs = [LOGS/(prefix+'-'+method) for method in METHODS]
    configs = [read(run/'config.json') for run in runs]
    normalized = json.loads(json.dumps(configs))
    for method, c in zip(METHODS, normalized, strict=True):
        if c['operator']['online']['input_evaluation'] != 'host_float64_shared':
            raise ValueError('New paired controls require shared prescribed-input evaluation')
        if c['nl']['method'] != {'nonlocal': 'sparse', 'local_ad': 'local_ad', 'fem_weak': 'fem_weak'}[method]:
            raise ValueError('Unexpected derivative method')
        c['nl']['method'] = 'sparse'
        del c['log']['run_name']
        if method == 'local_ad':
            if c['loss']['neumann_scale'] != 1 or c['loss']['gradnorm_bias'] != [1.]*6:
                raise ValueError('AD boundary terms changed')
            del c['loss']['neumann_scale']; c['loss']['gradnorm_bias'] = [1.]*5
        elif method == 'fem_weak':
            from examples.electropolish2d.fem_weak import SPEC
            if c['operator']['fem'] != SPEC:
                raise ValueError('FEM scheme changed')
            del c['operator']['fem']
    if normalized[0] != normalized[1] or normalized[0] != normalized[2]:
        raise ValueError('Unexpected physics/architecture/optimizer/monitor differences')
    base = runs[0]
    for run in runs:
        for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
            if (run/name).read_bytes() != (base/name).read_bytes():
                raise ValueError('Cold state differs: '+str(run/name))
        for name in ('online_coefficients.npy', 'online_accepted.npy'):
            np.testing.assert_array_equal(np.load(run/name), np.load(base/name))
        with np.load(run/'sampling.npz') as a, np.load(base/'sampling.npz') as b:
            if set(a.files) != set(b.files):
                raise ValueError('Sampling schemas differ')
            for key in a.files:
                if key != 'nonlocal_halo_counts':
                    np.testing.assert_array_equal(a[key], b[key], err_msg=key)
            np.testing.assert_array_equal(a['draw_ids'], np.arange(10000))
        with np.load(run/'paired_inputs.npz') as a, np.load(base/'paired_inputs.npz') as b:
            if set(a.files) != set(b.files):
                raise ValueError('Input schemas differ')
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        source_names = read(run/'source/files.json')
        for name in source_names:
            if (run/'source'/name).read_bytes() != (ROOT/name).read_bytes():
                raise ValueError('Current/frozen source mismatch: '+name)
    result = dict(status='passed', seed=42, updates=10000,
        common_cold_state_and_actual_inputs=True, cases={})
    for method, run, c in zip(METHODS, runs, configs, strict=True):
        monitor = [json.loads(line) for line in (run/'local_metrics.jsonl').read_text().splitlines()]
        if [r['step'] for r in monitor] != list(range(0, 10001, 250)):
            raise ValueError('Expected41actual common-local complete-test evaluations')
        monitoring = read(run/'local_monitor.json')
        ids = list(range(64))
        if (c['operator']['comparison_monitor']['indices'] != ids
                or monitoring['indices'] != ids or monitoring['split'] != 'test'
                or monitoring['count'] != 64):
            raise ValueError('Complete ordered64test cohort required throughout training')
        for row in monitor:
            if [item['index'] for item in row['conditions']] != ids:
                raise ValueError('Incomplete or reordered test monitoring cohort')
            values = np.array([[item['phi'], item['c']] for item in row['conditions']])
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('Nonfinite/negative test monitoring errors')
            np.testing.assert_allclose(values.mean(axis=0), [row['phi'], row['c']], rtol=2e-12, atol=1e-14)
        metadata = read(run/'spatial_evaluation64.json')
        if metadata['status'] != 'passed' or metadata['count'] != 64:
            raise ValueError('Complete common64evaluation required')
        errors = []
        with np.load(run/'spatial_evaluation64.npz') as saved:
            take = np.searchsorted(saved['time'], np.linspace(0, 1, 21))
            np.testing.assert_array_equal(saved['time'][take], np.linspace(0, 1, 21))
            for i in range(64):
                refpath = Path(c['operator']['comparison_monitor']['reference_root'])/f'condition_{i:03d}/local101.npz'
                with np.load(refpath) as ref:
                    np.testing.assert_array_equal(saved['time'], ref['time'])
                    np.testing.assert_array_equal(saved['X'], ref['X'])
                    np.testing.assert_array_equal(saved['Y'], ref['Y'])
                    actual = np.asarray(saved['prediction'][i, take], dtype=np.float64)
                    target = np.asarray(ref['fields'][take], dtype=np.float64)
                norms = np.sqrt(np.sum((actual-target)**2, axis=(0, 1, 2))/np.sum(target**2, axis=(0, 1, 2)))
                if not np.isfinite(norms).all():
                    raise ValueError('Nonfinite final physical field error')
                errors.append(norms.tolist())
        result['cases'][method] = dict(run=str(run), checkpoint=str(run/'model_final.eqx'),
            config=str(run/'config.json'), monitoring=str(run/'local_metrics.jsonl'),
            evaluation=str(run/'spatial_evaluation64.json'), predictions=str(run/'spatial_evaluation64.npz'),
            errors_phi_c=errors, mean=np.mean(errors, axis=0).tolist(), std=np.std(errors, axis=0, ddof=1).tolist(),
            timing=str(run/'timing.json'), memory=str(run/'memory.json'))
    result['scope'] = 'Complete64local-FD test inputs for all41records and final evaluation; one seed'
    write(bundle/'comparison.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--wait-journal', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--wait-start-ticks', type=int, required=True)
    args = parser.parse_args()
    args.bundle.mkdir(parents=True, exist_ok=False)
    state = dict(status='waiting', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
        wait_journal=str(args.wait_journal), wait_pid=args.wait_pid,
        wait_start_ticks=args.wait_start_ticks, prefix=args.prefix, stages=[])
    journal = args.bundle/'journal.json'
    env = dict(os.environ, JAX_PLATFORMS='cuda', XLA_FLAGS='', JAX_ENABLE_COMPILATION_CACHE='false',
        XLA_PYTHON_CLIENT_PREALLOCATE='true', XLA_PYTHON_CLIENT_MEM_FRACTION='.8',
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')

    def save():
        state['heartbeat_utc'] = datetime.now(timezone.utc).isoformat(); write(journal, state)

    def stage(name, command):
        current_gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version',
            '--format=csv,noheader'], text=True).strip()
        if current_gpu != state['gpu']:
            raise RuntimeError('GPU identity changed between three-method stages')
        gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if gpu:
            raise RuntimeError('GPU occupied before serial stage: '+gpu)
        entry = dict(name=name, command=command, status='running'); state['stages'].append(entry); save()
        start = time.perf_counter()
        with (args.bundle/(name+'.log')).open('x') as f:
            child = subprocess.Popen([sys.executable, *command], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            entry['pid'] = child.pid; save()
            while True:
                try:
                    code = child.wait(timeout=30); break
                except subprocess.TimeoutExpired:
                    save()
        entry.update(exit_code=code, seconds=time.perf_counter()-start, status='passed' if code == 0 else 'failed')
        save()
        if code:
            raise RuntimeError(name+' failed; later stages not launched')

    save()
    try:
        while True:
            dependency = read(args.wait_journal)
            if dependency['status'] == 'complete':
                break
            if dependency['status'] != 'running':
                raise RuntimeError('Dependency is terminal without success: '+dependency['status'])
            if process_ticks(args.wait_pid) != args.wait_start_ticks:
                raise RuntimeError('Dependency owner identity changed')
            save(); time.sleep(30)
        state['status'] = 'running'
        state['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version',
            '--format=csv,noheader'], text=True).strip()
        save()
        for method, config in zip(METHODS, ('train_operator_threeway', 'train_operator_threeway_local',
                                            'train_operator_threeway_fem'), strict=True):
            run = LOGS/(args.prefix+'-'+method)
            stage('train-'+method, ['-m', 'examples.electropolish2d.train',
                '--config=examples/electropolish2d/configs/'+config+'.py', '--seed=42', '--config.log.run_name='+run.name])
            stage('evaluate-'+method, ['-m', 'docs.tasks.electro_spatial_evaluate', '--run', str(run)])
        audit(args.bundle, args.prefix)
        state.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat()); save()
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}'); save(); raise


if __name__ == '__main__':
    main()
