"""Finite fresh three-method melting controls, serialized after another owner."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from docs.tasks.electro_threeway_queue import process_ticks, read, write
from docs.tasks.melting_fem_comparison_evaluate import evaluate, metrics

ROOT = Path(__file__).resolve().parents[2]
LOGS = Path('/root/autodl-tmp/tf-logs/icemelting3d')
METHODS = ('nonlocal', 'local_ad', 'fem')


def audit(bundle, prefix):
    runs = {method: LOGS/(prefix+'-'+method) for method in METHODS}
    evaluate(runs['nonlocal'], runs['fem'], bundle/'nonlocal_fem.json')
    pair = read(bundle/'nonlocal_fem.json')
    base = runs['nonlocal']
    original = read(base/'config.json')
    reference = Path(original['operator']['dataset_path'])
    manifest = read(reference/'manifest.json')
    if original['operator']['monitor_split'] != 'test':
        raise ValueError('New controls require complete-test monitoring')
    local = read(runs['local_ad']/'config.json')
    if local['operator']['spatial_discretization'] != 'local_ad':
        raise ValueError('Missing genuine operator-AD control')
    bc = local['operator'].pop('local_ad')
    expected_bc = dict(boundary_points=96, boundary_weight=1., boundary_norm='physical_normal_mse',
                       initial_derivatives='analytic_unique_projection')
    if bc != expected_bc:
        raise ValueError('AD physical boundary scheme changed')
    local['operator']['spatial_discretization'] = 'nonlocal'
    local['log']['run_name'] = original['log']['run_name']
    if local != original:
        raise ValueError('Unexpected AD physics/architecture/optimization changes')
    for method, run in runs.items():
        for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
            if (run/name).read_bytes() != (base/name).read_bytes():
                raise ValueError('Cold state mismatch '+str(run/name))
        with np.load(run/'sampling.npz') as actual, np.load(base/'sampling.npz') as expected:
            if set(actual.files) != set(expected.files):
                raise ValueError('Sampling schema mismatch')
            for key in actual.files:
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        for name in read(run/'source/files.json'):
            if (run/'source'/name).read_bytes() != (ROOT/name).read_bytes():
                raise ValueError('Current/frozen source mismatch '+name)
        monitor = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        monitor = [row for row in monitor if row['split'] == 'monitor']
        if [row['step'] for row in monitor] != list(range(0, 10001, 250)):
            raise ValueError('Require41actual native-reference monitor records')
        monitoring = read(run/'monitoring.json')
        ids = manifest['test_indices']
        if (monitoring['split'] != 'test' or monitoring['indices'] != ids
                or monitoring['count'] != len(ids) or len(ids) != 64
                or monitoring['reference'] != str(reference)):
            raise ValueError('All64native test ellipsoids required throughout training')
        for row in monitor:
            if [item['index'] for item in row['conditions']] != ids:
                raise ValueError('Incomplete or reordered native test cohort')
            values = np.array([[item['pooled_phi'], item['terminal_phi']] for item in row['conditions']])
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('Nonfinite/negative native monitoring errors')
            np.testing.assert_allclose(values.mean(axis=0), [row['pooled_phi'], row['terminal_phi']], rtol=2e-12, atol=1e-14)
    from ml_collections import ConfigDict
    from examples.icemelting3d.operator_local_ad import sample_boundary
    config = ConfigDict(read(runs['local_ad']/'config.json'))
    with np.load(runs['local_ad']/'boundary_sampling.npz') as trace:
        if trace['points'].shape != (10000, 96, 3):
            raise ValueError('Incomplete explicit AD boundary stream')
        for step in range(10000):
            points, normal = sample_boundary(config, step)
            np.testing.assert_array_equal(trace['points'][step], points)
            np.testing.assert_array_equal(trace['normals'], normal)
    results = {}
    for method, run in runs.items():
        native = []
        for index in manifest['test_indices']:
            with np.load(run/f'prediction_{index:03d}.npz') as prediction, np.load(reference/f'condition_{index:03d}.npz') as ref:
                for key in ('points', 'time', 'parameters'):
                    np.testing.assert_array_equal(prediction[key], ref[key])
                np.testing.assert_array_equal(prediction['target'], ref['phi'])
                native.append(dict(index=index, **metrics(prediction['prediction'], ref['phi'])))
        with np.load(run/'prediction_005.npz') as prediction, np.load(pair['local_reference']) as ref:
            for key in ('points', 'time', 'parameters'):
                np.testing.assert_array_equal(prediction[key], ref[key])
            independent = metrics(prediction['prediction'], ref['phi'])
        timing, summary = read(run/'timing.json'), read(run/'summary.json')
        inclusive = sum(np.asarray(timing[key][1:]) for key in ('update_seconds', 'sample_seconds', 'spatial_seconds'))
        results[method] = dict(run=str(run), checkpoint=str(run/'model_final.eqx'),
            native_records=native, native_mean=float(np.mean([r['pooled_phi'] for r in native])),
            native_std=float(np.std([r['pooled_phi'] for r in native], ddof=1)),
            local_case005=independent, monitor=str(run/'metrics.jsonl'),
            warm_inclusive_median_seconds=float(np.median(inclusive)),
            trainer_seconds=summary['trainer_seconds'], setup_seconds=summary['setup_seconds'],
            peak_active_gib=summary['peak_active_GiB'])
    write(bundle/'comparison.json', dict(status='passed', seed=42, steps=10000, results=results,
        paired_cold_states_and_all_inputs=True, independent_boundary_replay=True,
        reference=pair['native_reference'], local_reference=pair['local_reference'],
        strict_local_refinement_passed=pair['strict_local_refinement_passed'],
        native_to_local_case005=pair['native_to_local_case005'],
        metric='Unweighted complete space-time relative L2 per sample; then arithmetic sample mean',
        scope='All64native test cases at each of41evaluations;1separate user-accepted localCG2 case;one initialization',
        limitations=['Native graph references can favor their discretization; not continuum errors.',
                     'AD uses analytic known-IC geometry jets and network AD; explicit physical Neumann loss.',
                     'Core/halo metadata is prepared in all arms; AD evaluates only256PDEcores and96face points.']))


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
        wait_journal=str(args.wait_journal), wait_pid=args.wait_pid, wait_start_ticks=args.wait_start_ticks,
        prefix=args.prefix, stages=[])
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        JAX_PLATFORMS='cuda', XLA_FLAGS='', JAX_ENABLE_COMPILATION_CACHE='false',
        XLA_PYTHON_CLIENT_PREALLOCATE='true', XLA_PYTHON_CLIENT_MEM_FRACTION='.8', PYTHONUNBUFFERED='1')

    def save():
        state['heartbeat_utc'] = datetime.now(timezone.utc).isoformat(); write(args.bundle/'journal.json', state)

    def stage(name, command, cpu=False, reference=False):
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version',
            '--format=csv,noheader'], text=True).strip()
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if gpu != state['gpu'] or occupied:
            raise RuntimeError('GPU changed or busy before serial stage: '+occupied)
        row = dict(name=name, status='running', command=command); state['stages'].append(row); save()
        begin = time.perf_counter()
        with (args.bundle/(name+'.log')).open('x') as stream:
            child = subprocess.Popen([sys.executable, *command], cwd=ROOT,
                env=dict(env, JAX_PLATFORMS='cpu' if cpu else 'cuda',
                    JAX_ENABLE_X64='true' if reference else 'false'), stdout=stream, stderr=subprocess.STDOUT)
            row['pid'] = child.pid; save()
            while True:
                try:
                    code = child.wait(timeout=30); break
                except subprocess.TimeoutExpired:
                    save()
        row.update(exit_code=code, status='passed' if code == 0 else 'failed', seconds=time.perf_counter()-begin)
        save()
        if code:
            raise RuntimeError(name+' failed; remaining stages not launched')

    def command(method, memory):
        name = args.prefix+('-probe-' if memory else '-')+method
        cmd = ['-m', 'examples.icemelting3d.train', '--seed=42',
            '--config=examples/icemelting3d/configs/train_operator_compare_'+method+'.py',
            '--config.log.run_name='+name]
        if memory:
            cmd += ['--config.operator.action=memory', '--config.operator.probe_steps=12']
        return cmd

    save()
    try:
        while True:
            dependency = read(args.wait_journal)
            if dependency['status'] == 'complete':
                break
            if dependency['status'] not in ('waiting', 'running'):
                raise RuntimeError('Dependency stopped without success: '+dependency['status'])
            if process_ticks(args.wait_pid) != args.wait_start_ticks:
                raise RuntimeError('Dependency owner changed')
            save(); time.sleep(30)
        state['status'] = 'running'
        state['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version',
            '--format=csv,noheader'], text=True).strip(); save()
        stage('prepare-test64', ['-m', 'docs.tasks.melting_test64_reference',
            '--config', 'examples.icemelting3d.configs.train_operator_compare_nonlocal',
            '--source', '/root/autodl-tmp/data/icemelting3d/operator_ellipsoids_v1',
            '--output', '/root/autodl-tmp/data/icemelting3d/operator_ellipsoids_test64_20261114_v1',
            '--workers', '4', '--cores-per-worker', '2', '--extension-seed', '20261114'], cpu=True, reference=True)
        stage('probe-local_ad', command('local_ad', True))
        probe = read(LOGS/(args.prefix+'-probe-local_ad')/'summary.json')
        if probe['peak_active_GiB'] > 70:
            raise RuntimeError('AD probe exceeds explicit70GiBactive-memory budget')
        state['ad_probe'] = probe; save()
        for method in METHODS:
            stage('train-'+method, command(method, False))
        # Audit uses CPU only, after every trainer has exited.
        stage('audit', ['-m', 'docs.tasks.melting_threeway_audit', '--bundle', str(args.bundle), '--prefix', args.prefix], True)
        state.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat()); save()
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}'); save(); raise


if __name__ == '__main__':
    main()
