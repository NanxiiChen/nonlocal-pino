"""Cold50k Nonlocal/FEM continuation; Local AD explicitly omitted by user."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from docs.tasks.electro_threeway_queue import read, write
from docs.tasks.spinodal_threeway_support import normalized

ROOT = Path(__file__).resolve().parents[2]
LOGS = Path('/root/autodl-tmp/tf-logs/confined_ch2d')
METHODS = ('nonlocal', 'fem')


def preflight(prior_bundle, expected_gpu_uuid, maximum_hours):
    """Reuse verified measurements; reject changed physics, sources or GPU."""
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
                                  '--format=csv,noheader'], text=True).strip()
    if gpu != expected_gpu_uuid:
        raise RuntimeError('Comparison GPU identity changed')
    old = read(prior_bundle / 'resources.json')
    prior = read(prior_bundle / 'journal.json')
    if (old['status'] != 'passed' or old['pairing']['status'] != 'passed'
            or old['reference']['status'] != 'passed' or gpu not in prior['gpu']):
        raise ValueError('Qualified same-GPU original resources/reference required')
    rows = {row['method']: row for row in old['resources']}
    selected = []
    common = None
    from examples.confined_ch2d.configs.train_operator_compare_nonlocal import get_config as nonlocal_config
    from examples.confined_ch2d.configs.train_operator_compare_fem import get_config as fem_config
    for method, factory in zip(METHODS, (nonlocal_config, fem_config), strict=True):
        row = rows[method]
        run = Path(row['run'])
        saved = read(run / 'config.json')
        current = factory().to_dict()
        current['train']['iterations'] = 30
        current['train']['seed'] = 42
        if normalized(saved) != normalized(current):
            raise ValueError('Current configuration differs from the resource-qualified one')
        if common is None:
            common = run
        for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
            if (run / name).read_bytes() != (common / name).read_bytes():
                raise ValueError('Common initial state mismatch')
        for name in read(run / 'source/files.json'):
            if (run / 'source' / name).read_bytes() != (ROOT / name).read_bytes():
                raise ValueError('Qualified numerical source changed: ' + name)
        reference = Path(saved['operator']['comparison_reference']['path'])
        manifest = read(reference / 'manifest.json')
        if manifest['status'] != 'passed' or len(manifest['indices']) != 8:
            raise ValueError('All8qualified local reference cases required')
        for name in manifest['sources']:
            if (reference / 'source' / name).read_bytes() != (ROOT / name).read_bytes():
                raise ValueError('Reference source changed: ' + name)
        with np.load(run / 'update_timing.npz') as f:
            inclusive = f['update_seconds'] + f['batch_preparation_seconds']
        if inclusive.shape != (30,) or not np.isfinite(inclusive).all():
            raise ValueError('Complete finite30update resource record required')
        seconds = float(np.median(inclusive[10:]))
        if seconds != row['warm_inclusive_median_seconds'] or seconds * 50000 / 3600 > maximum_hours:
            raise ValueError('Resource measurement mismatch or execution budget exceeded')
        selected.append(row)
    return dict(status='passed', methods=list(METHODS), resources=selected,
                reference=old['reference'], original_audit=str(prior_bundle / 'resources.json'),
                omitted_method='local_ad: user requested omission after resource measurements')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--prior-bundle', type=Path, required=True)
    parser.add_argument('--expected-gpu-uuid', required=True)
    parser.add_argument('--max-projected-update-hours', type=float, required=True)
    args = parser.parse_args()
    if not 0 < args.max_projected_update_hours <= 24:
        raise ValueError('Explicit execution ceiling in(0,24]hours required')
    args.bundle.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', pid=os.getpid(), prefix=args.prefix,
                 started_utc=datetime.now(timezone.utc).isoformat(), stages=[],
                 methods=list(METHODS), omitted_method='local_ad: user request',
                 prior_bundle=str(args.prior_bundle), gpu_uuid=args.expected_gpu_uuid)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               XLA_FLAGS='', JAX_ENABLE_COMPILATION_CACHE='false', PYTHONUNBUFFERED='1',
               XLA_PYTHON_CLIENT_PREALLOCATE='true', XLA_PYTHON_CLIENT_MEM_FRACTION='.8')

    def save():
        state['heartbeat_utc'] = datetime.now(timezone.utc).isoformat()
        write(args.bundle / 'journal.json', state)

    def stage(name, command, platform):
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                            '--format=csv,noheader'], text=True).strip()
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
                                      '--format=csv,noheader'], text=True).strip()
        if occupied or gpu != args.expected_gpu_uuid:
            raise RuntimeError('GPU occupied or changed: ' + occupied)
        if shutil.disk_usage('/root/autodl-tmp').free < 8 * 1024**3:
            raise RuntimeError('Less than8GiBdisk headroom; no deletion or launch')
        row = dict(name=name, status='running', command=[sys.executable, *command])
        state['stages'].append(row)
        save()
        start = time.perf_counter()
        with (args.bundle / (name + '.log')).open('x') as stream:
            child = subprocess.Popen(row['command'], cwd=ROOT,
                env=dict(env, JAX_PLATFORMS=platform), stdout=stream, stderr=subprocess.STDOUT)
            row['pid'] = child.pid
            save()
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    save()
        row.update(status='passed' if code == 0 else 'failed', exit_code=code,
                   seconds=time.perf_counter() - start)
        save()
        if code:
            raise RuntimeError(name + ' failed; no later stage launched')

    save()
    try:
        resources = preflight(args.prior_bundle, args.expected_gpu_uuid, args.max_projected_update_hours)
        write(args.bundle / 'resources.json', resources)
        for method in METHODS:
            stage('train-' + method, ['-m', 'examples.confined_ch2d.train', '--seed=42',
                '--config=examples/confined_ch2d/configs/train_operator_compare_' + method + '.py',
                '--config.log.run_name=' + args.prefix + '-' + method], 'cuda')
        stage('final-audit', ['-m', 'docs.tasks.spinodal_nonlocal_fem_audit',
              '--bundle', str(args.bundle), '--prefix', args.prefix], 'cpu')
        state.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
        save()
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        save()
        raise


if __name__ == '__main__':
    main()
