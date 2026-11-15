"""Frozen-source finite paired probes/trainers/audit; no retries or default changes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
INPUTS = '/root/autodl-tmp/data/particle_corrosion2d/20261107-spaced-six-01/inputs64'
REFERENCE = '/root/autodl-tmp/data/particle_corrosion2d/local_test64_20261111-01'
EXCLUDED = '/root/autodl-tmp/data/particle_corrosion2d/20261107-spaced-six-01/local_dev4'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--prefix', required=True)
    a = p.parse_args(); bundle = a.bundle.resolve(); bundle.mkdir(parents=True, exist_ok=False)
    source = bundle/'source'
    ignored = shutil.ignore_patterns('__pycache__', '*.pyc')
    shutil.copytree(ROOT/'src', source/'src', ignore=ignored)
    shutil.copytree(ROOT/'examples/particle_corrosion2d', source/'examples/particle_corrosion2d', ignore=ignored)
    for name in ('ceramic_temporal_probe.py', 'ceramic_temporal_queue.py', 'ceramic_temporal_audit.py'):
        target = source/'docs/tasks'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/'docs/tasks'/name, target)
    def gpu():
        return subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version',
                                        '--format=csv,noheader'], text=True).strip()
    runs = {arm: '/root/autodl-tmp/tf-logs/particle_corrosion2d/'+a.prefix+'-'+arm for arm in ('film', 'joint')}
    state = dict(status='running', pid=os.getpid(), gpu=gpu(), source=str(source), runs=runs,
        inputs=INPUTS, reference=REFERENCE, excluded=EXCLUDED,
        started_utc=datetime.now(timezone.utc).isoformat(), stages=[])
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        JAX_PLATFORMS='cuda', XLA_FLAGS='', JAX_ENABLE_X64='false', JAX_ENABLE_COMPILATION_CACHE='false',
        XLA_PYTHON_CLIENT_PREALLOCATE='true', XLA_PYTHON_CLIENT_MEM_FRACTION='.90',
        PYTHONPATH=str(source)+':'+str(source/'src'), PYTHONUNBUFFERED='1')
    def save():
        (bundle/'journal.json').write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
    def stage(name, command, cpu=False):
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if gpu() != state['gpu'] or occupied: raise RuntimeError('GPU changed/busy: '+occupied)
        row = dict(name=name, command=command, status='running'); state['stages'].append(row); save()
        with (bundle/(name+'.log')).open('x') as stream:
            start = time.monotonic_ns()
            child = subprocess.Popen([sys.executable, *command], cwd=source, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(env, JAX_PLATFORMS='cpu' if cpu else 'cuda', CERAMIC_COLD_START_MONOTONIC_NS=str(start)))
            row.update(pid=child.pid, started_utc=datetime.now(timezone.utc).isoformat()); save()
            code = child.wait()
            elapsed = (time.monotonic_ns()-start)/1e9
        row.update(exit_code=code, seconds=elapsed, status='passed' if code == 0 else 'failed'); save()
        if code: raise RuntimeError(name+' failed; subsequent stages not launched')
    save()
    try:
        for arm in ('film', 'joint'):
            stage('probe-'+arm, ['-m', 'docs.tasks.ceramic_temporal_probe', '--config',
                'examples.particle_corrosion2d.configs.train_time_'+arm, '--inputs', INPUTS,
                '--reference', REFERENCE, '--max-active-gib', '85',
                '--output', str(bundle/('probe-'+arm+'.json'))])
        state['probes_passed'] = True; save()
        for arm in ('film', 'joint'):
            stage('train-'+arm, ['-m', 'examples.particle_corrosion2d.train', '--config',
                'examples/particle_corrosion2d/configs/train_time_'+arm+'.py', '--inputs', INPUTS,
                '--reference', REFERENCE, '--exclude-reference', EXCLUDED, '--output', runs[arm], '--seed', '42'])
        stage('audit', ['-m', 'docs.tasks.ceramic_temporal_audit', '--bundle', str(bundle)], cpu=True)
        state.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat()); save()
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}'); save(); raise


if __name__ == '__main__': main()
