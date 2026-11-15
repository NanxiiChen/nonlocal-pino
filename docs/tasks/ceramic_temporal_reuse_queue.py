"""User-directed single cold joint-query run, reusing the accepted FiLM model."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/root/autodl-tmp/tf-logs/particle_corrosion2d/ceramic-default-fixed-adjoint-20261113-01')
EVAL = Path('/root/autodl-tmp/tf-logs/ceramic-execution-isolation-20261113-01/default-cold/eval64')
BASE_JOURNAL = EVAL.parent/'journal.json'
QUALIFIED = Path('/root/autodl-tmp/tf-logs/ceramic-time-fusion-20261115-02')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args(); bundle = a.bundle.resolve(); bundle.mkdir(parents=True, exist_ok=False)
    if a.run.exists(): raise FileExistsError(a.run)
    prior = json.loads((QUALIFIED/'journal.json').read_text())
    for arm in ('film', 'joint'):
        probe = json.loads((QUALIFIED/('probe-'+arm+'.json')).read_text())
        if probe['status'] != 'passed' or not probe['persistent_gpu_bank']:
            raise ValueError('Require qualified full-bank probes; do not repeat training')
    old = json.loads((BASE_JOURNAL).read_text()); evaluation = json.loads((EVAL/'summary.json').read_text())
    if old['status'] != 'complete' or evaluation['status'] != 'passed' or evaluation['count'] != 64:
        raise ValueError('Missing completed original FiLM baseline/evaluation')
    source = Path(prior['source']).resolve(strict=True)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version',
                                   '--format=csv,noheader'], text=True).strip()
    if gpu != prior['gpu']: raise ValueError('GPU differs from qualified probes')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU is busy before the isolated joint run')
    state = dict(status='running', pid=os.getpid(), baseline_reused=True, source=str(source), gpu=gpu,
        started_utc=datetime.now(timezone.utc).isoformat(), stages=[],
        runs=dict(film=str(BASE), joint=str(a.run.resolve())),
        prediction_roots=dict(film=str(EVAL), joint=str(a.run.resolve())),
        reference=prior['reference'], inputs=prior['inputs'], excluded=prior['excluded'],
        baseline_journal=str(BASE_JOURNAL), baseline_evaluation=str(EVAL/'summary.json'),
        baseline_hardware=str(EVAL.parents[1]/'hardware.json'), qualified_bundle=str(QUALIFIED),
        monitor_scope=dict(film='21records/4development cases, weighted norms',
                           joint='41records/64test cases, weighted and equal-point norms'),
        final_scope='Both final20k weights on the identical64local-FEM cohort, equal-point norms')
    def save():
        (bundle/'journal.json').write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        JAX_PLATFORMS='cuda', XLA_FLAGS='', JAX_ENABLE_X64='false', JAX_ENABLE_COMPILATION_CACHE='false',
        XLA_PYTHON_CLIENT_PREALLOCATE='true', XLA_PYTHON_CLIENT_MEM_FRACTION='.90',
        PYTHONPATH=str(source)+':'+str(source/'src'), PYTHONUNBUFFERED='1')
    def stage(name, command, cpu=False):
        row = dict(name=name, status='running', command=command); state['stages'].append(row); save()
        with (bundle/(name+'.log')).open('x') as log:
            start = time.monotonic_ns()
            child = subprocess.Popen([sys.executable, *command], cwd=ROOT if cpu else source,
                stdout=log, stderr=subprocess.STDOUT, env=dict(env,
                    JAX_PLATFORMS='cpu' if cpu else 'cuda',
                    PYTHONPATH=str(ROOT)+':'+str(ROOT/'src') if cpu else env['PYTHONPATH'],
                    CERAMIC_COLD_START_MONOTONIC_NS=str(start)))
            row.update(pid=child.pid, started_utc=datetime.now(timezone.utc).isoformat()); save()
            code = child.wait(); elapsed = (time.monotonic_ns()-start)/1e9
        row.update(status='passed' if code == 0 else 'failed', exit_code=code, seconds=elapsed); save()
        if code: raise RuntimeError(name+' failed; no automatic restart')
    save()
    try:
        stage('train-joint', ['-m', 'examples.particle_corrosion2d.train', '--config',
            'examples/particle_corrosion2d/configs/train_time_joint.py', '--inputs', state['inputs'],
            '--reference', state['reference'], '--exclude-reference', state['excluded'],
            '--output', str(a.run.resolve()), '--seed', '42'])
        stage('audit', ['-m', 'docs.tasks.ceramic_temporal_reuse_audit', '--bundle', str(bundle)], cpu=True)
        state.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat()); save()
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}'); save(); raise


if __name__ == '__main__': main()
