"""Finish the live joint arm, then cold FiLM with matching64case equal-point monitoring."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from docs.tasks.ceramic_temporal_followup import process_identity


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predecessor', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--film-run', type=Path, required=True)
    args = parser.parse_args()
    predecessor = args.predecessor.resolve(strict=True)
    prior = read(predecessor/'journal.json')
    if prior['status'] not in ('running', 'complete') or prior['baseline_reused'] is not True:
        raise ValueError('Require the selected joint-only predecessor')
    identity = process_identity(prior['pid'])
    if identity is not None:
        command = (Path('/proc')/str(prior['pid'])/'cmdline').read_bytes().split(b'\0')
        if b'docs.tasks.ceramic_temporal_reuse_queue' not in command:
            raise ValueError('Unexpected predecessor owner')
    if args.film_run.exists():
        raise FileExistsError('A new cold FiLM directory is required: '+str(args.film_run))
    bundle = args.bundle.resolve(); bundle.mkdir(parents=True, exist_ok=False)
    source = Path(prior['source']).resolve(strict=True)
    state = dict(status='waiting_for_joint', pid=os.getpid(),
        started_utc=datetime.now(timezone.utc).isoformat(), stages=[], source=str(source), gpu=prior['gpu'],
        predecessor=str(predecessor), predecessor_pid=prior['pid'], predecessor_start_ticks=identity,
        process_poll_seconds=60, metrics_polled=False,
        runs=dict(film=str(args.film_run.resolve()), joint=prior['runs']['joint']),
        prediction_roots=dict(film=str(args.film_run.resolve()), joint=prior['runs']['joint']),
        reference=prior['reference'], inputs=prior['inputs'], excluded=prior['excluded'],
        metric='Equal-point per-channel space-time L2; then channel/sample mean',
        monitor_scope='Both64cases at all41evaluations, every500updates;20k cold training each',
        baseline_scope='New FiLM only; historical4case FiLM and interrupted02FiLM excluded')
    def save():
        temporary = bundle/'journal.tmp'
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
        temporary.replace(bundle/'journal.json')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        JAX_PLATFORMS='cuda', XLA_FLAGS='', JAX_ENABLE_X64='false', JAX_ENABLE_COMPILATION_CACHE='false',
        XLA_PYTHON_CLIENT_PREALLOCATE='true', XLA_PYTHON_CLIENT_MEM_FRACTION='.90',
        PYTHONPATH=str(source)+':'+str(source/'src'), PYTHONUNBUFFERED='1')
    def stage(name, command, cpu):
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
        if code:
            raise RuntimeError(name+' failed; no automatic restart')
    save()
    try:
        # Wait only on the owner identity, never read training metrics in this loop.
        deadline = time.monotonic()+12*3600
        while identity is not None and process_identity(prior['pid']) == identity:
            if time.monotonic() >= deadline:
                raise TimeoutError('Predecessor still live after12hours; trainer not changed')
            time.sleep(60)
        prior = read(predecessor/'journal.json')
        joint = [s for s in prior['stages'] if s['name'] == 'train-joint']
        if prior['status'] != 'complete' or len(joint) != 1 or joint[0]['status'] != 'passed':
            raise RuntimeError('Joint predecessor incomplete; no further training started')
        state['stages'].append(dict(copy.deepcopy(joint[0]), origin_journal=str(predecessor/'journal.json')))
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version',
                                       '--format=csv,noheader'], text=True).strip()
        if gpu != state['gpu']:
            raise ValueError('Device differs from the actual joint run/probes')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU busy before the isolated cold FiLM run')
        state['status'] = 'running'; save()
        stage('train-film', ['-m', 'examples.particle_corrosion2d.train', '--config',
            'examples/particle_corrosion2d/configs/train_time_film.py', '--inputs', state['inputs'],
            '--reference', state['reference'], '--exclude-reference', state['excluded'],
            '--output', state['runs']['film'], '--seed', '42'], cpu=False)
        stage('audit', ['-m', 'docs.tasks.ceramic_temporal_audit', '--bundle', str(bundle)], cpu=True)
        state.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat()); save()
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}'); save(); raise


if __name__ == '__main__':
    main()
