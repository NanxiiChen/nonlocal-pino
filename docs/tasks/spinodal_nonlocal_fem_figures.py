"""Finite CPU-only preview after the explicit two-method training audit."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from docs.tasks.electro_threeway_queue import process_ticks, read, write

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    'scripts/pino_comparison_data.py', 'scripts/pino_comparison_plot.py',
    'scripts/render_pino_three_method.py', 'scripts/trajectory_trend.py',
    'scripts/local_trend.py', 'docs/tasks/ceramic_fem_comparison_figures.py',
    'docs/tasks/spinodal_nonlocal_fem_figures.py',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--owner-pid', type=int, required=True)
    parser.add_argument('--owner-start-ticks', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    state = dict(status='waiting', pid=os.getpid(), stages=[],
                 started_utc=datetime.now(timezone.utc).isoformat(),
                 dependency=str(args.journal), owner_pid=args.owner_pid,
                 owner_start_ticks=args.owner_start_ticks, source_files=list(SOURCES),
                 training=False, inference=False, reference_solves=False, publication=False)

    def save():
        state['heartbeat_utc'] = datetime.now(timezone.utc).isoformat()
        write(output / 'journal.json', state)

    save()
    try:
        while True:
            dependency = read(args.journal)
            if dependency['pid'] != args.owner_pid or dependency['methods'] != ['nonlocal', 'fem']:
                raise RuntimeError('Training owner or declared methods changed')
            if dependency['status'] == 'complete':
                break
            if dependency['status'] not in ('waiting', 'running'):
                raise RuntimeError('Training stopped before successful completion')
            if process_ticks(args.owner_pid) != args.owner_start_ticks:
                raise RuntimeError('Original training process is no longer live')
            save()
            time.sleep(30)
        report = args.journal.parent / 'final.json'
        if read(report)['status'] != 'passed':
            raise ValueError('Final independent audit missing/failed')
        for name in SOURCES:
            if (ROOT / name).read_bytes() != (output / 'source' / name).read_bytes():
                raise RuntimeError('Rendering source changed: ' + name)
        if shutil.disk_usage(output).free < 512 * 1024**2:
            raise RuntimeError('Insufficient output headroom; nothing deleted')
        command = [sys.executable, '-m', 'scripts.render_pino_three_method', '--case', 'spinodal',
                   '--comparison', str(report), '--output', str(output / 'spinodal')]
        stage = dict(name='render-spinodal', case='spinodal', status='running', command=command)
        state['stages'].append(stage)
        state['status'] = 'running'
        save()
        env = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONPATH='/tmp/pino-pdf-audit',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PYTHONUNBUFFERED='1')
        start = time.perf_counter()
        with (output / 'spinodal.log').open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            stage['pid'] = child.pid
            save()
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    save()
        stage.update(status='passed' if code == 0 else 'failed', exit_code=code,
                     seconds=time.perf_counter() - start)
        save()
        if code:
            raise RuntimeError('Spinodal figure verification failed')
        provenance = output / 'spinodal/provenance.json'
        if read(provenance)['status'] != 'passed':
            raise ValueError('Missing verified figure provenance')
        stage['provenance'] = str(provenance)
        state.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
        save()
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        save()
        raise


if __name__ == '__main__':
    main()
