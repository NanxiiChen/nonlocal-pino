"""Serial full-budget electropolish spatial controls; no reference generation."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path('/root/nonlocal-pinn')
ROOT = Path('/root/autodl-tmp/tf-logs/electropolish2d')
PREFIX = '20261031-electro-spatial-01'
JOURNAL = ROOT / (PREFIX+'-process.json')
MODES = ('full', 'fixed_pool', 'rolling_pool')


def write(record):
    temporary = JOURNAL.with_suffix('.partial')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    temporary.replace(JOURNAL)


def main():
    if JOURNAL.exists():
        raise FileExistsError(JOURNAL)
    env = dict(os.environ, JAX_PLATFORMS='cuda', OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', XLA_PYTHON_CLIENT_MEM_FRACTION='0.40')
    record = dict(status='running', runner_pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
        controls=[], budget=5000, runtime_seed=42, gpu_jobs='serial', reference_solves=0)
    write(record)
    for mode in MODES:
        name = PREFIX+'-'+mode
        run = ROOT / name
        if run.exists():
            raise FileExistsError(run)
        command = [sys.executable, '-m', 'examples.electropolish2d.train',
            '--config=examples/electropolish2d/configs/train_operator_'+mode+'.py',
            '--seed=42', '--config.log.run_name='+name]
        start = time.perf_counter()
        log = ROOT / (name+'-training.log')
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
            item = dict(mode=mode, run=str(run), log=str(log), trainer_pid=child.pid, status='running')
            record['controls'].append(item)
            write(record)
            code = child.wait()
        item.update(status='complete' if code == 0 else 'failed', exit_code=code,
            process_wall_seconds=time.perf_counter()-start, trainer_pid=None)
        if code != 0:
            record.update(status='failed', finished_utc=datetime.now(timezone.utc).isoformat())
            write(record)
            raise RuntimeError(f'{mode} failed; inspect {log}')
        result = json.loads((run/'summary.json').read_text())
        if result['iterations'] != 5000:
            raise ValueError('Incomplete training budget')
        item['final_monitor'] = result['final']
        write(record)
    record.update(status='training_complete', finished_utc=datetime.now(timezone.utc).isoformat())
    write(record)


if __name__ == '__main__':
    main()
