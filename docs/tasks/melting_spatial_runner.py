"""Serial cold melting spatial controls; existing references only."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path('/root/nonlocal-pinn')
ROOT = Path('/root/autodl-tmp/tf-logs/icemelting3d')
PREFIX = '20261031-melting-spatial-01'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modes', nargs='+', choices=('full', 'fixed_pool', 'rolling_pool'), required=True)
    parser.add_argument('--journal', required=True)
    args = parser.parse_args()
    journal = Path(args.journal)
    if journal.exists() or len(set(args.modes)) != len(args.modes):
        raise ValueError('Require a fresh journal and unique explicit modes')
    record = dict(status='running', runner_pid=os.getpid(), controls=[], iterations=5000,
        runtime_seed=42, started_utc=datetime.now(timezone.utc).isoformat(), reference_solves=0)
    def write():
        temporary = journal.with_suffix('.partial')
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temporary.replace(journal)
    write()
    env = dict(os.environ, JAX_PLATFORMS='cuda', OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', XLA_PYTHON_CLIENT_MEM_FRACTION='0.80')
    for mode in args.modes:
        name = PREFIX+'-'+mode
        run = ROOT/name
        if run.exists():
            raise FileExistsError(run)
        command = [sys.executable, '-m', 'examples.icemelting3d.train',
            '--config=examples/icemelting3d/configs/train_operator_'+mode+'.py',
            '--seed=42', '--config.log.run_name='+name]
        log = ROOT/(name+'-training.log'); start = time.perf_counter()
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
            item = dict(mode=mode, run=str(run), log=str(log), trainer_pid=child.pid, status='running')
            record['controls'].append(item); write(); code = child.wait()
        item.update(status='complete' if code == 0 else 'failed', exit_code=code,
            process_wall_seconds=time.perf_counter()-start, trainer_pid=None)
        if code != 0:
            record.update(status='failed', finished_utc=datetime.now(timezone.utc).isoformat()); write()
            raise RuntimeError(f'{mode} failed: {log}')
        summary = json.loads((run/'summary.json').read_text())
        if summary['iterations'] != 5000:
            raise ValueError('Incomplete declared training budget')
        item['test'] = summary['test']; write()
    record.update(status='training_complete', finished_utc=datetime.now(timezone.utc).isoformat())
    write()


if __name__ == '__main__':
    main()
