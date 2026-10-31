"""Finish authorized spatial controls with serial evaluation and CPU audits."""
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
RECEIPT = REPO/'docs/tasks/electro-spatial-finish.json'


def main():
    receipt = dict(status='waiting_for_training', pid=os.getpid(), stages=[])
    def write():
        RECEIPT.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    write()
    while True:
        record = json.loads((ROOT/(PREFIX+'-process.json')).read_text())
        if record['status'] == 'failed':
            raise RuntimeError('A training control failed; evaluation not started')
        if record['status'] == 'training_complete':
            break
        time.sleep(10)
    base_env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    stages = [('draw_audit', 'cpu', ['-m', 'docs.tasks.electro_spatial_audit', 'runs'])]
    stages += [(mode+'_evaluate64', 'cuda', ['-m', 'docs.tasks.electro_spatial_evaluate',
        '--run='+str(ROOT/(PREFIX+'-'+mode))]) for mode in ('full', 'fixed_pool', 'rolling_pool')]
    stages += [('CPU_replay', 'cpu', ['-m', 'docs.tasks.electro_spatial_audit', 'cpu']),
               ('comparison', 'cpu', ['-m', 'docs.tasks.electro_spatial_report'])]
    for label, backend, args in stages:
        log = ROOT/(PREFIX+'-'+label+'.log')
        env = dict(base_env, JAX_PLATFORMS=backend, XLA_PYTHON_CLIENT_MEM_FRACTION='0.40')
        if backend == 'cpu':
            env['CUDA_VISIBLE_DEVICES'] = ''
        start = time.perf_counter()
        with log.open('x') as stream:
            child = subprocess.Popen([sys.executable, *args], cwd=REPO, env=env,
                stdout=stream, stderr=subprocess.STDOUT)
            entry = dict(stage=label, backend=backend, pid=child.pid, status='running', log=str(log))
            receipt['stages'].append(entry)
            receipt['status'] = 'running'
            write()
            code = child.wait()
        entry.update(status='passed' if code == 0 else 'failed', pid=None,
            seconds=time.perf_counter()-start, exit_code=code)
        if code != 0:
            receipt['status'] = 'failed'
            write()
            raise RuntimeError(f'{label} failed; inspect {log}')
        write()
    receipt.update(status='passed', finished_utc=datetime.now(timezone.utc).isoformat())
    write()


if __name__ == '__main__':
    main()
