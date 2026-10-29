"""Measure an entire canonical train.py subprocess, preserving raw stdout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=('nonlocal', 'local'))
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    output = Path(__file__).resolve().parent/f'electro-operator-match-{args.method}'
    config = ('examples/electropolish2d/configs/train_operator_online.py'
              if args.method == 'nonlocal' else 'examples/electropolish2d/configs/train_operator_local.py')
    command = [sys.executable, 'examples/electropolish2d/train.py', '--config='+config,
               '--config.log.run_name='+args.run_name]
    if args.method == 'nonlocal':
        command += ['--config.arch.local_input=zero']
    settings = dict(JAX_PLATFORMS='cuda', JAX_ENABLE_X64='false',
                    XLA_PYTHON_CLIENT_PREALLOCATE='false', OPENBLAS_NUM_THREADS='1',
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    environment = os.environ.copy(); environment.update(settings)
    start = time.perf_counter()
    with output.with_name(output.name+'-training.txt').open('x') as stream:
        result = subprocess.run(command, env=environment, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    record = dict(command=command, environment=settings,
                   wall_seconds=time.perf_counter()-start, returncode=result.returncode)
    output.with_name(output.name+'-process.json').write_text(json.dumps(record, indent=2)+'\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    print('COMPLETE', json.dumps(record), flush=True)
