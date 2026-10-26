"""Run the approved FP64 control through the canonical train.py entry."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT = ROOT/'reports/corrosion-fp64-accuracy-20261026'
NAME = '20261026-precision-float64-phase-seed42'
RUN = Path('/root/autodl-tmp/tf-logs/corrosion2d')/NAME


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    if RUN.exists():
        raise FileExistsError(RUN)
    command = [sys.executable, '-u', 'examples/corrosion2d/train.py', '--seed=42',
               '--config.train.compute_dtype=float64', f'--config.log.run_name={NAME}']
    env = dict(os.environ, JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
               XLA_FLAGS='--xla_gpu_deterministic_ops=true',
               JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-fp64-accuracy-cache')
    log = OUT/'train-float64.log'
    record = {'status': 'running', 'command': command, 'run': str(RUN),
              'stdout': str(log), 'maximum_GPU_workers': 1, 'updates': 5000,
              'sampling_initialization_policy': 'Common FP32 generated values promoted exactly'}
    manifest = Path(__file__).with_suffix('.json')
    manifest.write_text(json.dumps(record, indent=2)+'\n')
    start = time.perf_counter()
    with log.open('x') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record.update(status='completed' if result.returncode == 0 else 'failed',
                  returncode=result.returncode, wall_seconds=time.perf_counter()-start)
    manifest.write_text(json.dumps(record, indent=2)+'\n')
    if result.returncode:
        raise RuntimeError(log.read_text()[-6000:])
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
