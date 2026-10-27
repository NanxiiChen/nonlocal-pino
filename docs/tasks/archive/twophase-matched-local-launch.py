"""Fresh serial same-scheme local/nonlocal seed42 cold-start comparison."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT = ROOT/'reports/twophase-matched-local'


def main():
    OUT.mkdir(exist_ok=False)
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip()
    manifest = dict(status='running', seed=42, gpu=hardware, max_gpu_workers=1, runs=[])
    env = dict(os.environ, JAX_PLATFORMS='cuda', XLA_FLAGS='',
               JAX_ENABLE_COMPILATION_CACHE='false',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
    manifest['environment'] = {key:env[key] for key in ('JAX_PLATFORMS','XLA_FLAGS','JAX_ENABLE_COMPILATION_CACHE','XLA_PYTHON_CLIENT_PREALLOCATE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')}
    for method, preset in [('nonlocal','train_vortex_conserved'), ('local_ad','train_vortex_conserved_local')]:
        name = '20261027-vortex-matched-'+method
        run = Path('/root/autodl-tmp/tf-logs/twophase2d')/name
        if run.exists():raise FileExistsError(run)
        cmd = [sys.executable, '-u', 'examples/twophase2d/train.py',
               '--config=examples/twophase2d/configs/'+preset+'.py', '--seed=42',
               '--config.log.run_name='+name]
        item = dict(method=method, command=cmd, run=str(run), status='running')
        manifest['runs'].append(item)
        (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        start = time.perf_counter()
        with (OUT/(method+'.log')).open('x') as stream:
            process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        item.update(status='completed' if process.returncode==0 else 'failed',
                    returncode=process.returncode, external_wall_seconds=time.perf_counter()-start)
        (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        if process.returncode:raise RuntimeError((OUT/(method+'.log')).read_text()[-4000:])
        print(json.dumps(item), flush=True)
    manifest['status'] = 'completed'
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__=='__main__':main()
