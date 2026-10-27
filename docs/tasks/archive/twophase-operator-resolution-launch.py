"""Thin-interface spatial-resolution control; distinct architecture and budget."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT = ROOT/'reports/twophase-operator-resolution'

def main():
    OUT.mkdir(exist_ok=False)
    runs = [('cubic_resolved','train_vortex_cubic_resolved'), ('causal','train_vortex_causal')]
    manifest = dict(status='running', runs=[], max_gpu_workers=1)
    for arm, cfg in runs:
        name = '20261027-vortex-resolved-'+arm
        run = Path('/root/autodl-tmp/tf-logs/twophase2d')/name
        if run.exists():raise FileExistsError(run)
        cmd = [sys.executable, '-u', 'examples/twophase2d/train.py',
               '--config=examples/twophase2d/configs/'+cfg+'.py',
               '--config.log.run_name='+name]
        item = dict(arm=arm, command=cmd, run=str(run), status='running', seed=42)
        manifest['runs'].append(item)
        (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        env = dict(os.environ, JAX_PLATFORMS='cuda', XLA_FLAGS='',
                   XLA_PYTHON_CLIENT_PREALLOCATE='false', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
        start=time.perf_counter()
        with (OUT/(arm+'.log')).open('x') as log:
            process=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        item.update(status='completed' if process.returncode==0 else 'failed',
                    returncode=process.returncode, wall_seconds=time.perf_counter()-start)
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if process.returncode:raise RuntimeError((OUT/(arm+'.log')).read_text()[-4000:])
        print(item,flush=True)
    manifest['status']='completed'
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
