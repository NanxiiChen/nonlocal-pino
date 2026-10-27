"""Predeclared serial5000-update residual-norm controls, common seed42 provenance."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/twophase-vortex-residual'


def main():
    OUT.mkdir(exist_ok=False)
    manifest=dict(status='running',maximum_GPU_workers=1,runs=[])
    for mode in ['original','transport','both']:
        name='20261026-vortex-residual-'+mode
        run=Path('/root/autodl-tmp/tf-logs/twophase2d')/name
        if run.exists():raise FileExistsError(run)
        cmd=[sys.executable,'-u','examples/twophase2d/train.py',
             '--config=examples/twophase2d/configs/train_vortex_residual.py',
             f'--config.residual.mode={mode}',f'--config.log.run_name={name}']
        item=dict(mode=mode,eta=.1,seed_as_run_provenance=42,command=cmd,run=str(run),status='running')
        manifest['runs'].append(item)
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        env=dict(os.environ,JAX_PLATFORMS='cuda',JAX_ENABLE_X64='false',XLA_FLAGS='',
            XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
        start=time.perf_counter()
        with (OUT/(mode+'.log')).open('x') as log:
            process=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        item.update(status='completed' if process.returncode==0 else 'failed',returncode=process.returncode,
                    wall_seconds=time.perf_counter()-start)
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if process.returncode:raise RuntimeError((OUT/(mode+'.log')).read_text()[-4000:])
        if json.loads((run/'summary.json').read_text())['updates']!=5000:raise ValueError('Incomplete budget')
        print(item,flush=True)
    manifest['status']='completed'
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
