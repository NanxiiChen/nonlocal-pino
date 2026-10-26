"""Predeclared six full controls before canonical policy/default alignment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/ordinary-policy-defaults-20261026'
JOBS=(('electropolish2d','float32',42),('electropolish2d','float32',2026),
      ('electropolish2d','float64',42),('electropolish2d','float64',2026),
      ('corrosion2d','float64',2026),('icemelting3d','float32',42))


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','maximum_GPU_workers':1,'updates_each':5000,'runs':[]}
    for case,dtype,seed in JOBS:
        name=f'20261026-defaultcheck-ordinary-{dtype}-seed{seed}'
        run=Path('/root/autodl-tmp/tf-logs')/case/name
        if run.exists():raise FileExistsError(run)
        command=[sys.executable,'-u',f'examples/{case}/train.py',f'--seed={seed}',
                 '--config.train.iterations=5000','--config.train.gpu_policy=ordinary',f'--config.log.run_name={name}']
        if case!='icemelting3d':command.extend([f'--config.train.compute_dtype={dtype}','--config.experiment.checkpoint_every=200'])
        else:command.append('--config.eval.graph_reference_path=/root/nonlocal-pinn/reports/icemelting-weak-prior-20261024/reference.npz')
        env=dict(os.environ,JAX_PLATFORMS='cuda',JAX_ENABLE_X64='false',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                 XLA_FLAGS='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
                 JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-other-case-policy-cache' if case=='icemelting3d' else '/tmp/nonlocal-pinn-local-ad-performance-cache')
        log=OUT/f'{case}-{dtype}-{seed}.log';item={'case':case,'dtype':dtype,'seed_as_run_provenance':seed,
             'policy':'ordinary','run':str(run),'command':command,'stdout':str(log),'status':'running'}
        manifest['runs'].append(item);(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(f'Start {case}/{dtype}/{seed}',flush=True);start=time.perf_counter()
        with log.open('x') as stream:result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        item.update(returncode=result.returncode,wall_seconds=time.perf_counter()-start,status='completed' if result.returncode==0 else 'failed')
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if result.returncode:raise RuntimeError(log.read_text()[-7000:])
        if json.loads((run/'summary.json').read_text())['updates']!=5000:raise ValueError('Incomplete updates')
        print(f'Finished {case}/{dtype}/{seed}',flush=True)
    manifest['status']='completed';(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
