"""Supplementary ordinary-policy full-step graph timing, not a new default."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/local-ad-performance-20261026'


def main():
    for name in ('manifest.json','operator-manifest.json'):
        if json.loads((OUT/name).read_text())['status']!='completed':
            raise RuntimeError('Finish preceding GPU workers before ordinary full-step controls')
    script=Path(__file__).with_name('local-ad-performance.py')
    manifest={'status':'running','maximum_GPU_workers':1,'runs':[],
              'scope':'Change compiler policy only, no default or reproducibility claim'}
    for dtype in ('float32','float64'):
        result_path=OUT/f'graph-ordinary-{dtype}.json'
        if result_path.exists():raise FileExistsError(result_path)
        env=dict(os.environ,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',XLA_FLAGS='',
                 JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-local-ad-performance-cache')
        command=[sys.executable,'-u',str(script),'--worker','graph','--dtype',dtype,'--policy','ordinary']
        log=OUT/f'graph-ordinary-{dtype}.log';start=time.perf_counter()
        print(f'Start ordinary graph/{dtype}',flush=True)
        with log.open('x') as stream:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        manifest['runs'].append({'dtype':dtype,'command':command,'stdout':str(log),
                                 'returncode':result.returncode,'wall_seconds':time.perf_counter()-start})
        (OUT/'ordinary-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if result.returncode:raise RuntimeError(log.read_text()[-6000:])
    manifest['status']='completed';(OUT/'ordinary-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
