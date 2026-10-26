"""Queued serial GEMM-only compiler-policy controls; production stays unchanged."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/fp64-performance-20261026'


def main():
    target=OUT/'gemm-controls-manifest.json'
    if target.exists():raise FileExistsError(target)
    manifest={'status':'queued','maximum_GPU_workers':1,'runs':[]}
    target.write_text(json.dumps(manifest,indent=2)+'\n')
    while True:
        record=json.loads((OUT/'manifest.json').read_text())
        if any(v['returncode']!=0 for v in record['runs']):raise RuntimeError('Primary benchmark failed')
        if record['status']=='completed':break
        time.sleep(10)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',PYTHONPATH=str(ROOT),
             JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',
             JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-fp64-benchmark-cache')
    for policy in ('deterministic','ordinary'):
        for dtype in ('float32','float64'):
            env['XLA_FLAGS']='--xla_gpu_deterministic_ops=true' if policy=='deterministic' else ''
            name=f'gemm-{policy}-{dtype}';log=OUT/(name+'.log')
            output=OUT/(name+'.json');raw=OUT/f'gemm-{dtype}.json'
            if output.exists() or raw.exists():raise FileExistsError(output)
            command=[sys.executable,'-u',str(Path(__file__).with_name('fp64-performance.py')),
                     '--worker',dtype,'--suite','gemm']
            print('Starting '+name,flush=True);start=time.perf_counter()
            with log.open('x') as stream:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            manifest['runs'].append({'name':name,'command':command,'xla_flags':env['XLA_FLAGS'],
                                      'returncode':result.returncode,'wall_seconds':time.perf_counter()-start,
                                      'stdout':str(log),'output':str(output)})
            manifest['status']='running';target.write_text(json.dumps(manifest,indent=2)+'\n')
            if result.returncode:raise RuntimeError(log.read_text())
            raw.rename(output)
            print('Finished '+name,flush=True)
    manifest['status']='completed';target.write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
