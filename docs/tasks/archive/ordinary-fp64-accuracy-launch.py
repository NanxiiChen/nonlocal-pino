"""Full ordinary-policy FP64 cold start through canonical corrosion train.py."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/ordinary-fp64-accuracy-20261026'
NAME='20261026-full-graph_ordinary-float64-seed42'
RUN=Path('/root/autodl-tmp/tf-logs/corrosion2d')/NAME


def main():
    if RUN.exists():raise FileExistsError(RUN)
    OUT.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-u','examples/corrosion2d/train.py',
             '--config=examples/corrosion2d/configs/train_graph_ordinary.py','--seed=42',
             '--config.train.compute_dtype=float64','--config.train.iterations=5000',
             '--config.experiment.checkpoint_every=200',f'--config.log.run_name={NAME}']
    env=dict(os.environ,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',
             XLA_FLAGS='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
             JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-local-ad-performance-cache')
    record={'status':'running','run':str(RUN),'command':command,'updates':5000,
            'physical_seconds':50,'policy':'ordinary','compute_dtype':'float64',
            'maximum_GPU_workers':1,'stdout':str(OUT/'train.log'),
            'initialization':'Common FP32 values exactly promoted; fresh weights',
            'timing':'Actual compilation cache reused; wall includes launch/JIT/evaluation/save, excludes independent reference generation'}
    (OUT/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    start=time.perf_counter()
    with (OUT/'train.log').open('x') as stream:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    record.update(returncode=result.returncode,wall_seconds=time.perf_counter()-start,
                  status='completed' if result.returncode==0 else 'failed')
    (OUT/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    if result.returncode:raise RuntimeError((OUT/'train.log').read_text()[-6000:])
    summary=json.loads((RUN/'summary.json').read_text())
    if summary['updates']!=5000:raise ValueError('Incomplete training')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
