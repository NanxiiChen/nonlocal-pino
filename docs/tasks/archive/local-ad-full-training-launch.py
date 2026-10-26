"""Four predeclared full cold starts through the canonical corrosion train.py."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/local-ad-full-training-20261026'
LOGS=Path('/root/autodl-tmp/tf-logs/corrosion2d')
JOBS=(('graph_deterministic','train_base.py','deterministic'),
      ('graph_ordinary','train_graph_ordinary.py','ordinary'),
      ('local_tensor','train_local_tensor.py','deterministic'),
      ('local_joint','train_local_joint.py','deterministic'))


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','maximum_GPU_workers':1,'seed_as_run_provenance':42,
              'updates':5000,'physical_seconds':50,'dtype':'float32','runs':[]}
    for label,config,policy in JOBS:
        name=f'20261026-full-{label}-seed42'
        run=LOGS/name
        if run.exists():raise FileExistsError(run)
        if label.startswith('local'):
            reference=Path('/root/autodl-tmp/data/corrosion2d/results_local_ad_20261026.json')
            if not reference.is_file():raise FileNotFoundError(reference)
            if not json.loads(reference.read_text())['time_refinement_pass']:
                raise ValueError('Local reference time refinement failed')
        command=[sys.executable,'-u','examples/corrosion2d/train.py',
                 f'--config=examples/corrosion2d/configs/{config}','--seed=42',
                 '--config.train.iterations=5000','--config.train.compute_dtype=float32',
                 '--config.experiment.checkpoint_every=200',f'--config.log.run_name={name}']
        env=dict(os.environ,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
                 XLA_FLAGS='--xla_gpu_deterministic_ops=true' if policy=='deterministic' else '',
                 JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-local-ad-performance-cache')
        log=OUT/f'{label}.log';start=time.perf_counter()
        item={'label':label,'status':'running','command':command,'run':str(run),'stdout':str(log),'gpu_policy':policy}
        manifest['runs'].append(item)
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(f'Start {label}',flush=True)
        with log.open('x') as stream:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        item.update(returncode=result.returncode,wall_seconds=time.perf_counter()-start,
                    status='completed' if result.returncode==0 else 'failed')
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if result.returncode:raise RuntimeError(log.read_text()[-7000:])
        summary=json.loads((run/'summary.json').read_text())
        if summary['updates']!=5000:raise ValueError('Incomplete training')
        print(f'Finished {label}',flush=True)
    manifest['status']='completed';(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
