"""Two real GPU canonical starts, using selected defaults without policy overrides."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/ordinary-policy-defaults-20261026'


def main():
    rows=[]
    for case,dtype in [('corrosion2d','float64'),('electropolish2d','float32')]:
        run=OUT/('canonical-smoke-'+case)
        command=[sys.executable,f'examples/{case}/train.py','--config.train.iterations=2',
                 '--config.log.eval_every=1','--config.experiment.checkpoint_every=1',
                 f'--config.log.log_dir={OUT}','--config.log.run_name='+run.name]
        env=dict(os.environ,JAX_PLATFORMS='cuda',JAX_ENABLE_X64='false',XLA_FLAGS='',
                 XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
                 JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-local-ad-performance-cache')
        with (OUT/(case+'-canonical-smoke.log')).open('x') as log:
            p=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError(f'Failed smoke: {case}')
        c=json.loads((run/'config.json').read_text());s=json.loads((run/'summary.json').read_text());precision=json.loads((run/'precision.json').read_text())
        if c['train']['gpu_policy']!='ordinary' or c['train']['compute_dtype']!=dtype or s['updates']!=2:raise ValueError('Wrong canonical defaults')
        for key in ('parameters','parameters_final','data','optimizer_initial','optimizer_final'):
            if precision[key]['dtype']!=dtype:raise ValueError('Wrong canonical dtype')
        rows.append({'case':case,'command':command,'run':str(run),'returncode':p.returncode,'updates':s['updates'],'dtype':dtype,'policy':c['train']['gpu_policy']})
        print(rows[-1],flush=True)
    (OUT/'canonical-smokes.json').write_text(json.dumps({'status':'passed','runs':rows},indent=2)+'\n')


if __name__=='__main__':main()
