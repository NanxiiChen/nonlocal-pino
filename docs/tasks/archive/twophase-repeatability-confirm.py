"""Additional recordings and independent-process repeat of the symmetric probe."""
from pathlib import Path
import json
import os
import subprocess
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/twophase-repeatability'
records=[]
for name,adjoint,steps in [('ordinary-autodiff-spectrum','autodiff',2),('ordinary-symmetric-independent','symmetric',100)]:
    cmd=[str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).with_name('twophase-repeatability-probe.py')),
         '--policy=ordinary','--adjoint='+adjoint,'--steps='+str(steps),'--output-name='+name]
    env=dict(os.environ,JAX_PLATFORMS='cuda',XLA_FLAGS='',JAX_ENABLE_COMPILATION_CACHE='false',
             XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
    with (OUT/(name+'.log')).open('x') as log:result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    records.append(dict(command=cmd,returncode=result.returncode))
    (OUT/'confirmation-manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    if result.returncode:raise RuntimeError(name+' failed')
    print(name+' completed',flush=True)
