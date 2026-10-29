"""Entire canonical cold-start process wall time, with raw stdout."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

out=Path(__file__).resolve().parent
command=[sys.executable,'examples/icemelting3d/train.py',
    '--config=examples/icemelting3d/configs/train_operator.py',
    '--config.log.run_name=20261029-melting-operator-online-b16-01']
settings=dict(JAX_PLATFORMS='cuda',JAX_ENABLE_X64='false',XLA_PYTHON_CLIENT_PREALLOCATE='false',
    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
environment=os.environ.copy();environment.update(settings)
start=time.perf_counter()
with (out/'melting-operator-training.txt').open('x') as stream:
    result=subprocess.run(command,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=False)
record=dict(command=command,environment=settings,wall_seconds=time.perf_counter()-start,returncode=result.returncode)
(out/'melting-operator-process.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
if result.returncode:raise SystemExit(result.returncode)
