"""Validate cap11/raw/L1 field gain with seed7 after queued controls have started."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path('/root/autodl-tmp/tf-logs/corrosion2d')
PREFIX='20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-phase-seed'
LATER=[ROOT/(PREFIX+'42-tail3000-end1e-05'),ROOT/(PREFIX+'7'),ROOT/(PREFIX+'7-tail3000-end1e-05')]
NAME='20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-bounded-seed7'

def main():
    while True:
        args=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
        workers=sum(line.split()[:4]==[sys.executable,'-u','-m','examples.corrosion2d.residual_experiment'] for line in args)
        if workers<3 and all((p/'source/files.json').is_file() for p in LATER):break
        time.sleep(20)
    run=ROOT/NAME
    if run.exists():raise FileExistsError(run)
    log=Path('/tmp')/(NAME+'.log')
    command=[sys.executable,'-u','-m','examples.corrosion2d.residual_experiment','train',
        '--case','corrosion2d','--formulation','bounded','--bounded-eta','0.1','--seed','7',
        '--data-loss-mode','s_bc','--initial-sampling','matched_graph','--boundary-sampling','matched_graph',
        '--transport-evaluation','kks_raw_edges','--ch-norm','l1']
    env=dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_FLAGS='--xla_gpu_deterministic_ops=true',JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-transport-cache')
    print('Starting',NAME,flush=True)
    with log.open('x') as stream:result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if run.is_dir():(run/'train.log').write_bytes(log.read_bytes())
    record={'label':'bounded/kks_raw_edges/l1/seed7','path':str(run),'seed':7,'formulation':'bounded',
        'bounded_eta':.1,'lr_tail_start':0,'command':command,'returncode':result.returncode,'maximum_worker_processes':3}
    Path(__file__).with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    if result.returncode:raise RuntimeError(log.read_text())

if __name__=='__main__':main()
