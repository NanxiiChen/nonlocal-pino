"""One bounded/raw-edge/L1 control, queued until the first three workers exit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path('/root/autodl-tmp/tf-logs/corrosion2d')
FIRST=[ROOT/f'20261025-transport-{e}-ch{n}-s_bc-icmatched_graph-bcmatched_graph-phase-seed42'
       for e,n in (('physical_difference','l1'),('physical_difference','rms'),('kks_raw_edges','l1'))]
NAME='20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-bounded-seed42'

def main():
    while True:
        args=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
        workers=[line for line in args if line.split()[:4]==[sys.executable,'-u','-m','examples.corrosion2d.residual_experiment']]
        if all((p/'summary.json').is_file() for p in FIRST) and len(workers)<3:break
        time.sleep(40)
    run=ROOT/NAME
    if run.exists():raise FileExistsError(run)
    log=Path('/tmp')/(NAME+'.log')
    env=dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_FLAGS='--xla_gpu_deterministic_ops=true',JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-transport-cache')
    command=[sys.executable,'-u','-m','examples.corrosion2d.residual_experiment','train',
        '--case','corrosion2d','--formulation','bounded','--bounded-eta','0.1','--seed','42',
        '--data-loss-mode','s_bc','--initial-sampling','matched_graph','--boundary-sampling','matched_graph',
        '--transport-evaluation','kks_raw_edges','--ch-norm','l1']
    print('Starting',NAME,flush=True)
    with log.open('x') as stream:
        result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if run.is_dir():(run/'train.log').write_bytes(log.read_bytes())
    record={'path':str(run),'label':'bounded/kks_raw_edges/l1/seed42','formulation':'bounded',
        'transport_evaluation':'kks_raw_edges','ch_norm':'l1','bounded_eta':.1,'seed':42,
        'command':command,'returncode':result.returncode,'maximum_worker_processes':3}
    Path(__file__).with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    if result.returncode:raise RuntimeError(log.read_text())

if __name__=='__main__':main()
