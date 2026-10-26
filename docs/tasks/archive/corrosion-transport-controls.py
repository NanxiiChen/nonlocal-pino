"""Matched2x2 transport/norm phase controls and one latent check; cold5000 steps."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/root/autodl-tmp/tf-logs/corrosion2d')
PLAN=[('phase',evaluation,norm) for evaluation in ('physical_difference','kks_raw_edges') for norm in ('l1','rms')]
PLAN.append(('latent','kks_raw_edges','rms'))

def execute(form,evaluation,norm):
    name=f'20261025-transport-{evaluation}-ch{norm}-s_bc-icmatched_graph-bcmatched_graph-{form}-seed42'
    run=ROOT/name;log=Path('/tmp')/(name+'.log')
    if run.exists():raise FileExistsError(run)
    command=[sys.executable,'-u','-m','examples.corrosion2d.residual_experiment','train',
        '--case','corrosion2d','--formulation',form,'--seed','42','--data-loss-mode','s_bc',
        '--initial-sampling','matched_graph','--boundary-sampling','matched_graph',
        '--transport-evaluation',evaluation,'--ch-norm',norm]
    env=dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_FLAGS='--xla_gpu_deterministic_ops=true',
        JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-transport-cache')
    print('Starting',name,flush=True)
    with log.open('x') as stream:
        completed=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if run.is_dir():(run/'train.log').write_bytes(log.read_bytes())
    record={'path':str(run),'formulation':form,'transport_evaluation':evaluation,'ch_norm':norm,
        'seed':42,'command':command,'xla_flags':env['XLA_FLAGS'],'returncode':completed.returncode,
        'maximum_worker_processes':3}
    if completed.returncode:print('FAILED',name,log.read_text(),flush=True)
    else:print('Finished',name,json.dumps(json.loads((run/'summary.json').read_text())['final_metrics']),flush=True)
    return record

def main():
    records=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(execute,*item) for item in PLAN]
        for future in as_completed(futures):
            records.append(future.result())
            Path(__file__).with_suffix('.json').write_text(json.dumps(records,indent=2)+'\n')
    if any(r['returncode'] for r in records):raise SystemExit('Failed controls; inspect logs')

if __name__=='__main__':main()
