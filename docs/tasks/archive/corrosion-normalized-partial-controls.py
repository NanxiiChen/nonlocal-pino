"""Matched deterministic GPU controls; three independent processes at most.

Adopts the unchanged first worker after replacing its serial launcher.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/root/autodl-tmp/tf-logs/corrosion2d')
PLAN=[(form,mode) for form in ('phase','latent') for mode in ('physical','s_bc','s_bc_ic')]
BASELINE_PID=47641


def execute(form,mode):
    env=dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_FLAGS='--xla_gpu_deterministic_ops=true',
               JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-normalized-data-cache')
    name=f'20261025-data-{mode}-icmatched_graph-{form}-stagger0-seed42'
    run=ROOT/name;log=Path('/tmp')/f'{name}.log'
    command=[sys.executable,'-u','-m','examples.corrosion2d.residual_experiment','train',
             '--case','corrosion2d','--formulation',form,'--seed','42',
             '--data-loss-mode',mode,'--initial-sampling','matched_graph']
    adopted=(form,mode)==('phase','physical')
    if adopted:
        if not run.is_dir():raise FileNotFoundError(run)
        print('Adopting unchanged baseline worker',BASELINE_PID,flush=True)
        while not (run/'summary.json').is_file():
            try:os.kill(BASELINE_PID,0)
            except ProcessLookupError as error:raise RuntimeError('Baseline worker exited without summary') from error
            time.sleep(1)
        returncode=0
    else:
        if run.exists():raise FileExistsError(run)
        print('Starting',name,flush=True)
        with log.open('x') as stream:
            result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        returncode=result.returncode
    if run.is_dir():(run/'train.log').write_bytes(log.read_bytes())
    record={'path':str(run),'formulation':form,'data_loss_mode':mode,'seed':42,
            'returncode':returncode,'command':command,'xla_flags':env['XLA_FLAGS'],
            'execution':'adopted original worker' if adopted else 'independent concurrent worker',
            'maximum_worker_processes':3}
    if returncode:
        print('FAILED',name,log.read_text(),flush=True)
    else:
        summary=json.loads((run/'summary.json').read_text())
        print('Finished',name,json.dumps(summary['final_metrics']),flush=True)
    return record


def main():
    records=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(execute,*item) for item in PLAN]
        for future in as_completed(futures):
            records.append(future.result())
            Path(__file__).with_suffix('.json').write_text(json.dumps(records,indent=2)+'\n')
    if any(r['returncode'] for r in records):raise SystemExit('Run failures; see exact commands/logs')

if __name__=='__main__':main()
