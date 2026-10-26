"""Second-seed paired validation of the field-improving BC-only candidate."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/root/autodl-tmp/tf-logs/corrosion2d')

def execute(mode):
    name=f'20261025-data-{mode}-icmatched_graph-bcmatched_graph-phase-stagger0-seed7'
    run=ROOT/name;log=Path('/tmp')/(name+'.log')
    if run.exists():raise FileExistsError(run)
    env=dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_FLAGS='--xla_gpu_deterministic_ops=true',JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-normalized-data-cache')
    command=[sys.executable,'-u','-m','examples.corrosion2d.residual_experiment','train',
             '--case','corrosion2d','--formulation','phase','--seed','7',
             '--data-loss-mode',mode,'--initial-sampling','matched_graph','--boundary-sampling','matched_graph']
    print('Starting',name,flush=True)
    with log.open('x') as stream:
        completed=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if run.is_dir():(run/'train.log').write_bytes(log.read_bytes())
    record={'path':str(run),'data_loss_mode':mode,'seed':7,'command':command,
            'returncode':completed.returncode,'xla_flags':env['XLA_FLAGS']}
    if completed.returncode:print('FAILED',name,log.read_text(),flush=True)
    else:print('Finished',name,json.dumps(json.loads((run/'summary.json').read_text())['final_metrics']),flush=True)
    return record

def main():
    records=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(execute,mode) for mode in ('physical','s_bc')]
        for future in as_completed(tasks):
            records.append(future.result())
            Path(__file__).with_suffix('.json').write_text(json.dumps(records,indent=2)+'\n')
    if any(r['returncode'] for r in records):raise SystemExit('Second-seed controls failed')

if __name__=='__main__':main()
