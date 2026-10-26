"""Fresh deterministic GPU normalization controls, no overwrite or resume."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/root/autodl-tmp/tf-logs/corrosion2d')
PLAN=[(form,mode) for form in ('phase','latent') for mode in ('physical','s_bc','s_bc_ic')]

def main():
    env=dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_FLAGS='--xla_gpu_deterministic_ops=true',
               JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-normalized-data-cache')
    records=[]
    for form,mode in PLAN:
        name=f'20261025-data-{mode}-icmatched_graph-{form}-stagger0-seed42'
        run=ROOT/name;log=Path('/tmp')/f'{name}.log'
        if run.exists():raise FileExistsError(run)
        command=[sys.executable,'-u','-m','examples.corrosion2d.residual_experiment','train',
                 '--case','corrosion2d','--formulation',form,'--seed','42',
                 '--data-loss-mode',mode,'--initial-sampling','matched_graph']
        print('Starting',name,flush=True)
        with log.open('x') as stream:
            finished=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
        if run.is_dir():(run/'train.log').write_bytes(log.read_bytes())
        record={'path':str(run),'formulation':form,'data_loss_mode':mode,'seed':42,
                'returncode':finished.returncode,'command':command,'xla_flags':env['XLA_FLAGS']}
        records.append(record)
        Path(__file__).with_suffix('.json').write_text(json.dumps(records,indent=2)+'\n')
        if finished.returncode:
            print('FAILED',name,log.read_text(),flush=True)
        else:
            summary=json.loads((run/'summary.json').read_text())
            print('Finished',name,json.dumps(summary['final_metrics']),flush=True)
    if any(r['returncode'] for r in records):raise SystemExit('Run failures; see exact commands/logs')

if __name__=='__main__':main()
