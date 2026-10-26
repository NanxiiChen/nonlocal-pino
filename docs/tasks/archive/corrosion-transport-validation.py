"""Queued cold starts: raw/L1 seed7 plus paired late-rate seed42/7 controls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/root/autodl-tmp/tf-logs/corrosion2d')
PLAN=[(42,3000),(7,0),(7,3000)]
LIMIT=3

def workers():
    rows=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
    return sum(line.split()[:4]==[sys.executable,'-u','-m','examples.corrosion2d.residual_experiment'] for line in rows)

def launch(seed,tail):
    name=f'20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-phase-seed{seed}'
    if tail:name+=f'-tail{tail}-end1e-05'
    run=ROOT/name
    if run.exists():raise FileExistsError(run)
    logfile=Path('/tmp')/(name+'.log');stream=logfile.open('x')
    command=[sys.executable,'-u','-m','examples.corrosion2d.residual_experiment','train',
        '--case','corrosion2d','--formulation','phase','--seed',str(seed),
        '--data-loss-mode','s_bc','--initial-sampling','matched_graph','--boundary-sampling','matched_graph',
        '--transport-evaluation','kks_raw_edges','--ch-norm','l1']
    if tail:command+=['--lr-tail-start',str(tail),'--lr-tail-end','1e-5']
    env=dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_FLAGS='--xla_gpu_deterministic_ops=true',JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-transport-cache')
    label=f'phase/kks_raw_edges/l1/seed{seed}'+(f'/tail{tail}' if tail else '')
    process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
    print('Starting',name,process.pid,flush=True)
    return process,stream,logfile,{'label':label,'path':str(run),'seed':seed,'lr_tail_start':tail,
        'lr_tail_end':1e-5,'command':command,'maximum_worker_processes':LIMIT}

def main():
    # All first-cohort processes have already captured their original code;
    # their paired source/config records are preserved independently.
    pending=list(PLAN);active=[];records=[]
    while pending or active:
        while pending and workers()<LIMIT:
            active.append(launch(*pending.pop(0)))
        remaining=[]
        for process,stream,logfile,record in active:
            status=process.poll()
            if status is None:
                remaining.append((process,stream,logfile,record));continue
            stream.close();run=Path(record['path'])
            if run.is_dir():(run/'train.log').write_bytes(logfile.read_bytes())
            record['returncode']=status;records.append(record)
            destination=Path(__file__).with_suffix('.json');temporary=destination.with_suffix('.tmp')
            temporary.write_text(json.dumps(records,indent=2)+'\n');temporary.replace(destination)
            print('Finished',record['label'],status,flush=True)
            if status:print(logfile.read_text(),flush=True)
        active=remaining
        if pending or active:time.sleep(20)
    if any(r['returncode'] for r in records):raise SystemExit('Failed refinement/validation; inspect logs')

if __name__=='__main__':main()
