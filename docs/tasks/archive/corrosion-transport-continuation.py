"""Final paired cold starts:4000 L1 updates then1000 CH-RMS updates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path('/root/autodl-tmp/tf-logs/corrosion2d')

def workers():
    return sum(line.split()[:4]==[sys.executable,'-u','-m','examples.corrosion2d.residual_experiment']
        for line in subprocess.check_output(['ps','-eo','args'],text=True).splitlines())

def launch(seed):
    name=f'20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-phase-seed{seed}-rmsafter4000'
    run=ROOT/name
    if run.exists():raise FileExistsError(run)
    log=Path('/tmp')/(name+'.log');stream=log.open('x')
    command=[sys.executable,'-u','-m','examples.corrosion2d.residual_experiment','train',
        '--case','corrosion2d','--formulation','phase','--seed',str(seed),
        '--data-loss-mode','s_bc','--initial-sampling','matched_graph','--boundary-sampling','matched_graph',
        '--transport-evaluation','kks_raw_edges','--ch-norm','l1','--ch-rms-after','4000']
    env=dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_FLAGS='--xla_gpu_deterministic_ops=true',JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-transport-cache')
    process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
    record={'label':f'phase/kks_raw_edges/l1/seed{seed}/rmsafter4000','path':str(run),
        'seed':seed,'ch_rms_after':4000,'lr_tail_start':0,'command':command,'maximum_worker_processes':3}
    print('Starting',name,process.pid,flush=True)
    return process,stream,log,record

def main():
    pending=[42,7];active=[];records=[]
    while pending or active:
        while pending and workers()<3:active.append(launch(pending.pop(0)))
        remaining=[]
        for process,stream,log,record in active:
            status=process.poll()
            if status is None:
                remaining.append((process,stream,log,record));continue
            stream.close();run=Path(record['path'])
            if run.is_dir():(run/'train.log').write_bytes(log.read_bytes())
            record['returncode']=status;records.append(record)
            output=Path(__file__).with_suffix('.json');temporary=output.with_suffix('.tmp')
            temporary.write_text(json.dumps(records,indent=2)+'\n');temporary.replace(output)
            print('Finished',record['label'],status,flush=True)
            if status:print(log.read_text(),flush=True)
        active=remaining
        if pending or active:time.sleep(20)
    if any(r['returncode'] for r in records):raise SystemExit('Failed continuation control; inspect logs')

if __name__=='__main__':main()
