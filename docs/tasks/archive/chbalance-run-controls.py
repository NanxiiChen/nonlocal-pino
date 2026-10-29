"""Three serial cold starts through canonical train.py; stop on any failure."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=Path(__file__).parent
rows=[]
for label,config in [('joint','joint'),('gradnorm','gradnorm'),('stagger','stagger')]:
    name='20261028-chbalance-'+label+'-01'
    command=[sys.executable,'examples/confined_ch2d/train.py',
        '--config=examples/confined_ch2d/configs/train_coarsening_mixed_'+config+'.py',
        '--config.log.run_name='+name,'--seed=42']
    log=OUT/('chbalance-'+label+'-training.txt')
    row=dict(name=name,command=command,log=str(log),status='running')
    rows.append(row);path=OUT/'chbalance-processes.json'
    path.write_text(json.dumps(rows,indent=2)+'\n')
    start=time.perf_counter();print('Starting',name,flush=True)
    with log.open('x') as stream:
        process=subprocess.run(command,cwd=ROOT,env={**os.environ,'OPENBLAS_NUM_THREADS':'4'},stdout=stream,stderr=subprocess.STDOUT)
    row.update(returncode=process.returncode,process_seconds=time.perf_counter()-start,status='completed' if process.returncode==0 else 'failed')
    path.write_text(json.dumps(rows,indent=2)+'\n')
    if process.returncode:raise RuntimeError('Training failed: '+name+'; inspect '+str(log))
    print('Finished',name,row['process_seconds'],flush=True)
