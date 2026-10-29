from pathlib import Path
import subprocess,json,time
root=Path('/root/nonlocal-pinn');out=root/'docs/tasks/archive'
controls=[('20261028-chtanh-control-m8-01','train_coarsening_global_strong'),('20261028-chtanh-map-m8-01','train_coarsening_tanh'),('20261028-chtanh-jac-m8-01','train_coarsening_tanh_jacobian'),('20261028-chtanh-jac-m5-01','train_coarsening_tanh_lowfreq')]
records=[]
for name,config in controls:
    start=time.perf_counter();print('START',name,flush=True)
    with Path('/tmp/'+name+'.log').open('x') as log:
        result=subprocess.run([str(root/'.venv/bin/python'),'examples/confined_ch2d/train.py','--config=examples/confined_ch2d/configs/'+config+'.py','--config.log.run_name='+name,'--seed=42'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
    records.append(dict(name=name,config=config,exit_code=result.returncode,process_seconds=time.perf_counter()-start))
    (out/'chtanh-processes.json').write_text(json.dumps(records,indent=2)+'\n')
    print('END',records[-1],flush=True)
    if result.returncode:raise RuntimeError('Cold start failed; see /tmp/'+name+'.log')
