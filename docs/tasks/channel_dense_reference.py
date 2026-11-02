"""Wait for GPU audits, rebuild dense FEM exports, then evaluate the frozen NO."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def evaluate_frozen(run,output,log_name):
    root=Path(__file__).resolve().parents[2];run=Path(run);output=Path(output)
    gate=output/'reference-validation.json'
    if json.loads(gate.read_text())['status']!='passed':raise ValueError('A passed dense FEM gate is required')
    with (output/log_name).open('x') as stream:
        subprocess.run([str(root/'.venv/bin/python'),'docs/tasks/channel_dense_evaluate.py',
            '--run',str(run),'--validation',str(gate)],cwd=root,
            stdout=stream,stderr=subprocess.STDOUT,check=True)
    cpu=dict(os.environ);cpu['JAX_PLATFORMS']='cpu'
    subprocess.run([str(root/'.venv/bin/python'),'-m','examples.channel_nsch2d.evaluate',
        '--fields',str(run/'dense-fem-prediction.npz'),'--output',str(run/'dense-fem-figures')],cwd=root,env=cpu,check=True)
    print('Dense independently refined FEM and frozen-checkpoint evaluation completed',flush=True)


def main(pid,run,output):
    root=Path(__file__).resolve().parents[2];run=Path(run);output=Path(output)
    print(f'Waiting for GPU validation driver {pid}',flush=True)
    while Path(f'/proc/{pid}').exists():time.sleep(2)
    if json.loads((run/'validation.json').read_text())['status']!='replay_and_hard_constraints_passed':
        raise ValueError('Complete independent GPU replay must precede dense FEM')
    while True:
        active=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
            capture_output=True,text=True,check=True).stdout.strip()
        if not active:break
        print(f'Waiting for GPU processes to exit: {active}',flush=True);time.sleep(10)
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads((run/'config.json').read_text());a=config['geometry'];env=dict(os.environ)
    env.pop('VIRTUAL_ENV',None)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    fenics=Path('/root/miniconda3/envs/fenicsenv/bin/python')
    if not fenics.is_file():raise FileNotFoundError(fenics)
    records=[]
    for tag,grid,dt in (('coarse',(40,16),.025),('time',(64,24),.05),('fine',(64,24),.025)):
        path=output/f'bdf-{tag}.npz'
        command=[str(fenics),'-m','examples.channel_nsch2d.solver','--grid',*[str(v) for v in grid],
            '--dt',str(dt),'--query-grid','96','40','--upper',str(a['upper']),'--lower',str(a['lower']),'--output',str(path)]
        print(f'Solving dense FEM export {tag}',flush=True)
        with (output/f'{tag}.log').open('x') as stream:
            subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        records.append(dict(tag=tag,path=str(path),command=command))
        (output/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.validate_reference import assess
    c=ConfigDict(config);c.reference.root=str(output)
    assess(c,output/'bdf-coarse.npz',output/'bdf-time.npz',output/'bdf-fine.npz')
    evaluate_frozen(run,output,'model-evaluation.log')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--after-pid',required=True,type=int)
    p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--evaluation-only',action='store_true',help='Reuse passed FEM exports; retain original failed evaluation log')
    a=p.parse_args()
    if a.evaluation_only:evaluate_frozen(a.run,a.output,'model-evaluation-recovery.log')
    else:main(a.after_pid,a.run,a.output)
