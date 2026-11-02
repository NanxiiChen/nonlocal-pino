"""Wait for serial training to exit, then audit/plot each control sequentially."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time


def main(pid,tag):
    root=Path(__file__).resolve().parents[2];python=root/'.venv/bin/python'
    output=Path('/root/autodl-tmp/data/channel_nsch2d')
    runs=Path('/root/autodl-tmp/tf-logs/channel_nsch2d')
    print(f'Waiting for training driver {pid} to exit',flush=True)
    while Path(f'/proc/{pid}').exists():time.sleep(2)
    for mode in ('direct','stream','uniform'):
        run=runs/(f'{tag}-channel-stream-uniform-01' if mode=='uniform' else f'{tag}-channel-{mode}-gradnorm-01')
        summary=json.loads((run/'summary.json').read_text())
        if summary['status']!='completed':raise ValueError(f'Incomplete {mode} run')
        print(f'Auditing {mode}',flush=True)
        with (output/f'validate-{tag}-{mode}-01.log').open('x') as stream:
            subprocess.run([str(python),'-m','examples.channel_nsch2d.validate_run','--run',str(run)],
                cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True)
        subprocess.run([str(python),'-m','examples.channel_nsch2d.evaluate','--fields',str(run/'prediction.npz'),
            '--output',str(run/'figures')],cwd=root,check=True)
    environment=os.environ.copy();environment['JAX_PLATFORMS']='cpu'
    subprocess.run([str(python),'docs/tasks/channel_gradnorm_compare.py','--tag',tag],cwd=root,env=environment,check=True)
    print('All independent GPU audits, CPU array/weight audits and plots completed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--driver-pid',type=int,required=True)
    p.add_argument('--tag',required=True)
    args=p.parse_args();main(args.driver_pid,args.tag)
