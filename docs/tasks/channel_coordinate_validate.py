"""Serial independent replay of map controls, then actual-stream comparison."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time


def main(pid,tag,baseline_tag,incomplete_map_only):
    root=Path(__file__).resolve().parents[2];python=root/'.venv/bin/python'
    data=Path('/root/autodl-tmp/data/channel_nsch2d')
    runs=Path('/root/autodl-tmp/tf-logs/channel_nsch2d')
    print(f'Waiting for coordinate-control driver {pid}',flush=True)
    while Path(f'/proc/{pid}').exists():time.sleep(2)
    modes=('tanh-preconditioned',) if incomplete_map_only else ('tanh','tanh-preconditioned')
    for mode in modes:
        run=runs/f'{tag}-channel-stream-{mode}-01'
        if json.loads((run/'summary.json').read_text())['status']!='completed':
            raise ValueError(f'Incomplete {mode} control')
        print(f'Auditing {mode}',flush=True)
        with (data/f'validate-{tag}-{mode}-01.log').open('x') as stream:
            subprocess.run([str(python),'-m','examples.channel_nsch2d.validate_run',
                '--run',str(run)],cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True)
        environment=os.environ.copy();environment['JAX_PLATFORMS']='cpu'
        subprocess.run([str(python),'-m','examples.channel_nsch2d.evaluate',
            '--fields',str(run/'prediction.npz'),'--output',str(run/'figures')],
            cwd=root,env=environment,check=True)
        subprocess.run([str(python),'-m','examples.channel_nsch2d.evaluate',
            '--fields',str(run/'diagnostic_fields.npz'),'--output',str(run/'dense-prediction'),
            '--field-label','Prediction'],cwd=root,env=environment,check=True)
    environment=os.environ.copy();environment['JAX_PLATFORMS']='cpu'
    command=[str(python),'docs/tasks/channel_gradnorm_compare.py',
        '--tag',baseline_tag,'--coordinate-tag',tag]
    if incomplete_map_only:command.append('--incomplete-map-only')
    subprocess.run(command,cwd=root,env=environment,check=True)
    print('Coordinate controls: physical replay and matched initial fields/inputs audited',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--driver-pid',type=int,required=True);p.add_argument('--tag',required=True)
    p.add_argument('--baseline-tag',required=True)
    p.add_argument('--incomplete-map-only',action='store_true')
    a=p.parse_args();main(a.driver_pid,a.tag,a.baseline_tag,a.incomplete_map_only)
