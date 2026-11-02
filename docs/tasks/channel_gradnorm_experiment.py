"""Run explicit cold/continued recipes serially on the current GPU; no FEM."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time


def main(tag,modes,after_pid,baseline_results):
    root=Path(__file__).resolve().parents[2]
    output=Path('/root/autodl-tmp/data/channel_nsch2d')
    runs=Path('/root/autodl-tmp/tf-logs/channel_nsch2d')
    if len(set(modes))!=len(modes):raise ValueError('Control modes must be unique')
    (output/f'{tag}-driver.pid').write_text(str(os.getpid())+'\n')
    if after_pid is not None:
        if baseline_results is None:raise ValueError('Queued controls require an explicit preceding validation receipt')
        print(f'Waiting for preceding validation driver {after_pid}',flush=True)
        while Path(f'/proc/{after_pid}').exists():time.sleep(2)
    if baseline_results is not None:
        prior=json.loads(Path(baseline_results).read_text())
        if prior['actual_draws_and_known_inputs_identical'] is not True or prior['initialization_audit']['common_trunk_exact'] is not True:
            raise ValueError('Preceding matched-control audit did not pass')
    hardware=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version',
        '--format=csv'],capture_output=True,text=True,check=True).stdout
    (output/f'{tag}-hardware.txt').write_text(hardware)
    records=[]
    recipes=dict(direct=('direct-gradnorm','train_direct_gradnorm.py'),
        stream=('stream-gradnorm','train_stream_gradnorm.py'),
        uniform=('stream-uniform','train_operator.py'),
        stagger=('stream-stagger','train_stream_stagger.py'),
        regions=('stream-regions','train_stream_regions.py'),
        tanh=('stream-tanh','train_stream_tanh.py'),
        tanh_preconditioned=('stream-tanh-preconditioned','train_stream_tanh_preconditioned.py'),
        single_bdf2=('single-bdf2','train_single_bdf2.py'),
        single_sbp=('single-sbp','train_single_conservative.py'),
        single_refined=('single-sbp-refined','train_single_conservative_refined.py'))
    for mode in modes:
        suffix,config=recipes[mode];name=f'{tag}-channel-{suffix}-01'
        log=output/f'train-{tag}-{mode}-01.log'
        command=[str(root/'.venv/bin/python'),'-m','examples.channel_nsch2d.train',
            f'--config=examples/channel_nsch2d/configs/{config}',
            f'--config.log.run_name={name}']
        start=time.perf_counter()
        print(f'Starting {mode}: {log}',flush=True)
        with log.open('x') as stream:subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True)
        summary=json.loads((runs/name/'summary.json').read_text())
        if summary['status']!='completed':raise ValueError(f'Incomplete {mode} control')
        records.append(dict(mode=mode,command=command,run=str(runs/name),log=str(log),
            process_seconds=time.perf_counter()-start,summary=summary))
        (output/f'{tag}-controls.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
        print(f'Completed {mode}: phi={summary["metrics"]["phi"]:.6f}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--modes',nargs='+',choices=('direct','stream','uniform','stagger','regions','tanh','tanh_preconditioned','single_bdf2','single_sbp','single_refined'),required=True)
    p.add_argument('--after-pid',type=int)
    p.add_argument('--baseline-results')
    args=p.parse_args();main(args.tag,args.modes,args.after_pid,args.baseline_results)
