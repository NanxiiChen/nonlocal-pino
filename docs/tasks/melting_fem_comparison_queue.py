"""Finite paired 3D melting probes, cold10k runs and independent final audit."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
LOGS=Path('/root/autodl-tmp/tf-logs/icemelting3d')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prefix',required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),stages=[])
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             PYTHONUNBUFFERED='1',JAX_ENABLE_COMPILATION_CACHE='false',XLA_FLAGS='',
             XLA_PYTHON_CLIENT_PREALLOCATE='true',XLA_PYTHON_CLIENT_MEM_FRACTION='.90')
    def save():
        p=args.output/'journal.partial';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(args.output/'journal.json')
    def stage(name,command,cpu=False):
        row=dict(name=name,status='running',command=command);state['stages'].append(row);start=time.perf_counter();save()
        with (args.output/(name+'.log')).open('x') as stream:
            child=subprocess.Popen([sys.executable,*command],cwd=ROOT,env=dict(env,JAX_PLATFORMS='cpu' if cpu else 'cuda'),stdout=stream,stderr=subprocess.STDOUT)
            row['pid']=child.pid;save()
            while True:
                try:code=child.wait(timeout=30);break
                except subprocess.TimeoutExpired:state['heartbeat_utc']=datetime.now(timezone.utc).isoformat();save()
        row.update(status='passed' if code==0 else 'failed',exit_code=code,seconds=time.perf_counter()-start);save()
        if code:raise RuntimeError(f'{name} failed; inspect its preserved log')
    def command(method,name,memory):
        cmd=['-m','examples.icemelting3d.train',f'--config=examples/icemelting3d/configs/train_operator_compare_{method}.py','--seed=42',f'--config.log.run_name={name}']
        if memory:cmd+=['--config.operator.action=memory','--config.operator.probe_steps=12']
        return cmd
    save()
    try:
        probes={}
        for method in ('nonlocal','fem'):
            name=args.prefix+'-probe-'+method;stage('probe-'+method,command(method,name,True))
            summary=json.loads((LOGS/name/'summary.json').read_text());probes[method]=summary
            if summary['peak_active_GiB']>75:raise RuntimeError('Probe active memory exceeds the declared75GiB budget')
        state['probes']=probes;save()
        runs={}
        for method in ('nonlocal','fem'):
            name=args.prefix+'-'+method;runs[method]=str(LOGS/name)
            stage('train-'+method,command(method,name,False))
        state['runs']=runs;save()
        stage('evaluate',['-m','docs.tasks.melting_fem_comparison_evaluate','--nonlocal-run',runs['nonlocal'],
                         '--fem-run',runs['fem'],'--output',str(args.output/'comparison.json')],True)
        state.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat());save()
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');save();raise


if __name__=='__main__':main()
