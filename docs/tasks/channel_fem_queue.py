"""Finite full-coordinate comparison: screen, current-card probes, FE20k, audits."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
LOGS=Path('/root/autodl-tmp/tf-logs/channel_nsch2d')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--prefix',required=True)
    args=parser.parse_args();args.bundle.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,JAX_PLATFORMS='cuda',XLA_FLAGS='',JAX_ENABLE_COMPILATION_CACHE='false',
        XLA_PYTHON_CLIENT_PREALLOCATE='true',XLA_PYTHON_CLIENT_MEM_FRACTION='.97',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
    state=dict(status='running',pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
        gpu=gpu,prefix=args.prefix,stages=[],environment={k:env[k]for k in (
            'XLA_FLAGS','JAX_ENABLE_COMPILATION_CACHE','XLA_PYTHON_CLIENT_MEM_FRACTION','OPENBLAS_NUM_THREADS')})

    def save():
        tmp=args.bundle/'journal.partial';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(args.bundle/'journal.json')

    def stage(name,command,*,cpu=False):
        entry=dict(name=name,command=command,status='running');state['stages'].append(entry);save()
        start=time.perf_counter()
        with (args.bundle/(name+'.log')).open('x')as stream:
            child=subprocess.Popen([sys.executable,*command],cwd=ROOT,
                env=dict(env,JAX_PLATFORMS='cpu'if cpu else 'cuda'),stdout=stream,stderr=subprocess.STDOUT)
            entry['pid']=child.pid;save()
            while True:
                try:code=child.wait(timeout=30);break
                except subprocess.TimeoutExpired:
                    state['heartbeat_utc']=datetime.now(timezone.utc).isoformat();save()
        entry.update(status='passed'if code==0 else 'failed',exit_code=code,seconds=time.perf_counter()-start);save()
        if code:raise RuntimeError(f'{name} failed; inspect preserved log')

    def support(action):
        return ['-m','docs.tasks.channel_fem_support','--action',action,'--bundle',str(args.bundle),'--prefix',args.prefix]

    def train(config,name,probe):
        command=['-m','examples.channel_nsch2d.train',f'--config=examples/channel_nsch2d/configs/{config}.py',
                 '--seed=42',f'--config.log.run_name={name}']
        if probe:command+=['--config.train.iterations=60']
        return command

    save()
    try:
        stage('screen',support('screen'),cpu=True)
        stage('probe-nonlocal',train('train_full_fem_comparison_probe',args.prefix+'-probe-nonlocal',True))
        stage('probe-fem',train('train_operator_fem_weak',args.prefix+'-probe-fem',True))
        stage('qualify',support('qualify'),cpu=True)
        run=LOGS/(args.prefix+'-fem-cold20k')
        stage('train-fem',train('train_operator_fem_weak',run.name,False))
        stage('validate',['-m','examples.channel_nsch2d.validate_run','--run',str(run)])
        stage('cohort',['-m','examples.channel_nsch2d.cohort_evaluate','--run',str(run),
            '--manifest','/root/autodl-tmp/data/channel_nsch2d/random-walls-20261102-02/unseen3/manifest.json',
            '--output',str(run/'unseen3')])
        stage('finish',support('finish'),cpu=True)
        state.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat());save()
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');save();raise


if __name__=='__main__':main()
