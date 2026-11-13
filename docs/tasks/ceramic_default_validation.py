"""Bounded canonical-default cold20k, independent input audit and final64 evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
INPUTS=Path('/root/autodl-tmp/data/particle_corrosion2d/20261107-spaced-six-01/inputs64')
REFERENCE=Path('/root/autodl-tmp/data/particle_corrosion2d/20261107-spaced-six-01/local_dev4')
TEST=Path('/root/autodl-tmp/data/particle_corrosion2d/local_test64_20261111-01')
CONTROL=Path('/root/autodl-tmp/tf-logs/particle_corrosion2d/ceramic-fem-comparison-20261113-02-nonlocal')


def utc():return datetime.now(timezone.utc).isoformat()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',started_utc=utc(),pid=os.getpid(),run=str(a.run),seed=a.seed,stages=[])
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             MPLBACKEND='Agg',PYTHONUNBUFFERED='1',JAX_ENABLE_COMPILATION_CACHE='false',
             XLA_FLAGS='',XLA_PYTHON_CLIENT_PREALLOCATE='true',XLA_PYTHON_CLIENT_MEM_FRACTION='.90')
    def save():
        partial=a.output/'journal.partial'
        partial.write_text(json.dumps(state,indent=2)+'\n');partial.replace(a.output/'journal.json')
    def stage(name,command,cpu):
        row=dict(name=name,command=command,status='running',started_utc=utc());state['stages'].append(row)
        stamp=time.perf_counter();save()
        with (a.output/(name+'.log')).open('x') as stream:
            child=subprocess.Popen([sys.executable,*command],cwd=ROOT,
                env=dict(env,JAX_PLATFORMS='cpu' if cpu else 'cuda'),stdout=stream,stderr=subprocess.STDOUT)
            row['pid']=child.pid;save()
            while True:
                try:code=child.wait(timeout=30);break
                except subprocess.TimeoutExpired:state['heartbeat_utc']=utc();save()
        row.update(exit_code=code,seconds=time.perf_counter()-stamp,status='passed' if code==0 else 'failed');save()
        if code:raise RuntimeError(f'{name} exited{code}; inspect retained log')
    save()
    try:
        stage('train',['-m','examples.particle_corrosion2d.train','--inputs',str(INPUTS),
                       '--reference',str(REFERENCE),'--exclude-reference',str(TEST),
                       '--seed',str(a.seed),'--output',str(a.run)],False)
        stage('online',['-m','examples.particle_corrosion2d.online_validation','--run',str(a.run),'--workers','4'],True)
        if a.seed==json.loads((CONTROL/'runtime.json').read_text())['seed']:
            for name in ('model_initial.eqx','optimizer_initial.eqx'):
                if (a.run/name).read_bytes()!=(CONTROL/name).read_bytes():raise ValueError('Initial state mismatch: '+name)
            for name in ('sampling.npz','online_sampling.npz','online_initial_inputs.npz','online_final_inputs.npz'):
                with np.load(a.run/name) as actual,np.load(CONTROL/name) as expected:
                    if set(actual.files)!=set(expected.files):raise ValueError('Array schema mismatch: '+name)
                    for key in actual.files:np.testing.assert_array_equal(actual[key],expected[key],err_msg=name+'/'+key)
            state['cold_and_complete_inputs_match']=True;save()
        stage('eval64',['-m','docs.tasks.pino_unified_ceramic_evaluate','--run',str(a.run),
                        '--reference',str(TEST),'--output',str(a.output/'eval64')],False)
        state['status']='complete';state['finished_utc']=utc();save()
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}',finished_utc=utc());save();raise


if __name__=='__main__':main()
