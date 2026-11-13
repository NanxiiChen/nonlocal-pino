"""One original-source cold replay after the authorized three-arm queue finishes."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from docs.tasks.pino_unified_sampling_queue import gpu

ROOT=Path(__file__).resolve().parents[2]
PARENT=Path('/root/autodl-tmp/tf-logs/ceramic-fem-comparison-20261113-02')
OLD=Path('/root/autodl-tmp/tf-logs/particle_corrosion2d/pino-rolling-ad-cold-20261111-01-particle_corrosion2d-s42-nonlocal')
TEST=Path('/root/autodl-tmp/data/particle_corrosion2d/local_test64_20261111-01')


def read(p):return json.loads(p.read_text())
def utc():return datetime.now(timezone.utc).isoformat()


def main(output):
    output.mkdir(parents=True,exist_ok=False)
    (output/'runner.py').write_bytes(Path(__file__).read_bytes())
    state=dict(status='waiting',pid=os.getpid(),started_utc=utc(),dependency=str(PARENT),
               historical=str(OLD),stages=[])
    def save():
        p=output/'journal.partial';p.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
        p.replace(output/'journal.json')
    def interrupted(signum,frame):raise InterruptedError(f'Owner stopped by signal{signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    save()
    def stage(name,args,cwd,cpu):
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 MPLBACKEND='Agg',PYTHONUNBUFFERED='1',XLA_FLAGS='',
                 JAX_ENABLE_COMPILATION_CACHE='false',JAX_PLATFORMS='cpu' if cpu else 'cuda',
                 XLA_PYTHON_CLIENT_PREALLOCATE='true',XLA_PYTHON_CLIENT_MEM_FRACTION='.90')
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        else:
            card=gpu()
            if card['uuid']!='GPU-1e537971-8529-b7c1-1682-82073d2cae12' or card['used_MiB']>1024:
                raise RuntimeError(f'Expected same idle PRO6000: {card}')
        row=dict(label=name,status='running',command=args,cwd=str(cwd),started_utc=utc())
        state['stages'].append(row);child=None;tick=time.perf_counter()
        try:
            with (output/(name+'.log')).open('x') as f:
                child=subprocess.Popen([sys.executable,*args],cwd=cwd,env=env,
                    stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                row['pid']=child.pid;save()
                while True:
                    try:code=child.wait(timeout=30);break
                    except subprocess.TimeoutExpired:
                        state['heartbeat_utc']=utc();save()
            row.update(status='passed' if code==0 else 'failed',exit_code=code,seconds=time.perf_counter()-tick);save()
            if code:raise RuntimeError(name+' failed; no implicit retry')
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                row.update(status='interrupted',exit_code=child.returncode);save()
    try:
        while True:
            parent=read(PARENT/'journal.json')
            if parent['status']=='complete':break
            if parent['status']!='running':raise RuntimeError('Comparison not complete: '+parent['status'])
            state['heartbeat_utc']=utc();save();time.sleep(30)
        state['status']='running';save()
        source=OLD/'source'
        # Execute the historical canonical entrypoint from its saved directory,
        # with its original preset, seed and launcher flags. No source mutation.
        runtime=read(OLD/'runtime.json')
        run=Path('/root/autodl-tmp/tf-logs/particle_corrosion2d')/output.name
        state['run']=str(run);save()
        stage('old-source-train',['-m','examples.particle_corrosion2d.train','--config',
            str(source/'examples/particle_corrosion2d/configs/train_physical_spaced_w256_method_nonlocal.py'),
            '--inputs',runtime['inputs'],'--reference',runtime['reference'],
            '--seed',str(runtime['seed']),'--output',str(run)],source,False)
        stage('old-source-online',['-m','examples.particle_corrosion2d.online_validation',
            '--run',str(run),'--workers','4'],source,True)
        import numpy as np
        if read(run/'config.json')!=read(OLD/'config.json'):raise ValueError('Original replay configuration changed')
        for name in ('model_initial.eqx','optimizer_initial.eqx'):
            if (run/name).read_bytes()!=(OLD/name).read_bytes():raise ValueError('Original cold state differs')
        for name in ('sampling.npz','online_sampling.npz','core_trace.npz',
                     'online_initial_inputs.npz','online_final_inputs.npz'):
            with np.load(run/name) as a,np.load(OLD/name) as b:
                if set(a.files)!=set(b.files):raise ValueError('Replay schema differs: '+name)
                for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=name+'/'+key)
        from examples.particle_corrosion2d.online import geometry_identity
        seen={geometry_identity(g) for g in read(Path(runtime['inputs'])/'manifest.json')['geometries']}
        for line in (run/'online_refresh.jsonl').read_text().splitlines():
            seen.update(geometry_identity(r['geometry']) for r in json.loads(line)['records'])
        heldout={geometry_identity(r['geometry']) for r in read(TEST/'manifest.json')['cases']}
        if seen & heldout:raise ValueError('Historical replay overlaps the final reference cohort')
        state['original_inputs_and_cold_state_identical']=True;save()
        stage('old-source-eval64',['-m','docs.tasks.pino_unified_ceramic_evaluate',
            '--run',str(run),'--reference',str(TEST),'--output',str(output/'eval64')],ROOT,False)
        state['status']='complete'
    except InterruptedError as error:state.update(status='stopped_by_user',reason=str(error))
    except Exception as error:state.update(status='failed',failure=str(error));raise
    finally:state['finished_utc']=utc();save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
