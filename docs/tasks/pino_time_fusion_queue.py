"""Finite two-case nonlocal temporal-fusion controls after the melting owner."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from docs.tasks.electro_threeway_queue import process_ticks,read,write

ROOT=Path(__file__).resolve().parents[2]
LOGS=Path('/root/autodl-tmp/tf-logs')
CASES=(('electropolish','electropolish2d'),('melting','icemelting3d'))


def train_command(case,example,prefix,probe):
    run=prefix+'-'+case+('-probe' if probe else '-joint_query')
    command=['-m','examples.'+example+'.train','--seed=42',
        '--config=examples/'+example+'/configs/train_operator_joint_query.py',
        '--config.log.run_name='+run]
    if probe:
        if case=='electropolish':
            command+=['--config.train.iterations=30','--config.log.checkpoint_step=0']
        else:
            command+=['--config.operator.action=memory','--config.operator.probe_steps=12']
    return command,LOGS/example/run


def qualify_probe(case,run):
    records=read(run/'memory.json')
    key='allocator_stats' if case=='electropolish' else 'allocator'
    peak=max(r[key]['peak_bytes_in_use'] for r in records)/1024**3
    if peak>70:raise RuntimeError('Joint-query probe exceeds explicit70GiBactive-memory budget')
    times=read(run/'timing.json')
    n=30 if case=='electropolish' else 12
    if any(len(times[k])!=n for k in ('update_seconds','sample_seconds','spatial_seconds')):
        raise ValueError('Incomplete synchronized resource probe')
    seconds=sum(sum(times[k][1:]) for k in ('update_seconds','sample_seconds','spatial_seconds'))/(n-1)
    if not 0<seconds<3600 or not 0<peak<70:raise ValueError('Invalid measured probe resources')
    return dict(peak_active_gib=peak,warm_inclusive_mean_seconds=seconds,
        projected_10000_update_hours=seconds*10000/3600,
        scope='Measured probe includes sampling and updates; projection excludes setup/full-test evaluation')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--prefix',required=True)
    p.add_argument('--wait-journal',type=Path,required=True)
    p.add_argument('--wait-pid',type=int,required=True);p.add_argument('--wait-start-ticks',type=int,required=True)
    p.add_argument('--electro-baseline',type=Path,required=True)
    p.add_argument('--melting-baseline',type=Path,required=True)
    args=p.parse_args();args.bundle.mkdir(parents=True,exist_ok=False)
    state=dict(status='waiting',pid=os.getpid(),prefix=args.prefix,stages=[],
        started_utc=datetime.now(timezone.utc).isoformat(),wait_journal=str(args.wait_journal),
        wait_pid=args.wait_pid,wait_start_ticks=args.wait_start_ticks)
    environment=dict(os.environ,XLA_FLAGS='',JAX_ENABLE_X64='false',JAX_ENABLE_COMPILATION_CACHE='false',
        XLA_PYTHON_CLIENT_PREALLOCATE='true',XLA_PYTHON_CLIENT_MEM_FRACTION='.8',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    def save():
        state['heartbeat_utc']=datetime.now(timezone.utc).isoformat();write(args.bundle/'journal.json',state)
    def stage(name,command,cpu=False,plot=False):
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version',
            '--format=csv,noheader'],text=True).strip()
        occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if gpu!=state['gpu'] or occupied:raise RuntimeError('GPU changed or busy before serial stage: '+occupied)
        entry=dict(name=name,status='running',command=command);state['stages'].append(entry);save()
        env=dict(environment,JAX_PLATFORMS='cpu' if cpu else 'cuda')
        if plot:env['PYTHONPATH']='/tmp/pino-pdf-audit'
        start=time.perf_counter()
        with (args.bundle/(name+'.log')).open('x') as stream:
            child=subprocess.Popen([sys.executable,*command],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            entry['pid']=child.pid;save()
            while True:
                try:code=child.wait(timeout=30);break
                except subprocess.TimeoutExpired:save()
        entry.update(status='passed' if code==0 else 'failed',exit_code=code,seconds=time.perf_counter()-start);save()
        if code:raise RuntimeError(name+' failed; remaining stages not launched')
    save()
    try:
        while True:
            dependency=read(args.wait_journal)
            if dependency['pid']!=args.wait_pid:raise RuntimeError('Dependency owner changed')
            if dependency['status']=='complete':break
            if dependency['status'] not in ('waiting','running'):raise RuntimeError('Dependency did not complete successfully')
            if process_ticks(args.wait_pid)!=args.wait_start_ticks:raise RuntimeError('Dependency process changed')
            save();time.sleep(30)
        state['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version',
            '--format=csv,noheader'],text=True).strip();state['status']='running';save()
        comparisons={}
        for (case,example),baseline in zip(CASES,(args.electro_baseline,args.melting_baseline),strict=True):
            if read(baseline/'comparison.json')['status']!='passed':raise ValueError('Passed nonlocal baseline required')
            command,probe=train_command(case,example,args.prefix,True);stage('probe-'+case,command)
            state[case+'_probe']=qualify_probe(case,probe);save()
            command,run=train_command(case,example,args.prefix,False);stage('train-'+case,command)
            if case=='electropolish':stage('evaluate-'+case,['-m','docs.tasks.electro_spatial_evaluate','--run',str(run)])
            report=args.bundle/(case+'-comparison.json')
            stage('audit-'+case,['-m','docs.tasks.pino_time_fusion_support','--case',case,
                '--baseline-report',str(baseline/'comparison.json'),'--run',str(run),
                '--baseline-journal',str(baseline/'journal.json'),'--joint-journal',str(args.bundle/'journal.json'),
                '--output',str(report)],cpu=True)
            stage('render-'+case,['-m','scripts.render_pino_time_fusion','--comparison',str(report),
                '--output',str(args.bundle/(case+'-figures')),'--x-axis','step'],cpu=True,plot=True)
            comparisons[case]=str(report)
        write(args.bundle/'comparison.json',dict(status='passed',comparisons=comparisons,
            test_count=64,steps=10000,spatial_derivatives='nonlocal',seed=42))
        state.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat());save()
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');save();raise


if __name__=='__main__':main()
