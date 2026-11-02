"""Finish the adopted4-time probe, measure40-time, then one finite cold5k run."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
CONFIG='examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_online'


def main(a):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    if a.seed!=42:raise ValueError('This matched cold control requires the existing initialization42')
    pidfile=Path(f'/proc/{a.first_probe_pid}/cmdline')
    if pidfile.exists():
        command=pidfile.read_text()
        if ('channel_time_batch_probe.py' not in command or str(Path(a.probe4).parent) not in command
                or a.control not in command):
            raise ValueError('The adopted PID is not the exact declared4-time resource probe')
    elif not Path(a.probe4).is_file():
        raise ValueError('No live declared4-time probe or completed result exists')
    record=dict(status='waiting_for_4_time_probe',owner_pid=os.getpid(),adopted_probe_pid=a.first_probe_pid,
        started_utc=datetime.now(timezone.utc).isoformat(),stages=[],config=CONFIG,run_name=a.run_name,
        requested_updates=5000,time_samples=40,time_microbatch=8,seed=a.seed,
        automatic_goal=None,causal_runs_queued=0,new_reference_solves=0,
        scope='Finite resource preflight and one cold5k uniform-loss control; no retry or subsequent training')
    def save():
        p=output/'status.partial';p.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        p.replace(output/'status.json')
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    save()
    for name in ('docs/tasks/channel_time40_runner.py','docs/tasks/channel_time_batch_probe.py',
                 'docs/tasks/channel_online_train_runner.py'):
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
             NUMEXPR_NUM_THREADS='4',JAX_PLATFORMS='cuda',PYTHONUNBUFFERED='1')

    def execute(stage,args):
        record['status']=stage
        entry=dict(stage=stage,arguments=args,started_utc=datetime.now(timezone.utc).isoformat())
        record['stages'].append(entry);save()
        with (output/(stage+'.log')).open('x') as stream:
            child=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            entry['pid']=child.pid;save()
            try:
                start=time.monotonic()
                while child.poll() is None:
                    if time.monotonic()-start>a.stage_timeout_seconds:
                        raise TimeoutError('Explicit finite stage deadline exceeded')
                    record['heartbeat_utc']=datetime.now(timezone.utc).isoformat();save()
                    time.sleep(5)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=45)
            entry.update(exit_code=child.returncode,completed_utc=datetime.now(timezone.utc).isoformat());save()
            if child.returncode:raise subprocess.CalledProcessError(child.returncode,args)

    try:
        start=time.monotonic()
        while pidfile.exists():
            if time.monotonic()-start>a.probe_wait_seconds:raise TimeoutError('Adopted probe wait expired')
            time.sleep(5)
        first=json.loads(Path(a.probe4).read_text())
        if (first['status']!='finite_production_probe_passed' or first['time_samples']!=4
                or first['control']!=a.control):raise ValueError('First resource probe failed')
        record['probe4']=first;save()
        execute('probe40',['docs/tasks/channel_time_batch_probe.py','--config='+CONFIG,
            '--control='+a.control,'--output='+a.probe40_output,'--updates=20'])
        second=json.loads((Path(a.probe40_output)/'result.json').read_text())
        if (second['status']!='finite_production_probe_passed' or second['time_samples']!=40
                or second['microbatch']!=dict(kind='time',size=8)
                or second['optimizer_updates_per_time_batch']!=1):raise ValueError('40-time resource probe failed')
        record['probe40']=second
        record['warm_time_ratio_40_over_4']=second['warm_update_seconds']/first['warm_update_seconds'];save()
        execute('cold5k_and_audits',['docs/tasks/channel_online_train_runner.py','--config='+CONFIG,
            '--run-name='+a.run_name,'--seed='+str(a.seed),'--reference-root='+a.reference_root,
            '--construction-screen='+a.construction_screen])
        journal=Path('/root/autodl-tmp/data/channel_nsch2d')/(a.run_name+'-process.json')
        done=json.loads(journal.read_text())
        if done['status']!='completed_with_saved_run_audits':raise ValueError('Cold5k audits incomplete')
        record.update(status='completed_cold5k_and_finite_audits',journal=str(journal),
            final_metrics=done['final_metrics'],completed_utc=datetime.now(timezone.utc).isoformat());save()
    except BaseException as error:
        record.update(status='failed_or_interrupted_no_retry',error=repr(error));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('output','probe4','probe40-output','control','run-name','reference-root','construction-screen'):
        p.add_argument('--'+key,required=True)
    for key in ('first-probe-pid','seed'):p.add_argument('--'+key,required=True,type=int)
    for key in ('probe-wait-seconds','stage-timeout-seconds'):p.add_argument('--'+key,required=True,type=float)
    a=p.parse_args()
    if a.probe_wait_seconds<=0 or a.stage_timeout_seconds<=0:raise ValueError('Explicit positive deadlines required')
    main(a)
