"""Wait for two owned input/resource checks, then one cold10k and finite audits."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def process_token(pid):
    path=Path(f'/proc/{pid}/stat')
    try:
        return path.read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:
        return None  # Normal completion of a finite owned check.


def main(a):
    root=Path(a.output)
    if not root.is_absolute():raise ValueError('Absolute capacity output root required')
    journal=root/'training-pipeline.json'
    if journal.exists():raise FileExistsError(journal)
    owners=[]
    for pid,name,artifact in ((a.probe_pid,'channel_time_batch_probe.py',root/'resource-probe/result.json'),
                              (a.screen_pid,'channel_operator_prior_probe.py',root/'input-screen.json')):
        token=process_token(pid)
        if token is None:
            if not artifact.is_file():raise ValueError(f'Check exited without its completion artifact: {artifact}')
        else:
            command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            if name not in command or str(root) not in command or a.config not in command:
                raise ValueError(f'PID does not own the declared capacity check: {pid}')
        owners.append(dict(pid=pid,token=token,artifact=str(artifact)))
    record=dict(status='waiting_for_owned_checks',runner_pid=os.getpid(),started=time.time(),
        run_name=a.run_name,config=a.config,control=a.control,owners=owners,
        requested_updates=10000,automatic_retries=0,FEM_solves=0,active_child_pid=None)
    (root/'training-pipeline-owner.py').write_bytes(Path(__file__).read_bytes())

    def save():
        temporary=journal.with_suffix('.partial')
        temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        temporary.replace(journal)

    def terminate(signum,frame):
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    save()
    try:
        deadline=time.monotonic()+3600
        while any(o['token'] is not None and process_token(o['pid'])==o['token'] for o in owners):
            if time.monotonic()>deadline:raise TimeoutError('Owned resource/input checks exceeded the finite1hour wait budget')
            record['heartbeat']=time.time();save();time.sleep(5)
        import importlib
        from examples.channel_nsch2d.capacity_validation import comparison_config
        from examples.channel_nsch2d.online import validate_construction_screen
        c=importlib.import_module(a.config).get_config();c.train.seed=42;c.log.run_name=a.run_name
        comparison_config(c,a.control)
        screen=validate_construction_screen(c,root/'input-screen.json')
        resource=json.loads((root/'resource-probe/result.json').read_text())
        if (resource['status']!='finite_production_probe_passed' or resource['updates']!=7
                or resource['memory']['peak_bytes_in_use']>28*2**30):
            raise ValueError('Capacity resource check failed or leaves insufficient32GiB headroom')
        record.update(status='qualified',input_screen=dict(online_draws=screen['online_draws'],seconds=screen['seconds']),
            resource_summary=dict(warm_update_seconds=resource['warm_update_seconds'],
                peak_bytes=resource['memory']['peak_bytes_in_use'],parameters=resource['capacity_trial']['parameter_count']))
        save()
        env=dict(os.environ,JAX_PLATFORMS='cuda',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        arguments=['docs/tasks/channel_online_train_runner.py','--config',a.config,'--run-name',a.run_name,
            '--seed','42','--reference-root',c.reference.root,'--construction-screen',str(root/'input-screen.json'),
            '--capacity-control',a.control,'--capacity-probe',str(root/'resource-probe')]
        with (root/'training-and-audits.log').open('x') as stream:
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),*arguments],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            record.update(status='cold10000_and_audits',active_child_pid=child.pid,arguments=arguments);save()
            try:code=child.wait()
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30)
        record.update(active_child_pid=None,child_exit_code=code);save()
        if code:raise RuntimeError(f'Cold10000 or owned audit failed: exit{code}; no retry')
        owner=Path('/root/autodl-tmp/data/channel_nsch2d')/(a.run_name+'-process.json')
        result=json.loads(owner.read_text())
        if result['status']!='completed_with_saved_run_audits':raise ValueError('Capacity run/audits incomplete')
        record.update(status='completed',finished=time.time(),run=result['run'],metrics=result['final_metrics'],
            finite_cohort=result['finite_cohort'],source_byte_checks=result['source_byte_checks'])
        save()
    except BaseException as error:
        record.update(status='failed_or_interrupted',error=repr(error),active_child_pid=None);save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','control','output','run-name'):p.add_argument('--'+key,required=True)
    for key in ('probe-pid','screen-pid'):p.add_argument('--'+key,required=True,type=int)
    main(p.parse_args())
