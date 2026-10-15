"""Supervise one recomputed zero prefix and its same-run audited continuation."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tail_controls(a,prefix):
    """Freeze only this run's independently audited prefix as continuation input."""
    result = json.loads((prefix/'result.json').read_text())
    audit = json.loads((prefix/'independent_review/audit.json').read_text())
    identity = json.loads((prefix/'identity.json').read_text())
    parent = json.loads((prefix/f'state_{a.prefix_index:04d}/result.json').read_text())
    if (not result['complete'] or not audit['complete'] or not audit['pass_replay']
            or identity['own_zero'] is not True or parent['load'] != a.prefix_load
            or parent['index'] != a.prefix_index or parent['q_d_synchronized'] is not True
            or result['curve'][-1]['index'] != a.prefix_index):
        raise ValueError('this run must finish and independently audit its fresh prefix')
    names = ['identity.json','parameterization.npz','independent_review/audit.json']
    names += [f'state_{a.prefix_index:04d}/{name}' for name in
              ('result.json','fields.npz','network.eqx','optimizer.eqx')]
    keys = ('load_step','transition_load','end_load','inner_updates','max_outer_sweeps',
        'damage_change_limit','displacement_change_limit','minimum_sweeps','consecutive_passes',
        'learning_rate','checkpoints','oracle_loads','sampling','loss_norm','optimizer',
        'omega','symmetry_constraint','advance_on_budget_exit','residual_is_stopping_gate','cpu_threads')
    controls = {key:a[key] for key in keys}
    controls.update(output_dir=a.suite_root,source_root=str(prefix),source_index=a.prefix_index,
        source_load=a.prefix_load,source_optimizer_count=parent['optimizer_count'],
        source_pins={name:sha(prefix/name) for name in names})
    return controls


def main(a,config_path):
    repo = Path(__file__).resolve().parents[3]; python = str(repo/'.venv/bin/python')
    suite = Path(a.suite_root); prefix = suite/'prefix'/a.prefix_case; tail = suite/'fine'
    if suite.exists(): raise ValueError('refusing existing fresh-trajectory run directory')
    busy = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
        capture_output=True,text=True,check=True)
    if busy.stdout.strip(): raise RuntimeError('GPU busy; serialize numerical workers')
    suite.mkdir(parents=True)
    def record(name,data):
        data = dict(data,utc=datetime.now(timezone.utc).isoformat())
        path=suite/name;tmp=path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
    def interrupted(signum,frame):
        raise InterruptedError(f'fresh-trajectory supervisor signal{signum}')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(sig,interrupted)
    env = dict(os.environ,PYTHONPATH='.:src',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',CUDA_VISIBLE_DEVICES='0',JAX_ENABLE_X64='true',
        PYTHONUNBUFFERED='1',JAX_PLATFORMS='cuda')
    cpu = dict(env,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
    children=[];codes={};success=False;failure=None
    def spawn(role,module,args,environment):
        with (suite/f'{role}.log').open('w') as log:
            process=subprocess.Popen([python,'-m',module,*args],cwd=repo,env=environment,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(process)
        record(f'{role}_launch.json',dict(pid=process.pid,module=module,args=args))
        return process
    def wait_phase(worker,audit,monitor,solution):
        while worker.poll() is None:
            if monitor.poll() is not None:
                rp=tail/'result.json'
                if monitor.returncode != 0 or not rp.exists() or not json.loads(rp.read_text())['complete']:
                    raise RuntimeError(f'field monitor unexpectedly exits{monitor.returncode}')
            if audit.poll() is not None:
                rp=solution/'result.json'
                if audit.returncode != 0 or not rp.exists() or not json.loads(rp.read_text())['complete']:
                    raise RuntimeError(f'native auditor unexpectedly exits{audit.returncode}')
            time.sleep(1)
        if worker.wait()!=0: raise RuntimeError(f'numerical worker exits{worker.returncode}')
        while audit.poll() is None:
            if monitor.poll() is not None and monitor.returncode != 0:
                raise RuntimeError(f'field monitor exits{monitor.returncode}')
            time.sleep(1)
        if audit.returncode!=0: raise RuntimeError(f'native auditor exits{audit.returncode}')
    try:
        record('launch.json',dict(supervisor_pid=os.getpid(),config_path=config_path,
            config=a.to_dict(),own_zero=True,source_checkpoint=None,end_load=a.end_load))
        monitor=spawn('monitor','examples.fracture2d._legacy.cold_full_damage_monitor',
            ['--config',str(repo/a.monitor_config)],cpu)
        audit=spawn('prefix_audit','examples.fracture2d._legacy.full_damage_loading_audit',
            ['--root',str(prefix),'--watch'],cpu)
        worker=spawn('prefix_training','examples.fracture2d._legacy.full_damage_loading',
            ['--config',str(repo/a.prefix_config)],env)
        record('status.json',dict(phase='fresh_prefix',complete=False,worker_pid=worker.pid,
            audit_pid=audit.pid,monitor_pid=monitor.pid,end_load=a.end_load,own_zero=True))
        wait_phase(worker,audit,monitor,prefix)
        codes.update(prefix_worker=worker.returncode,prefix_audit=audit.returncode)
        controls=tail_controls(a,prefix)
        record('tail_driver_config.json',controls)
        audit=spawn('tail_audit','examples.fracture2d._legacy.best_full_damage_loading_audit',
            ['--root',str(tail)],cpu)
        worker=spawn('tail_training','examples.fracture2d._legacy.best_full_damage_loading',
            ['--config',str(repo/a.continuation_config)],env)
        record('status.json',dict(phase='critical_loading',complete=False,worker_pid=worker.pid,
            audit_pid=audit.pid,monitor_pid=monitor.pid,end_load=a.end_load,own_zero=True,
            source_optimizer_count=controls['source_optimizer_count']))
        wait_phase(worker,audit,monitor,tail)
        codes.update(tail_worker=worker.returncode,tail_audit=audit.returncode)
        while monitor.poll() is None: time.sleep(1)
        codes['monitor']=monitor.returncode
        if monitor.returncode!=0: raise RuntimeError(f'final field monitor exits{monitor.returncode}')
        completed=json.loads((tail/'result.json').read_text())
        monitored=json.loads((suite/'tensorboard_status.json').read_text())
        if (not completed['complete'] or len(completed['curve'])!=round(a.end_load/a.load_step)
                or not monitored['complete'] or len(monitored['published_loads'])!=len(completed['curve'])):
            raise RuntimeError('fresh trajectory or its complete endpoint monitoring is incomplete')
        success=True
    except BaseException as error:
        failure=repr(error);raise
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: child.kill();child.wait()
        record('exit.json',dict(exit_code=0 if success else 1,process_exit_codes=codes,failure=failure))
        record('status.json',dict(phase='complete' if success else 'interrupted_or_failed',complete=True,
            exit_code=0 if success else 1,end_load=a.end_load,own_zero=True,failure=failure))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    main(runpy.run_path(args.config)['get_config'](),args.config)
