"""Detached single-compilation25k rolling-pool continuation and final CPU audit."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def main(prefix):
    root=Path(__file__).resolve().parents[2]
    base=Path('/root/autodl-tmp/tf-logs/confined_ch2d');receipts=root/'docs/tasks'
    run=base/(prefix+'-train');journal=base/(prefix+'-process.json')
    gate=json.loads((receipts/'confined-ch-online-cpu.json').read_text())
    if gate['status']!='passed' or not gate['exact_model_and_optimizer_bytes']:
        raise ValueError('Production CPU producer/checkpoint gate must pass')
    for path in (run,journal):
        if path.exists():raise FileExistsError(path)
    required=2*2**30
    if shutil.disk_usage(base).free<required:raise RuntimeError('Need2GiB for25paired continuation checkpoints and reserves')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    env.pop('JAX_PLATFORMS',None)
    record=dict(status='starting',runner_pid=os.getpid(),run=str(run),start_step=25000,
        requested_steps=50000,stages=[],started_utc=datetime.now(timezone.utc).isoformat(),
        parent=gate['parent'],checkpoint=gate['checkpoint'],required_free_disk_bytes=required,
        cpu_gate=str(receipts/'confined-ch-online-cpu.json'),
        scope='Full paired25kstate continuation; online rolling geometry inputs only, same frozen16/8native monitors. Single production compilation, no default promotion.')
    def write():
        temporary=journal.with_suffix('.json.partial');temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        os.replace(temporary,journal)
    def terminate(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    def execute(stage,command,environment):
        log=base/(prefix+'-'+stage+'.log')
        with log.open('x') as stream:
            child=subprocess.Popen(command,cwd=root,env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            entry=dict(stage=stage,pid=child.pid,command=command,log=str(log),started_utc=datetime.now(timezone.utc).isoformat())
            record['stages'].append(entry);record.update(status=stage,active_child_pid=child.pid);write()
            try:
                while child.poll() is None:
                    record['heartbeat_utc']=datetime.now(timezone.utc).isoformat()
                    if stage=='training':
                        for filename,key in [('metrics.jsonl','latest_evaluation'),('loss.jsonl','latest_loss')]:
                            path=run/filename
                            if path.is_file():
                                lines=path.read_bytes().splitlines(keepends=True)
                                complete=[line for line in lines if line.endswith(b'\n')]
                                if complete:
                                    value=json.loads(complete[-1]);record[key]={k:v for k,v in value.items() if k!='conditions'}
                        for name in ('initial_restore_audit','initial_field_audit','online_preflight'):
                            path=run/(name+'.json')
                            if path.is_file():record[name]=json.loads(path.read_text())
                    write();time.sleep(15)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30)
            entry.update(exit_code=child.returncode,completed_utc=datetime.now(timezone.utc).isoformat());write()
            if child.returncode:raise RuntimeError(f'{stage} failed, see {log}')
    write()
    try:
        execute('training',[str(root/'.venv/bin/python'),'examples/confined_ch2d/train.py',
            '--config=examples/confined_ch2d/configs/train_operator_online_rolling.py',
            '--config.log.run_name='+run.name],env)
        summary=json.loads((run/'summary.json').read_text())
        if summary['start_step']!=25000 or summary['new_updates']!=25000 or summary['final_metrics']['step']!=50000:
            raise ValueError('Wrong continuation endpoint')
        output=receipts/'confined-ch-online-audit.json'
        if output.exists():raise FileExistsError(output)
        execute('cpu_audit',[str(root/'.venv/bin/python'),'-m','docs.tasks.confined_ch_online_gate',
            '--action=audit','--run='+str(run),'--output='+str(output)],dict(env,JAX_PLATFORMS='cpu'))
        record.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),active_child_pid=None,
            audit=str(output),final_metrics={k:v for k,v in summary['final_metrics'].items() if k!='conditions'},online=summary['online'])
        write()
    except BaseException as error:
        record.update(status='failed_or_interrupted',error=repr(error),active_child_pid=None)
        write();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prefix',required=True)
    main(parser.parse_args().prefix)
