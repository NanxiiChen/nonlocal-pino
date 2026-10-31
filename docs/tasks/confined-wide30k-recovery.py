"""Owned, detached fixed30k recovery run and native-only validation pipeline.

Launch this script in its own OS session from the repository .venv. It preserves
an earlier incomplete run and imports only the complete20k model/SOAP state.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess


def main():
    root=Path(__file__).resolve().parents[2]
    base=Path('/root/autodl-tmp/tf-logs/confined_ch2d')
    run=base/'20261030-confined-geometry-b16-t16-wide96-30k-retry-01'
    parent=base/'20261030-confined-geometry-b16-t16-wide96-continue-01'
    previous=base/'20261030-confined-geometry-b16-t16-wide96-30k-01'
    journal=base/'20261030-confined-wide30k-retry-process.json'
    if run.exists() or journal.exists():raise FileExistsError('Recovery run/journal already exists')
    assert json.loads((parent/'summary.json').read_text())['final_metrics']['step']==20000
    assert not (previous/'summary.json').exists()
    python=str(root/'.venv/bin/python')
    preset='examples.confined_ch2d.configs.train_operator_smallscale10_512_wide_30k'
    prefix='docs/tasks/confined-fourier-wide96-30k-retry'
    record=dict(status='starting',runner_pid=os.getpid(),run=str(run),parent_run=str(parent),
        previous_incomplete_run=str(previous),started_utc=datetime.now(timezone.utc).isoformat(),
        requested_final_step=30000,optimization_changed=False,
        scope='Same20kfullstate/config; retry incomplete stage, not a new seed or selected checkpoint')
    def write():
        temporary=journal.with_name(journal.name+'.partial')
        temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        temporary.replace(journal)
    write()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(root))
    env.pop('JAX_PLATFORMS',None)
    command=[python,'examples/confined_ch2d/train.py',
        '--config=examples/confined_ch2d/configs/train_operator_smallscale10_512_wide_30k.py',
        '--config.log.run_name='+run.name]
    with (base/'20261030-confined-wide30k-retry-train.log').open('x') as log:
        child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True)
        record.update(status='training',training_pid=child.pid,training_session=os.getsid(child.pid),command=command)
        write();print(json.dumps(record),flush=True)
        code=child.wait()
    record['training_exit_code']=code
    if code:
        record['status']='training_failed';write();raise RuntimeError(f'Training exited {code}')
    summary=json.loads((run/'summary.json').read_text())
    assert summary['final_metrics']['step']==30000
    record.update(status='postprocessing',final_metrics={k:v for k,v in summary['final_metrics'].items()
        if k!='conditions'},completed_training_utc=datetime.now(timezone.utc).isoformat());write()
    cpu_env=dict(env,JAX_PLATFORMS='cpu')
    commands=[
        [python,'-m','examples.confined_ch2d.operator_evaluate','--run',str(run),'--output',str(run/'evaluation')],
        [python,'-m','examples.confined_ch2d.operator_geometry_effect','--run',str(run),
            '--dataset','/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_512_v1',
            '--output',str(run/'geometry_effect')],
        [python,'docs/tasks/confined-fourier-training-report.py','--run',str(run),'--output',str(run/'training_report')],
        [python,'docs/tasks/confined-fourier-dense-evaluate.py','--run',str(run),
            '--dense','/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_dense_validation',
            '--step','30000','--output',str(run/'dense_evaluation')],
        [python,'docs/tasks/confined-fourier-modal-error.py','--run',str(run),'--output',prefix+'-modal-error.json'],
        [python,'docs/tasks/confined-fourier-audit.py','--run',str(run),'--prior',str(parent),
            '--output',prefix+'-audit.json','--config',preset]]
    with (base/'20261030-confined-wide30k-retry-validation.log').open('x') as log:
        for command in commands:
            record['postprocess_command']=command;write()
            result=subprocess.run(command,cwd=root,env=cpu_env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                record.update(status='validation_failed',validation_exit_code=result.returncode);write()
                raise RuntimeError(f'Validation exited {result.returncode}: {command}')
    record.update(status='complete',audit=prefix+'-audit.json',completed_utc=datetime.now(timezone.utc).isoformat());write()
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
