"""Own one already-authorized random-wall pilot after its current FEM process exits.

Finite queue: qualified reference plots, resource probe, one cold5000 through
train.py and the existing owner/audits/three-held-out cases. No retries or FEM.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

ROOT=Path(__file__).resolve().parents[2]
CONTROL='/root/autodl-tmp/tf-logs/channel_nsch2d/20261102-5090-integral-time40-jvp-stronger-online-width128-01'
CONFIG='examples.channel_nsch2d.configs.train_operator_random_walls'


def process_token(pid):
    path=Path(f'/proc/{pid}/stat')
    try:return path.read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None  # Expected process completion, not a numerical fallback.


def main(a):
    root=Path(a.reference_root);journal=root/'training-pipeline.json'
    if journal.exists():raise FileExistsError(journal)
    token=process_token(a.after_pid)
    if token is None:raise ValueError('Expected reference process must still be active at handoff')
    command=Path(f'/proc/{a.after_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if 'channel_random_wall_references.py' not in command or str(root) not in command:
        raise ValueError('Reference PID does not own the declared attempt')
    record=dict(status='waiting_for_owned_FEM',runner_pid=os.getpid(),reference_pid=a.after_pid,
        reference_process_token=token,started=time.time(),reference_root=str(root),run_name=a.run_name,
        requested_updates=5000,automatic_retries=0,FEM_solves=0,stages=[])
    def save():
        tmp=journal.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');tmp.replace(journal)
    save()
    (root/'training-pipeline-owner.py').write_bytes(Path(__file__).read_bytes())
    env=dict(os.environ,JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def execute(stage,args,environment):
        log=root/(stage+'.log')
        with log.open('x') as stream:
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),*args],cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            entry=dict(stage=stage,pid=child.pid,started=time.time(),log=str(log));record['stages'].append(entry)
            record.update(status=stage,active_child_pid=child.pid);save()
            try:code=child.wait()
            except BaseException:
                child.terminate();child.wait(timeout=30);raise
            entry.update(exit_code=code,finished=time.time());record['active_child_pid']=None;save()
            if code:raise RuntimeError(f'{stage} failed: exit{code}; see {log}')
    try:
        while process_token(a.after_pid)==token:time.sleep(2)
        progress=json.loads((root/'progress.json').read_text())
        gate=json.loads((root/'unseen3'/'cohort-reference-validation.json').read_text())
        if progress['stage']!='completed' or gate['status']!='passed' or gate['cases']!=3:
            raise ValueError('All four independent FEM qualifications must pass before GPU work')
        from examples.channel_nsch2d.configs.train_operator_random_walls import get_config
        from examples.channel_nsch2d.online import validate_construction_screen
        from examples.channel_nsch2d.random_wall_validation import comparison_config
        c=get_config();c.train.seed=42;c.log.run_name=a.run_name
        if Path(c.reference.root)!=root/'development':raise ValueError('Pilot configuration/reference root mismatch')
        comparison_config(c,CONTROL);validate_construction_screen(c,a.construction_screen)
        execute('reference_response',['docs/tasks/channel_random_wall_fields.py','--root',str(root),
            '--output',str(root/'reference-response')],env)
        response=json.loads((root/'reference-response'/'response.json').read_text())
        record['reference_response']=response['centroid_aligned_pairwise'];save()
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPU occupied; finite pilot stops rather than overlap another experiment')
        gpu=dict(env,JAX_PLATFORMS='cuda');probe=root/'resource-probe'
        execute('resource_probe',['docs/tasks/channel_time_batch_probe.py','--config',CONFIG,'--control',CONTROL,
            '--output',str(probe),'--updates','7'],gpu)
        resource=json.loads((probe/'result.json').read_text())
        if resource['status']!='finite_production_probe_passed':raise ValueError('Resource probe did not qualify')
        if resource['memory']['peak_bytes_in_use']>28*2**30:raise RuntimeError('Probe leaves insufficient32GiB pool headroom')
        record['resource_summary']=dict(warm_update_seconds=resource['warm_update_seconds'],
            peak_bytes=resource['memory']['peak_bytes_in_use']);save()
        execute('cold5000_and_audits',['docs/tasks/channel_online_train_runner.py','--config',CONFIG,
            '--run-name',a.run_name,'--seed','42','--reference-root',str(root/'development'),
            '--construction-screen',a.construction_screen,'--random-wall-control',CONTROL],gpu)
        path=Path(c.log.log_dir)/a.run_name
        owner=Path('/root/autodl-tmp/data/channel_nsch2d')/(a.run_name+'-process.json')
        final=json.loads(owner.read_text())
        if final['status']!='completed_with_saved_run_audits':raise ValueError('Cold run/audits incomplete')
        # Direct byte checks, no hashes or alternate model loading mechanism.
        sources=[]
        for f in (path/'source').rglob('*.py'):
            relative=f.relative_to(path/'source')
            if not (ROOT/relative).is_file() or f.read_bytes()!=(ROOT/relative).read_bytes():raise ValueError(f'Source changed: {relative}')
            sources.append(str(relative))
        record.update(status='completed',run=str(path),finished=time.time(),
            source_byte_checks=len(sources),source_files=sources,metrics=final['final_metrics'],
            unseen_evaluation=str(path/'unseen3'/'evaluation.json'))
        save()
    except BaseException as exc:
        record.update(status='failed_or_interrupted',error=repr(exc),active_child_pid=None);save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--after-pid',required=True,type=int)
    p.add_argument('--reference-root',required=True);p.add_argument('--construction-screen',required=True)
    p.add_argument('--run-name',required=True);main(p.parse_args())
