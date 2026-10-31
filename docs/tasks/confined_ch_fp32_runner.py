"""Detached sequential precision probe, canonical cold run and final audits."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main(prefix,prior):
    root=Path(__file__).resolve().parents[2]
    base=Path('/root/autodl-tmp/tf-logs/confined_ch2d')
    python=str(root/'.venv/bin/python')
    run=base/(prefix.replace('-confined-width128-fp32-','-confined-geometry-b16-t16-width128-fp32-30k-'))
    journal=base/(prefix+'-process.json');probe=base/(prefix+'-probe')
    initial=root/'docs/tasks/confined-ch-fp32-initial-audit.json'
    for p in [run,journal,probe,initial]:
        if p.exists():raise FileExistsError(p)
    receipt=json.loads((root/'docs/tasks/confined-ch-fp32-switch.json').read_text())
    assert receipt['status']=='intentionally_stopped_for_fp32' and receipt['run']==str(prior)
    if shutil.disk_usage(base).free<4*2**30:raise RuntimeError('Need at least4GiB free for preserved states/new run')
    # Wait only for asynchronous release by the just-terminated owned process.
    for _ in range(40):
        used=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip().splitlines()
        if len(used)!=1:raise RuntimeError('Expected one existing GPU')
        if int(used[0])<1024:break
        time.sleep(.5)
    else:raise RuntimeError('GPU has not become idle; refuse competing worker')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    env.pop('JAX_PLATFORMS',None)
    cpu=dict(env,JAX_PLATFORMS='cpu')
    record=dict(status='starting',runner_pid=os.getpid(),runner_session=os.getsid(0),run=str(run),
        prior_fp64_partial=str(prior),probe=str(probe),requested_steps=30000,stages=[],
        started_utc=datetime.now(timezone.utc).isoformat(),scope='SameGPU cold precision-only control, no default promotion')
    def write():
        temporary=journal.with_name(journal.name+'.partial')
        temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');os.replace(temporary,journal)
    write()
    def execute(stage,command,environment,monitor):
        log=base/(prefix+'-'+stage+'.log')
        with log.open('x') as stream:
            child=subprocess.Popen(command,cwd=root,env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            entry=dict(stage=stage,pid=child.pid,command=command,log=str(log),started_utc=datetime.now(timezone.utc).isoformat())
            record['stages'].append(entry);record.update(status=stage,active_child_pid=child.pid);write()
            while True:
                try:code=child.wait(timeout=15);break
                except subprocess.TimeoutExpired:
                    if monitor and not initial.exists() and (run/'metrics.jsonl').exists():
                        raw=(run/'metrics.jsonl').read_text()
                        if raw.endswith('\n'):
                            with (base/(prefix+'-initial-audit.log')).open('x') as auditlog:
                                result=subprocess.run([python,'-m','docs.tasks.confined_ch_fp32_probe','--action','initial',
                                    '--prior',str(prior),'--run',str(run),'--output',str(initial)],cwd=root,env=cpu,stdout=auditlog,stderr=subprocess.STDOUT)
                            if result.returncode:
                                child.terminate();child.wait(timeout=60)
                                raise RuntimeError('Production initialization audit failed')
                            record['initial_audit']=str(initial)
                    if monitor and (run/'metrics.jsonl').exists():
                        rows=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines(keepends=True) if s.endswith('\n')]
                        if rows:record['latest_evaluation']={k:v for k,v in rows[-1].items() if k!='conditions'}
                    record['heartbeat_utc']=datetime.now(timezone.utc).isoformat();write()
        entry.update(exit_code=code,completed_utc=datetime.now(timezone.utc).isoformat())
        record['active_child_pid']=None;write()
        if code:raise RuntimeError(f'{stage} exited{code}; inspect{log}')
    try:
        execute('gpu_probe',[python,'-m','docs.tasks.confined_ch_fp32_probe','--action','probe','--prior',str(prior),'--output',str(probe)],env,False)
        execute('probe_cpu',[python,'-m','docs.tasks.confined_ch_fp32_probe','--action','cpu','--prior',str(prior),'--output',str(probe)],cpu,False)
        preflight=json.loads((probe/'probe.json').read_text());preflight['cpu_replay']=json.loads((probe/'cpu_audit.json').read_text())
        preflight['probe_path']=str(probe)
        (root/'docs/tasks/confined-ch-fp32-preflight.json').write_text(json.dumps(preflight,indent=2,allow_nan=False)+'\n')
        preset='examples/confined_ch2d/configs/train_operator_smallscale10_512_width128_fp32.py'
        execute('training',[python,'examples/confined_ch2d/train.py','--config='+preset,'--config.log.run_name='+run.name],env,True)
        assert json.loads((run/'summary.json').read_text())['final_metrics']['step']==30000
        assert json.loads(initial.read_text())['status']=='passed'
        cold96=base/'20261030-confined-geometry-b16-t16-wide96-soap-01'
        stages=[
            ('native',[python,'-m','examples.confined_ch2d.operator_evaluate','--run',str(run),'--output',str(run/'evaluation')]),
            ('dense61',[python,'docs/tasks/confined-fourier-dense-evaluate.py','--run',str(run),'--dense','/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_dense_validation','--step','30000','--output',str(run/'dense_evaluation')]),
            ('geometry',[python,'-m','examples.confined_ch2d.operator_geometry_effect','--run',str(run),'--dataset','/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_512_v1','--output',str(run/'geometry_effect')]),
            ('report',[python,'docs/tasks/confined-fourier-training-report.py','--run',str(run),'--output',str(run/'training_report')]),
            ('cpu_audit',[python,'docs/tasks/confined-fourier-audit.py','--run',str(run),'--prior',str(cold96),'--output',str(root/'docs/tasks/confined-ch-fp32-audit.json'),'--config','examples.confined_ch2d.configs.train_operator_smallscale10_512_width128_fp32']),
        ]
        for stage,command in stages:execute(stage,command,cpu,False)
        from examples.confined_ch2d.operator_history import training_records
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');histories={}
        for name,path in [('FP64 beta09 (partial)',prior),('FP32 beta09',run)]:
            rows=training_records(path,'metrics.jsonl');histories[name]={r['step']:r for r in rows}
            for axis,x in zip(axes,([r['step'] for r in rows],[r['elapsed_seconds']/60 for r in rows]),strict=True):
                axis.plot(x,[100*r['test_phi_pooled'] for r in rows],label=name)
        matched=[];a,b=histories.values()
        for step in sorted(set(a)&set(b)):
            matched.append(dict(step=step,fp64_phi=a[step]['test_phi_pooled'],fp32_phi=b[step]['test_phi_pooled'],fp64_terminal=a[step]['test_phi_terminal'],fp32_terminal=b[step]['test_phi_terminal']))
        for axis,label in zip(axes,('Updates','Elapsed minutes'),strict=True):
            axis.set(xlabel=label,ylabel='Reserved development phase L2 (%)',yscale='log');axis.grid(alpha=.25);axis.legend()
        fig.savefig(run/'precision_comparison.png',dpi=180);fig.savefig(run/'precision_comparison.pdf');plt.close(fig)
        result=dict(status='complete',run=str(run),prior_partial=str(prior),matched_steps=matched,
            native=json.loads((run/'evaluation/metrics.json').read_text()),dense61=json.loads((run/'dense_evaluation/metrics.json').read_text()),
            scope='One common rounded initialization/sampler, native8reuseddevelopment domains, same GPU. Prior FP64 run deliberately interrupted; no completedFP64 30k ranking or default promotion.')
        (root/'docs/tasks/confined-ch-fp32-results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        record.update(status='complete',results='docs/tasks/confined-ch-fp32-results.json',completed_utc=datetime.now(timezone.utc).isoformat());write()
    except Exception as error:
        record.update(status='failed',error=repr(error),failed_utc=datetime.now(timezone.utc).isoformat());write();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',required=True);p.add_argument('--prior',type=Path,required=True)
    a=p.parse_args();main(a.prefix,a.prior.resolve())
