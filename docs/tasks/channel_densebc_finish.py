"""Finite post-run strain, full-time residual and independent BC audits for a pair."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def main(a):
    pair=Path(a.comparison);output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    owner=json.loads((pair/'status.json').read_text())['runner_pid']
    command=Path(f'/proc/{owner}/cmdline').read_text()
    if 'channel_densebc_compare.py' not in command or str(pair) not in command:
        raise ValueError('Require the exact live finite-pair owner, not a stale/reused PID')
    record=dict(status='waiting_for_finite_pair',pid=os.getpid(),pair_owner=owner,
        comparison=str(pair),started_utc=datetime.now(timezone.utc).isoformat(),stages=[],
        scope='Two completed saved-model diagnostics only; no training, FEM, CH reintegration, retry or promotion')
    def save():
        p=output/'status.partial';p.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');p.replace(output/'status.json')
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    save();(output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    begin=time.monotonic()
    env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    try:
        while True:
            status=json.loads((pair/'status.json').read_text())
            if status['status']=='finite_pair_completed_with_exact_initial_and_actual_input_parity':break
            if status['status'].startswith('failed'):raise RuntimeError('Finite pair failed; no retry or saved-model audit')
            if time.monotonic()-begin>a.max_wait_seconds:raise TimeoutError('Finite pair wait expired')
            if not Path(f'/proc/{owner}').exists():raise RuntimeError('Pair owner disappeared before completion')
            time.sleep(5)
        for arm in status['arms']:
            label=arm['arm'];run=arm['run']
            for name in ('network.py','losses.py','problem.py','data.py','operators.py','online.py','temporal.py','balance.py'):
                relative=Path('examples/channel_nsch2d')/name
                if (ROOT/relative).read_bytes()!=(Path(run)/'source'/relative).read_bytes():
                    raise ValueError(f'Saved-run numerical source changed before diagnostics: {relative}')
            commands=[('strain',['docs/tasks/channel_velocity_shape_audit.py','--run',run,
                '--numerical',a.numerical,'--output',str(output/(label+'-strain.json'))]),
                ('boundary',['docs/tasks/channel_boundary_trace.py','--run',run,
                    '--output',str(output/(label+'-boundary'))]),
                ('residual',['docs/tasks/channel_pde_bc_diagnosis.py','--run',run,
                    '--output',str(output/(label+'-residual')),'--residual-only'])]
            for stage,args in commands:
                for relative in (Path(args[0]),Path('docs/tasks/channel_frozen_ch.py')):
                    target=output/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True)
                    if not target.exists():target.write_bytes((ROOT/relative).read_bytes())
                record['status']=label+'_'+stage
                entry=dict(arm=label,stage=stage,arguments=args);record['stages'].append(entry);save()
                with (output/(label+'-'+stage+'.log')).open('x') as stream:
                    child=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,env=env,
                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    entry['pid']=child.pid;save()
                    try:code=child.wait(timeout=a.stage_timeout_seconds)
                    finally:
                        if child.poll() is None:
                            os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=45)
                    entry['exit_code']=code;save()
                    if code:raise subprocess.CalledProcessError(code,args)
        record.update(status='all_finite_saved_model_diagnostics_completed',completed_utc=datetime.now(timezone.utc).isoformat())
        save()
    except BaseException as error:
        record.update(status='failed_or_interrupted_no_retry',error=repr(error));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('comparison','output','numerical'):p.add_argument('--'+key,required=True)
    p.add_argument('--max-wait-seconds',type=float,required=True)
    p.add_argument('--stage-timeout-seconds',type=float,required=True)
    main(p.parse_args())
