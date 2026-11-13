"""Finite single-geometry FEM check, then three cold paired20k/64case arms."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from docs.tasks.pino_unified_sampling_queue import gpu

ROOT=Path(__file__).resolve().parents[2]
LOGS=Path('/root/autodl-tmp/tf-logs/particle_corrosion2d')
INPUTS=Path('/root/autodl-tmp/data/particle_corrosion2d/20261107-spaced-six-01/inputs64')
DEV=INPUTS.parent/'local_dev4'
TEST=Path('/root/autodl-tmp/data/particle_corrosion2d/local_test64_20261111-01')
CHANNEL=Path('/root/autodl-tmp/tf-logs/channel-uniform-pro6000-20261113-02/journal.json')
PREFLIGHT=Path('/root/autodl-tmp/tf-logs/ceramic-fem-preflight-20261113-01/summary.json')


def utc():return datetime.now(timezone.utc).isoformat()


def recovered_fem(bundle):
    """Explicit continuation after a completed FEM arm's regional-only failure."""
    from ml_collections import ConfigDict
    from examples.particle_corrosion2d.compare import run_costs
    journal=json.loads((bundle/'journal.json').read_text())
    expected=[('fixed-fem2k','passed'),('fem_weak-train','passed'),
              ('fem_weak-online','passed'),('fem_weak-eval64','failed')]
    if journal['status']!='failed' or [(s['label'],s['status']) for s in journal['stages']]!=expected:
        raise ValueError('Continuation requires the explicit completed-FEM/regional-failure state')
    run=Path(journal['stages'][1]['run']).resolve(strict=True)
    evaluation=bundle/'fem_weak-eval64'
    score=json.loads((evaluation/'summary.json').read_text())
    regional=json.loads((evaluation/'regional_metrics.json').read_text())
    replay=json.loads((bundle/'reduction_roundoff_audit.json').read_text())
    audit=json.loads((run/'online_validation.json').read_text())
    if (score['status']!='passed' or score['steps']!=20000 or score['count']!=64
            or Path(score['run']).resolve()!=run or Path(score['reference']).resolve()!=TEST
            or regional['status']!='passed' or regional['count']!=64
            or replay['status']!='passed' or replay['headline_reproduced_bitwise'] is not True
            or audit['status']!='passed' or audit['updates']!=20000
            or not (run/'model_final.eqx').is_file() or not (run/'optimizer_final.eqx').is_file()):
        raise ValueError('Completed FEM state,64case replay and regional recovery required')
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    if c.nl.method!='fem_weak' or c.train.iterations!=20000:
        raise ValueError('Continuation requires the original FEM20k control')
    record=dict(method='fem_weak',run=str(run),evaluation=str(evaluation),
                statistics=score['statistics'],costs=run_costs(run,c),
                reused_completed_training=True)
    return run,record


def main(output,continue_from):
    if json.loads(CHANNEL.read_text())['status']!='completed_resources_only':
        raise RuntimeError('Finish channel resource-only workflow before ceramic GPU work')
    if json.loads(PREFLIGHT.read_text())['status']!='passed':
        raise RuntimeError('Real64geometry weak-support preflight must pass')
    card=gpu()
    if card['uuid']!='GPU-1e537971-8529-b7c1-1682-82073d2cae12' or card['used_MiB']>1024:
        raise RuntimeError(f'Expected idle current PRO6000, found{card}')
    reused=None if continue_from is None else recovered_fem(continue_from.resolve(strict=True))
    output.mkdir(parents=True,exist_ok=False)
    (output/'runner.py').write_bytes(Path(__file__).read_bytes())
    state=dict(status='running',pid=os.getpid(),started_utc=utc(),gpu=card,stages=[],
               continue_from=None if continue_from is None else str(continue_from.resolve()))
    def save():
        p=output/'journal.partial';p.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
        p.replace(output/'journal.json')
    def interrupted(signum,frame):raise InterruptedError(f'Owner interrupted by signal{signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    save()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
    def stage(label,command,*,cpu,run):
        if not cpu:
            current=gpu()
            if current['uuid']!=card['uuid'] or current['used_MiB']>1024:
                raise RuntimeError(f'Same idle GPU required before{label}: {current}')
        environment=dict(env,JAX_PLATFORMS='cpu' if cpu else 'cuda')
        if cpu:environment['CUDA_VISIBLE_DEVICES']=''
        else:environment.update(XLA_PYTHON_CLIENT_PREALLOCATE='true',XLA_PYTHON_CLIENT_MEM_FRACTION='.8')
        record=dict(label=label,command=command,status='running',started_utc=utc(),run=str(run) if run else None)
        state['stages'].append(record);tick=time.perf_counter();child=None
        try:
            with (output/(label+'.log')).open('x') as f:
                child=subprocess.Popen([sys.executable,*command],cwd=ROOT,env=environment,
                    stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                record['pid']=child.pid;save()
                while True:
                    try:code=child.wait(timeout=30);break
                    except subprocess.TimeoutExpired:
                        observation=dict(utc=utc(),stage=label,pid=child.pid,seconds=time.perf_counter()-tick)
                        if run:
                            for name in ('loss','metrics'):
                                p=run/(name+'.jsonl')
                                if p.is_file():
                                    lines=p.read_text().rsplit('\n',1)[0].splitlines()
                                    if lines:observation[name]=json.loads(lines[-1])
                        state['progress']=observation;save()
            record.update(status='passed' if code==0 else 'failed',exit_code=code,seconds=time.perf_counter()-tick)
            save()
            if code:raise RuntimeError(f'{label} failed; retained log; no implicit retry')
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                record.update(status='interrupted',exit_code=child.returncode);save()
    try:
        if reused is None:
            pilot=LOGS/(output.name+'-fixed-fem2k')
            stage('fixed-fem2k',['-m','docs.tasks.ceramic_fem_fixed_probe','--reference',str(DEV),'--output',str(pilot)],cpu=False,run=pilot)
            report=json.loads((pilot/'summary.json').read_text())
            if report['status']!='passed' or report['steps']!=2000:raise ValueError('Fixed FEM pilot did not finish')
            records=[];control=None
            arms=(('fem_weak','weak'),('nonlocal','nonlocal'),('local_ad','local_ad'))
        else:
            control,record=reused;records=[record]
            arms=(('nonlocal','nonlocal'),('local_ad','local_ad'))
            state['reused_fem']=record;save()
            (output/'results.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
        for method,preset in arms:
            run=LOGS/(output.name+'-'+method)
            args=['-m','examples.particle_corrosion2d.train','--config',
                  f'examples/particle_corrosion2d/configs/train_fem_comparison_{preset}.py',
                  '--inputs',str(INPUTS),'--reference',str(DEV),'--exclude-reference',str(TEST),
                  '--output',str(run)]
            args+=['--seed','42'] if control is None else ['--rolling-method-control',str(control)]
            stage(method+'-train',args,cpu=False,run=run)
            stage(method+'-online',['-m','examples.particle_corrosion2d.online_validation',
                  '--run',str(run),'--workers','4'],cpu=True,run=None)
            if control is not None:
                stage(method+'-pair',['-m','examples.particle_corrosion2d.method_validation','--run',str(run),'--control',str(control)],cpu=True,run=None)
            evaluation=output/(method+'-eval64')
            stage(method+'-eval64',['-m','docs.tasks.pino_unified_ceramic_evaluate','--run',str(run),
                '--reference',str(TEST),'--output',str(evaluation)],cpu=False,run=None)
            from ml_collections import ConfigDict
            from examples.particle_corrosion2d.compare import run_costs
            c=ConfigDict(json.loads((run/'config.json').read_text()))
            score=json.loads((evaluation/'summary.json').read_text())
            records.append(dict(method=method,run=str(run),statistics=score['statistics'],costs=run_costs(run,c)))
            (output/'results.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
            control=run
        state['status']='complete'
    except InterruptedError as error:state.update(status='stopped_by_user',reason=str(error))
    except Exception as error:
        state.update(status='failed',failure=str(error));raise
    finally:state['finished_utc']=utc();save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--continue-from',type=Path,help='Explicit completed-FEM bundle with repaired regional postprocessing')
    a=p.parse_args();main(a.output,a.continue_from)
