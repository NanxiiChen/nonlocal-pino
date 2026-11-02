"""One finite input screen and independent local-FEM refinement trio; no GPU work."""
import argparse
from datetime import datetime,timezone
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main(config,seed,workers,query_grid):
    from examples.channel_nsch2d.problem import validate_config,identity
    from nonlocal_pinn.training.runtime import with_run_seed
    c=with_run_seed(importlib.import_module(config).get_config(),seed)
    validate_config(c)
    if workers<1 or len(query_grid)!=2 or min(query_grid)<1:
        raise ValueError('Explicit positive worker and query-grid counts required')
    output=Path(c.reference.root)
    if not output.is_absolute()or output.exists():
        raise ValueError('Reference root must be a fresh absolute path')
    fenics=Path('/root/miniconda3/envs/fenicsenv/bin/python')
    if not fenics.is_file():raise FileNotFoundError(fenics)
    def gpu_idle():
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPU work must exit before FEM; no environment overlap')
    gpu_idle()
    output.mkdir(parents=True)
    saved=output/'config.json';saved.write_text(c.to_json(indent=2)+'\n')
    (output/'owner.py').write_bytes(Path(__file__).read_bytes())
    record=dict(status='preflight_passed',runner_pid=os.getpid(),config=config,
        saved_config=str(saved),identity=identity(c),stages=[],active_child_pid=None,
        started_utc=datetime.now(timezone.utc).isoformat(),GPU_work=0,
        scope='Finite input-only screen and three FEM solves; no training, retries or later queue')
    journal=output/'process.json'
    def write():
        temporary=journal.with_suffix('.json.partial')
        temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        os.replace(temporary,journal)
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def execute(stage,command,environment):
        gpu_idle()
        with (output/(stage+'.log')).open('x')as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
            item=dict(stage=stage,pid=child.pid,command=command,
                started_utc=datetime.now(timezone.utc).isoformat())
            record['stages'].append(item)
            record.update(status=stage,active_child_pid=child.pid);write()
            try:code=child.wait()
            finally:
                if child.poll()is None:
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30)
            item.update(exit_code=code,completed_utc=datetime.now(timezone.utc).isoformat())
            record['active_child_pid']=None;write()
            if code:raise RuntimeError(f'{stage} failed with exit{code}')
    cpu=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    fem_env=dict(cpu);fem_env.pop('VIRTUAL_ENV',None)
    fem_env['PATH']=str(fenics.parent)+os.pathsep+cpu['PATH']
    fem_env['CONDA_PREFIX']=str(fenics.parent.parent)
    fem_env['CONDA_DEFAULT_ENV']='fenicsenv'
    screen=output/'construction-screen.json'
    draws=((c.train.iterations-1)//c.operator.online.refresh_every+c.operator.online.prefetch_batches)*c.operator.online.refresh_size
    write()
    try:
        execute('input_screen',[str(ROOT/'.venv/bin/python'),
            'docs/tasks/channel_operator_prior_probe.py','--config='+config,
            '--draws='+str(draws),'--workers='+str(workers),'--output='+str(screen)],cpu)
        from examples.channel_nsch2d.online import validate_construction_screen
        record['construction_screen']=validate_construction_screen(c,screen);write()
        for tag,grid,dt in (
                ('coarse',c.reference.coarse_grid,c.reference.refined_dt),
                ('time',c.reference.fine_grid,c.reference.dt),
                ('fine',c.reference.fine_grid,c.reference.refined_dt)):
            execute('fem_'+tag,[str(fenics),'-m','examples.channel_nsch2d.solver',
                '--saved-config='+str(saved),'--grid',*[str(v)for v in grid],
                '--dt='+str(dt),'--query-grid',*[str(v)for v in query_grid],
                '--upper='+str(c.geometry.upper),'--lower='+str(c.geometry.lower),
                '--output='+str(output/('bdf-'+tag+'.npz'))],fem_env)
        from examples.channel_nsch2d.validate_reference import assess
        gate=assess(c,output/'bdf-coarse.npz',output/'bdf-time.npz',output/'bdf-fine.npz')
        record.update(status='reference_and_input_screen_passed',reference=gate['reference'],
            completed_utc=datetime.now(timezone.utc).isoformat(),active_child_pid=None)
        write()
    except BaseException as error:
        record.update(status='failed_or_interrupted',error=repr(error),active_child_pid=None)
        write();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--workers',type=int,required=True)
    p.add_argument('--query-grid',type=int,nargs=2,required=True)
    a=p.parse_args();main(a.config,a.seed,a.workers,a.query_grid)
